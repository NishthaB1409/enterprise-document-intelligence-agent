"""POST /ingest — a PDF becomes retrievable, citable chunks."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from langfuse import get_client
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.api.deps import ServicesDep, SettingsDep
from app.ingest.parser import UnreadableDocument
from app.ingest.pipeline import ingest_pdf
from app.observability.trace_io import publish_trace_io
from app.vectorstore.store import CollectionSchemaError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingestion"])

# Read in 1 MiB bites so the size limit is enforced while the upload streams,
# rather than after the whole thing is already resident.
_READ_CHUNK_BYTES = 1024 * 1024


class IngestResponse(BaseModel):
    # Derived from the file's content, so re-uploading the same PDF returns the
    # same id and replaces the previous copy instead of duplicating it.
    doc_id: str
    source: str
    pages: int
    chunks: int
    trace_id: str | None = None


async def _read_capped(upload: UploadFile, limit: int) -> bytes:
    buffer = bytearray()
    while segment := await upload.read(_READ_CHUNK_BYTES):
        buffer.extend(segment)
        if len(buffer) > limit:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"the upload exceeds the {limit} byte limit",
            )
    return bytes(buffer)


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    request: Request,
    services: ServicesDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="The PDF to index.")],
) -> IngestResponse:
    data = await _read_capped(file, settings.max_upload_bytes)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "the uploaded file is empty")

    filename = file.filename or "upload.pdf"

    try:
        # Parsing and embedding are CPU-bound and synchronous. Off the event
        # loop, or one large upload stalls every other request in the process.
        result = await run_in_threadpool(
            ingest_pdf,
            data,
            filename=filename,
            embedder=services.embedder,
            store=services.store,
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            # None in dense-only mode. Taken from the services rather than the
            # settings so a document is never indexed for a search the running
            # retriever cannot perform.
            sparse_embedder=services.sparse_embedder,
        )
    except UnreadableDocument as exc:
        # The uploader's problem to fix (corrupt, encrypted, or a scan with no
        # text layer), so it gets a 422 and the reason, not a 500.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except CollectionSchemaError as exc:
        # Nothing wrong with the upload — the index on disk predates hybrid
        # retrieval. The operator has to act, so the remedy travels with the
        # error instead of being buried in a 500 and a stack trace.
        logger.error("Cannot ingest into the existing collection: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    publish_trace_io(
        request,
        input={"filename": filename, "bytes": len(data)},
        output={"doc_id": result.doc_id, "pages": result.pages, "chunks": result.chunks},
    )

    return IngestResponse(
        doc_id=result.doc_id,
        source=result.source,
        pages=result.pages,
        chunks=result.chunks,
        trace_id=get_client().get_current_trace_id(),
    )
