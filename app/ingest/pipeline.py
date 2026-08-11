"""The one path from uploaded bytes to retrievable, citable chunks.

parse -> chunk -> embed -> upsert. Each step already exists on its own; this is
the only place that knows the order, so a route handler never has to.

Ingestion is synchronous and blocking on purpose: parsing and embedding are
CPU-bound, and the route pushes the whole thing to a worker thread rather than
pretending it is async.
"""

import hashlib
import logging
from dataclasses import dataclass

from langfuse import observe

from app.ingest.chunking import build_splitter, chunk_pages
from app.ingest.embedding import Embedder
from app.ingest.parser import parse_pdf
from app.vectorstore.store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IngestResult:
    doc_id: str
    source: str
    pages: int
    chunks: int


def document_id(data: bytes) -> str:
    """Identify a document by its content, not its filename.

    Uploading the same PDF twice — under a different name, or after a failed
    run — re-derives the same id, so the chunk ids are the same and the upsert
    overwrites in place. Two genuinely different documents that happen to share
    a filename stay separate.
    """
    return hashlib.sha256(data).hexdigest()


@observe(name="ingest-document")
def ingest_pdf(
    data: bytes,
    *,
    filename: str,
    embedder: Embedder,
    store: VectorStore,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> IngestResult:
    """Raises `UnreadableDocument` if the bytes are not a PDF with a text layer."""
    doc_id = document_id(data)

    pages = parse_pdf(data)
    chunks = chunk_pages(
        pages, doc_id, build_splitter(chunk_size_words, chunk_overlap_words)
    )

    store.ensure_collection(embedder.dimension)
    # Re-ingesting with different chunk settings produces a different number of
    # chunks; without this the leftovers from the previous run stay searchable
    # and the same passage gets cited twice under two ids.
    store.delete_document(doc_id)

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    store.upsert(chunks, vectors, source=filename)

    logger.info(
        "Ingested %r (doc_id=%s): %d pages, %d chunks", filename, doc_id, len(pages), len(chunks)
    )
    return IngestResult(doc_id=doc_id, source=filename, pages=len(pages), chunks=len(chunks))
