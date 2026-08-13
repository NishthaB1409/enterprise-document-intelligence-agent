"""POST /query — retrieve, answer, and hand back the evidence.

The response is built so a reviewer never has to take the answer on trust:
`claims` says which sentence rests on which source, `citations` says exactly
where each source is (document, page, character span), and `unsupported_claims`
names anything the model asserted without resolvable backing.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from langfuse import get_client
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.api.deps import ServicesDep, SettingsDep
from app.generation.answerer import GenerationError
from app.generation.citations import ground
from app.observability.trace_io import publish_trace_io

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # Overrides `retrieval_top_k` for one request. Useful when demoing recall
    # against precision; the ceiling keeps a single query from sending the whole
    # corpus to the model.
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    doc_id: str
    source: str
    page: int
    # -1 when the exact span could not be located in the page text; the page
    # number is still exact. See `app.ingest.chunking`.
    char_start: int
    char_end: int
    score: float
    text: str


class Claim(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    text: str
    citations: list[Citation]


class QueryResponse(BaseModel):
    answer: str
    # False means the retrieved documents do not contain the answer — a correct
    # outcome, and a different thing from an answer that happens to be short.
    answerable: bool
    claims: list[Claim]
    citations: list[Citation]
    unsupported_claims: list[str]
    trace_id: str | None = None


@router.post("/query", response_model=QueryResponse)
async def query(
    payload: QueryRequest,
    request: Request,
    services: ServicesDep,
    settings: SettingsDep,
) -> QueryResponse:
    if not settings.generation_configured:
        # Checked up front: ingestion works without an LLM key, so an
        # otherwise-healthy deployment can reach this endpoint unconfigured.
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"answer generation is not configured; set {settings.generation_key_variable}",
        )

    # Both are blocking calls (ONNX inference, then an HTTP round trip to the
    # model), so both go to a worker thread rather than blocking the loop.
    chunks = await run_in_threadpool(
        services.retriever.retrieve, payload.question, payload.top_k
    )

    try:
        generated = await run_in_threadpool(services.answerer.answer, payload.question, chunks)
    except GenerationError as exc:
        logger.warning("Answer generation failed: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    # `chunks` must be the same sequence, in the same order, that the answerer
    # was given — the model's source numbers are positions in it.
    grounded = ground(generated, chunks)

    publish_trace_io(
        request,
        input={"question": payload.question, "retrieved": len(chunks)},
        output={
            "answer": grounded.answer,
            "answerable": grounded.answerable,
            "citations": len(grounded.citations),
            "unsupported_claims": len(grounded.unsupported_claims),
        },
    )

    return QueryResponse(
        answer=grounded.answer,
        answerable=grounded.answerable,
        claims=[Claim.model_validate(claim) for claim in grounded.claims],
        citations=[Citation.model_validate(citation) for citation in grounded.citations],
        unsupported_claims=grounded.unsupported_claims,
        trace_id=get_client().get_current_trace_id(),
    )
