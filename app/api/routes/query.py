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
from app.generation.citations import ground
from app.graph.pipeline import describe
from app.llm import LLMError
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


class Critique(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    supported: bool
    confidence: float
    concerns: list[str]


class QueryResponse(BaseModel):
    answer: str
    # False means the retrieved documents do not contain the answer — a correct
    # outcome, and a different thing from an answer that happens to be short.
    answerable: bool
    claims: list[Claim]
    citations: list[Citation]
    unsupported_claims: list[str]
    # Null when the answer was not reviewed — either the agentic graph is off,
    # or its critic could not be reached. Not the same as reviewed and clean,
    # which is what phase 4's gate will need to distinguish.
    critique: Critique | None = None
    # Which nodes ran, in order. The cheapest way to explain a surprising
    # answer: a thin response after `grade:0/5` is a retrieval problem, the
    # same response after `route:direct` is the router declining.
    steps: list[str] = Field(default_factory=list)
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

    # The whole pipeline is blocking (ONNX inference, then one or more HTTP
    # round trips to the model), so it goes to a worker thread rather than
    # blocking the loop. One call whether it is the single pass or the graph.
    try:
        result = await run_in_threadpool(
            services.pipeline.run, payload.question, payload.top_k
        )
    except LLMError as exc:
        logger.warning("Answer generation failed: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    # `result.chunks` is what the generator actually saw, which is not always
    # what retrieval returned — the grader may have dropped some. The model's
    # source numbers are positions in this list, so grounding has to use it.
    grounded = ground(result.answer, result.chunks)

    if result.dropped:
        logger.info("Grader dropped %d source(s): %s", len(result.dropped), result.dropped)

    publish_trace_io(
        request,
        input={"question": payload.question, "retrieved": len(result.chunks)},
        output={
            "answer": grounded.answer,
            "answerable": grounded.answerable,
            "citations": len(grounded.citations),
            "unsupported_claims": len(grounded.unsupported_claims),
            "path": describe(result.steps),
            "confidence": result.critique.confidence if result.critique else None,
        },
    )

    return QueryResponse(
        answer=grounded.answer,
        answerable=grounded.answerable,
        claims=[Claim.model_validate(claim) for claim in grounded.claims],
        citations=[Citation.model_validate(citation) for citation in grounded.citations],
        unsupported_claims=grounded.unsupported_claims,
        critique=Critique.model_validate(result.critique) if result.critique else None,
        steps=result.steps,
        trace_id=get_client().get_current_trace_id(),
    )
