"""A deliberately trivial endpoint that exercises the full tracing path.

There is no real logic here yet. Its job is to prove the observability wiring
end to end before any of it exists: one HTTP request produces one trace, with
nested spans, trace-level input/output, and a trace id handed back to the caller.

The span shape it produces is the same shape the real pipeline will use:

    POST /api/v1/echo              <- root span, opened by TracingMiddleware
    └── echo-handler               <- the unit of work
        └── transform-message      <- a nested step

Delete this endpoint once real endpoints cover the same ground.
"""

from fastapi import APIRouter, Request
from langfuse import get_client, observe
from pydantic import BaseModel, Field

from app.observability.trace_io import publish_trace_io

router = APIRouter(tags=["diagnostics"])


class EchoRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


class EchoResponse(BaseModel):
    echoed: str
    # Returned so a caller (or a test) can look this exact request up in Langfuse.
    trace_id: str | None = None


@observe(name="transform-message")
def _transform(message: str) -> str:
    """A nested step, so the trace has depth rather than a single flat span."""
    return message.strip().upper()


@observe(name="echo-handler")
def _handle(message: str) -> str:
    return _transform(message)


@router.post("/echo", response_model=EchoResponse)
async def echo(payload: EchoRequest, request: Request) -> EchoResponse:
    langfuse = get_client()

    echoed = _handle(payload.message)

    # What shows up in the Langfuse trace list should be the request's meaning,
    # not its transport details. This writes to the root span, which carries the
    # trace's I/O in Langfuse v4 (`set_current_trace_io` is deprecated there).
    publish_trace_io(request, input={"message": payload.message}, output={"echoed": echoed})

    return EchoResponse(echoed=echoed, trace_id=langfuse.get_current_trace_id())
