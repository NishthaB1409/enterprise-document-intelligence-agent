"""Let a handler publish the *meaning* of a request to its trace.

In Langfuse v4 the root span carries the trace's input/output (the older
`set_current_trace_io` is deprecated and slated for removal). The root span is
opened by the middleware, before routing, where the only thing knowable is
transport detail — method, path, query.

So the middleware stashes the root span on `request.state`, and a handler that
knows what the request actually *means* publishes it here. Handlers that don't
call this fall back to the transport summary.

This is opt-in on purpose: this service ingests PDFs and contracts, and shipping
raw document bytes to a trace backend is both expensive and a data-governance
problem. Handlers publish the distilled inputs worth keeping.
"""

from typing import Any

from starlette.requests import Request

_ROOT_SPAN = "langfuse_root_span"
_PUBLISHED = "langfuse_trace_io_published"


def attach_root_span(request: Request, span: Any) -> None:
    """Called by the tracing middleware, not by application code."""
    setattr(request.state, _ROOT_SPAN, span)


def trace_io_published(request: Request) -> bool:
    return getattr(request.state, _PUBLISHED, False)


def publish_trace_io(
    request: Request, *, input: Any = None, output: Any = None
) -> None:
    """Set the trace's headline input/output from inside a route handler."""
    span = getattr(request.state, _ROOT_SPAN, None)
    if span is None:
        # Untraced path, or tracing disabled. Not an error.
        return

    fields = {k: v for k, v in (("input", input), ("output", output)) if v is not None}
    if fields:
        span.update(**fields)
        setattr(request.state, _PUBLISHED, True)
