"""Per-request tracing middleware.

Opens one root span per HTTP request so that everything the request touches —
retrieval decisions, LLM calls, graph nodes, human-in-the-loop pauses — lands
under a single trace rather than as orphaned spans.
"""

import logging
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from langfuse import get_client, propagate_attributes
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings

logger = logging.getLogger(__name__)

# Client-supplied correlation IDs. Both are optional; Langfuse groups traces by
# session so a multi-turn document Q&A thread stays together.
SESSION_HEADER = "x-session-id"
USER_HEADER = "x-user-id"
TRACE_HEADER = "x-trace-id"


class TracingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings) -> None:
        super().__init__(app)
        self._settings = settings

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in self._settings.untraced_paths:
            return await call_next(request)

        langfuse = get_client()

        # propagate_attributes wraps the root span so trace-level attributes reach
        # every child span, including ones created deeper in the call stack.
        with propagate_attributes(
            user_id=request.headers.get(USER_HEADER),
            session_id=request.headers.get(SESSION_HEADER),
        ):
            # Named with the raw path for now; renamed to the route template below
            # once routing has resolved, to keep trace names low-cardinality.
            with langfuse.start_as_current_observation(
                name=f"{request.method} {request.url.path}",
                as_type="span",
                # Deliberately *not* the request body: this service ingests PDFs and
                # contracts, and shipping those to a trace backend is both expensive
                # and a data-governance problem. Handlers attach the inputs that matter.
                input={
                    "method": request.method,
                    "path": request.url.path,
                    "query": dict(request.query_params),
                },
            ) as span:
                trace_id = langfuse.get_current_trace_id()
                try:
                    response = await call_next(request)
                except Exception as exc:
                    span.update(level="ERROR", status_message=repr(exc))
                    raise

                # Available only after routing; "/echo/{message}" rather than
                # "/echo/hello-world", so traces aggregate per endpoint.
                route = request.scope.get("route")
                if route is not None and getattr(route, "path", None):
                    span.update(name=f"{request.method} {route.path}")

                span.update(output={"status_code": response.status_code})
                if response.status_code >= 500:
                    span.update(level="ERROR", status_message=f"HTTP {response.status_code}")
                elif response.status_code >= 400:
                    span.update(level="WARNING", status_message=f"HTTP {response.status_code}")

                # Lets a support ticket ("request X was wrong") be resolved straight
                # to the trace that produced it.
                if trace_id:
                    response.headers[TRACE_HEADER] = trace_id

                return response
