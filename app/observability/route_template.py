"""Resolve the full route template for a request.

Span names must be low-cardinality: `/documents/{doc_id}` rather than one distinct
name per document id, or the trace list becomes unusable and per-endpoint
aggregation is impossible.

`request.scope["route"].path` is *not* that template. Under Starlette 1.6 /
FastAPI 0.141, a router included with a prefix stores the router-local path on the
route ("/echo"), exposes no prefix attribute, and leaves `root_path` empty — so
the prefix has to be recovered from the concrete path.

The route template always matches the *tail* of the concrete path, so the leading
segments are exactly the prefix. That relationship holds regardless of how many
routers are nested, which makes this independent of the framework internals that
changed here.
"""

from starlette.requests import Request


def resolve_route_template(request: Request) -> str | None:
    """Return e.g. "/api/v1/echo/{msg}", or None if routing has not resolved."""
    route = request.scope.get("route")
    template = getattr(route, "path_format", None) or getattr(route, "path", None)
    if not template:
        return None

    concrete = request.url.path
    if concrete == template:
        return template

    template_segments = [s for s in template.split("/") if s]
    concrete_segments = [s for s in concrete.split("/") if s]

    # A template longer than the path it supposedly matched means our assumption
    # is broken; fall back rather than emit a nonsense name.
    if len(template_segments) > len(concrete_segments):
        return template

    prefix_segments = concrete_segments[: len(concrete_segments) - len(template_segments)]
    combined = prefix_segments + template_segments
    return "/" + "/".join(combined) if combined else "/"
