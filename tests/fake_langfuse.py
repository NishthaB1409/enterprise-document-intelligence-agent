"""A stand-in Langfuse server that captures what the SDK actually puts on the wire.

Langfuse exports spans as protobuf OTLP over HTTP to
`{host}/api/public/otel/v1/traces`. This server speaks just enough of that to
accept the export and decode it, which lets the test suite assert on real
serialised spans rather than on mocks. If the instrumentation stops emitting a
span, or emits it with the wrong parent or attributes, the tests fail here.
"""

import gzip
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

OTEL_TRACES_PATH = "/api/public/otel/v1/traces"
PROJECTS_PATH = "/api/public/projects"


@dataclass
class CapturedSpan:
    name: str
    span_id: str
    parent_span_id: str | None
    trace_id: str
    attributes: dict[str, Any]

    def attr(self, key: str) -> Any:
        return self.attributes.get(key)

    def json_attr(self, key: str) -> Any:
        """Langfuse serialises structured input/output attributes as JSON strings."""
        raw = self.attributes.get(key)
        return json.loads(raw) if isinstance(raw, str) else raw


@dataclass
class SpanCollector:
    spans: list[CapturedSpan] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, span: CapturedSpan) -> None:
        with self._lock:
            self.spans.append(span)

    def by_name(self, name: str) -> CapturedSpan:
        with self._lock:
            matches = [s for s in self.spans if s.name == name]
        if not matches:
            available = sorted(s.name for s in self.spans)
            raise AssertionError(f"No span named {name!r}. Captured: {available}")
        if len(matches) > 1:
            raise AssertionError(f"Expected one span named {name!r}, got {len(matches)}")
        return matches[0]

    def names(self) -> list[str]:
        with self._lock:
            return [s.name for s in self.spans]


def _attr_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue into a plain Python value."""
    which = value.WhichOneof("value")
    if which == "string_value":
        return value.string_value
    if which == "bool_value":
        return value.bool_value
    if which == "int_value":
        return value.int_value
    if which == "double_value":
        return value.double_value
    if which == "array_value":
        return [_attr_value(v) for v in value.array_value.values]
    return None


def _make_handler(collector: SpanCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep pytest output readable
            pass

        def do_GET(self) -> None:
            # `Langfuse.auth_check()` calls this at startup.
            if self.path.rstrip("/").endswith(PROJECTS_PATH.rstrip("/")):
                # Must satisfy the SDK's Project model or auth_check raises a
                # validation error rather than returning False.
                body = json.dumps(
                    {
                        "data": [
                            {
                                "id": "test-project",
                                "name": "test-project",
                                "organization": {"id": "test-org", "name": "test-org"},
                                "metadata": {},
                            }
                        ]
                    }
                ).encode()
                self._respond(200, body, "application/json")
            else:
                self._respond(404, b"{}", "application/json")

        def do_POST(self) -> None:
            if not self.path.startswith(OTEL_TRACES_PATH):
                self._respond(404, b"", "application/json")
                return

            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)

            request = ExportTraceServiceRequest()
            request.ParseFromString(body)

            for resource_span in request.resource_spans:
                for scope_span in resource_span.scope_spans:
                    for span in scope_span.spans:
                        collector.add(
                            CapturedSpan(
                                name=span.name,
                                span_id=span.span_id.hex(),
                                parent_span_id=span.parent_span_id.hex() or None,
                                trace_id=span.trace_id.hex(),
                                attributes={
                                    a.key: _attr_value(a.value) for a in span.attributes
                                },
                            )
                        )

            self._respond(
                200, ExportTraceServiceResponse().SerializeToString(), "application/x-protobuf"
            )

        def _respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class FakeLangfuseServer:
    """Context manager yielding a running server plus the spans it captured."""

    def __init__(self) -> None:
        self.collector = SpanCollector()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.collector))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeLangfuseServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
