"""End-to-end proof that one HTTP request produces one correctly-shaped trace.

These assertions run against protobuf spans decoded off the wire by the fake
Langfuse server — not against mocks — so they fail if the instrumentation stops
exporting, loses its parent/child structure, or drops trace attributes.
"""

from langfuse import LangfuseOtelSpanAttributes as Attr

ECHO_PATH = "/api/v1/echo"


def test_request_is_traced_end_to_end(client, spans, flush):
    response = client.post(ECHO_PATH, json={"message": "  hello langfuse  "})
    flush()

    assert response.status_code == 200
    body = response.json()
    assert body["echoed"] == "HELLO LANGFUSE"

    # The caller gets a trace id back, in the body and as a response header.
    trace_id = body["trace_id"]
    assert trace_id
    assert response.headers["x-trace-id"] == trace_id

    # Every span the request produced actually reached the server.
    assert spans.names(), "no spans were exported"
    root = spans.by_name(f"POST {ECHO_PATH}")
    handler = spans.by_name("echo-handler")
    transform = spans.by_name("transform-message")

    # ...and they form one trace, matching the id handed to the caller.
    assert {root.trace_id, handler.trace_id, transform.trace_id} == {trace_id}

    # ...with the nesting the real pipeline will rely on:
    #   POST /api/v1/echo -> echo-handler -> transform-message
    assert root.parent_span_id is None
    assert handler.parent_span_id == root.span_id
    assert transform.parent_span_id == handler.span_id


def test_root_span_is_named_after_the_route_template(client, spans, flush):
    """Names come from the route, not the concrete URL, so traces aggregate
    per endpoint instead of exploding into one name per path value."""
    client.post(ECHO_PATH, json={"message": "x"})
    flush()

    assert f"POST {ECHO_PATH}" in spans.names()


def test_trace_carries_business_io_not_transport_details(client, spans, flush):
    client.post(ECHO_PATH, json={"message": "contract clause 4.2"})
    flush()

    root = spans.by_name(f"POST {ECHO_PATH}")
    assert root.json_attr(Attr.TRACE_INPUT) == {"message": "contract clause 4.2"}
    assert root.json_attr(Attr.TRACE_OUTPUT) == {"echoed": "CONTRACT CLAUSE 4.2"}


def test_environment_and_release_are_tagged(client, spans, flush):
    """Without these, dev traffic and prod traffic are indistinguishable in one
    Langfuse project, and a regression can't be tied back to a build."""
    client.post(ECHO_PATH, json={"message": "x"})
    flush()

    root = spans.by_name(f"POST {ECHO_PATH}")
    assert root.attr(Attr.ENVIRONMENT) == "test"
    assert root.attr(Attr.RELEASE) == "test-release"


def test_session_and_user_headers_propagate_to_all_spans(client, spans, flush):
    """A multi-turn document Q&A thread has to group under one session, and the
    attributes must reach child spans, not just the root."""
    client.post(
        ECHO_PATH,
        json={"message": "x"},
        headers={"x-session-id": "session-abc", "x-user-id": "user-42"},
    )
    flush()

    for name in (f"POST {ECHO_PATH}", "echo-handler", "transform-message"):
        span = spans.by_name(name)
        assert span.attr(Attr.TRACE_SESSION_ID) == "session-abc", name
        assert span.attr(Attr.TRACE_USER_ID) == "user-42", name


def test_failed_requests_are_traced_as_errors(client, spans, flush):
    """A 4xx/5xx must still produce a trace — the failures are exactly the ones
    worth looking at later."""
    response = client.post(ECHO_PATH, json={"message": ""})  # violates min_length
    flush()

    assert response.status_code == 422
    root = spans.by_name(f"POST {ECHO_PATH}")
    assert root.attr(Attr.OBSERVATION_LEVEL) == "WARNING"
    assert root.json_attr(Attr.OBSERVATION_OUTPUT) == {"status_code": 422}


def test_health_endpoint_is_not_traced(client, spans, flush):
    """Probes fire constantly; tracing them would bury the real traffic."""
    assert client.get("/health").status_code == 200
    flush()

    assert spans.names() == []
