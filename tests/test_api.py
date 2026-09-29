"""The two endpoints, end to end, over the real pipeline with stub leaves."""

import pytest
from fastapi.testclient import TestClient

from app.generation.answerer import GeneratedAnswer, GenerationError
from tests.fakes import InMemoryVectorStore, StubAnswerer
from tests.pdfs import build_pdf

CONTRACT = [
    "Section 1. Term. This agreement begins on 1 January 2026.",
    "Section 2. Termination. Either party may terminate on thirty days notice.",
]


def _upload(client: TestClient, data: bytes, filename: str = "contract.pdf"):
    return client.post(
        "/api/v1/ingest",
        files={"file": (filename, data, "application/pdf")},
    )


@pytest.fixture
def override_settings(client: TestClient):
    """Swap in adjusted settings for one test. The app is session-scoped, so the
    original has to go back afterwards."""
    original = client.app.state.settings

    def _override(**changes):
        client.app.state.settings = original.model_copy(update=changes)

    yield _override
    client.app.state.settings = original


# --- the front door ---------------------------------------------------------


def test_the_root_url_sends_you_to_the_docs(client: TestClient):
    """A bare 404 on `/` is the same thing a stopped server looks like from a
    browser, so someone checking whether the service is up gets told it is not.
    The redirect costs nothing and answers the question actually being asked."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 302)
    assert response.headers["location"] == "/docs"


def test_the_docs_are_actually_there(client: TestClient):
    """The redirect is only useful if its destination exists — and `/docs`
    disappears the moment someone sets `openapi_url=None`."""
    assert client.get("/docs").status_code == 200


# --- ingestion --------------------------------------------------------------


def test_ingest_returns_what_was_indexed(client: TestClient, store: InMemoryVectorStore):
    response = _upload(client, build_pdf(CONTRACT))

    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 2
    assert body["chunks"] == len(store.points) > 0
    assert body["source"] == "contract.pdf"
    assert body["doc_id"]
    # Lets a support ticket about a bad ingest be traced to the run that did it.
    assert body["trace_id"]


def test_ingest_is_idempotent_for_the_same_file(client: TestClient, store: InMemoryVectorStore):
    data = build_pdf(CONTRACT)

    first = _upload(client, data).json()
    second = _upload(client, data, filename="contract-copy.pdf").json()

    assert second["doc_id"] == first["doc_id"]
    assert len(store.points) == first["chunks"]


def test_ingesting_a_scan_is_the_uploaders_error_not_a_server_fault(client: TestClient):
    response = _upload(client, build_pdf(["", ""]))

    assert response.status_code == 422
    assert "OCR" in response.json()["detail"]


def test_ingesting_a_non_pdf_is_rejected(client: TestClient):
    response = _upload(client, b"just some bytes", filename="notes.txt")

    assert response.status_code == 422


def test_an_empty_upload_is_rejected(client: TestClient):
    assert _upload(client, b"").status_code == 400


def test_an_oversized_upload_is_refused(client: TestClient, override_settings):
    override_settings(max_upload_bytes=512)

    response = _upload(client, build_pdf(CONTRACT))

    assert response.status_code == 413
    assert "512" in response.json()["detail"]


# --- query ------------------------------------------------------------------


def test_query_answers_with_resolvable_citations(client: TestClient, answerer: StubAnswerer):
    _upload(client, build_pdf(CONTRACT))

    response = client.post(
        "/api/v1/query", json={"question": "What notice is needed to terminate?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answerable"] is True
    assert body["unsupported_claims"] == []

    (claim,) = body["claims"]
    (citation,) = claim["citations"]
    # Every field a reviewer needs to go and check the source themselves.
    assert citation["source"] == "contract.pdf"
    assert citation["page"] in {1, 2}
    assert citation["text"]
    assert citation["chunk_id"]
    assert body["citations"][0]["chunk_id"] == citation["chunk_id"]

    # The model was given the retrieved chunks, and nothing else.
    question, chunks = answerer.calls[-1]
    assert question == "What notice is needed to terminate?"
    assert chunks and all(chunk.source == "contract.pdf" for chunk in chunks)


def test_top_k_bounds_what_the_model_is_shown(client: TestClient, answerer: StubAnswerer):
    _upload(client, build_pdf(CONTRACT))

    client.post("/api/v1/query", json={"question": "termination", "top_k": 1})

    _, chunks = answerer.calls[-1]
    assert len(chunks) == 1


def _answer_with_an_uncited_claim(question, chunks) -> GeneratedAnswer:
    return GeneratedAnswer.model_validate(
        {
            "answer": "Notice is thirty days and penalties accrue at 5%.",
            "answerable": True,
            "claims": [
                {"text": "Notice is thirty days.", "sources": [1]},
                # Cites a source that was never offered.
                {"text": "Penalties accrue at 5% monthly.", "sources": [99]},
            ],
        }
    )


def test_an_uncited_claim_is_reported_rather_than_hidden(
    client: TestClient, answerer: StubAnswerer, override_settings
):
    # Review off, so the grounding report itself is what comes back. With it on
    # the same answer is held — see the review tests below.
    override_settings(review_enabled=False)
    _upload(client, build_pdf(CONTRACT))
    answerer.respond = _answer_with_an_uncited_claim

    response = client.post("/api/v1/query", json={"question": "What are the terms?"})

    assert response.status_code == 200
    body = response.json()
    assert body["unsupported_claims"] == ["Penalties accrue at 5% monthly."]
    assert body["claims"][1]["citations"] == []
    assert body["review"] is None


def test_querying_an_empty_corpus_says_so_without_calling_the_model(
    client: TestClient, answerer: StubAnswerer
):
    body = client.post("/api/v1/query", json={"question": "What is the notice period?"}).json()

    assert body["answerable"] is False
    assert body["citations"] == []
    _, chunks = answerer.calls[-1]
    assert chunks == []


def test_query_without_a_model_key_is_unavailable_not_broken(
    client: TestClient, override_settings
):
    override_settings(openai_api_key=None)

    response = client.post("/api/v1/query", json={"question": "anything"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_a_failing_model_is_reported_as_an_upstream_failure(
    client: TestClient, answerer: StubAnswerer
):
    _upload(client, build_pdf(CONTRACT))

    def _fail(question, chunks):
        raise GenerationError("the answer exceeded max_tokens (8000) and was cut off")

    answerer.respond = _fail

    response = client.post("/api/v1/query", json={"question": "What are the terms?"})

    # The request was fine; the dependency was not.
    assert response.status_code == 502
    assert "max_tokens" in response.json()["detail"]


def test_an_empty_question_is_rejected(client: TestClient):
    assert client.post("/api/v1/query", json={"question": ""}).status_code == 422
    assert client.post("/api/v1/query", json={}).status_code == 422


# --- human review -------------------------------------------------------------


def _held(client: TestClient, answerer: StubAnswerer, question: str = "What are the terms?"):
    _upload(client, build_pdf(CONTRACT))
    answerer.respond = _answer_with_an_uncited_claim
    return client.post("/api/v1/query", json={"question": question})


def test_a_clean_answer_is_released_without_review(client: TestClient):
    _upload(client, build_pdf(CONTRACT))

    response = client.post("/api/v1/query", json={"question": "termination notice"})

    assert response.status_code == 200
    assert response.json()["review"] is None
    assert client.get("/api/v1/reviews").json() == []


def test_an_answer_with_an_uncited_claim_is_held(client: TestClient, answerer: StubAnswerer):
    response = _held(client, answerer)

    assert response.status_code == 202
    body = response.json()
    review = body["review"]
    assert review["status"] == "pending"
    assert review["reasons"] == [
        "claim cites no retrieved source: 'Penalties accrue at 5% monthly.'"
    ]
    # Nothing the requester could act on before a human has looked.
    assert review["id"] in body["answer"]
    assert "penalties" not in body["answer"].lower()
    assert body["claims"] == body["citations"] == body["unsupported_claims"] == []
    # The path and trace still come back: they explain the hold, not the answer.
    assert body["steps"] == ["retrieve", "generate"]
    assert body["trace_id"]


def test_the_reviewer_sees_the_whole_answer_and_why_it_was_held(
    client: TestClient, answerer: StubAnswerer
):
    review_id = _held(client, answerer).json()["review"]["id"]

    review = client.get(f"/api/v1/reviews/{review_id}").json()

    assert review["status"] == "pending"
    assert review["question"] == "What are the terms?"
    assert review["reasons"][0].startswith("claim cites no retrieved source")
    released = review["response"]
    assert released["answer"] == "Notice is thirty days and penalties accrue at 5%."
    assert released["unsupported_claims"] == ["Penalties accrue at 5% monthly."]
    assert released["claims"][0]["citations"][0]["source"] == "contract.pdf"
    assert released["review"] is None


def test_the_queue_lists_pending_reviews_oldest_first(
    client: TestClient, answerer: StubAnswerer
):
    first = _held(client, answerer, "first question").json()["review"]["id"]
    second = _held(client, answerer, "second question").json()["review"]["id"]

    pending = client.get("/api/v1/reviews").json()

    assert [r["id"] for r in pending] == [first, second]


def test_approving_releases_the_stored_answer(client: TestClient, answerer: StubAnswerer):
    review_id = _held(client, answerer).json()["review"]["id"]

    response = client.post(
        f"/api/v1/reviews/{review_id}/decision",
        json={"decision": "approve", "reviewer": "legal-ops", "note": "5% is in schedule B"},
    )

    assert response.status_code == 200
    review = response.json()
    assert review["status"] == "approved"
    assert review["reviewer"] == "legal-ops"
    assert review["decided_at"]
    # The requester polls the same resource and now gets the answer.
    fetched = client.get(f"/api/v1/reviews/{review_id}").json()
    assert fetched["status"] == "approved"
    assert fetched["response"]["answer"].startswith("Notice is thirty days")
    # Decided reviews leave the pending queue but are still listable.
    assert client.get("/api/v1/reviews").json() == []
    assert [r["id"] for r in client.get("/api/v1/reviews?status=approved").json()] == [
        review_id
    ]


def test_a_decision_is_final(client: TestClient, answerer: StubAnswerer):
    """Two reviewers must not silently overrule each other."""
    review_id = _held(client, answerer).json()["review"]["id"]
    url = f"/api/v1/reviews/{review_id}/decision"

    client.post(url, json={"decision": "reject", "reviewer": "alice"})
    second = client.post(url, json={"decision": "approve", "reviewer": "bob"})

    assert second.status_code == 409
    assert "alice" in second.json()["detail"]
    assert client.get(f"/api/v1/reviews/{review_id}").json()["status"] == "rejected"


def test_a_decision_needs_a_reviewer(client: TestClient, answerer: StubAnswerer):
    review_id = _held(client, answerer).json()["review"]["id"]

    response = client.post(
        f"/api/v1/reviews/{review_id}/decision", json={"decision": "approve", "reviewer": ""}
    )

    assert response.status_code == 422


def test_an_unknown_review_is_not_found(client: TestClient):
    assert client.get("/api/v1/reviews/nope").status_code == 404
    assert (
        client.post(
            "/api/v1/reviews/nope/decision", json={"decision": "approve", "reviewer": "x"}
        ).status_code
        == 404
    )


def test_an_honest_no_answer_is_not_held(client: TestClient):
    """Declining for lack of evidence is the correct outcome. Queueing it would
    teach reviewers that most of the queue needs no attention."""
    response = client.post("/api/v1/query", json={"question": "What is the notice period?"})

    assert response.status_code == 200
    assert response.json()["answerable"] is False
    assert response.json()["review"] is None
