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


def test_an_uncited_claim_is_reported_rather_than_hidden(
    client: TestClient, answerer: StubAnswerer
):
    _upload(client, build_pdf(CONTRACT))
    answerer.respond = lambda question, chunks: GeneratedAnswer.model_validate(
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

    body = client.post("/api/v1/query", json={"question": "What are the terms?"}).json()

    assert body["unsupported_claims"] == ["Penalties accrue at 5% monthly."]
    assert body["claims"][1]["citations"] == []


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
    override_settings(anthropic_api_key=None)

    response = client.post("/api/v1/query", json={"question": "anything"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


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
