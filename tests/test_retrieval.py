"""Retrieval: does the question actually reach the right chunk?"""

from app.ingest.embedding import Embedder
from app.ingest.pipeline import ingest_pdf
from app.retrieval.retriever import DenseRetriever
from app.vectorstore.store import VectorStore
from tests.fakes import InMemoryVectorStore, StubEmbedder
from tests.pdfs import build_pdf

PAGES = [
    "Section 2. Termination. Either party may terminate this agreement on thirty days notice.",
    "Section 9. Governing law. This agreement is governed by the laws of Delaware.",
]


def _indexed() -> tuple[Embedder, VectorStore]:
    embedder = StubEmbedder()
    store = InMemoryVectorStore()
    ingest_pdf(
        build_pdf(PAGES),
        filename="contract.pdf",
        embedder=embedder,
        store=store,
        chunk_size_words=40,
        chunk_overlap_words=5,
    )
    return embedder, store


def test_the_relevant_chunk_ranks_first():
    embedder, store = _indexed()

    hits = DenseRetriever(embedder, store, top_k=5).retrieve(
        "What notice is required to terminate?"
    )

    assert "terminate" in hits[0].chunk.text
    assert hits[0].score > hits[-1].score


def test_top_k_bounds_what_reaches_the_model():
    embedder, store = _indexed()
    retriever = DenseRetriever(embedder, store, top_k=1)

    assert len(retriever.retrieve("termination notice")) == 1
    # The per-request override wins over the configured default.
    assert len(retriever.retrieve("termination notice", top_k=2)) == 2


def test_an_empty_index_returns_nothing_rather_than_failing():
    retriever = DenseRetriever(StubEmbedder(), InMemoryVectorStore(), top_k=5)

    assert retriever.retrieve("anything at all") == []
