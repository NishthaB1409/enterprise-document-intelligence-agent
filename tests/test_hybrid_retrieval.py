"""Hybrid retrieval: does adding BM25 actually find what dense search missed?

Vectors are built by hand rather than produced by an embedder. The claim under
test is about *fusion* — that a chunk ranked poorly by one arm and well by the
other survives — and a test that went through a real embedder would be asserting
that claim only indirectly, through whatever the model happened to think of the
wording. Here the two rankings are exactly what the test says they are.

`eval/` is where the same question gets answered against real embeddings and a
real corpus. This file pins the mechanism; that measures the benefit.
"""

import pytest

from app.ingest.chunking import Chunk
from app.ingest.sparse import SparseVector
from app.retrieval.retriever import DenseRetriever, HybridRetriever, RerankingRetriever
from tests.fakes import InMemoryVectorStore, StubReranker

DIMENSION = 4


def _chunk(index: int, text: str = "", doc_id: str = "doc-1") -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        doc_id=doc_id,
        index=index,
        page=1,
        text=text or f"Clause {index}.",
        char_start=0,
        char_end=len(text or f"Clause {index}."),
    )


def _dense(*values: float) -> list[float]:
    return list(values)


class _FixedEmbedder:
    """Returns one preset vector for any question.

    The documents' vectors are put into the store directly, so this only has to
    supply the query side — which is what makes each arm's ranking predictable.
    """

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    @property
    def dimension(self) -> int:
        return DIMENSION

    def embed_documents(self, texts):
        raise AssertionError("documents are indexed directly in these tests")

    def embed_query(self, text: str) -> list[float]:
        return self._vector


class _FixedSparseEmbedder:
    def __init__(self, vector: SparseVector) -> None:
        self._vector = vector

    def embed_documents(self, texts):
        raise AssertionError("documents are indexed directly in these tests")

    def embed_query(self, text: str) -> SparseVector:
        return self._vector


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


def test_hybrid_surfaces_a_chunk_that_dense_search_ranks_out_of_reach(store):
    """The reason this phase exists.

    `lexical-match` is the chunk containing the exact term the question asks
    about, and dense search ranks it last of four — the failure mode BM25 is
    there to cover. Three decoys sit above it in vector space and none of them
    contain the term.
    """
    query_dense = _dense(1.0, 0.0, 0.0, 0.0)
    # Decoys: close to the query in vector space, no term overlap.
    for index in range(3):
        store.upsert(
            [_chunk(index, f"Topically adjacent passage {index}.")],
            [_dense(0.99 - index * 0.01, 0.1, 0.0, 0.0)],
            [SparseVector(indices=[900 + index], values=[1.0])],
            source="decoy.pdf",
        )
    # The answer: nearly orthogonal to the query vector, exact term match.
    store.upsert(
        [_chunk(9, "The indemnification cap is set out in Section 12.4.")],
        [_dense(0.0, 0.0, 0.0, 1.0)],
        [SparseVector(indices=[42], values=[3.0])],
        source="contract.pdf",
    )

    embedder = _FixedEmbedder(query_dense)
    sparse = _FixedSparseEmbedder(SparseVector(indices=[42], values=[1.0]))

    dense_hits = DenseRetriever(embedder, store, top_k=2).retrieve("Section 12.4 cap?")
    hybrid_hits = HybridRetriever(
        embedder, sparse, store, top_k=2, candidates=10
    ).retrieve("Section 12.4 cap?")

    # Dense alone cannot see it: it is 4th of 4, and top_k is 2.
    assert "chunk-9" not in {hit.chunk.id for hit in dense_hits}
    # Fusion promotes it on the strength of being the only lexical match.
    assert hybrid_hits[0].chunk.id == "chunk-9"


def test_a_chunk_both_arms_like_beats_one_that_only_one_arm_likes(store):
    """RRF rewards agreement, which is the whole point of fusing rather than
    concatenating two result lists."""
    store.upsert(
        [_chunk(1, "Ranked well by both.")],
        [_dense(0.9, 0.1, 0.0, 0.0)],
        [SparseVector(indices=[7], values=[5.0])],
        source="a.pdf",
    )
    store.upsert(
        [_chunk(2, "Dense only.")],
        [_dense(1.0, 0.0, 0.0, 0.0)],
        [SparseVector(indices=[999], values=[1.0])],
        source="b.pdf",
    )

    hits = HybridRetriever(
        _FixedEmbedder(_dense(1.0, 0.0, 0.0, 0.0)),
        _FixedSparseEmbedder(SparseVector(indices=[7], values=[1.0])),
        store,
        top_k=2,
        candidates=10,
    ).retrieve("anything")

    # chunk-2 wins the dense arm outright, but chunk-1 places in both.
    assert [hit.chunk.id for hit in hits] == ["chunk-1", "chunk-2"]


def test_a_question_of_only_stopwords_falls_back_to_dense(store):
    """An empty sparse vector means BM25 has nothing to match. Fusing it would
    re-rank the dense list at the cost of an extra search, so the retriever
    should skip it — and must still return dense results, not nothing."""
    store.upsert(
        [_chunk(1, "Some content.")],
        [_dense(1.0, 0.0, 0.0, 0.0)],
        [SparseVector(indices=[7], values=[1.0])],
        source="a.pdf",
    )

    hits = HybridRetriever(
        _FixedEmbedder(_dense(1.0, 0.0, 0.0, 0.0)),
        _FixedSparseEmbedder(SparseVector(indices=[], values=[])),
        store,
        top_k=3,
        candidates=10,
    ).retrieve("of the and")

    assert [hit.chunk.id for hit in hits] == ["chunk-1"]


def test_a_dense_only_document_stays_invisible_to_the_lexical_arm(store):
    """Ingesting without a sparse embedder is allowed, and the consequence has
    to be the one documented: still retrievable, never via BM25."""
    store.upsert(
        [_chunk(1, "Indexed without sparse vectors.")],
        [_dense(1.0, 0.0, 0.0, 0.0)],
        None,
        source="legacy.pdf",
    )

    assert store.sparse_search(SparseVector(indices=[42], values=[1.0]), top_k=5) == []
    # ...but dense search still finds it.
    assert len(store.search(_dense(1.0, 0.0, 0.0, 0.0), top_k=5)) == 1


class TestRerankingHandoff:
    """What the reranker is handed, and what happens to what it returns."""

    def _retriever(self, store, reranker, *, top_k=2, candidates=8):
        hybrid = HybridRetriever(
            _FixedEmbedder(_dense(1.0, 0.0, 0.0, 0.0)),
            _FixedSparseEmbedder(SparseVector(indices=[7], values=[1.0])),
            store,
            top_k,
            candidates=candidates,
        )
        if reranker is None:
            return hybrid
        return RerankingRetriever(hybrid, reranker, top_k, candidates=candidates)

    def _index(self, store, count: int) -> None:
        for index in range(count):
            store.upsert(
                [_chunk(index, f"Passage {index}.")],
                [_dense(1.0 - index * 0.05, 0.0, 0.0, 0.0)],
                [SparseVector(indices=[7], values=[float(count - index)])],
                source="corpus.pdf",
            )

    def test_the_reranker_sees_the_candidate_pool_not_the_final_k(self, store):
        """A reranker handed only top_k could reorder those k but never rescue
        the chunk sitting just below the cutoff — which is most of its value."""
        self._index(store, count=6)
        reranker = StubReranker()

        self._retriever(store, reranker, top_k=2, candidates=6).retrieve("q")

        (_, candidates), = reranker.calls
        assert len(candidates) == 6

    def test_the_reranked_order_is_what_the_caller_gets(self, store):
        """The stub reverses the fused order, so if the caller's list is still
        in fusion order the reranker's output was silently discarded."""
        self._index(store, count=4)
        fused = self._retriever(store, None, top_k=4, candidates=4).retrieve("q")
        reranked = self._retriever(store, StubReranker(), top_k=4, candidates=4).retrieve("q")

        assert [h.chunk.id for h in reranked] == [h.chunk.id for h in fused][::-1]

    def test_scores_come_from_the_reranker_not_from_fusion(self, store):
        self._index(store, count=3)
        reranker = StubReranker(score=lambda question, chunk: 42.0)

        hits = self._retriever(store, reranker, top_k=3, candidates=3).retrieve("q")

        assert [hit.score for hit in hits] == [42.0, 42.0, 42.0]

    def test_a_per_request_top_k_above_the_candidate_pool_still_fills(self, store):
        """`candidates` is a floor, not a ceiling: asking for more results than
        the configured pool must widen the search rather than truncate."""
        self._index(store, count=6)

        hits = self._retriever(store, None, top_k=2, candidates=3).retrieve("q", top_k=5)

        assert len(hits) == 5
