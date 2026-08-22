"""What reaches the model, and how it got chosen.

Two retrievers behind one Protocol, and a wrapper that composes with either:

    DenseRetriever      embed the question, take the nearest k.
    HybridRetriever     dense + BM25, fused by reciprocal rank.
    RerankingRetriever  wraps either, re-scoring a wider shortlist.

Which combination is actually best is a question about your corpus, not a
question with a universal answer — see `eval/README.md` for what it measured on
this one, and `Settings.retrieval_mode` for how to switch. The pieces are kept
separable so that stays measurable rather than becoming folklore.

Questions go through `embed_query`, not `embed_documents` — see the note in
`app.ingest.embedding` about what asymmetric models do to recall when a question
is embedded as if it were a passage.
"""

from typing import Protocol, runtime_checkable

from langfuse import get_client, observe

from app.ingest.embedding import Embedder
from app.ingest.sparse import SparseEmbedder
from app.retrieval.reranking import Reranker
from app.vectorstore.store import ScoredChunk, VectorStore


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]: ...


class DenseRetriever:
    """Embedding nearest-neighbours, and nothing else. The measured baseline."""

    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    @observe(name="retrieve")
    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        vector = self._embedder.embed_query(question)
        return self._store.search(vector, top_k or self._top_k)


class HybridRetriever:
    """Dense + BM25, fused by reciprocal rank.

    Fusion is reciprocal-rank rather than a weighted sum of scores, because the
    two arms produce incomparable numbers: cosine similarity is bounded in
    [-1, 1] and clusters tightly near the top, while BM25 is unbounded and
    scales with term rarity and document length. Summing them would silently
    let whichever arm happened to have the wider spread dominate, and the
    balance would drift as the corpus grew. RRF discards the magnitudes and
    keeps only the ranks, which is the one thing the two arms agree on the
    meaning of. It is also why there is no weight to tune here.
    """

    def __init__(
        self,
        embedder: Embedder,
        sparse_embedder: SparseEmbedder,
        store: VectorStore,
        top_k: int,
        *,
        candidates: int,
    ) -> None:
        self._embedder = embedder
        self._sparse_embedder = sparse_embedder
        self._store = store
        self._top_k = top_k
        self._candidates = candidates

    @observe(name="retrieve")
    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        wanted = top_k or self._top_k
        # `candidates` bounds how deep each arm looks, never how many results
        # come back. A caller asking for more than the configured depth widens
        # the search rather than getting a short list.
        candidates = max(self._candidates, wanted)

        hits = self._store.hybrid_search(
            self._embedder.embed_query(question),
            self._sparse_embedder.embed_query(question),
            wanted,
            candidates=candidates,
        )

        # Recorded on the span so a thin answer can be diagnosed as a retrieval
        # problem from the trace, without re-running the query.
        get_client().update_current_span(
            metadata={
                "retrieval.mode": "hybrid",
                "retrieval.candidates": candidates,
                "retrieval.returned": len(hits),
            }
        )
        return hits


class RerankingRetriever:
    """Any retriever, plus a cross-encoder pass over a wider shortlist.

    A wrapper rather than a flag on each retriever, because reranking is
    orthogonal to how the candidates were found — and because keeping it
    separable is what let `eval/` measure the two changes independently instead
    of reporting one combined number for hybrid-and-reranking together. That
    turned out to matter: the two do not move the score in the same direction.

    The inner retriever is asked for `candidates`, not for `top_k`. Handing the
    reranker only the final k would let it reorder those k but never promote the
    chunk sitting just below the cutoff, which is most of what it is for.
    """

    def __init__(
        self,
        inner: Retriever,
        reranker: Reranker,
        top_k: int,
        *,
        candidates: int,
    ) -> None:
        self._inner = inner
        self._reranker = reranker
        self._top_k = top_k
        self._candidates = candidates

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        wanted = top_k or self._top_k
        shortlist = self._inner.retrieve(question, top_k=max(self._candidates, wanted))
        return self._reranker.rerank(question, shortlist, wanted)
