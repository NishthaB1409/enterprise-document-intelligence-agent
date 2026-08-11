"""Dense top-k retrieval.

Deliberately the whole of it for now: embed the question, take the nearest k
chunks. Hybrid search (BM25 alongside the dense vectors) and reranking land in
phase 2, and the point of keeping this baseline intact is to have a before
number to measure that lift against.

The question goes through `embed_query`, not `embed_documents` — see the note in
`app.ingest.embedding` about what asymmetric models do to recall when a question
is embedded as if it were a passage.
"""

from langfuse import observe

from app.ingest.embedding import Embedder
from app.vectorstore.store import ScoredChunk, VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore, top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k

    @observe(name="retrieve")
    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        vector = self._embedder.embed_query(question)
        return self._store.search(vector, top_k or self._top_k)
