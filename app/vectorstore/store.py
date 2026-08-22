"""The persistence boundary: chunks in, scored chunks out.

Behind a Protocol for the same reason the embedder is — the real store needs a
running Qdrant, and the test suite needs neither a container nor a network. The
ingestion and retrieval code is written against this interface, so the in-memory
stand-in and the real thing are interchangeable.

A retrieved chunk carries everything a citation needs (document, page, span,
filename), because the alternative is a second lookup at answer time to resolve
what a hit actually points at.

Two searches, not one. `search` is dense-only and `hybrid_search` fuses dense
with BM25; both stay on the interface because the dense path is the measured
baseline the hybrid lift is quoted against, not dead code. `eval/` runs them
side by side, so deleting the "old" one would delete the evidence.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.ingest.chunking import Chunk
from app.ingest.sparse import SparseVector


class CollectionSchemaError(RuntimeError):
    """The collection exists but is not shaped the way this code needs.

    In practice this means a collection created before hybrid retrieval landed:
    it has dense vectors and no sparse index, and Qdrant will not add one to a
    populated collection. Raised with the remedy rather than left to surface as
    an opaque engine error at query time.
    """


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    # The uploaded filename. Kept alongside the chunk because a citation that
    # says "page 7" is useless without saying page 7 *of what*.
    source: str
    # Comparable only within one result list. Cosine similarity from a dense
    # search, a fused rank score from `hybrid_search`, or a cross-encoder
    # logit once a reranker has run — three different scales, and none of them
    # a probability that the chunk answers the question.
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it is missing. Must be idempotent — it runs
        at startup and again on every ingest.

        Raises `CollectionSchemaError` if a collection exists that cannot serve
        the searches below.
        """

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector] | None = None,
        *,
        source: str,
    ) -> None:
        """`sparse_vectors` omitted indexes the chunks for dense search only;
        they stay retrievable, but BM25 will never surface them."""

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document. Not an error if there are none."""

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]: ...

    def hybrid_search(
        self,
        vector: Sequence[float],
        sparse_vector: SparseVector,
        top_k: int,
        *,
        candidates: int,
    ) -> list[ScoredChunk]:
        """Dense and BM25 rankings, fused.

        `candidates` is how deep each arm searches before fusion. It has to
        exceed `top_k` for fusion to do anything: fusing two copies of the same
        top-k list can only reorder those k, whereas a chunk ranked 15th by the
        dense arm and 2nd by BM25 is precisely the result hybrid exists to
        surface — and it is only reachable if both arms looked past k.
        """
