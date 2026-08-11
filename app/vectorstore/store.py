"""The persistence boundary: chunks in, scored chunks out.

Behind a Protocol for the same reason the embedder is — the real store needs a
running Qdrant, and the test suite needs neither a container nor a network. The
ingestion and retrieval code is written against this interface, so the in-memory
stand-in and the real thing are interchangeable.

A retrieved chunk carries everything a citation needs (document, page, span,
filename), because the alternative is a second lookup at answer time to resolve
what a hit actually points at.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.ingest.chunking import Chunk


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    # The uploaded filename. Kept alongside the chunk because a citation that
    # says "page 7" is useless without saying page 7 *of what*.
    source: str
    score: float


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it is missing. Must be idempotent — it runs
        at startup and again on every ingest."""

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, source: str
    ) -> None: ...

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of a document. Not an error if there are none."""

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]: ...
