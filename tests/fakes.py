"""In-process stand-ins for the three things that would otherwise need the world.

The real embedder downloads ONNX weights, the real store needs a Qdrant, and the
real answerer needs an API key and a network. Each sits behind a Protocol for
exactly this reason, so the pipeline under test is the production pipeline —
only the leaves are swapped.

The stub embedder is a real (if crude) embedding: a hashed bag of words. That
matters more than it looks. A stub returning constant or random vectors would
make every retrieval test vacuous; this one actually ranks a chunk sharing words
with the question above one that doesn't, so retrieval assertions mean something.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence

from app.generation.answerer import GeneratedAnswer
from app.ingest.chunking import Chunk
from app.vectorstore.store import ScoredChunk

_DIMENSION = 64


def _bucket(word: str) -> int:
    digest = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _DIMENSION


def _vectorize(text: str) -> list[float]:
    vector = [0.0] * _DIMENSION
    for word in text.lower().split():
        vector[_bucket(word.strip(".,;:()[]\"'"))] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class StubEmbedder:
    @property
    def dimension(self) -> int:
        return _DIMENSION

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [_vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vectorize(text)


class InMemoryVectorStore:
    """Cosine search over a dict. Same contract as the Qdrant store, including
    that `delete_document` on an unknown id is not an error."""

    def __init__(self) -> None:
        self.points: dict[str, tuple[Chunk, str, list[float]]] = {}
        self.collection_dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.collection_dimension = dimension

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], *, source: str
    ) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.id] = (chunk, source, list(vector))

    def delete_document(self, doc_id: str) -> None:
        for chunk_id in [
            chunk_id
            for chunk_id, (chunk, _, _) in self.points.items()
            if chunk.doc_id == doc_id
        ]:
            del self.points[chunk_id]

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=chunk, source=source, score=_cosine(vector, stored))
            for chunk, source, stored in self.points.values()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        self.points.clear()
        self.collection_dimension = None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    # Both sides are already unit-normalized by `_vectorize`, so the dot product
    # is the cosine.
    return sum(a * b for a, b in zip(left, right, strict=True))


class StubAnswerer:
    """Returns whatever the test tells it to, and records what it was asked.

    Default behaviour cites the first source, which is the shape a passing
    end-to-end test should produce.
    """

    def __init__(
        self, respond: Callable[[str, Sequence[ScoredChunk]], GeneratedAnswer] | None = None
    ) -> None:
        self.respond = respond
        self.calls: list[tuple[str, list[ScoredChunk]]] = []

    def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        self.calls.append((question, list(chunks)))

        if self.respond is not None:
            return self.respond(question, chunks)

        if not chunks:
            return GeneratedAnswer(
                answer="No indexed document contains anything relevant to this question.",
                claims=[],
                answerable=False,
            )

        return GeneratedAnswer.model_validate(
            {
                "answer": "The notice period is thirty days.",
                "claims": [{"text": "The notice period is thirty days.", "sources": [1]}],
                "answerable": True,
            }
        )

    def reset(self) -> None:
        self.respond = None
        self.calls.clear()
