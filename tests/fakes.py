"""In-process stand-ins for the three things that would otherwise need the world.

The real embedder downloads ONNX weights, the real store needs a Qdrant, and the
real answerer needs an API key and a network. Each sits behind a Protocol for
exactly this reason, so the pipeline under test is the production pipeline —
only the leaves are swapped.

The stub embedder is a real (if crude) embedding: a hashed bag of words. That
matters more than it looks. A stub returning constant or random vectors would
make every retrieval test vacuous; this one actually ranks a chunk sharing words
with the question above one that doesn't, so retrieval assertions mean something.

The sparse stub is deliberately *not* the same function. It hashes whole words
to term ids with no bucketing collisions worth speaking of, so it matches exact
tokens and nothing else — which is the property that makes a hybrid test able to
show BM25 finding something dense retrieval missed. If both stubs scored text
the same way, fusing them would be a no-op and every hybrid assertion would pass
for the wrong reason.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from app.generation.answerer import GeneratedAnswer
from app.ingest.chunking import Chunk
from app.ingest.sparse import SparseVector
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


class StubSparseEmbedder:
    """Exact-token matching, standing in for BM25.

    Term ids are hashes of the whole word, so unlike `StubEmbedder` there is no
    dimensional bucketing and no accidental similarity between different words.
    Document side weights by term frequency; query side is all-ones, mirroring
    how `Qdrant/bm25` leaves the IDF to the engine.
    """

    @staticmethod
    def _terms(text: str) -> list[str]:
        return [word.strip(".,;:()[]\"'") for word in text.lower().split()]

    @staticmethod
    def _term_id(word: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(word.encode("utf-8"), digest_size=4).digest(), "big"
        )

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        vectors = []
        for text in texts:
            counts: dict[int, float] = {}
            for word in self._terms(text):
                if word:
                    counts[self._term_id(word)] = counts.get(self._term_id(word), 0.0) + 1.0
            vectors.append(
                SparseVector(indices=list(counts), values=[counts[i] for i in counts])
            )
        return vectors

    def embed_query(self, text: str) -> SparseVector:
        ids = sorted({self._term_id(word) for word in self._terms(text) if word})
        return SparseVector(indices=ids, values=[1.0] * len(ids))


class StubReranker:
    """Reorders by a caller-supplied key, and records what it was given.

    Default behaviour reverses the incoming order. That is deliberately not a
    plausible relevance ranking — it makes it unmistakable in a test whether the
    reranker actually ran, which a subtle reordering would not.
    """

    def __init__(self, score: Callable[[str, ScoredChunk], float] | None = None) -> None:
        self.score = score
        self.calls: list[tuple[str, list[ScoredChunk]]] = []

    def rerank(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        self.calls.append((question, list(chunks)))

        if self.score is not None:
            rescored = [replace(c, score=self.score(question, c)) for c in chunks]
        else:
            # Score ascending with incoming position, so the sort below hands
            # back the exact reverse of what came in.
            rescored = [
                replace(chunk, score=float(rank)) for rank, chunk in enumerate(chunks)
            ]

        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]


@dataclass
class _Point:
    chunk: Chunk
    source: str
    vector: list[float]
    sparse: SparseVector | None = None


class InMemoryVectorStore:
    """Cosine + BM25-ish search over a dict, fused by RRF.

    Same contract as the Qdrant store, including that `delete_document` on an
    unknown id is not an error.

    The fusion here re-implements what Qdrant does server-side, which is a real
    duplication and worth being honest about: the two will not produce identical
    *scores*. They are held to producing the same *ordering*, which is all the
    pipeline depends on, and `tests/test_qdrant_store.py` pins the real engine's
    behaviour against a real (embedded) Qdrant so the two cannot drift silently.
    """

    def __init__(self) -> None:
        self.points: dict[str, _Point] = {}
        self.collection_dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        self.collection_dimension = dimension

    def upsert(
        self,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[SparseVector] | None = None,
        *,
        source: str,
    ) -> None:
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            self.points[chunk.id] = _Point(
                chunk=chunk,
                source=source,
                vector=list(vector),
                sparse=sparse_vectors[index] if sparse_vectors is not None else None,
            )

    def delete_document(self, doc_id: str) -> None:
        for chunk_id in [
            chunk_id
            for chunk_id, point in self.points.items()
            if point.chunk.doc_id == doc_id
        ]:
            del self.points[chunk_id]

    def search(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        scored = [
            ScoredChunk(chunk=p.chunk, source=p.source, score=_cosine(vector, p.vector))
            for p in self.points.values()
        ]
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def sparse_search(self, sparse_vector: SparseVector, top_k: int) -> list[ScoredChunk]:
        query = set(sparse_vector.indices)
        scored = []
        for point in self.points.values():
            if point.sparse is None:
                # Indexed dense-only: invisible to the lexical arm, exactly as
                # it would be in Qdrant.
                continue
            overlap = sum(
                value
                for index, value in zip(point.sparse.indices, point.sparse.values, strict=True)
                if index in query
            )
            if overlap:
                scored.append(
                    ScoredChunk(chunk=point.chunk, source=point.source, score=overlap)
                )
        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:top_k]

    def hybrid_search(
        self,
        vector: Sequence[float],
        sparse_vector: SparseVector,
        top_k: int,
        *,
        candidates: int,
    ) -> list[ScoredChunk]:
        if not sparse_vector.indices:
            return self.search(vector, top_k)

        return _reciprocal_rank_fusion(
            [self.search(vector, candidates), self.sparse_search(sparse_vector, candidates)],
            top_k,
        )

    def clear(self) -> None:
        self.points.clear()
        self.collection_dimension = None


# Qdrant's RRF constant. Named because the value is the whole of the algorithm:
# it sets how fast a result's contribution decays with rank, and therefore how
# much a strong showing in one arm can outweigh absence from the other.
_RRF_K = 2


def _reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], top_k: int
) -> list[ScoredChunk]:
    fused: dict[str, float] = {}
    hits: dict[str, ScoredChunk] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
            hits.setdefault(hit.chunk.id, hit)

    ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    return [replace(hits[chunk_id], score=fused[chunk_id]) for chunk_id in ordered[:top_k]]


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
