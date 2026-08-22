"""Text to sparse (lexical) vectors — the BM25 half of hybrid retrieval.

Dense embeddings match on meaning, which is what makes them worth having, and
also what makes them miss. Ask about "Section 12.4", `SOC 2 Type II`, or a party
name that appeared nowhere in the embedding model's training data, and the
nearest neighbours are passages that are *about the same topic* rather than the
one passage containing that exact token. BM25 has the opposite failure: it
cannot match a paraphrase, but a rare exact term is the thing it ranks best.

Neither subsumes the other, so the retriever runs both and fuses the rankings.

Behind a Protocol for the same reason `Embedder` is: the real model pulls a
tokenizer and stopword list from disk, and the test suite substitutes a
deterministic stand-in.

A note on where the IDF lives. `Qdrant/bm25` emits raw term frequencies on the
document side and all-ones on the query side; the inverse-document-frequency
term is applied by the engine at query time, which is why the collection's
sparse vector must be configured with `models.Modifier.IDF`. That split is
deliberate on Qdrant's part — IDF depends on the whole corpus, so it cannot be
computed correctly while embedding one document in isolation. Configure the
collection without the modifier and nothing errors; the ranking just quietly
becomes term-frequency-only, which is worse in exactly the case BM25 was added
for (rare terms stop being treated as rare).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from fastembed import SparseTextEmbedding


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A term-weight vector, held as parallel index/value arrays.

    Defined here rather than reusing `qdrant_client.models.SparseVector` so the
    ingestion and retrieval code does not import the storage client to describe
    its own data — the same reason `Embedder` returns plain lists of floats.
    """

    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


@runtime_checkable
class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class FastEmbedSparseEmbedder:
    """BM25 term weights, computed locally.

    Unlike the dense model this pulls only a tokenizer and a stopword list — a
    few hundred kilobytes, not 130MB — so it is cheap to add to the ingest path.
    """

    def __init__(self, model_name: str = "Qdrant/bm25", cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: SparseTextEmbedding | None = None

    def _ensure_model(self) -> SparseTextEmbedding:
        # Deferred for the same reason the dense embedder defers: constructing
        # the service must not touch the disk cache or the network.
        if self._model is None:
            self._model = SparseTextEmbedding(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        if not texts:
            return []
        return [
            _to_sparse_vector(embedding)
            for embedding in self._ensure_model().embed(list(texts))
        ]

    def embed_query(self, text: str) -> SparseVector:
        # `query_embed`, not `embed`: the query side deliberately drops term
        # frequency weighting, because a term repeated in the question says
        # nothing about which document matches it best.
        return _to_sparse_vector(next(iter(self._ensure_model().query_embed(text))))


def _to_sparse_vector(embedding: object) -> SparseVector:
    # fastembed returns numpy arrays; the store and the tests want plain Python,
    # and `.tolist()` also converts numpy scalar types that would otherwise
    # survive into the payload sent to Qdrant.
    return SparseVector(
        indices=[int(index) for index in embedding.indices.tolist()],  # type: ignore[attr-defined]
        values=[float(value) for value in embedding.values.tolist()],  # type: ignore[attr-defined]
    )
