"""Text to vectors.

Behind a Protocol on purpose. The real embedder downloads ~130MB of ONNX
weights on first use, which is fine in a container and intolerable in a test
suite — so tests substitute a deterministic stand-in, and the ingestion and
retrieval code never knows which one it has.

Queries and documents go through different methods because asymmetric embedding
models (BGE, E5, and friends) expect an instruction prefix on the query side
only. Embedding a question as if it were a passage measurably degrades recall,
and it fails silently: the vectors are still valid, just aimed slightly wrong.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from fastembed import TextEmbedding


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """Vector width. The collection is created against this, so it must be
        known before anything is embedded."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _dimension_of(model_name: str) -> int:
    for model in TextEmbedding.list_supported_models():
        if model["model"].lower() == model_name.lower():
            return int(model["dim"])
    raise ValueError(
        f"unknown embedding model {model_name!r}; "
        "see fastembed.TextEmbedding.list_supported_models()"
    )


class FastEmbedEmbedder:
    """Local ONNX embeddings. No API key, no per-document cost, no network at
    query time once the weights are cached."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        # Read from the model registry rather than by embedding a probe string,
        # so constructing an embedder stays free and offline.
        self._dimension = _dimension_of(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_model(self) -> TextEmbedding:
        # Deferred so that importing the app, or booting it with no documents to
        # ingest, never pays for the weight download.
        if self._model is None:
            self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        return [vector.tolist() for vector in model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        return next(iter(model.query_embed(text))).tolist()
