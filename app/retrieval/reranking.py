"""Cross-encoder reranking — the second, more expensive look.

Retrieval scores a question against a chunk without the two ever meeting: each
is embedded alone, and the score is the distance between those summaries. That
is what makes it fast enough to run over the whole corpus, and it is also what
it gets wrong. A chunk can sit near the question in vector space while answering
a neighbouring question, and nothing in a bi-encoder score can tell the
difference.

A cross-encoder reads the question and the chunk *together* and scores the pair
directly. Far more accurate, and far too slow to run over a corpus — the cost is
one forward pass per candidate, so it only works on a shortlist that cheap
retrieval has already narrowed. Hence the shape of the pipeline: retrieve wide
with fusion, rerank the shortlist, keep top_k.

The reranker cannot rescue a chunk that retrieval never returned. Widening
`candidates` is what raises the ceiling; reranking only orders what came back.
"""

import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol, runtime_checkable

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langfuse import observe

from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Local ONNX cross-encoder. ~80MB for the default MiniLM model.

    Scores replace the fusion scores they came in with. They are relevance
    logits — unbounded, and freely negative for a genuinely irrelevant chunk —
    so they are not comparable to the cosine similarities a dense search
    returns, and must not be thresholded as if they were probabilities.
    """

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextCrossEncoder | None = None

    def _ensure_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
        return self._model

    @observe(name="rerank")
    def rerank(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        scores = list(self._ensure_model().rerank(question, [c.chunk.text for c in chunks]))
        rescored = [
            replace(chunk, score=float(score))
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        # Stable sort, so chunks the cross-encoder scores identically keep the
        # order fusion gave them rather than an arbitrary one.
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]
