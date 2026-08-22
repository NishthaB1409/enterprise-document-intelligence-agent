"""Run the golden set through each retrieval configuration and report the lift.

    python -m eval.retrieval_eval

Everything runs locally and offline once the models are cached: the corpus is in
the repo, embeddings and reranking are ONNX models on disk, and the vector store
is an embedded Qdrant in a temporary directory. No API key, and nothing here
calls a generation model — this measures retrieval, which is the only part the
change in phase 2 touched. Answer quality is phase 5's harness.

Four configurations, laid out as a 2x2 so that adding BM25 and adding a
reranker can be read independently:

    dense            the phase 1 baseline
    dense+rerank     baseline + cross-encoder over a wider dense shortlist
    hybrid           baseline + BM25, fused by reciprocal rank
    hybrid+rerank    both

The 2x2 is the point. Reporting only `dense` against `hybrid+rerank` would have
shown a single combined number and hidden the fact that the two changes move the
score in opposite directions on this corpus.

The store is rebuilt per configuration rather than shared. It costs a few
seconds and removes the possibility that one configuration is reading an index
another one warmed.
"""

import argparse
import json
import logging
import shutil
import statistics
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient

from app.ingest.chunking import build_splitter, chunk_pages
from app.ingest.embedding import FastEmbedEmbedder
from app.ingest.sparse import FastEmbedSparseEmbedder
from app.retrieval.reranking import CrossEncoderReranker
from app.retrieval.retriever import (
    DenseRetriever,
    HybridRetriever,
    RerankingRetriever,
    Retriever,
)
from app.vectorstore.qdrant_store import QdrantVectorStore
from eval.corpus import CORPUS, GOLD_DOCUMENTS
from eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank, relevance
from eval.questions import QUESTIONS, Question

# Matches the shipped defaults, so the numbers describe the deployed system
# rather than a configuration tuned for the report.
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
TOP_K = 5
CANDIDATES = 30


@dataclass
class Scores:
    """One configuration's results over one slice of the question set."""

    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    count: int = 0
    misses: list[str] = field(default_factory=list)

    @classmethod
    def over(cls, per_question: Sequence[tuple[Question, list[bool]]]) -> "Scores":
        if not per_question:
            return cls()
        return cls(
            hit_at_1=statistics.mean(hit_at_k(j, 1) for _, j in per_question),
            hit_at_3=statistics.mean(hit_at_k(j, 3) for _, j in per_question),
            hit_at_5=statistics.mean(hit_at_k(j, 5) for _, j in per_question),
            mrr=statistics.mean(reciprocal_rank(j) for _, j in per_question),
            ndcg_at_5=statistics.mean(ndcg_at_k(j, 5) for _, j in per_question),
            count=len(per_question),
            misses=[q.question for q, j in per_question if not any(j[:TOP_K])],
        )

    def as_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "hit@1": round(self.hit_at_1, 4),
            "hit@3": round(self.hit_at_3, 4),
            "hit@5": round(self.hit_at_5, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@5": round(self.ndcg_at_5, 4),
            "questions": self.count,
            "misses": self.misses,
        }


def index_corpus(store: QdrantVectorStore, embedder, sparse_embedder) -> int:
    """Chunk and index every document. Returns the chunk count.

    Goes through `chunk_pages` — the production chunker — but not through the
    PDF parser: the corpus is already text. Re-encoding it to PDF only to
    extract it again would add a typesetting round trip that could reflow a gold
    span, and would be measuring pypdf rather than retrieval.
    """
    splitter = build_splitter(CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
    store.ensure_collection(embedder.dimension)

    total = 0
    for document in CORPUS:
        chunks = chunk_pages(document.pages, document.doc_id, splitter)
        texts = [chunk.text for chunk in chunks]
        store.upsert(
            chunks,
            embedder.embed_documents(texts),
            sparse_embedder.embed_documents(texts) if sparse_embedder else None,
            source=document.source,
        )
        total += len(chunks)
    return total


def evaluate(retriever: Retriever) -> list[tuple[Question, list[bool]]]:
    results = []
    for question in QUESTIONS:
        hits = retriever.retrieve(question.question, top_k=TOP_K)
        results.append((question, relevance(hits, question.gold_span)))
    return results


def _format_table(rows: Sequence[tuple[str, Scores]], baseline: Scores | None) -> str:
    header = (
        "| configuration | hit@1 | hit@3 | hit@5 | MRR | nDCG@5 |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = []
    for name, scores in rows:
        delta = ""
        if baseline is not None and scores is not baseline:
            change = scores.mrr - baseline.mrr
            delta = f" ({change:+.3f})" if abs(change) > 1e-9 else " (=)"
        lines.append(
            f"| {name} | {scores.hit_at_1:.2f} | {scores.hit_at_3:.2f} | "
            f"{scores.hit_at_5:.2f} | {scores.mrr:.3f}{delta} | {scores.ndcg_at_5:.3f} |"
        )
    return header + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        type=Path,
        help="also write the full results here, misses included",
    )
    args = parser.parse_args()

    # Langfuse is unconfigured here and says so on every span. The tracing is
    # not what is being measured, and 200 identical warnings bury the table.
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5")
    sparse_embedder = FastEmbedSparseEmbedder()
    reranker = CrossEncoderReranker()

    print(
        f"corpus: {len(CORPUS)} documents  "
        f"questions: {len(QUESTIONS)}  "
        f"top_k: {TOP_K}  candidates: {CANDIDATES}\n"
    )

    configurations: list[tuple[str, bool, bool]] = [
        # name, needs BM25, needs reranker
        ("dense", False, False),
        ("dense+rerank", False, True),
        ("hybrid", True, False),
        ("hybrid+rerank", True, True),
    ]

    results: dict[str, list[tuple[Question, list[bool]]]] = {}

    for name, hybrid, rerank in configurations:
        directory = tempfile.mkdtemp(prefix=f"edia-eval-{name.replace('+', '-')}-")
        client = QdrantClient(path=directory)
        try:
            store = QdrantVectorStore(client=client, collection="eval")
            chunks = index_corpus(store, embedder, sparse_embedder if hybrid else None)

            retriever: Retriever = (
                HybridRetriever(
                    embedder, sparse_embedder, store, TOP_K, candidates=CANDIDATES
                )
                if hybrid
                else DenseRetriever(embedder, store, TOP_K)
            )
            if rerank:
                retriever = RerankingRetriever(
                    retriever, reranker, TOP_K, candidates=CANDIDATES
                )

            print(f"  {name}: indexed {chunks} chunks, running {len(QUESTIONS)} questions...")
            results[name] = evaluate(retriever)
        finally:
            client.close()
            shutil.rmtree(directory, ignore_errors=True)

    overall = {name: Scores.over(rows) for name, rows in results.items()}
    baseline = overall["dense"]

    print("\n## Overall\n")
    print(_format_table(list(overall.items()), baseline))

    payload: dict[str, object] = {
        "config": {
            "top_k": TOP_K,
            "candidates": CANDIDATES,
            "chunk_size_words": CHUNK_SIZE_WORDS,
            "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
            "questions": len(QUESTIONS),
            "documents": len(CORPUS),
        },
        "overall": {name: scores.as_dict() for name, scores in overall.items()},
        "by_kind": {},
    }

    for kind in ("lexical", "semantic"):
        sliced = {
            name: Scores.over([(q, j) for q, j in rows if q.kind == kind])
            for name, rows in results.items()
        }
        payload["by_kind"][kind] = {n: s.as_dict() for n, s in sliced.items()}  # type: ignore[index]
        print(f"\n## {kind} questions ({sliced['dense'].count})\n")
        print(_format_table(list(sliced.items()), sliced["dense"]))

    print("\n## Questions with no relevant chunk in the top 5\n")
    for name, scores in overall.items():
        if scores.misses:
            print(f"  {name}:")
            for miss in scores.misses:
                print(f"    - {miss}")
        else:
            print(f"  {name}: none")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
