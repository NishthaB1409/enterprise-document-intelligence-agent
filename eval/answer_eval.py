"""Answer quality, end to end: does the agentic graph beat the single pass?

    python -m eval.answer_eval [--pipelines simple,agent] [--gate] [--limit N]

Needs OPENAI_API_KEY and the eval extra (`uv sync --extra eval`). Runs every
question in the golden set, plus a set the corpus cannot answer, through each
pipeline over the same in-process index, then scores the answers. The call
count is printed before anything is spent.

This is the measurement phase 3 deferred. The graph costs three to five model
calls per question against the single pass's one, and until now the only part
of it that had been measured was routing. Whether the rest buys anything is the
question every number here is laid out to answer, so every pipeline is scored
on the same questions, in the same table, with its cost beside it.

What is measured, and by what:

    faithfulness        Ragas, judge model. The share of the answer's
                        statements the shown chunks support. Scored only on
                        answers that answered: a decline has no claims to check.

    answer relevancy    Ragas, judge model plus local embeddings. Does the
                        answer address the question? Ragas scores
                        noncommittal answers as zero, so a wrong decline on an
                        answerable question costs here.

    answer correctness  Judge model, against the gold span: the share of the
                        span's claims that the answer states. The one the
                        README's four targets were missing — an answer can be
                        faithful to a distractor, quoting the wrong contract's
                        thirty days perfectly, and only this catches it.

                        Built on Ragas's claim decomposition and NLI steps, but
                        not on either of its scores, both of which misread a
                        fragment as a reference. `AnswerAccuracy` expects a full
                        reference answer and rated every correct answer 0.5 as
                        a "partial match" for a sentence fragment.
                        `FactualCorrectness(mode="recall")` counts the answer's
                        claims the reference supports as its numerator, so a
                        correct answer adding one true detail ("...with manager
                        approval") scored 0.0. See `ReferenceRecall` below.

    context precision   Deterministic, from gold spans (see `eval/metrics.py`).
    context recall      Deterministic. Was the answering passage shown at all?

    unanswerable        Deterministic. Declined, or answered-but-held by the
                        review gate, both count as safe. Answered and released
                        is the failure.

    review gate         Deterministic. How often each pipeline's answers would
                        be held, using the shipped `review_min_confidence`.

    cost                Calls and tokens, read off each client's `Usage`.

With LANGFUSE_* configured, every question's pipeline run is traced, with the
graph's own route, grade, and critique spans inside it, and each score above is
attached to that trace. A low faithfulness score then opens onto the exact run
that earned it, rather than a row in a JSON file. Without the keys, tracing is a
no-op and nothing changes.

Why the judge is a model at all, when two of the metrics are computed exactly:
there is no gold answer text, only gold passages, and "is this sentence
supported by that passage" is not a string match. The judge is `AGENT_MODEL` by
default. That is the same family as the generator, which is known to flatter it;
`--judge-model` is there to check the ranking holds under a different judge.

What this cannot tell you. Twenty-six questions, one sample each: a difference
of one or two questions between pipelines is inside the noise, and should be
read as "no measured difference". Temperature 0 did not change that much: the
judge varies too, and some answers still differ between runs. The corpus is
synthetic and small. See `eval/ANSWERS.md` for the numbers and the caveats.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import statistics
import sys
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.generation.answerer import LLMAnswerer
from app.generation.citations import ground
from app.graph.pipeline import QueryPipeline
from app.ingest.embedding import FastEmbedEmbedder
from app.llm import OpenAILLM
from app.observability.langfuse_client import init_langfuse
from app.retrieval.retriever import DenseRetriever
from app.review.contradictions import collect_contradictions
from app.review.gate import review_reasons
from app.services import build_llm, build_pipeline
from app.vectorstore.qdrant_store import QdrantVectorStore
from eval.metrics import context_precision, context_recall, relevance
from eval.questions import QUESTIONS, UNANSWERABLE
from eval.retrieval_eval import TOP_K, index_corpus

# What `--gate` enforces: regression floors, set below the first measured runs
# (eval/ANSWERS.md) by roughly the run-to-run noise seen between them. A CI gate
# exists to catch a change that makes answers worse, and it can only do that if
# today's system passes it.
#
# These are deliberately not the README's targets. The shipped single pass was
# measured against those for the first time in phase 5 and misses three of four,
# so a gate built on them would fail every build — and a gate that always fails
# gets switched off. The targets stay as goals; raise a floor when a change
# measurably lifts the baseline.
THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.65,
    "answer_correctness": 0.60,
    "context_precision": 0.65,
    "context_recall": 0.75,
}

# The README's targets, reported beside each score but never enforced.
TARGETS: dict[str, float] = {
    "faithfulness": 0.90,
    "answer_relevancy": 0.85,
    "context_precision": 0.80,
    "context_recall": 0.80,
}

# Judge calls per scored answer: faithfulness makes two (extract statements,
# then check them), relevancy three (one per generated question, Ragas's
# default strictness), correctness two (decompose the reference, verify it).
_JUDGE_CALLS = {"faithfulness": 2, "answer_relevancy": 3, "answer_correctness": 2}

# The committed record eval/ANSWERS.md cites. Written only when `--json` asks for
# it: a gate run in CI, or a quick check, must not silently replace the run the
# write-up describes. (The first gate run did exactly that.)
RESULTS_PATH = Path(__file__).with_name("answer_results.json")


@dataclass
class Sample:
    """One question through one pipeline, and everything scored about it."""

    pipeline: str
    question: str
    # "lexical" or "semantic" for the golden set, "unanswerable" for the rest.
    kind: str
    gold_span: str | None
    answer: str = ""
    answered: bool = False
    # Exactly the chunks the generator was shown, in order.
    contexts: list[str] = field(default_factory=list)
    # Whether each shown chunk contains the gold span.
    judgements: list[bool] = field(default_factory=list)
    unsupported_claims: int = 0
    held: bool = False
    reasons: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    error: str | None = None
    # The Langfuse trace this run was recorded under; scores attach to it.
    trace_id: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    answer_correctness: float | None = None

    @property
    def answerable(self) -> bool:
        return self.kind != "unanswerable"


# --- running the pipelines ------------------------------------------------------


class _SerialRetriever:
    """Embedded Qdrant is not safe to share across threads. The model calls are
    what is slow and they run in parallel; retrieval takes milliseconds and
    queues behind this lock."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def retrieve(self, question: str, top_k: int | None = None):
        with self._lock:
            return self._inner.retrieve(question, top_k)


@dataclass
class PipelineRun:
    name: str
    pipeline: QueryPipeline
    answer_llm: OpenAILLM
    agent_llm: OpenAILLM | None


def build_runs(settings: Settings, retriever, names: Sequence[str]) -> list[PipelineRun]:
    """Each pipeline gets its own clients, so each one's cost is its own.

    Built through `build_pipeline`, the same function the service uses, so what
    is measured is the graph that ships rather than a reconstruction of it.
    """
    runs = []
    for name in names:
        answer_llm = _client(settings, settings.answer_model)
        agent_llm = _client(settings, settings.agent_model) if name == "agent" else None
        configured = settings.model_copy(update={"agent_enabled": name == "agent"})
        pipeline = build_pipeline(
            configured,
            retriever,
            LLMAnswerer(answer_llm, max_tokens=settings.answer_max_tokens),
            answer_llm,
            agent_llm=agent_llm,
        )
        runs.append(PipelineRun(name, pipeline, answer_llm, agent_llm))
    return runs


def _client(settings: Settings, model: str) -> OpenAILLM:
    # Through the service's own factory, so every setting that shapes a model
    # call — temperature included — is the one that ships.
    client = build_llm(settings, model=model)
    assert isinstance(client, OpenAILLM)  # the eval reads its token usage
    return client


def run_one(run: PipelineRun, sample: Sample, min_confidence: float) -> Sample:
    """Answer one question and record everything that is scored deterministically.

    Mirrors what `/query` does after the pipeline returns — grounding,
    contradiction resolution, the review gate — so `held` is what the service
    would actually have done.
    """
    from langfuse import get_client

    with get_client().start_as_current_observation(
        name=f"answer-eval/{run.name}",
        as_type="span",
        input={"question": sample.question, "kind": sample.kind},
        metadata={"pipeline": run.name, "gold_span": sample.gold_span},
    ) as span:
        sample.trace_id = span.trace_id
        _answer(run, sample, min_confidence)
        span.update(
            output={
                "answer": sample.answer,
                "answered": sample.answered,
                "held": sample.held,
                "reasons": sample.reasons,
                "path": " -> ".join(sample.steps),
                "error": sample.error,
            }
        )
    return sample


def _answer(run: PipelineRun, sample: Sample, min_confidence: float) -> None:
    try:
        result = run.pipeline.run(sample.question)
    except Exception as exc:  # noqa: BLE001 — one failed question must not end the run
        sample.error = f"{type(exc).__name__}: {exc}"
        return

    grounded = ground(result.answer, result.chunks)
    # From the answerer on every pipeline, and the critic when the graph ran.
    contradictions = collect_contradictions(
        result.answer, result.critique.conflicts if result.critique else [], result.chunks
    )
    sample.answer = grounded.answer
    sample.answered = grounded.answerable
    sample.contexts = [c.chunk.text for c in result.chunks]
    sample.judgements = relevance(result.chunks, sample.gold_span) if sample.gold_span else []
    sample.unsupported_claims = len(grounded.unsupported_claims)
    sample.reasons = review_reasons(
        grounded,
        critique=result.critique,
        critique_failed=result.critique_failed,
        contradictions=contradictions,
        min_confidence=min_confidence,
    )
    sample.held = bool(sample.reasons)
    sample.steps = result.steps


def questions() -> list[tuple[str, str, str | None]]:
    """(question, kind, gold span) for every case, golden set first."""
    return [(q.question, q.kind, q.gold_span) for q in QUESTIONS] + [
        (u.question, "unanswerable", None) for u in UNANSWERABLE
    ]


# --- judging ------------------------------------------------------------------------


def metrics_for(sample: Sample) -> list[str]:
    """Which judged metrics apply to a sample. Pure, so it can be tested.

    Nothing is judged on an unanswerable question: the right answer is to
    decline, which is scored exactly. Faithfulness needs claims, so a decline is
    skipped rather than scored — its cost shows up in relevancy and correctness
    instead, which is where a wrong decline belongs.
    """
    if sample.error or not sample.answerable:
        return []
    applicable = ["answer_relevancy", "answer_correctness"]
    if sample.answered and sample.contexts:
        applicable.insert(0, "faithfulness")
    return applicable


def _reference_recall(llm):
    """The share of the reference's claims the answer supports.

    One direction only: decompose the gold span, check each claim against the
    answer. Extra correct detail in the answer cannot lower it, and a wrong
    figure or a decline scores zero. Half the calls of `FactualCorrectness`,
    which also checks the answer's claims against the reference — the half
    that punishes an answer for saying more than a fragment does.

    Leans on `_decompose_and_verify_claims`, which is private. Checked here so
    a Ragas upgrade that renames it fails loudly at startup instead of scoring
    every answer NaN.
    """
    from ragas.metrics.collections import FactualCorrectness
    from ragas.metrics.result import MetricResult

    if not hasattr(FactualCorrectness, "_decompose_and_verify_claims"):
        raise SystemExit(
            "this Ragas version no longer has FactualCorrectness._decompose_and_verify_claims; "
            "answer correctness needs updating"
        )

    class ReferenceRecall(FactualCorrectness):
        async def ascore(self, response: str, reference: str) -> MetricResult:
            verdicts = await self._decompose_and_verify_claims(reference, response)
            return MetricResult(value=reference_recall(list(verdicts)))

    return ReferenceRecall(llm=llm, name="answer_correctness")


def reference_recall(verdicts: Sequence[bool]) -> float:
    """NaN when the reference decomposed into no claims: unscored, not zero."""
    return sum(verdicts) / len(verdicts) if verdicts else float("nan")


def _local_embeddings(settings: Settings):
    """The service's own FastEmbed model, behind Ragas's embedding interface.

    Answer relevancy compares the question against questions regenerated from
    the answer, by embedding. Running that locally keeps a second vendor, and a
    second bill, out of the eval — and it is the model retrieval already uses.
    """
    from ragas.embeddings.base import BaseRagasEmbedding

    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embedder.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(embedder.embed_query, text)

    return LocalEmbeddings()


async def judge(
    samples: Sequence[Sample], settings: Settings, judge_model: str, concurrency: int
) -> int:
    """Score every applicable metric in place. Returns how many judge calls failed.

    Ragas is imported here rather than at module scope, so the rest of this
    module — and its tests — work without the eval extra installed.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    llm = llm_factory(judge_model, client=client)
    scorers = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=_local_embeddings(settings)),
        "answer_correctness": _reference_recall(llm),
    }
    gate = asyncio.Semaphore(concurrency)
    failures = 0

    async def score(sample: Sample, metric: str) -> None:
        nonlocal failures
        kwargs: dict[str, Any] = {"response": sample.answer}
        if metric == "faithfulness":
            kwargs |= {"user_input": sample.question, "retrieved_contexts": sample.contexts}
        elif metric == "answer_relevancy":
            kwargs["user_input"] = sample.question
        else:  # correctness judges the answer against the reference alone
            kwargs["reference"] = sample.gold_span
        async with gate:
            try:
                result = await scorers[metric].ascore(**kwargs)
                value = result.value
            except Exception as exc:  # noqa: BLE001 — record, don't abort the run
                logging.getLogger(__name__).warning(
                    "%s failed on %r: %s", metric, sample.question, exc
                )
                failures += 1
                return
        # Ragas returns NaN where a metric is undefined (no statements to check).
        # Treated as unscored, not as zero.
        if value is not None and value == value:
            setattr(sample, metric, float(value))

    await asyncio.gather(
        *(score(sample, metric) for sample in samples for metric in metrics_for(sample))
    )
    return failures


# --- publishing ---------------------------------------------------------------------


def scores_for(sample: Sample) -> dict[str, float]:
    """Every score worth attaching to a sample's trace. Pure, so it can be tested.

    Judged metrics where they were scored, the exact context metrics for golden
    questions, and whether the review gate held the answer. Nothing for a failed
    run: a trace with an error on it already says what happened.
    """
    if sample.error:
        return {}
    scores = {
        metric: value
        for metric in ("faithfulness", "answer_relevancy", "answer_correctness")
        if (value := getattr(sample, metric)) is not None
    }
    if sample.answerable and sample.gold_span:
        scores["context_precision"] = context_precision(sample.judgements)
        scores["context_recall"] = context_recall(sample.judgements)
    scores["held_for_review"] = 1.0 if sample.held else 0.0
    return scores


def publish_scores(samples: Sequence[Sample]) -> int:
    """Attach each sample's scores to its trace. Returns how many were sent.

    A no-op when Langfuse is not configured: the client discards scores the same
    way it discards spans, so this needs no branch of its own.
    """
    from langfuse import get_client

    client = get_client()
    sent = 0
    for sample in samples:
        if sample.trace_id is None:
            continue
        for name, value in scores_for(sample).items():
            client.create_score(
                trace_id=sample.trace_id,
                name=name,
                value=value,
                data_type="BOOLEAN" if name == "held_for_review" else "NUMERIC",
            )
            sent += 1
    client.flush()
    return sent


# --- summarising ----------------------------------------------------------------


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.mean(present) if present else None


def summarise(samples: Sequence[Sample]) -> dict[str, Any]:
    """The headline numbers for one pipeline. Pure, so it can be tested."""
    golden = [s for s in samples if s.answerable and not s.error]
    unanswerable = [s for s in samples if not s.answerable and not s.error]
    released_wrongly = [s for s in unanswerable if s.answered and not s.held]
    return {
        "questions": len(samples),
        "errors": sum(1 for s in samples if s.error),
        "answered": _rate([s.answered for s in golden]),
        "faithfulness": _mean([s.faithfulness for s in golden]),
        "answer_relevancy": _mean([s.answer_relevancy for s in golden]),
        "answer_correctness": _mean([s.answer_correctness for s in golden]),
        "context_precision": _mean([context_precision(s.judgements) for s in golden]),
        "context_recall": _mean([context_recall(s.judgements) for s in golden]),
        "chunks_shown": _mean([float(len(s.contexts)) for s in golden]),
        "held": _rate([s.held for s in samples if not s.error]),
        "unanswerable_declined": _rate([not s.answered for s in unanswerable]),
        "unanswerable_released": len(released_wrongly),
    }


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def gate_failures(summary: dict[str, Any]) -> list[str]:
    """Every threshold a pipeline misses. An unscored metric is a failure: a
    gate that passes because nothing was measured is not a gate."""
    failures = []
    for metric, threshold in THRESHOLDS.items():
        value = summary.get(metric)
        if value is None:
            failures.append(f"{metric}: not measured")
        elif value < threshold:
            failures.append(f"{metric}: {value:.3f} < {threshold:.2f}")
    return failures


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(summaries: dict[str, dict[str, Any]], costs: dict[str, dict[str, Any]]) -> str:
    rows = [
        ("answered (golden)", "answered", 3),
        ("faithfulness", "faithfulness", 3),
        ("answer relevancy", "answer_relevancy", 3),
        ("answer correctness", "answer_correctness", 3),
        ("context precision", "context_precision", 3),
        ("context recall", "context_recall", 3),
        ("chunks shown", "chunks_shown", 2),
        ("held for review", "held", 3),
        ("unanswerable declined", "unanswerable_declined", 3),
    ]
    names = list(summaries)
    width = 24
    lines = [f"{'':{width}}" + "".join(f"{n:>12}" for n in names)]
    lines[0] += f"{'floor':>9}{'target':>9}"
    for label, key, digits in rows:
        floor = THRESHOLDS.get(key)
        target = TARGETS.get(key)
        lines.append(
            f"{label:{width}}"
            + "".join(f"{_fmt(summaries[n][key], digits):>12}" for n in names)
            + f"{_fmt(floor, 2):>9}{_fmt(target, 2):>9}"
        )
    lines.append(
        f"{'unanswerable released':{width}}"
        + "".join(f"{summaries[n]['unanswerable_released']:>12}" for n in names)
    )
    lines.append(
        f"{'model calls / question':{width}}"
        + "".join(f"{costs[n]['calls_per_question']:>12.2f}" for n in names)
    )
    lines.append(
        f"{'tokens / question':{width}}"
        + "".join(f"{costs[n]['tokens_per_question']:>12.0f}" for n in names)
    )
    return "\n".join(lines)


def _cost(run: PipelineRun, answered_questions: int) -> dict[str, Any]:
    clients = [run.answer_llm] + ([run.agent_llm] if run.agent_llm else [])
    calls = sum(c.usage.calls for c in clients)
    prompt = sum(c.usage.prompt_tokens for c in clients)
    completion = sum(c.usage.completion_tokens for c in clients)
    n = max(answered_questions, 1)
    return {
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "calls_per_question": calls / n,
        "tokens_per_question": (prompt + completion) / n,
    }


# --- entry point ----------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipelines", default="simple,agent", help="comma-separated: simple, agent")
    parser.add_argument("--judge-model", default=None, help="default: AGENT_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="first N questions only (smoke test)")
    parser.add_argument("--concurrency", type=int, default=4, help="parallel pipeline runs and judge calls")
    parser.add_argument("--gate", action="store_true", help="exit 1 if any pipeline misses a threshold")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation before spending")
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help=f"write per-question results here; the committed record is eval/{RESULTS_PATH.name}",
    )
    args = parser.parse_args()

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    logging.getLogger("langfuse").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    # Before any pipeline runs, so their spans have a client to go to. Without
    # LANGFUSE_* keys this builds a disabled client and tracing costs nothing.
    init_langfuse(settings)
    if not settings.generation_configured:
        raise SystemExit(f"{settings.generation_key_variable} is not set — this eval calls the model.")
    names = [n.strip() for n in args.pipelines.split(",") if n.strip()]
    if unknown := set(names) - {"simple", "agent"}:
        raise SystemExit(f"unknown pipeline(s): {', '.join(sorted(unknown))}")
    judge_model = args.judge_model or settings.agent_model

    cases = questions()[: args.limit] if args.limit else questions()
    golden = sum(1 for _, kind, _ in cases if kind != "unanswerable")
    # Upper bounds: the graph makes fewer calls when it routes away or grading
    # keeps something first time, and declines skip faithfulness.
    pipeline_calls = sum(len(cases) * (5 if n == "agent" else 1) for n in names)
    judge_calls = golden * sum(_JUDGE_CALLS.values()) * len(names)
    print(
        f"answer model: {settings.answer_model}  agent model: {settings.agent_model}  "
        f"judge: {judge_model}\n"
        f"pipelines: {', '.join(names)}  questions: {len(cases)} ({golden} answerable)\n"
        f"at most ~{pipeline_calls} pipeline calls + ~{judge_calls} judge calls"
    )
    if not args.yes and sys.stdin.isatty():
        if input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("aborted; nothing was spent")

    embedder = FastEmbedEmbedder(
        model_name=settings.embedding_model, cache_dir=settings.embedding_cache_dir
    )
    directory = tempfile.mkdtemp(prefix="edia-answer-eval-")
    client = QdrantClient(path=directory)
    try:
        store = QdrantVectorStore(client=client, collection="eval")
        chunks = index_corpus(store, embedder, None)
        # Dense, top-5: the shipped retrieval configuration, so the numbers
        # describe the deployed system.
        retriever = _SerialRetriever(DenseRetriever(embedder, store, TOP_K))
        runs = build_runs(settings, retriever, names)
        print(f"indexed {chunks} chunks\n")

        samples: dict[str, list[Sample]] = {}
        for run in runs:
            print(f"running {run.name}...")
            pending = [Sample(run.name, q, kind, gold) for q, kind, gold in cases]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                samples[run.name] = list(
                    pool.map(
                        lambda s, r=run: run_one(r, s, settings.review_min_confidence), pending
                    )
                )
    finally:
        client.close()
        shutil.rmtree(directory, ignore_errors=True)

    print("judging...")
    everything = [s for rows in samples.values() for s in rows]
    judge_failures = asyncio.run(judge(everything, settings, judge_model, args.concurrency * 2))

    if settings.langfuse_configured:
        sent = publish_scores(everything)
        print(f"attached {sent} scores to {len(everything)} Langfuse traces")
    summaries = {name: summarise(rows) for name, rows in samples.items()}
    costs = {
        run.name: _cost(run, sum(1 for s in samples[run.name] if not s.error)) for run in runs
    }

    print("\n## Results\n")
    print(_table(summaries, costs))
    if judge_failures:
        print(f"\n{judge_failures} judge call(s) failed and were left unscored.")

    _print_failures(samples)

    if args.json is None:
        print("\nper-question results not written (pass --json to keep them)")
    else:
        _write_results(args.json, settings, judge_model, cases, summaries, costs, samples)
        print(f"\nwrote {args.json}")

    if args.gate:
        failed = {n: f for n in summaries if (f := gate_failures(summaries[n]))}
        if failed:
            print("\n## Gate: FAILED\n")
            for name, reasons in failed.items():
                for reason in reasons:
                    print(f"  {name}: {reason}")
            raise SystemExit(1)
        print("\n## Gate: passed")


def _write_results(path: Path, settings, judge_model, cases, summaries, costs, samples) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {
                    "answer_model": settings.answer_model,
                    "agent_model": settings.agent_model,
                    "judge_model": judge_model,
                    "top_k": TOP_K,
                    "llm_temperature": settings.llm_temperature,
                    "review_min_confidence": settings.review_min_confidence,
                    "questions": len(cases),
                },
                "summary": summaries,
                "cost": costs,
                "samples": {
                    name: [
                        {k: v for k, v in asdict(s).items() if k != "contexts"} for s in rows
                    ]
                    for name, rows in samples.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_failures(samples: dict[str, list[Sample]]) -> None:
    """The questions behind the numbers. An average hides which answers failed,
    and that is the part worth reading."""
    print("\n## Worth reading\n")
    for name, rows in samples.items():
        for s in rows:
            problems = []
            if s.error:
                problems.append(f"error: {s.error}")
            elif not s.answerable:
                if s.answered and not s.held:
                    problems.append("answered an unanswerable question, and released it")
            else:
                if not s.answered:
                    problems.append("declined")
                if s.judgements and not any(s.judgements):
                    problems.append("gold passage never shown")
                if s.answer_correctness is not None and s.answer_correctness < 0.5:
                    problems.append(f"correctness {s.answer_correctness:.2f}")
                if s.faithfulness is not None and s.faithfulness < 0.8:
                    problems.append(f"faithfulness {s.faithfulness:.2f}")
            if problems:
                print(f"  [{name}] {s.question}\n      {'; '.join(problems)}")


if __name__ == "__main__":
    main()
