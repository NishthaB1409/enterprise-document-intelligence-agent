"""The answer eval's deterministic half: what gets scored, and how it adds up.

The judged metrics come from Ragas and a model, and are the eval's business.
What is pinned here is everything around them that could quietly skew a result
without any model being wrong: which samples a metric is applied to, how an
unanswerable question is scored, whether a gate can pass on nothing measured,
and whether the cost figures come from the clients that actually spent them.

None of this imports Ragas, so it runs on a plain `uv sync`.
"""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.generation.answerer import ANSWER_SCHEMA, GeneratedAnswer
from app.graph.pipeline import AgentPipeline, SimplePipeline
from app.llm import OpenAILLM
from eval.answer_eval import (
    THRESHOLDS,
    Sample,
    build_runs,
    gate_failures,
    metrics_for,
    PipelineRun,
    publish_scores,
    questions,
    reference_recall,
    run_one,
    scores_for,
    summarise,
)
from eval.metrics import context_precision, context_recall
from eval.questions import QUESTIONS, UNANSWERABLE


def _sample(**fields) -> Sample:
    defaults = dict(pipeline="simple", question="q", kind="lexical", gold_span="gold")
    return Sample(**{**defaults, **fields})


# --- context metrics ------------------------------------------------------------


@pytest.mark.parametrize(
    ("judgements", "expected"),
    [
        ([True, False, False], 1.0),
        ([False, True], 0.5),
        ([False, False, True], 1 / 3),
        # Two relevant: mean of precision@1 (1/1) and precision@3 (2/3).
        ([True, False, True], (1 + 2 / 3) / 2),
        ([False, False], 0.0),
        ([], 0.0),
    ],
)
def test_context_precision_is_ragas_rank_weighted_precision(judgements, expected):
    assert context_precision(judgements) == pytest.approx(expected)


def test_context_recall_is_whether_the_gold_passage_was_shown():
    assert context_recall([False, True]) == 1.0
    assert context_recall([False, False]) == 0.0
    assert context_recall([]) == 0.0


# --- which metrics apply ----------------------------------------------------------


def test_an_answered_question_gets_every_judged_metric():
    sample = _sample(answered=True, contexts=["chunk"])
    assert metrics_for(sample) == ["faithfulness", "answer_relevancy", "answer_correctness"]


def test_a_decline_is_not_scored_for_faithfulness_but_still_costs_relevancy():
    """A decline has no claims to check. Scoring it 1.0 would reward declining;
    scoring it 0.0 would call an honest answer unfaithful. Its cost belongs in
    relevancy and correctness, where a wrong decline is exactly the failure."""
    sample = _sample(answered=False, contexts=["chunk"])
    assert metrics_for(sample) == ["answer_relevancy", "answer_correctness"]


def test_unanswerable_questions_are_scored_without_a_judge():
    assert metrics_for(_sample(kind="unanswerable", gold_span=None, answered=True)) == []


def test_a_failed_question_is_not_judged():
    assert metrics_for(_sample(error="LLMError: boom", answered=True, contexts=["c"])) == []


# --- correctness ------------------------------------------------------------------


def test_correctness_is_the_share_of_reference_claims_the_answer_states():
    assert reference_recall([True, True]) == 1.0
    assert reference_recall([True, False]) == 0.5
    assert reference_recall([False]) == 0.0


def test_a_reference_with_no_claims_is_unscored_not_zero():
    value = reference_recall([])
    assert value != value  # NaN, which the judge loop records as unscored


# --- summarising --------------------------------------------------------------------


def test_the_summary_separates_the_golden_set_from_the_unanswerable_one():
    samples = [
        _sample(answered=True, judgements=[True], contexts=["a"], faithfulness=1.0,
                answer_relevancy=0.9, answer_correctness=1.0),
        _sample(answered=False, judgements=[False], contexts=["b"], faithfulness=None,
                answer_relevancy=0.0, answer_correctness=0.0),
        _sample(kind="unanswerable", gold_span=None, answered=False),
        # Answered, but the gate held it: safe.
        _sample(kind="unanswerable", gold_span=None, answered=True, held=True, reasons=["r"]),
        # Answered and released: the failure.
        _sample(kind="unanswerable", gold_span=None, answered=True),
    ]

    summary = summarise(samples)

    assert summary["answered"] == 0.5
    # The decline was skipped, not scored as zero.
    assert summary["faithfulness"] == 1.0
    assert summary["answer_relevancy"] == pytest.approx(0.45)
    assert summary["context_recall"] == 0.5
    assert summary["unanswerable_declined"] == pytest.approx(1 / 3)
    assert summary["unanswerable_released"] == 1


def test_errors_are_counted_and_kept_out_of_the_averages():
    samples = [
        _sample(answered=True, judgements=[True], faithfulness=1.0),
        _sample(error="boom"),
    ]
    summary = summarise(samples)
    assert summary["errors"] == 1
    assert summary["answered"] == 1.0
    assert summary["context_recall"] == 1.0


# --- the gate -----------------------------------------------------------------------


def _passing() -> dict:
    return {metric: 1.0 for metric in THRESHOLDS}


def test_the_gate_passes_when_every_target_is_met():
    assert gate_failures(_passing()) == []


def test_the_gate_names_each_missed_floor():
    floor = THRESHOLDS["faithfulness"]
    summary = _passing() | {"faithfulness": floor - 0.05}
    assert gate_failures(summary) == [f"faithfulness: {floor - 0.05:.3f} < {floor:.2f}"]


def test_the_gate_floors_sit_below_the_readme_targets():
    """Floors catch regressions; targets are goals. A floor above its target
    would mean the gate demands more than the goal does."""
    from eval.answer_eval import TARGETS

    for metric, target in TARGETS.items():
        assert THRESHOLDS[metric] <= target


def test_an_unmeasured_metric_fails_the_gate():
    """A gate that passes because nothing was scored is not a gate."""
    summary = _passing() | {"answer_relevancy": None}
    assert gate_failures(summary) == ["answer_relevancy: not measured"]


# --- the cases ----------------------------------------------------------------------


def test_every_question_is_asked_once_golden_set_first():
    cases = questions()
    assert len(cases) == len(QUESTIONS) + len(UNANSWERABLE)
    assert [kind for _, kind, _ in cases[len(QUESTIONS):]] == ["unanswerable"] * len(UNANSWERABLE)
    assert all(gold is None for _, kind, gold in cases if kind == "unanswerable")


# --- cost accounting ------------------------------------------------------------------


def test_each_pipeline_gets_its_own_clients():
    """Shared clients would put the graph's calls on the single pass's bill."""
    settings = Settings(_env_file=None, openai_api_key="sk-test")

    simple, agent = build_runs(settings, retriever=object(), names=["simple", "agent"])

    assert isinstance(simple.pipeline, SimplePipeline)
    assert simple.agent_llm is None
    assert isinstance(agent.pipeline, AgentPipeline)
    assert agent.agent_llm is not None
    assert agent.answer_llm is not simple.answer_llm
    # The graph's nodes run on the injected client, not one built internally —
    # otherwise its usage would be invisible to the eval.
    assert agent.pipeline._router._llm is agent.agent_llm
    assert agent.pipeline._critic._llm is agent.agent_llm


def test_usage_is_counted_from_what_the_api_reports():
    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test")
    payload = GeneratedAnswer(answer="Thirty days.", claims=[], answerable=True)
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=30),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(refusal=None, content=payload.model_dump_json()),
                finish_reason="stop",
            )
        ],
    )
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )

    for _ in range(2):
        llm.complete(
            system="s", prompt="p", schema=ANSWER_SCHEMA, schema_name="a", model=GeneratedAnswer
        )

    assert (llm.usage.calls, llm.usage.prompt_tokens, llm.usage.completion_tokens) == (2, 240, 60)
    assert llm.usage.total_tokens == 300


# --- Langfuse ------------------------------------------------------------------------


def test_scores_for_a_golden_answer():
    sample = _sample(
        answered=True, judgements=[False, True], faithfulness=0.8,
        answer_relevancy=0.9, answer_correctness=1.0,
    )
    assert scores_for(sample) == {
        "faithfulness": 0.8,
        "answer_relevancy": 0.9,
        "answer_correctness": 1.0,
        "context_precision": 0.5,
        "context_recall": 1.0,
        "held_for_review": 0.0,
    }


def test_unscored_metrics_are_left_off_rather_than_sent_as_zero():
    sample = _sample(answered=False, judgements=[True], answer_relevancy=0.0)
    assert "faithfulness" not in scores_for(sample)
    assert scores_for(sample)["answer_relevancy"] == 0.0


def test_an_unanswerable_question_gets_no_context_scores():
    sample = _sample(kind="unanswerable", gold_span=None, answered=False)
    assert scores_for(sample) == {"held_for_review": 0.0}


def test_a_failed_run_sends_no_scores():
    assert scores_for(_sample(error="boom")) == {}


def test_each_eval_run_is_traced_and_its_scores_land_on_that_trace(client, spans, flush):
    """The point of the integration: a score in Langfuse opens onto the run that
    earned it. Asserted against what the SDK actually sent to the fake server,
    so a score that reaches Langfuse with the wrong trace id fails here."""
    from app.generation.answerer import GeneratedAnswer
    from app.graph.pipeline import PipelineResult
    from app.ingest.chunking import Chunk
    from app.vectorstore.store import ScoredChunk

    text = "Invoices are payable within thirty (30) days."
    chunk = ScoredChunk(
        chunk=Chunk(id="c1", doc_id="d", index=0, page=1, text=text, char_start=0,
                    char_end=len(text)),
        source="msa.pdf",
        score=0.9,
    )
    answer = GeneratedAnswer.model_validate(
        {"answer": "Thirty days.", "answerable": True,
         "claims": [{"text": "Thirty days.", "sources": [1]}]}
    )

    class _Pipeline:
        def run(self, question, top_k=None):
            return PipelineResult(answer=answer, chunks=[chunk], steps=["retrieve", "generate"])

    run = PipelineRun("simple", _Pipeline(), answer_llm=None, agent_llm=None)
    sample = run_one(run, _sample(gold_span="payable within thirty (30) days"), 0.7)
    sample.faithfulness = 1.0

    sent = publish_scores([sample])
    flush()

    span = spans.by_name("answer-eval/simple")
    assert sample.trace_id == span.trace_id
    received = {s["name"]: s for s in spans.scores}
    assert sent == len(received) == 4  # faithfulness, two context scores, held
    assert all(s["traceId"] == span.trace_id for s in received.values())
    assert received["faithfulness"]["value"] == 1.0
    assert received["context_recall"]["value"] == 1.0
    assert received["held_for_review"]["dataType"] == "BOOLEAN"
