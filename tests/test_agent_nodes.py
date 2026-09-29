"""How each node behaves, especially when the model does not cooperate.

The happy paths here are nearly trivial — the node asks for a shape and gets it.
What is worth pinning is the other half: every one of these four calls is an
*addition* to a pipeline that worked without it, so a node that fails must
degrade to the old behaviour rather than take the query down with it. Routing
failure should search anyway; grading failure should keep everything; a bad
rewrite should not burn the retry budget; a dead critic should say "unreviewed"
rather than "fine".

Those are the assertions below. They are also the ones a stub can actually make
— whether the grader has good taste is a question for `eval/`, not for a test.
"""

import pytest

from app.graph.nodes import Critic, Grader, Rewriter, Router
from app.ingest.chunking import Chunk
from app.llm import LLMError
from app.vectorstore.store import ScoredChunk


def _chunk(index: int, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"chunk-{index}",
            doc_id="doc-1",
            index=index,
            page=1,
            text=text,
            char_start=0,
            char_end=len(text),
        ),
        source="contract.pdf",
        score=0.9,
    )


CHUNKS = [
    _chunk(0, "Either party may terminate on ninety days notice."),
    _chunk(1, "The office is closed on public holidays."),
]


class _LLM:
    """Returns a preset payload, or raises. Records what it was asked."""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self._payload = payload
        self._error = error
        self.calls: list[dict] = []

    def complete(self, *, system, prompt, schema, schema_name, model, max_tokens=None):
        self.calls.append(
            {"system": system, "prompt": prompt, "max_tokens": max_tokens}
        )
        if self._error is not None:
            raise self._error
        return model.model_validate(self._payload)


class TestRouter:
    def test_it_reports_the_models_decision(self):
        router = Router(_LLM({"needs_documents": False, "reason": "greeting"}))

        decision = router.route("hello there")

        assert decision.needs_documents is False
        assert decision.reason == "greeting"

    def test_a_failed_route_searches_anyway(self):
        """Routing is an optimisation, not a correctness requirement. If it is
        unavailable the right move is to search — the failure then surfaces at
        generation, on the user's actual question, rather than on a preliminary
        about it."""
        router = Router(_LLM(error=LLMError("upstream is down")))

        decision = router.route("What is the notice period?")

        assert decision.needs_documents is True

    def test_it_asks_for_a_bounded_answer(self):
        """A route decision is a boolean and a sentence. A generous ceiling
        would not make it better, and the graph pays this on every query."""
        llm = _LLM({"needs_documents": True, "reason": "about documents"})

        Router(llm).route("q")

        assert llm.calls[0]["max_tokens"] == 2000


class TestGrader:
    def test_it_keeps_only_what_was_graded_relevant(self):
        llm = _LLM(
            {
                "grades": [
                    {"source": 1, "relevant": True, "reason": "on point"},
                    {"source": 2, "relevant": False, "reason": "unrelated"},
                ]
            }
        )

        kept, dropped = Grader(llm).keep_relevant("notice period?", CHUNKS)

        assert [c.chunk.id for c in kept] == ["chunk-0"]
        assert len(dropped) == 1
        assert "unrelated" in dropped[0]

    def test_it_preserves_the_order_it_was_given(self):
        """The generator numbers sources by position. Reordering here would
        renumber every citation for no reason."""
        llm = _LLM(
            {
                "grades": [
                    {"source": 2, "relevant": True, "reason": "b"},
                    {"source": 1, "relevant": True, "reason": "a"},
                ]
            }
        )

        kept, _ = Grader(llm).keep_relevant("q", CHUNKS)

        assert [c.chunk.id for c in kept] == ["chunk-0", "chunk-1"]

    def test_an_ungraded_source_is_kept(self):
        """Dropping whatever the grader forgot to mention would silently lose
        evidence on a malformed response — a failure that looks like a thin
        answer rather than like a bug."""
        llm = _LLM({"grades": [{"source": 1, "relevant": True, "reason": "on point"}]})

        kept, _ = Grader(llm).keep_relevant("q", CHUNKS)

        assert [c.chunk.id for c in kept] == ["chunk-0", "chunk-1"]

    def test_a_grade_for_a_source_that_was_never_offered_is_ignored(self):
        """A verdict on source 9 when two were given says nothing about source
        2, so it is dropped rather than clamped onto one."""
        llm = _LLM(
            {
                "grades": [
                    {"source": 9, "relevant": False, "reason": "hallucinated"},
                    {"source": 1, "relevant": False, "reason": "unrelated"},
                ]
            }
        )

        kept, _ = Grader(llm).keep_relevant("q", CHUNKS)

        # Source 1 dropped on its own verdict; source 2 kept because nothing
        # valid was said about it.
        assert [c.chunk.id for c in kept] == ["chunk-1"]

    def test_a_failed_grade_keeps_everything(self):
        """Grading exists to raise precision. Failing it should cost precision,
        not the answer."""
        grader = Grader(_LLM(error=LLMError("upstream is down")))

        kept, dropped = grader.keep_relevant("q", CHUNKS)

        assert len(kept) == 2
        assert dropped == []

    def test_no_chunks_means_no_call(self):
        llm = _LLM(error=AssertionError("must not be called"))

        assert Grader(llm).keep_relevant("q", []) == ([], [])


class TestRewriter:
    def test_it_returns_the_rewritten_query(self):
        llm = _LLM({"query": "termination for convenience notice", "reason": "formalised"})

        assert Rewriter(llm).rewrite("how do I quit early", "how do I quit early") == (
            "termination for convenience notice"
        )

    def test_a_failed_rewrite_falls_back_to_the_previous_query(self):
        rewriter = Rewriter(_LLM(error=LLMError("upstream is down")))

        assert rewriter.rewrite("original", "previous") == "previous"

    def test_an_empty_rewrite_falls_back_rather_than_searching_for_nothing(self):
        rewriter = Rewriter(_LLM({"query": "   ", "reason": "blank"}))

        assert rewriter.rewrite("original", "previous") == "previous"

    def test_the_previous_attempt_is_shown_only_when_it_differs(self):
        """On the first rewrite there is no failed attempt to avoid repeating,
        and saying "already tried: <the question>" invites the model to treat
        the question itself as the thing that failed."""
        first = _LLM({"query": "x", "reason": "r"})
        Rewriter(first).rewrite("same", "same")
        assert "Already tried" not in first.calls[0]["prompt"]

        second = _LLM({"query": "y", "reason": "r"})
        Rewriter(second).rewrite("same", "a previous attempt")
        assert "Already tried: a previous attempt" in second.calls[0]["prompt"]


class TestCritic:
    def test_it_returns_the_review(self):
        llm = _LLM({"supported": False, "confidence": 0.25, "concerns": ["figure not in source"]})

        critique = Critic(llm).review("q", "an answer", CHUNKS)

        assert critique is not None
        assert critique.supported is False
        assert critique.confidence == 0.25

    def test_it_returns_conflicts_between_sources(self):
        llm = _LLM(
            {
                "supported": True,
                "confidence": 0.8,
                "concerns": [],
                "conflicts": [{"sources": [1, 2], "description": "90 vs 60 days"}],
            }
        )

        critique = Critic(llm).review("q", "an answer", CHUNKS)

        (conflict,) = critique.conflicts
        assert conflict.sources == [1, 2]

    def test_contradiction_detection_costs_no_extra_call(self):
        """It rides on the critique the graph already pays for."""
        llm = _LLM({"supported": True, "confidence": 0.9, "concerns": [], "conflicts": []})

        Critic(llm).review("q", "an answer", CHUNKS)

        assert len(llm.calls) == 1
        assert "contradict each other" in llm.calls[0]["system"]

    def test_the_model_must_commit_to_a_conflicts_list(self):
        """Required in the strict schema, so "none" is an answer the model gave
        rather than a field it left out."""
        from app.graph.nodes import CRITIQUE_SCHEMA

        assert "conflicts" in CRITIQUE_SCHEMA["required"]

    def test_a_failed_critique_is_none_rather_than_a_pass(self):
        """A default clean verdict would let a broken critic look like a clean
        bill of health, which is precisely what a review gate must not act on."""
        critic = Critic(_LLM(error=LLMError("upstream is down")))

        assert critic.review("q", "an answer", CHUNKS) is None

    def test_nothing_to_review_means_no_call(self):
        llm = _LLM(error=AssertionError("must not be called"))

        assert Critic(llm).review("q", "an answer", []) is None

    def test_the_answer_and_the_sources_both_reach_the_prompt(self):
        llm = _LLM({"supported": True, "confidence": 0.9, "concerns": []})

        Critic(llm).review("What is the notice period?", "Ninety days.", CHUNKS)

        prompt = llm.calls[0]["prompt"]
        assert "Ninety days." in prompt
        assert "ninety days notice" in prompt
        assert "What is the notice period?" in prompt


def test_every_node_numbers_sources_the_same_way():
    """A grade, a critique, and a citation all say "source 3". They have to mean
    the same chunk, and divergent numbering would be invisible in testing."""
    grader_llm = _LLM({"grades": []})
    critic_llm = _LLM({"supported": True, "confidence": 1.0, "concerns": []})

    Grader(grader_llm).keep_relevant("q", CHUNKS)
    Critic(critic_llm).review("q", "a", CHUNKS)

    for llm in (grader_llm, critic_llm):
        prompt = llm.calls[0]["prompt"]
        assert "[1] contract.pdf, page 1" in prompt
        assert "[2] contract.pdf, page 1" in prompt
