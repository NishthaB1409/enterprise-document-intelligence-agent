"""The paths through the agentic graph, and what happens on each.

Every node is stubbed. The graph's job is not to be clever — it is to take the
right edge, keep the retry budget, and hand the generator the right chunks — and
those are properties of the wiring, not of any model's judgement. Stubbing the
nodes is what makes them assertable at all: with a real model the same test
would be checking whether the model happened to agree today.

`steps` is asserted directly rather than inferred from the output. Two different
paths can produce the same answer — a question routed away and a question that
retrieved nothing both come back unanswerable — and a test that only looked at
the answer would pass while the graph took the wrong route.
"""

from app.generation.answerer import GeneratedAnswer
from app.graph.nodes import Critique, Route
from app.graph.pipeline import OFF_TOPIC_ANSWER, AgentPipeline, SimplePipeline
from app.ingest.chunking import Chunk
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
        score=1.0 - index * 0.1,
    )


TERMINATION = _chunk(0, "Either party may terminate on ninety days written notice.")
LIABILITY = _chunk(1, "Aggregate liability shall not exceed the fees paid.")
IRRELEVANT = _chunk(2, "The office is closed on public holidays.")


class _Retriever:
    """Returns a preset result per query, and records what it was asked.

    Keyed by query so a rewrite can be made to return something different from
    the original question — which is the only way to test that rewriting
    actually changed the search rather than just incrementing a counter.
    """

    def __init__(self, results, default=None) -> None:
        self._results = results
        self._default = default if default is not None else []
        self.queries: list[str] = []

    def retrieve(self, question: str, top_k: int | None = None):
        self.queries.append(question)
        return list(self._results.get(question, self._default))


class _Answerer:
    def __init__(self, answer: GeneratedAnswer | None = None) -> None:
        self._answer = answer
        self.calls: list[tuple[str, list[ScoredChunk]]] = []

    def answer(self, question: str, chunks):
        self.calls.append((question, list(chunks)))
        if self._answer is not None:
            return self._answer
        if not chunks:
            return GeneratedAnswer(answer="Nothing relevant.", claims=[], answerable=False)
        return GeneratedAnswer(
            answer="Ninety days.",
            claims=[{"text": "Ninety days.", "sources": [1]}],
            answerable=True,
        )


class _Router:
    def __init__(self, needs_documents: bool) -> None:
        self._needs = needs_documents
        self.calls: list[str] = []

    def route(self, question: str) -> Route:
        self.calls.append(question)
        return Route(needs_documents=self._needs, reason="because")


class _Grader:
    """Keeps chunks whose text contains a marker, drops the rest."""

    def __init__(self, keep_containing: str | None) -> None:
        self._keep = keep_containing
        self.calls: list[list[ScoredChunk]] = []

    def keep_relevant(self, question: str, chunks):
        self.calls.append(list(chunks))
        if self._keep is None:
            return [], [f"[{i}] dropped" for i in range(1, len(chunks) + 1)]
        kept = [c for c in chunks if self._keep in c.chunk.text]
        dropped = [f"[{i}] dropped" for i, c in enumerate(chunks, 1) if c not in kept]
        return kept, dropped


class _Rewriter:
    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten
        self.calls: list[tuple[str, str]] = []

    def rewrite(self, question: str, previous: str) -> str:
        self.calls.append((question, previous))
        return self._rewritten


class _Critic:
    def __init__(self, critique: Critique | None) -> None:
        self._critique = critique
        self.calls: list[tuple[str, str, list[ScoredChunk]]] = []

    def review(self, question: str, answer: str, chunks):
        self.calls.append((question, answer, list(chunks)))
        return self._critique


def _pipeline(**overrides) -> AgentPipeline:
    defaults = dict(
        retriever=_Retriever({}, default=[TERMINATION]),
        answerer=_Answerer(),
        router=_Router(True),
        grader=_Grader("terminate"),
        rewriter=_Rewriter("termination for convenience notice period"),
        critic=_Critic(Critique(supported=True, confidence=0.9, concerns=[])),
        max_rewrites=1,
    )
    # An explicit None in `overrides` disables that node, which is a case the
    # graph has to handle and several tests below rely on.
    return AgentPipeline(**{**defaults, **overrides})


class TestRouting:
    def test_an_off_topic_question_never_reaches_retrieval(self):
        retriever = _Retriever({}, default=[TERMINATION])
        answerer = _Answerer()
        pipeline = _pipeline(router=_Router(False), retriever=retriever, answerer=answerer)

        result = pipeline.run("Who won the World Cup?")

        # The point of routing here is that it declines. Answering from the
        # model's own knowledge is the failure this node exists to prevent.
        assert result.answer.answer == OFF_TOPIC_ANSWER
        assert result.answer.answerable is False
        assert retriever.queries == []
        assert answerer.calls == []
        assert any(step.startswith("route:direct") for step in result.steps)

    def test_a_document_question_proceeds_to_retrieval(self):
        retriever = _Retriever({}, default=[TERMINATION])
        pipeline = _pipeline(router=_Router(True), retriever=retriever)

        result = pipeline.run("What is the notice period?")

        assert retriever.queries == ["What is the notice period?"]
        assert result.answer.answerable is True
        assert "route:retrieve" in result.steps

    def test_routing_can_be_disabled_without_changing_the_rest(self):
        pipeline = _pipeline(router=None)

        result = pipeline.run("What is the notice period?")

        assert "route:skipped" in result.steps
        assert result.answer.answerable is True


class TestGrading:
    def test_only_graded_chunks_reach_the_generator(self):
        """The generator numbers its citations by position in what it is given,
        so a chunk dropped after generation would misnumber every citation. It
        has to be dropped before."""
        retriever = _Retriever({}, default=[TERMINATION, IRRELEVANT])
        answerer = _Answerer()
        pipeline = _pipeline(retriever=retriever, answerer=answerer, grader=_Grader("terminate"))

        result = pipeline.run("What is the notice period?")

        (_, given), = answerer.calls
        assert [c.chunk.id for c in given] == [TERMINATION.chunk.id]
        # And the caller gets the same list, so grounding resolves against it.
        assert [c.chunk.id for c in result.chunks] == [TERMINATION.chunk.id]

    def test_dropped_sources_are_reported_rather_than_vanishing(self):
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[TERMINATION, IRRELEVANT]),
            grader=_Grader("terminate"),
        )

        result = pipeline.run("What is the notice period?")

        assert len(result.dropped) == 1

    def test_grading_can_be_disabled_and_everything_reaches_the_generator(self):
        answerer = _Answerer()
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[TERMINATION, IRRELEVANT]),
            answerer=answerer,
            grader=None,
        )

        result = pipeline.run("What is the notice period?")

        (_, given), = answerer.calls
        assert len(given) == 2
        assert "grade:skipped" in result.steps


class TestRewriteLoop:
    def test_nothing_relevant_triggers_a_rewrite_and_a_second_search(self):
        """The whole point of the loop: a question phrased unlike the document
        gets re-asked in the document's vocabulary."""
        rewritten = "termination for convenience notice period"
        retriever = _Retriever(
            {"How do I get out early?": [IRRELEVANT], rewritten: [TERMINATION]}
        )
        pipeline = _pipeline(
            retriever=retriever,
            grader=_Grader("terminate"),
            rewriter=_Rewriter(rewritten),
        )

        result = pipeline.run("How do I get out early?")

        assert retriever.queries == ["How do I get out early?", rewritten]
        assert [c.chunk.id for c in result.chunks] == [TERMINATION.chunk.id]
        assert any(step.startswith("rewrite:") for step in result.steps)

    def test_the_rewriter_always_sees_the_original_question(self):
        """Rewriting a rewrite compounds drift — two hops in, the search is for
        something the user did not ask."""
        rewriter = _Rewriter("second attempt")
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[IRRELEVANT]),
            grader=_Grader(None),
            rewriter=rewriter,
            max_rewrites=2,
        )

        pipeline.run("original question")

        assert [question for question, _ in rewriter.calls] == [
            "original question",
            "original question",
        ]
        # ...while the previous attempt is passed separately, so the rewriter
        # can avoid repeating a search that already failed.
        assert [previous for _, previous in rewriter.calls] == [
            "original question",
            "second attempt",
        ]

    def test_the_retry_budget_is_enforced(self):
        retriever = _Retriever({}, default=[IRRELEVANT])
        pipeline = _pipeline(
            retriever=retriever, grader=_Grader(None), rewriter=_Rewriter("x"), max_rewrites=2
        )

        pipeline.run("unanswerable")

        # One initial search plus exactly two retries, then it stops.
        assert len(retriever.queries) == 3

    def test_exhausting_the_budget_answers_honestly_rather_than_raising(self):
        """A loop that ran out is not an error. The corpus genuinely may not
        cover the question, and saying so is the correct answer."""
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[IRRELEVANT]),
            grader=_Grader(None),
            rewriter=_Rewriter("x"),
            max_rewrites=1,
        )

        result = pipeline.run("unanswerable")

        assert result.answer.answerable is False
        assert result.chunks == []

    def test_no_rewrite_happens_when_grading_keeps_something(self):
        rewriter = _Rewriter("should not be used")
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[TERMINATION]),
            grader=_Grader("terminate"),
            rewriter=rewriter,
        )

        pipeline.run("What is the notice period?")

        assert rewriter.calls == []


class TestCritique:
    def test_the_critique_reaches_the_caller(self):
        critique = Critique(supported=False, confidence=0.3, concerns=["unsupported figure"])
        pipeline = _pipeline(critic=_Critic(critique))

        result = pipeline.run("What is the notice period?")

        assert result.critique is not None
        assert result.critique.confidence == 0.3
        assert result.critique.concerns == ["unsupported figure"]

    def test_the_critic_reviews_the_answer_against_the_graded_chunks(self):
        """Critiquing against the pre-grading chunks would let it defend a claim
        using a source the generator never saw."""
        critic = _Critic(Critique(supported=True, confidence=1.0, concerns=[]))
        pipeline = _pipeline(
            retriever=_Retriever({}, default=[TERMINATION, IRRELEVANT]),
            grader=_Grader("terminate"),
            critic=critic,
        )

        pipeline.run("What is the notice period?")

        (_, answer, chunks), = critic.calls
        assert answer == "Ninety days."
        assert [c.chunk.id for c in chunks] == [TERMINATION.chunk.id]

    def test_an_unavailable_critic_is_null_rather_than_a_pass(self):
        """None has to mean "not reviewed". A default clean verdict would let a
        broken critic look like a clean bill of health, which is exactly what
        phase 4's gate must not act on."""
        pipeline = _pipeline(critic=_Critic(None))

        result = pipeline.run("What is the notice period?")

        assert result.critique is None

    def test_critique_can_be_disabled(self):
        pipeline = _pipeline(critic=None)

        result = pipeline.run("What is the notice period?")

        assert result.critique is None
        assert "critique:skipped" in result.steps


class TestSimplePipeline:
    def test_it_retrieves_then_generates_and_nothing_else(self):
        retriever = _Retriever({}, default=[TERMINATION])
        answerer = _Answerer()

        result = SimplePipeline(retriever, answerer).run("What is the notice period?")

        assert result.steps == ["retrieve", "generate"]
        assert result.critique is None
        assert result.dropped == []
        assert [c.chunk.id for c in result.chunks] == [TERMINATION.chunk.id]

    def test_the_per_request_top_k_reaches_the_retriever(self):
        class _Recording(_Retriever):
            def __init__(self):
                super().__init__({}, default=[TERMINATION])
                self.ks: list[int | None] = []

            def retrieve(self, question, top_k=None):
                self.ks.append(top_k)
                return super().retrieve(question, top_k)

        retriever = _Recording()
        SimplePipeline(retriever, _Answerer()).run("q", top_k=9)

        assert retriever.ks == [9]


def test_the_agent_pipeline_also_honours_a_per_request_top_k():
    """It travels through the graph state rather than as an argument, which is
    exactly the kind of wiring that silently stops working."""

    class _Recording(_Retriever):
        def __init__(self):
            super().__init__({}, default=[TERMINATION])
            self.ks: list[int | None] = []

        def retrieve(self, question, top_k=None):
            self.ks.append(top_k)
            return super().retrieve(question, top_k)

    retriever = _Recording()
    _pipeline(retriever=retriever).run("What is the notice period?", top_k=7)

    assert retriever.ks == [7]
