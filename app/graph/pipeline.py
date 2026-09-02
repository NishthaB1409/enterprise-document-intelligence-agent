"""Question in, grounded answer out — by one of two routes.

`SimplePipeline` is phase 1: retrieve, generate. `AgentPipeline` is the graph:

        route ──not about the documents──────────────────────▶ end
          │
          │ needs documents
          ▼
        retrieve ──▶ grade ──nothing relevant, budget left──▶ rewrite
          ▲                                                     │
          └─────────────────────────────────────────────────────┘
                          │ relevant, or out of retries
                          ▼
                       generate ──▶ critique ──▶ end

Both satisfy `QueryPipeline`, so `/query` does not know which it has and the
choice is one setting. Keeping the simple one is the same decision as keeping
`DenseRetriever` in phase 2: it is the baseline the graph has to justify itself
against, and deleting it would delete the ability to compare.

Why a graph rather than five function calls in a row. The control flow here is
not a line — grading can send the question back to retrieval, and that edge can
be taken more than once. Written as nested conditionals the retry budget ends up
as a loop counter threaded through five signatures, and "which paths exist" is
something you reconstruct by reading. As a state machine the edges are declared
in one place, the budget lives in the state, and the path actually taken comes
out in `steps` for free — which is what makes a surprising answer diagnosable
after the fact rather than by re-running it.

LangGraph's own recursion limit is a backstop, not the mechanism: the rewrite
budget is enforced on the edge, so exhausting it produces an honest answer from
whatever was retrieved rather than an exception.
"""

import logging
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, START, StateGraph

from app.generation.answerer import Answerer, GeneratedAnswer
from app.graph.nodes import Critic, Critique, Grader, Router, Rewriter
from app.retrieval.retriever import Retriever
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)

# What the router produces for a question the corpus cannot answer. Deliberately
# not a model-knowledge answer: the value of routing here is that it *declines*,
# so a question about the World Cup gets an honest non-answer instead of a
# confident one with an empty citation list.
OFF_TOPIC_ANSWER = (
    "This service only answers questions about the documents indexed in it. "
    "That question does not appear to be about them, so there is nothing to cite."
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    answer: GeneratedAnswer
    # Exactly the chunks the generator saw, in the order it saw them. The
    # citation numbers it returned are positions in this list, so it has to be
    # this list that grounding resolves against — not what retrieval originally
    # returned, which grading may have shortened.
    chunks: list[ScoredChunk]
    # None means the answer was not reviewed, which is not the same as reviewed
    # and found clean. Phase 4's gate has to be able to tell those apart.
    critique: Critique | None = None
    # The nodes that actually ran, in order. Returned to the caller and attached
    # to the trace, because "why did this answer come back thin" is usually
    # answered by which path it took.
    steps: list[str] = field(default_factory=list)
    # Chunks the grader discarded, with its reasoning. Kept because a wrongly
    # dropped source is otherwise invisible: the answer just quietly lacks it.
    dropped: list[str] = field(default_factory=list)


@runtime_checkable
class QueryPipeline(Protocol):
    def run(self, question: str, top_k: int | None = None) -> PipelineResult: ...


class SimplePipeline:
    """Retrieve, then generate. The phase 1 path, and the graph's baseline."""

    def __init__(self, retriever: Retriever, answerer: Answerer) -> None:
        self._retriever = retriever
        self._answerer = answerer

    def run(self, question: str, top_k: int | None = None) -> PipelineResult:
        chunks = self._retriever.retrieve(question, top_k)
        return PipelineResult(
            answer=self._answerer.answer(question, chunks),
            chunks=chunks,
            steps=["retrieve", "generate"],
        )


class AgentState(TypedDict, total=False):
    """What flows between nodes.

    `question` is the user's, and never changes. `search_query` is what
    retrieval is actually given and is what the rewriter replaces — keeping them
    apart is what stops a second rewrite drifting away from what was asked.
    """

    question: str
    search_query: str
    # Per-request override of the retriever's configured k, carried in the state
    # because the node that needs it is two hops from the caller.
    top_k: int | None
    chunks: list[ScoredChunk]
    dropped: list[str]
    rewrites: int
    answer: GeneratedAnswer
    critique: Critique | None
    # Appended to by every node rather than overwritten, so the value at the end
    # is the whole path.
    steps: Annotated[list[str], operator.add]


class AgentPipeline:
    """The routed, self-correcting path.

    Costs three to five model calls per question against the simple path's one:
    route, grade, generate, critique, plus one per rewrite. That is the price of
    the graph and it should be spent on purpose — `Settings.agent_enabled` turns
    it off, and the nodes that are pure overhead on a well-behaved corpus
    (routing, critique) can be disabled individually.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        answerer: Answerer,
        router: Router | None,
        grader: Grader | None,
        rewriter: Rewriter | None,
        critic: Critic | None,
        max_rewrites: int = 1,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._router = router
        self._grader = grader
        self._rewriter = rewriter
        self._critic = critic
        self._max_rewrites = max_rewrites
        self._graph = self._build()

    # -- nodes ------------------------------------------------------------

    def _route(self, state: AgentState) -> dict[str, Any]:
        if self._router is None:
            return {"steps": ["route:skipped"]}

        decision = self._router.route(state["question"])
        if decision.needs_documents:
            return {"steps": ["route:retrieve"]}

        return {
            "answer": GeneratedAnswer(
                answer=OFF_TOPIC_ANSWER, claims=[], answerable=False
            ),
            "chunks": [],
            "steps": [f"route:direct ({decision.reason})"],
        }

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        query = state.get("search_query") or state["question"]
        chunks = self._retriever.retrieve(query, state.get("top_k"))
        return {"chunks": chunks, "steps": ["retrieve"]}

    def _grade(self, state: AgentState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        if self._grader is None:
            return {"steps": ["grade:skipped"]}

        kept, dropped = self._grader.keep_relevant(state["question"], chunks)
        return {
            "chunks": kept,
            # Accumulated across retries: a source dropped on the first attempt
            # is still a source that was dropped.
            "dropped": state.get("dropped", []) + dropped,
            "steps": [f"grade:{len(kept)}/{len(chunks)}"],
        }

    def _rewrite(self, state: AgentState) -> dict[str, Any]:
        previous = state.get("search_query") or state["question"]
        rewritten = (
            self._rewriter.rewrite(state["question"], previous)
            if self._rewriter is not None
            else previous
        )
        return {
            "search_query": rewritten,
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [f"rewrite:{rewritten!r}"],
        }

    def _generate(self, state: AgentState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        return {
            "answer": self._answerer.answer(state["question"], chunks),
            "steps": ["generate"],
        }

    def _critique(self, state: AgentState) -> dict[str, Any]:
        if self._critic is None:
            return {"steps": ["critique:skipped"]}

        critique = self._critic.review(
            state["question"], state["answer"].answer, state.get("chunks", [])
        )
        label = "unavailable" if critique is None else f"{critique.confidence:.2f}"
        return {"critique": critique, "steps": [f"critique:{label}"]}

    # -- edges ------------------------------------------------------------

    def _after_route(self, state: AgentState) -> str:
        # The router writes an answer only when it decided not to search, so its
        # presence here is the decision.
        return END if state.get("answer") is not None else "retrieve"

    def _after_grade(self, state: AgentState) -> str:
        if state.get("chunks"):
            return "generate"
        if state.get("rewrites", 0) < self._max_rewrites:
            return "rewrite"
        # Out of retries with nothing relevant. Generating on an empty list is
        # deliberate: the answerer's response to no evidence is "no indexed
        # document covers this", which is the correct answer and a more useful
        # one than an error.
        return "generate"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)
        graph.add_node("critique", self._critique)

        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._after_route, {"retrieve": "retrieve", END: END})
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade", self._after_grade, {"generate": "generate", "rewrite": "rewrite"}
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", "critique")
        graph.add_edge("critique", END)
        return graph.compile()

    # -- entry point ------------------------------------------------------

    def run(self, question: str, top_k: int | None = None) -> PipelineResult:
        final: AgentState = self._graph.invoke(
            {
                "question": question,
                "search_query": question,
                "top_k": top_k,
                "rewrites": 0,
                "steps": [],
            }
        )

        answer = final.get("answer")
        if answer is None:
            # Unreachable while every terminal path writes an answer, and a
            # 500 with no explanation if that ever stops being true.
            raise RuntimeError("the graph finished without producing an answer")

        return PipelineResult(
            answer=answer,
            chunks=final.get("chunks", []),
            critique=final.get("critique"),
            steps=final.get("steps", []),
            dropped=final.get("dropped", []),
        )


def describe(steps: Sequence[str]) -> str:
    """The path taken, for a log line or a trace attribute."""
    return " -> ".join(steps)
