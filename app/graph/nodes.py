"""The decisions the agent makes, one class each.

Each node is a prompt, a Pydantic model describing the answer it wants, and the
handful of lines that turn one into the other. They know nothing about the graph
that runs them — they are ordinary objects with ordinary methods, which is what
lets every one of them be tested without a graph, a model, or a network.

Why these four, specifically:

    Router     stops the pipeline answering "who won the World Cup" out of the
               model's own memory. A document agent that will answer anything
               is a chatbot with a citation field.

    Grader     retrieval returns the k nearest chunks whether or not any of them
               are relevant — "nearest" is not "relevant", and with a small
               corpus the k-th result is often noise. Dropping the ones that
               cannot support an answer is what stops the generator being handed
               a plausible-looking distractor and citing it.

    Rewriter   a question phrased unlike the document that answers it is the
               most common retrieval failure. Re-asking in the document's
               vocabulary is the cheapest fix, and it is bounded because a loop
               that can retry forever will.

    Critic     the generator marks its own homework: it decides which claims it
               made and which sources support them. An independent pass over
               the same evidence is what turns "the model says it cited this"
               into something a reviewer can act on, and it is one of the
               signals the human-review gate keys on. It also reports sources
               that contradict each other — in the same call, because it is
               already reading every source the answer was written from, and a
               separate contradiction pass would be a second read of the same
               tokens.

All four are classification and paraphrase, not the reasoning-heavy step, so
they are built with their own `StructuredLLM` — `AGENT_MODEL`, cheaper than the
one that writes the answer. Running four extra calls per query on the answering
model is the easiest way to make an agentic graph cost several times what it
should for no measurable gain.
"""

import logging
from collections.abc import Sequence

from langfuse import observe
from pydantic import BaseModel, Field

from app.llm import LLMError, StructuredLLM, schema_of
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)

# These answers are a boolean and a sentence, not an essay. A generous ceiling
# would not make them better, and a truncated response is an `LLMError` rather
# than a partial verdict.
_CHEAP_TOKENS = 2000


def _numbered_sources(chunks: Sequence[ScoredChunk]) -> str:
    """The same 1-based numbering the generator uses.

    Shared so a grade, a critique, and a citation all refer to source 3 and mean
    the same chunk. Divergent numbering between nodes would be invisible in
    testing and wrong in production.
    """
    return "\n\n".join(
        f"[{number}] {chunk.source}, page {chunk.chunk.page}\n{chunk.chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )


# --------------------------------------------------------------------------
# Route


class Route(BaseModel):
    needs_documents: bool
    reason: str


ROUTE_SCHEMA = schema_of(Route, required=["needs_documents", "reason"])

ROUTE_SYSTEM = """\
You decide whether a question can be answered from an enterprise document \
library — contracts, policies, agreements, filings, handbooks.

You are not judging whether the question mentions a document. It will not — \
users ask about their situation, not about the filing system. You are judging \
whether a library of such documents could plausibly contain the answer.

So set needs_documents to true whenever the subject matter is the kind of thing \
these documents govern: terms, obligations, entitlements, deadlines, payments, \
notice, liability, coverage, process. Everyday phrasing is not evidence against \
it. "How much holiday do I get?" is a leave-entitlement question, "can I take \
the laptop home?" is an equipment-policy question, and "who pays if the \
shipment arrives damaged?" is a risk-of-loss question — all true. Rephrasing a \
question into the document's vocabulary happens later and is not your job.

Set it to false only when no document library could hold the answer, whatever it \
contained: greetings and small talk, questions about you or your capabilities, \
general knowledge and trivia, arithmetic, or requests to write something \
unrelated to the documents.

When in doubt, say true. Searching and finding nothing is a cheap, honest \
outcome that costs one retrieval; refusing wrongly denies the user an answer the \
library was holding, and tells them nothing about why.\
"""


class Router:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @observe(name="route")
    def route(self, question: str) -> Route:
        try:
            return self._llm.complete(
                system=ROUTE_SYSTEM,
                prompt=f"Question: {question}",
                schema=ROUTE_SCHEMA,
                schema_name="route",
                model=Route,
                max_tokens=_CHEAP_TOKENS,
            )
        except LLMError as exc:
            # Routing is an optimisation, not a correctness requirement. If the
            # model cannot be reached the right move is to search anyway — the
            # failure then surfaces at generation, where it is the user's actual
            # question that failed rather than a preliminary about it.
            logger.warning("Routing failed, assuming the question needs documents: %s", exc)
            return Route(needs_documents=True, reason="routing unavailable")


# --------------------------------------------------------------------------
# Grade


class ChunkGrade(BaseModel):
    source: int = Field(description="The 1-based source number being graded.")
    relevant: bool
    reason: str


class Grades(BaseModel):
    grades: list[ChunkGrade]


GRADE_SCHEMA = schema_of(Grades, required=["grades"])

GRADE_SYSTEM = """\
You judge whether each retrieved source could help answer a question.

A source is relevant if it contains information the answer would draw on — the \
clause, figure, definition, or obligation being asked about. It is not relevant \
merely because it is the same kind of document, mentions the same parties, or \
discusses a neighbouring topic.

Judge each source against the question, never against the other sources. A \
source that states the same term differently from another — a different \
period, amount, date, or party — is relevant: the disagreement is something the \
answer must report. Deciding which version is right is not your job, and \
dropping one is how the reader ends up never knowing there were two.

Be strict. A source kept in error becomes a citation on a claim it does not \
support. A source dropped in error is recoverable: the question gets rephrased \
and asked again.

Grade every source you are given, once each, by its number.\
"""


class Grader:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @observe(name="grade-documents")
    def keep_relevant(
        self, question: str, chunks: Sequence[ScoredChunk]
    ) -> tuple[list[ScoredChunk], list[str]]:
        """Return the chunks worth answering from, and why the rest went.

        Order is preserved: the generator numbers sources by position, and a
        reordered list would renumber every citation for no reason.
        """
        if not chunks:
            return [], []

        try:
            graded = self._llm.complete(
                system=GRADE_SYSTEM,
                prompt=f"Question: {question}\n\n<sources>\n{_numbered_sources(chunks)}\n</sources>",
                schema=GRADE_SCHEMA,
                schema_name="grades",
                model=Grades,
                max_tokens=_CHEAP_TOKENS,
            )
        except LLMError as exc:
            # Keep everything. Grading exists to raise precision; failing it
            # should cost precision, not the answer.
            logger.warning("Grading failed, keeping all retrieved chunks: %s", exc)
            return list(chunks), []

        # Only numbers that were actually offered. A grade for source 9 when six
        # were given says nothing about source 6, so it is dropped rather than
        # clamped.
        verdicts = {
            grade.source: grade
            for grade in graded.grades
            if 1 <= grade.source <= len(chunks)
        }

        kept: list[ScoredChunk] = []
        dropped: list[str] = []
        for number, chunk in enumerate(chunks, start=1):
            verdict = verdicts.get(number)
            # An ungraded source is kept. The alternative — dropping whatever
            # the grader forgot to mention — silently loses evidence on a
            # malformed response.
            if verdict is None or verdict.relevant:
                kept.append(chunk)
            else:
                dropped.append(f"[{number}] {chunk.source} p{chunk.chunk.page}: {verdict.reason}")

        return kept, dropped


# --------------------------------------------------------------------------
# Rewrite


class RewrittenQuery(BaseModel):
    query: str
    reason: str


REWRITE_SCHEMA = schema_of(RewrittenQuery, required=["query", "reason"])

REWRITE_SYSTEM = """\
A search over an enterprise document library returned nothing useful for this \
question. Rewrite it so it matches how the document would be worded.

Enterprise documents use formal register and defined terms. A question asking \
how to "walk away from the contract early" is answered by a clause about \
"termination for convenience"; one about "getting money back for receipts" by a \
clause about "expense reimbursement". Move the question toward that vocabulary.

Keep the information need identical. Do not narrow it, broaden it, or answer it. \
Return the rewritten search query alone, with no commentary.\
"""


class Rewriter:
    def __init__(self, llm: StructuredLLM) -> None:
        self._llm = llm

    @observe(name="rewrite-query")
    def rewrite(self, question: str, previous: str) -> str:
        """Return a new search query, or `previous` unchanged if rewriting fails.

        `question` is always the user's original. Rewriting a rewrite compounds
        drift — two hops in and the search is for something the user did not ask.
        """
        attempted = "" if previous == question else f"\nAlready tried: {previous}"
        try:
            rewritten = self._llm.complete(
                system=REWRITE_SYSTEM,
                prompt=f"Question: {question}{attempted}",
                schema=REWRITE_SCHEMA,
                schema_name="rewritten_query",
                model=RewrittenQuery,
                max_tokens=_CHEAP_TOKENS,
            )
        except LLMError as exc:
            logger.warning("Rewriting failed, retrying with the original question: %s", exc)
            return previous

        query = rewritten.query.strip()
        # An empty or unchanged rewrite would burn the retry budget re-running
        # the identical search.
        return query or previous


# --------------------------------------------------------------------------
# Critique


class SourceConflict(BaseModel):
    sources: list[int] = Field(
        description="The 1-based numbers of the sources that disagree — at least two."
    )
    description: str = Field(
        description="What they disagree about, quoting each source's version."
    )


class Critique(BaseModel):
    supported: bool = Field(
        description="Whether every claim in the answer is backed by the sources."
    )
    confidence: float = Field(
        description="Confidence from 0.0 to 1.0 that the answer is correct and grounded."
    )
    concerns: list[str] = Field(
        description="Specific problems found. Empty when there are none."
    )
    # Defaulted so a critique built without it means "none found" rather than
    # failing validation. The schema below still requires it, so the model
    # always has to commit to an answer.
    conflicts: list[SourceConflict] = Field(
        default_factory=list,
        description="Places where the sources contradict each other. Empty when they agree.",
    )


CRITIQUE_SCHEMA = schema_of(
    Critique, required=["supported", "confidence", "concerns", "conflicts"]
)

CRITIQUE_SYSTEM = """\
You are reviewing an answer that was generated from a fixed set of sources, on \
behalf of someone who will act on it.

Check four things:

Support — is every factual assertion in the answer traceable to the sources? \
Flag anything asserted that the sources do not say, including detail that sounds \
reasonable but is not written down.

Fidelity — are figures, dates, durations, defined terms, and party names \
reproduced exactly as the sources write them? A rounded number or a normalised \
date is a defect.

Scope — does the answer overreach? Presenting a partial answer as complete, or \
generalising one document's terms to a situation it does not cover, is a defect \
even when every individual sentence is supported.

Consistency — do any of the sources contradict each other on something the \
question turns on? Two sources giving different figures, dates, durations, \
parties, amounts, or governing terms for the same thing is a conflict, whichever \
one the answer used: the reader needs to know the documents disagree. List each \
one under conflicts with the numbers of the sources involved. Sources about \
different things do not conflict — a thirty-day payment term and a ninety-day \
notice period are two terms, and an exception or a pro-rata rule refines a term \
rather than contradicting it. Leave conflicts empty when the sources agree.

Set confidence to reflect the answer as a whole. Be willing to use the low end: \
an answer that says the sources do not cover the question, and is right about \
that, deserves high confidence — while a fluent answer resting on one \
tangential source does not. List concerns specifically enough to be checked.\
"""


class Critic:
    def __init__(self, llm: StructuredLLM, *, max_tokens: int = 4000) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    @observe(name="critique")
    def review(
        self, question: str, answer: str, chunks: Sequence[ScoredChunk]
    ) -> Critique | None:
        """Return a review, or None if one could not be produced.

        None rather than a default verdict: an unavailable critic must not look
        like a clean bill of health, and phase 4's gate has to be able to tell
        "reviewed and fine" from "not reviewed".
        """
        if not chunks:
            return None

        prompt = (
            f"Question: {question}\n\n"
            f"<sources>\n{_numbered_sources(chunks)}\n</sources>\n\n"
            f"<answer>\n{answer}\n</answer>"
        )
        try:
            return self._llm.complete(
                system=CRITIQUE_SYSTEM,
                prompt=prompt,
                schema=CRITIQUE_SCHEMA,
                schema_name="critique",
                model=Critique,
                max_tokens=self._max_tokens,
            )
        except LLMError as exc:
            logger.warning("Critique failed; the answer is returned unreviewed: %s", exc)
            return None
