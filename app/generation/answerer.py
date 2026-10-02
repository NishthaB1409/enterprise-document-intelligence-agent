"""Answer generation, grounded in the retrieved chunks and nothing else.

The model is asked for a structured answer rather than prose: a summary plus a
list of claims, each carrying the source numbers that support it. Prose with
`[1]`-style markers would read the same, but there is no way to *verify* it —
a marker in text is a character, while a claim with an empty `sources` list is
a fact the pipeline can catch and act on. Phase 4's human-review gate needs the
latter.

The model never sees chunk ids. It gets 1-based source numbers, which are short,
hard to hallucinate plausibly, and trivially range-checked; `app.generation.
citations` maps them back to real chunks and drops anything that doesn't resolve.

There is one implementation for both vendors. The prompt, the schema, and the
handling of an empty chunk list are the whole of what answering *is*; which
provider carries the request is `app.llm`'s problem, and keeping that split is
what stopped phase 3's four new model calls from doubling into eight.
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langfuse import observe
from pydantic import BaseModel, Field

from app.llm import LLMError, StructuredLLM
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)

# The failure mode callers already handle by name. It is `LLMError` under a
# different name rather than a subclass: generation has no failure that other
# model calls do not, and two exception types for one condition would mean every
# handler eventually catching both.
GenerationError = LLMError


class Claim(BaseModel):
    text: str
    # 1-based indices into the sources the prompt listed, in the order given.
    sources: list[int] = Field(default_factory=list)


class SourceConflict(BaseModel):
    """Two or more sources stating the same thing differently.

    Defined here rather than with the critic because two things report it: the
    answerer, which is the only one present on the default single-pass
    pipeline, and the critic, when the agentic graph runs.
    """

    sources: list[int] = Field(
        description="The 1-based numbers of the sources that disagree — at least two."
    )
    description: str = Field(
        description="What they disagree about, quoting each source's version."
    )


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    # Stated by the model rather than inferred from an empty claim list, so
    # "the documents don't say" is distinguishable from "the model forgot to
    # cite". They call for different follow-ups.
    answerable: bool
    # Sources that contradict each other. The prompt has always asked the model
    # to say so in prose, but prose is something a reader may notice and the
    # review gate cannot. This is the same observation as a field the gate can
    # act on, at no extra call.
    conflicts: list[SourceConflict] = Field(default_factory=list)


# Written by hand rather than derived from the Pydantic model: the structured
# output APIs require `additionalProperties: false` and an explicit `required`
# on every object, which `model_json_schema()` does not emit. The model above
# still validates what comes back.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The answer to the question, in prose, drawn only from the sources. "
                "If the sources do not answer it, say so plainly here."
            ),
        },
        "claims": {
            "type": "array",
            "description": (
                "Every factual assertion the answer makes, one entry each, with the "
                "sources supporting it. Empty when the question is unanswerable."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "One self-contained factual assertion from the answer.",
                    },
                    "sources": {
                        "type": "array",
                        "description": (
                            "Source numbers supporting this claim. Never empty; never "
                            "a number that was not listed."
                        ),
                        "items": {"type": "integer"},
                    },
                },
                "required": ["text", "sources"],
                "additionalProperties": False,
            },
        },
        "answerable": {
            "type": "boolean",
            "description": "Whether the sources actually contain the answer.",
        },
        "conflicts": {
            "type": "array",
            "description": (
                "Every place two or more sources state the same term differently. "
                "Empty when the sources agree."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "description": "The numbers of the sources that disagree; at least two.",
                        "items": {"type": "integer"},
                    },
                    "description": {
                        "type": "string",
                        "description": "What they disagree about, quoting each source's version.",
                    },
                },
                "required": ["sources", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "claims", "answerable", "conflicts"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """\
You answer questions about enterprise documents (contracts, policies, filings) \
for reviewers who will act on what you say.

Answer only from the numbered sources given to you. They are the whole of what \
you know for this question; your own background knowledge about the companies, \
laws, or standards involved is not evidence and must not appear in the answer.

Every factual assertion you make must appear in `claims` with at least one \
source number that supports it. A source supports a claim only if the claim can \
be read off that source's text directly — not if it merely sounds consistent \
with it. If you cannot support an assertion, leave it out of the answer rather \
than citing something adjacent.

Quote figures, dates, durations, defined terms, and party names exactly as the \
source writes them. Do not convert units, round numbers, or normalise dates.

When the sources do not answer the question, set `answerable` to false, say in \
`answer` what is missing, and return no claims. This is a correct outcome, not \
a failure — a reviewer can go find the right document. An answer that fills the \
gap by inference is worse than no answer.

When the sources disagree with each other, say so and cite each side. Do not \
silently pick one. Also list each disagreement under `conflicts`, with the \
numbers of the sources involved: that list is what routes the answer to a human \
reviewer. Only the same term stated differently is a conflict — a thirty-day \
payment term and a ninety-day notice period are two terms, and an exception or \
a pro-rata rule refines a term rather than contradicting it. Leave `conflicts` \
empty when the sources agree.\
"""

NO_EVIDENCE = "No indexed document contains anything relevant to this question."


@runtime_checkable
class Answerer(Protocol):
    def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer: ...


def build_prompt(question: str, chunks: Sequence[ScoredChunk]) -> str:
    """Numbered sources, then the question.

    The question goes last so it is the most recent thing in context, and so the
    (stable) source block stays a cacheable prefix if this ever grows a shared
    preamble.
    """
    sources = "\n\n".join(
        f"[{number}] {chunk.source}, page {chunk.chunk.page}\n{chunk.chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )
    return f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"


class LLMAnswerer:
    """Generation over any `StructuredLLM`."""

    def __init__(self, llm: StructuredLLM, *, max_tokens: int = 8000) -> None:
        self._llm = llm
        self._max_tokens = max_tokens

    @observe(name="generate-answer", as_type="generation")
    def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        if not chunks:
            # Nothing retrieved means no grounding to reason over, so asking the
            # model would only invite it to answer from memory. Returned rather
            # than raised: "the corpus does not cover this" is a correct answer.
            return GeneratedAnswer(answer=NO_EVIDENCE, claims=[], answerable=False)

        return self._llm.complete(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(question, chunks),
            schema=ANSWER_SCHEMA,
            schema_name="grounded_answer",
            model=GeneratedAnswer,
            max_tokens=self._max_tokens,
        )
