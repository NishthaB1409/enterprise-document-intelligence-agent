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
"""

import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from anthropic import Anthropic
from langfuse import observe
from pydantic import BaseModel, Field, ValidationError

from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)


class GenerationError(RuntimeError):
    """The model did not return a usable answer. Callers turn this into a 502:
    the request was fine, the upstream dependency was not."""


class Claim(BaseModel):
    text: str
    # 1-based indices into the sources the prompt listed, in the order given.
    sources: list[int] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    # Stated by the model rather than inferred from an empty claim list, so
    # "the documents don't say" is distinguishable from "the model forgot to
    # cite". They call for different follow-ups.
    answerable: bool


# Written by hand rather than derived from the Pydantic models: the structured
# output API requires `additionalProperties: false` and an explicit `required`
# on every object, which `model_json_schema()` does not emit. The Pydantic
# models above still validate what comes back.
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
    },
    "required": ["answer", "claims", "answerable"],
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
silently pick one.\
"""


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


class AnthropicAnswerer:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 8000,
        effort: str = "medium",
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._effort = effort
        self._client: Anthropic | None = None

    def _ensure_client(self) -> Anthropic:
        # Deferred: constructing the client without a key raises, and the app
        # must still boot (and serve /health) when generation is unconfigured.
        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    @observe(name="generate-answer", as_type="generation")
    def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        if not chunks:
            # Nothing retrieved: there is no grounding to reason over, so asking
            # the model would only invite it to answer from memory.
            return GeneratedAnswer(
                answer="No indexed document contains anything relevant to this question.",
                claims=[],
                answerable=False,
            )

        response = self._ensure_client().messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, chunks)}],
            output_config={
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
            },
        )

        if response.stop_reason == "refusal":
            raise GenerationError("the model declined to answer this question")
        if response.stop_reason == "max_tokens":
            # The JSON is truncated, so there is nothing to salvage. Surfaced
            # rather than retried: silently doubling the budget hides a prompt
            # or chunk-size problem that will keep recurring.
            raise GenerationError(
                f"the answer exceeded max_tokens ({self._max_tokens}) and was cut off"
            )

        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise GenerationError("the model returned no text content")

        try:
            return GeneratedAnswer.model_validate_json(text)
        except ValidationError as exc:
            # Structured outputs make this close to impossible; if it happens,
            # the raw payload is the only way to tell what changed upstream.
            logger.error("Unparseable answer payload: %s", text[:2000])
            raise GenerationError(f"the model returned malformed JSON: {exc}") from exc
