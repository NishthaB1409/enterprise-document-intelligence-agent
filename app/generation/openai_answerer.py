"""OpenAI-backed answer generation.

The same contract as `AnthropicAnswerer`, the same prompt, and the same JSON
schema — only the transport differs. Both satisfy the `Answerer` protocol, so
`app.services` picks one and nothing downstream (retrieval, grounding, the route,
the tests) knows which is in play.

`ANSWER_SCHEMA` is reused verbatim rather than re-declared. It already carries
`additionalProperties: false` and lists every property in `required`, which is
exactly what OpenAI's strict structured outputs require — so the two providers
are held to the same output shape by construction, not by two schemas kept in
sync by hand.
"""

import logging
from collections.abc import Sequence

from langfuse import observe
from openai import APIConnectionError, APIStatusError, AuthenticationError, OpenAI, RateLimitError
from pydantic import ValidationError

from app.generation.answerer import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    GeneratedAnswer,
    GenerationError,
    build_prompt,
)
from app.vectorstore.store import ScoredChunk

logger = logging.getLogger(__name__)


class OpenAIAnswerer:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8000,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client: OpenAI | None = None

    def _ensure_client(self) -> OpenAI:
        # Deferred for the same reason as the Anthropic client: constructing it
        # without a key raises, and the app must still boot and serve /health
        # when generation is unconfigured.
        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    @observe(name="generate-answer", as_type="generation")
    def answer(self, question: str, chunks: Sequence[ScoredChunk]) -> GeneratedAnswer:
        if not chunks:
            # Nothing retrieved means no grounding to reason over; asking anyway
            # would only invite an answer from the model's own memory.
            return GeneratedAnswer(
                answer="No indexed document contains anything relevant to this question.",
                claims=[],
                answerable=False,
            )

        try:
            response = self._ensure_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(question, chunks)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "grounded_answer",
                        # Without strict, the schema is a hint and the model may
                        # return a shape that parses as JSON but not an answer.
                        "strict": True,
                        "schema": ANSWER_SCHEMA,
                    },
                },
                # The newer spelling of `max_tokens`; the only one reasoning
                # models accept, and equivalent on the rest.
                max_completion_tokens=self._max_tokens,
            )
        # Each of these is something an operator can act on, so each says what
        # to do. Left to propagate they would all become an identical 500 with
        # a stack trace, and the fix would have to be guessed from the logs.
        except AuthenticationError as exc:
            raise GenerationError(
                "OpenAI rejected the API key; check OPENAI_API_KEY in .env"
            ) from exc
        except RateLimitError as exc:
            raise GenerationError(
                "OpenAI rate-limited the request, or the account is out of credit"
            ) from exc
        except APIConnectionError as exc:
            raise GenerationError(f"could not reach OpenAI: {exc}") from exc
        except APIStatusError as exc:
            if exc.status_code == 404:
                # Overwhelmingly a model name the account cannot use, which
                # reads as "not found" and looks nothing like a config error.
                raise GenerationError(
                    f"OpenAI does not recognise the model {self._model!r}, or this "
                    "account cannot access it; check ANSWER_MODEL"
                ) from exc
            raise GenerationError(f"OpenAI returned {exc.status_code}: {exc}") from exc

        choice = response.choices[0]

        if choice.message.refusal:
            raise GenerationError(f"the model declined to answer: {choice.message.refusal}")
        if choice.finish_reason == "length":
            # The JSON is truncated, so there is nothing to salvage. Surfaced
            # rather than retried: silently doubling the budget hides a prompt
            # or chunk-size problem that will keep recurring.
            raise GenerationError(
                f"the answer exceeded max_completion_tokens ({self._max_tokens}) and was cut off"
            )

        text = choice.message.content
        if not text:
            raise GenerationError("the model returned no content")

        try:
            return GeneratedAnswer.model_validate_json(text)
        except ValidationError as exc:
            # Strict structured outputs make this close to impossible; if it
            # happens, the raw payload is the only way to tell what changed.
            logger.error("Unparseable answer payload: %s", text[:2000])
            raise GenerationError(f"the model returned malformed JSON: {exc}") from exc
