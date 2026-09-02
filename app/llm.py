"""One way to ask a model for a typed answer.

Phase 1 needed a single LLM call and each provider's version of it carried its
own error handling. Phase 3 adds four more calls — routing, grading, rewriting,
critique — and the same code across two providers would have meant eight copies
of the same `except AuthenticationError` ladder, drifting apart the first time
one of them was fixed.

So the transport lives here, once per provider, and every caller above it is a
prompt plus a Pydantic model. What a node knows how to do is describe the answer
it wants; what this module knows is how to get one and how to explain a refusal.

Errors are translated rather than propagated. A rejected key, an exhausted
account, and a model the account cannot reach are three different problems with
three different fixes; raw they are one identical 500 and the fix has to be
guessed from a stack trace. Each becomes an `LLMError` naming what to change.
"""

import json
import logging
from typing import Any, Protocol, TypeVar, runtime_checkable

import anthropic
import openai
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The model did not return a usable answer.

    Callers turn this into a 502: the request was fine, the upstream dependency
    was not. The message is written for whoever has to fix it.
    """


@runtime_checkable
class StructuredLLM(Protocol):
    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: type[T],
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        """Return an instance of `model`, or raise `LLMError`.

        `schema` is passed to the provider to constrain generation and `model`
        validates what comes back. Both, deliberately: the schema makes a wrong
        shape unlikely, and the validation makes it impossible to act on one.

        `effort` is honoured where the provider has the concept and ignored
        where it does not, so a caller can ask for a cheap classification
        without first asking which vendor is configured.
        """


def _validate(model: type[T], text: str | None) -> T:
    if not text:
        raise LLMError("the model returned no content")
    try:
        return model.model_validate_json(text)
    except ValidationError as exc:
        # Structured outputs make this close to impossible. If it happens, the
        # raw payload is the only evidence of what changed upstream.
        logger.error("Unparseable %s payload: %s", model.__name__, text[:2000])
        raise LLMError(f"the model returned malformed JSON: {exc}") from exc


class AnthropicLLM:
    """Anthropic via `output_config.format`.

    `effort` is a per-request knob on this provider, so it is set per call
    rather than per client: grading a chunk and writing a cited answer are not
    equally hard, and paying answer-grade reasoning for a yes/no classification
    is the easiest cost mistake to make in an agentic graph.
    """

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
        self._client: anthropic.Anthropic | None = None

    def _ensure_client(self) -> anthropic.Anthropic:
        # Deferred: constructing the client without a key raises, and the app
        # must still boot and serve /health when generation is unconfigured.
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: type[T],
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        budget = max_tokens or self._max_tokens
        try:
            response = self._ensure_client().messages.create(
                model=self._model,
                max_tokens=budget,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={
                    "effort": effort or self._effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError(
                "Anthropic rejected the API key; check ANTHROPIC_API_KEY in .env"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(
                "Anthropic rate-limited the request, or the account is out of credit"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"could not reach Anthropic: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 404:
                raise LLMError(
                    f"Anthropic does not recognise the model {self._model!r}, or this "
                    "account cannot access it; check ANSWER_MODEL"
                ) from exc
            raise LLMError(f"Anthropic returned {exc.status_code}: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("the model declined to answer this question")
        if response.stop_reason == "max_tokens":
            # The JSON is truncated, so there is nothing to salvage. Surfaced
            # rather than retried: silently doubling the budget hides a prompt
            # or chunk-size problem that will keep recurring.
            raise LLMError(f"the response exceeded max_tokens ({budget}) and was cut off")

        text = next((block.text for block in response.content if block.type == "text"), None)
        return _validate(model, text)


class OpenAILLM:
    """OpenAI via strict `json_schema` response format.

    The schema is shared with the Anthropic path verbatim. Strict mode requires
    `additionalProperties: false` and every property listed in `required`, which
    the schemas in `app.graph.prompts` already satisfy — so both providers are
    held to one shape by construction rather than by two schemas kept in sync.
    """

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
        self._client: openai.OpenAI | None = None

    def _ensure_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        model: type[T],
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        budget = max_tokens or self._max_tokens
        try:
            response = self._ensure_client().chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        # Without strict, the schema is a hint and the model may
                        # return something that parses as JSON but is not the
                        # thing that was asked for.
                        "strict": True,
                        "schema": schema,
                    },
                },
                # The newer spelling of `max_tokens`; the only one reasoning
                # models accept, and equivalent on the rest.
                max_completion_tokens=budget,
            )
        except openai.AuthenticationError as exc:
            raise LLMError("OpenAI rejected the API key; check OPENAI_API_KEY in .env") from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                "OpenAI rate-limited the request, or the account is out of credit"
            ) from exc
        except openai.APIConnectionError as exc:
            raise LLMError(f"could not reach OpenAI: {exc}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 404:
                # Overwhelmingly a model name the account cannot use, which
                # reads as "not found" and looks nothing like a config error.
                raise LLMError(
                    f"OpenAI does not recognise the model {self._model!r}, or this "
                    "account cannot access it; check ANSWER_MODEL"
                ) from exc
            raise LLMError(f"OpenAI returned {exc.status_code}: {exc}") from exc

        choice = response.choices[0]
        if choice.message.refusal:
            raise LLMError(f"the model declined to answer: {choice.message.refusal}")
        if choice.finish_reason == "length":
            raise LLMError(
                f"the response exceeded max_completion_tokens ({budget}) and was cut off"
            )

        return _validate(model, choice.message.content)


def schema_of(model: type[BaseModel], *, required: list[str]) -> dict[str, Any]:
    """A provider-acceptable JSON schema for a Pydantic model.

    `model_json_schema()` alone is not enough: strict structured outputs require
    `additionalProperties: false` on every object and every property listed in
    `required`, and Pydantic emits neither for optional fields. Rather than hand
    every node a hand-written schema to keep in sync with its model, this
    derives one and patches those two rules in.

    `required` is explicit rather than "all properties", because the caller is
    the one who knows which fields the model must actually commit to.
    """
    schema = model.model_json_schema()
    _harden(schema)
    schema["required"] = required
    return schema


def _harden(node: Any) -> None:
    """Recursively add `additionalProperties: false` to every object."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node.setdefault("required", sorted(node.get("properties", {})))
        for value in node.values():
            _harden(value)
    elif isinstance(node, list):
        for item in node:
            _harden(item)


def dumps(value: Any) -> str:
    """Compact, key-sorted JSON for embedding in a prompt.

    Sorted so the same inputs render the same bytes — otherwise the prompt
    prefix changes between identical requests and prompt caching never hits.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
