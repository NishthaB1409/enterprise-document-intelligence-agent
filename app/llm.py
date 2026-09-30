"""One way to ask a model for a typed answer.

Five things in this service call a model — answering, routing, grading,
rewriting, critique — and without a seam each would carry its own copy of the
same `except AuthenticationError` ladder, drifting apart the first time one of
them was fixed. So the transport lives here once, and every caller above it is a
prompt plus a Pydantic model. What a node knows how to do is describe the answer
it wants; what this module knows is how to get one and how to explain a refusal.

`StructuredLLM` stays a Protocol even though there is only one implementation.
It is what lets every node be tested against a scripted stand-in rather than a
network call — `tests/test_agent_nodes.py` is entirely written against it — and
what makes "cheap model for classification, better model for the answer" a
matter of constructing two clients rather than threading a flag through five
call sites.

Errors are translated rather than propagated. A rejected key, an exhausted
account, and a model the account cannot reach are three different problems with
three different fixes; raw they are one identical 500 and the fix has to be
guessed from a stack trace. Each becomes an `LLMError` naming what to change.
"""

import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

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
    ) -> T:
        """Return an instance of `model`, or raise `LLMError`.

        `schema` is passed to the provider to constrain generation and `model`
        validates what comes back. Both, deliberately: the schema makes a wrong
        shape unlikely, and the validation makes it impossible to act on one.
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


@dataclass
class Usage:
    """Calls and tokens spent by one client since it was built.

    What makes "the graph costs three to five calls per query" a measurement
    rather than a figure worked out from the diagram: `eval/answer_eval.py`
    reads it off the answering client and the agent client separately, so the
    cost of each pipeline comes out next to its quality scores.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OpenAILLM:
    """OpenAI via strict `json_schema` response format.

    Strict mode is not optional here. Without it the schema is a hint, and the
    model may return something that parses as JSON but is not the thing that was
    asked for — which for the answering path means a citation contract that
    holds most of the time, and that is the same as not holding. It requires
    `additionalProperties: false` and every property listed in `required`, which
    is what `schema_of` below enforces.

    The model is fixed per instance rather than per call, so wanting a cheaper
    model for the graph's classification steps means building a second client —
    see `build_pipeline` in `app.services`.
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
        self.usage = Usage()
        # FastAPI calls one client from several worker threads, and `+=` on an
        # attribute is not atomic.
        self._usage_lock = threading.Lock()

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

        # Counted before any of the checks below: a truncated or refused
        # response was still billed.
        if response.usage is not None:
            with self._usage_lock:
                self.usage.calls += 1
                self.usage.prompt_tokens += response.usage.prompt_tokens
                self.usage.completion_tokens += response.usage.completion_tokens

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
