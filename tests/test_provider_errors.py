"""What an operator sees when the model provider says no.

A rejected key, an exhausted account, and an inaccessible model are three
different problems with three different fixes. Left unhandled they are one
identical 500 with a stack trace, and the fix has to be guessed from the logs —
so each one is asserted to name what to change.

These run against `app.llm`, which is where the two SDKs' error hierarchies are
translated. Every model call in the service — answering, routing, grading,
rewriting, critique — goes through it, so one pass over this seam covers all of
them rather than one suite per caller.
"""

import anthropic
import httpx
import openai
import pytest

from app.generation.answerer import ANSWER_SCHEMA, GeneratedAnswer, LLMAnswerer
from app.ingest.chunking import Chunk
from app.llm import AnthropicLLM, LLMError, OpenAILLM
from app.vectorstore.store import ScoredChunk

CHUNKS = [
    ScoredChunk(
        chunk=Chunk(
            id="chunk-1",
            doc_id="doc-1",
            index=0,
            page=1,
            text="Either party may terminate on thirty days notice.",
            char_start=0,
            char_end=48,
        ),
        source="contract.pdf",
        score=0.9,
    )
]

_REQUEST = httpx.Request("POST", "https://example.invalid/v1/messages")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST, json={"error": {"message": "nope"}})


class _Boom:
    """Stands in for the vendor SDK's entry point, raising on call."""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def __call__(self, **kwargs):
        raise self._error


def _openai_llm(error: Exception, model: str = "gpt-4o-mini") -> OpenAILLM:
    llm = OpenAILLM(model=model, api_key="sk-test")
    llm._client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": type("C", (), {"create": _Boom(error)})()})()},
    )()
    return llm


def _anthropic_llm(error: Exception, model: str = "claude-opus-5") -> AnthropicLLM:
    llm = AnthropicLLM(model=model, api_key="sk-ant-test")
    llm._client = type(
        "FakeClient", (), {"messages": type("M", (), {"create": _Boom(error)})()}
    )()
    return llm


def _ask(llm) -> GeneratedAnswer:
    return llm.complete(
        system="You answer questions.",
        prompt="What is the notice period?",
        schema=ANSWER_SCHEMA,
        schema_name="grounded_answer",
        model=GeneratedAnswer,
    )


def test_a_rejected_openai_key_names_the_setting_to_fix():
    llm = _openai_llm(openai.AuthenticationError("bad key", response=_response(401), body=None))

    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        _ask(llm)


def test_an_exhausted_openai_account_says_so():
    llm = _openai_llm(openai.RateLimitError("quota", response=_response(429), body=None))

    with pytest.raises(LLMError, match="out of credit"):
        _ask(llm)


def test_an_inaccessible_openai_model_points_at_answer_model():
    llm = _openai_llm(
        openai.NotFoundError("no model", response=_response(404), body=None),
        model="gpt-9-imaginary",
    )

    with pytest.raises(LLMError, match="ANSWER_MODEL"):
        _ask(llm)


def test_an_unreachable_openai_is_reported_as_a_connection_problem():
    llm = _openai_llm(openai.APIConnectionError(request=_REQUEST))

    with pytest.raises(LLMError, match="could not reach OpenAI"):
        _ask(llm)


def test_a_rejected_anthropic_key_names_the_setting_to_fix():
    llm = _anthropic_llm(
        anthropic.AuthenticationError("bad key", response=_response(401), body=None)
    )

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        _ask(llm)


def test_an_exhausted_anthropic_account_says_so():
    llm = _anthropic_llm(anthropic.RateLimitError("quota", response=_response(429), body=None))

    with pytest.raises(LLMError, match="out of credit"):
        _ask(llm)


def test_an_inaccessible_anthropic_model_points_at_answer_model():
    llm = _anthropic_llm(
        anthropic.NotFoundError("no model", response=_response(404), body=None),
        model="claude-imaginary",
    )

    with pytest.raises(LLMError, match="ANSWER_MODEL"):
        _ask(llm)


def test_an_unreachable_anthropic_is_reported_as_a_connection_problem():
    llm = _anthropic_llm(anthropic.APIConnectionError(request=_REQUEST))

    with pytest.raises(LLMError, match="could not reach Anthropic"):
        _ask(llm)


@pytest.mark.parametrize(
    ("build", "status_error"),
    # The two SDKs have entirely separate exception hierarchies, so each client
    # must catch its own — a shared base class would be a false comfort here.
    [
        (_openai_llm, openai.APIStatusError),
        (_anthropic_llm, anthropic.APIStatusError),
    ],
    ids=["openai", "anthropic"],
)
def test_an_unexpected_status_still_arrives_as_an_llm_error(build, status_error):
    llm = build(status_error("server error", response=_response(500), body=None))

    # Not a bare 500 from our own process: the request was fine, the dependency
    # was not, and the route turns this into a 502.
    with pytest.raises(LLMError, match="500"):
        _ask(llm)


@pytest.mark.parametrize("build", [_openai_llm, _anthropic_llm], ids=["openai", "anthropic"])
def test_no_retrieved_chunks_means_no_call_to_the_provider(build):
    # The stand-in raises on any call, so reaching a result proves none was made.
    answerer = LLMAnswerer(build(AssertionError("the provider must not be called")))

    result = answerer.answer("What is the notice period?", [])

    assert result.answerable is False
    assert result.claims == []
