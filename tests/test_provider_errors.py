"""What an operator sees when the model provider says no.

A rejected key, an exhausted account, and an inaccessible model are three
different problems with three different fixes. Left unhandled they are one
identical 500 with a stack trace, and the fix has to be guessed from the logs —
so each one is asserted to name what to change.
"""

import anthropic
import httpx
import openai
import pytest

from app.generation.answerer import AnthropicAnswerer, GenerationError
from app.generation.openai_answerer import OpenAIAnswerer
from app.ingest.chunking import Chunk
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


def _openai_answerer(error: Exception, model: str = "gpt-4o-mini") -> OpenAIAnswerer:
    answerer = OpenAIAnswerer(model=model, api_key="sk-test")
    answerer._client = type(
        "FakeClient",
        (),
        {"chat": type("Chat", (), {"completions": type("C", (), {"create": _Boom(error)})()})()},
    )()
    return answerer


def _anthropic_answerer(error: Exception, model: str = "claude-opus-5") -> AnthropicAnswerer:
    answerer = AnthropicAnswerer(model=model, api_key="sk-ant-test")
    answerer._client = type(
        "FakeClient", (), {"messages": type("M", (), {"create": _Boom(error)})()}
    )()
    return answerer


def test_a_rejected_openai_key_names_the_setting_to_fix():
    answerer = _openai_answerer(
        openai.AuthenticationError("bad key", response=_response(401), body=None)
    )

    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        answerer.answer("What is the notice period?", CHUNKS)


def test_an_exhausted_openai_account_says_so():
    answerer = _openai_answerer(
        openai.RateLimitError("quota", response=_response(429), body=None)
    )

    with pytest.raises(GenerationError, match="out of credit"):
        answerer.answer("What is the notice period?", CHUNKS)


def test_an_inaccessible_openai_model_points_at_answer_model():
    answerer = _openai_answerer(
        openai.NotFoundError("no model", response=_response(404), body=None),
        model="gpt-9-imaginary",
    )

    with pytest.raises(GenerationError, match="ANSWER_MODEL"):
        answerer.answer("What is the notice period?", CHUNKS)


def test_an_unreachable_openai_is_reported_as_a_connection_problem():
    answerer = _openai_answerer(openai.APIConnectionError(request=_REQUEST))

    with pytest.raises(GenerationError, match="could not reach OpenAI"):
        answerer.answer("What is the notice period?", CHUNKS)


def test_a_rejected_anthropic_key_names_the_setting_to_fix():
    answerer = _anthropic_answerer(
        anthropic.AuthenticationError("bad key", response=_response(401), body=None)
    )

    with pytest.raises(GenerationError, match="ANTHROPIC_API_KEY"):
        answerer.answer("What is the notice period?", CHUNKS)


@pytest.mark.parametrize(
    ("build", "status_error"),
    # The two SDKs have entirely separate exception hierarchies, so each
    # answerer must catch its own — a shared base class would be a false
    # comfort here.
    [
        (_openai_answerer, openai.APIStatusError),
        (_anthropic_answerer, anthropic.APIStatusError),
    ],
    ids=["openai", "anthropic"],
)
def test_an_unexpected_status_still_arrives_as_a_generation_error(build, status_error):
    answerer = build(status_error("server error", response=_response(500), body=None))

    # Not a bare 500 from our own process: the request was fine, the dependency
    # was not, and the route turns this into a 502.
    with pytest.raises(GenerationError, match="500"):
        answerer.answer("What is the notice period?", CHUNKS)


@pytest.mark.parametrize("build", [_openai_answerer, _anthropic_answerer])
def test_no_retrieved_chunks_means_no_call_to_the_provider(build):
    # The stand-in raises on any call, so reaching a result proves none was made.
    answerer = build(AssertionError("the provider must not be called"))

    result = answerer.answer("What is the notice period?", [])

    assert result.answerable is False
    assert result.claims == []
