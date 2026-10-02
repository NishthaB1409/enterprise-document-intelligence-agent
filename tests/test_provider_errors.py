"""What an operator sees when the provider says no.

A rejected key, an exhausted account, and an inaccessible model are three
different problems with three different fixes. Left unhandled they are one
identical 500 with a stack trace, and the fix has to be guessed from the logs —
so each one is asserted to name what to change.

These run against `app.llm`, which is where the SDK's error hierarchy is
translated. Every model call in the service — answering, routing, grading,
rewriting, critique — goes through it, so one pass over this seam covers all of
them rather than one suite per caller.
"""

import httpx
import openai
import pytest

from app.generation.answerer import ANSWER_SCHEMA, GeneratedAnswer, LLMAnswerer
from app.ingest.chunking import Chunk
from app.llm import LLMError, OpenAILLM
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


def test_an_unexpected_status_still_arrives_as_an_llm_error():
    llm = _openai_llm(openai.APIStatusError("server error", response=_response(500), body=None))

    # Not a bare 500 from our own process: the request was fine, the dependency
    # was not, and the route turns this into a 502.
    with pytest.raises(LLMError, match="500"):
        _ask(llm)


def test_no_retrieved_chunks_means_no_call_to_the_provider():
    # The stand-in raises on any call, so reaching a result proves none was made.
    answerer = LLMAnswerer(_openai_llm(AssertionError("the provider must not be called")))

    result = answerer.answer("What is the notice period?", [])

    assert result.answerable is False
    assert result.claims == []


# --- temperature ----------------------------------------------------------------


def _capturing_llm(**kwargs):
    """An OpenAILLM whose client records the arguments of each call."""
    from types import SimpleNamespace

    sent: list[dict] = []
    payload = GeneratedAnswer(answer="a", claims=[], answerable=False).model_dump_json()

    def create(**call):
        sent.append(call)
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                message=SimpleNamespace(refusal=None, content=payload), finish_reason="stop"
            )],
        )

    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test", **kwargs)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm, sent


def test_the_configured_temperature_is_sent():
    llm, sent = _capturing_llm(temperature=0.0)
    _ask(llm)
    assert sent[0]["temperature"] == 0.0


def test_no_temperature_means_the_parameter_is_omitted():
    """Reasoning models reject `temperature` outright, even at its default.
    Omitting it is the only thing they accept."""
    llm, sent = _capturing_llm(temperature=None)
    _ask(llm)
    assert "temperature" not in sent[0]


def test_the_shipped_default_is_deterministic():
    from app.config import Settings

    assert Settings(_env_file=None).llm_temperature == 0.0


@pytest.mark.parametrize("raw", ["none", "None", ""])
def test_temperature_can_be_switched_off_from_the_environment(raw, monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("LLM_TEMPERATURE", raw)
    assert Settings(_env_file=None).llm_temperature is None


def test_a_model_that_rejects_temperature_says_which_setting_to_change():
    error = openai.BadRequestError(
        "Unsupported parameter: 'temperature' is not supported with this model.",
        response=_response(400),
        body=None,
    )
    with pytest.raises(LLMError, match="LLM_TEMPERATURE=none"):
        _ask(_openai_llm(error, model="o4-mini"))


def test_every_client_the_service_builds_gets_the_temperature():
    from app.config import Settings
    from app.services import build_llm

    settings = Settings(_env_file=None, openai_api_key="sk-test", llm_temperature=0.3)
    assert build_llm(settings)._temperature == 0.3
    assert build_llm(settings, model=settings.agent_model)._temperature == 0.3


# --- timeout --------------------------------------------------------------------


def test_the_client_is_built_with_the_configured_timeout():
    """The SDK default is ten minutes per attempt, retried twice. A stalled
    request once held an eval run for 26 minutes before this was set."""
    llm = OpenAILLM(model="gpt-4o-mini", api_key="sk-test", timeout=12.5)
    assert llm._ensure_client().timeout == 12.5


def test_the_service_passes_its_timeout_setting_to_every_client():
    from app.config import Settings
    from app.services import build_llm

    settings = Settings(_env_file=None, openai_api_key="sk-test", llm_timeout_seconds=7)
    assert build_llm(settings)._timeout == 7
    assert Settings(_env_file=None).llm_timeout_seconds == 60.0


def test_a_timeout_is_reported_as_one_rather_than_as_unreachable():
    """APITimeoutError subclasses APIConnectionError. Reported as "could not
    reach OpenAI", it would send someone to check a network that is fine."""
    with pytest.raises(LLMError, match="did not respond within"):
        _ask(_openai_llm(openai.APITimeoutError(request=_REQUEST)))
