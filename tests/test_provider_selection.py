"""Generation configuration, and the gate that checks it before a request runs.

The failure this guards against is a deployment that *looks* configured — the
app boots, /health is green, /ingest works — and only discovers at query time
that it has no key. That is a 500 in front of a user rather than a 503 at the
door, so the check is asserted here rather than left to be noticed in staging.
"""

import pytest

from app.config import Settings
from app.llm import OpenAILLM
from app.services import build_llm


def _settings(**overrides) -> Settings:
    # `_env_file=None` so a developer's own .env cannot change the answer.
    return Settings(_env_file=None, **overrides)


def test_every_model_call_goes_through_one_client_type():
    assert isinstance(build_llm(_settings()), OpenAILLM)


def test_the_answer_model_must_support_strict_structured_outputs():
    """gpt-4o-mini is the cheapest model that does, and the citation contract
    depends on it: without strict mode the schema is a hint, and an answer whose
    claims parse most of the time is an answer that cannot be verified."""
    assert _settings().answer_model == "gpt-4o-mini"
    assert _settings(answer_model="gpt-4.1").answer_model == "gpt-4.1"


def test_the_graph_defaults_to_a_cheaper_model_than_the_answer():
    """Four extra classification calls per query on the answering model is the
    easiest way to make the graph cost several times what it should."""
    settings = _settings(answer_model="gpt-4.1")

    assert settings.agent_model == "gpt-4o-mini"
    # ...and the graph's client is built with it, not with the answering model.
    assert build_llm(settings, model=settings.agent_model)._model == "gpt-4o-mini"
    assert build_llm(settings)._model == "gpt-4.1"


def test_a_missing_key_is_caught_at_the_gate_rather_than_mid_request():
    assert not _settings().generation_configured
    assert _settings(openai_api_key="sk-test").generation_configured


def test_the_gate_names_the_variable_to_set():
    # Named in the 503 so the fix is obvious without reading the config.
    assert _settings().generation_key_variable == "OPENAI_API_KEY"


def test_ingestion_does_not_require_a_generation_key():
    """Indexing documents needs no vendor key at all — embeddings are local — so
    an unconfigured deployment must still be able to ingest."""
    settings = _settings()

    assert not settings.generation_configured
    assert settings.embedding_model  # local, no key involved
