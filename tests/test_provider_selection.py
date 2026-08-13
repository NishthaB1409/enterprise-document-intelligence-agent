"""Choosing an answer provider is one setting, and it must be all of one setting.

The failure this guards against is a half-switch: `LLM_PROVIDER=openai` with a
Claude model name still attached, or a deployment that looks configured because
it holds the *other* vendor's key.
"""

import pytest

from app.config import Settings
from app.generation.answerer import AnthropicAnswerer
from app.generation.openai_answerer import OpenAIAnswerer
from app.services import build_answerer


def _settings(**overrides) -> Settings:
    # `_env_file=None` so a developer's own .env cannot change the answer.
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("anthropic", AnthropicAnswerer), ("openai", OpenAIAnswerer)],
)
def test_the_provider_setting_picks_the_implementation(provider, expected):
    answerer = build_answerer(_settings(llm_provider=provider))

    assert isinstance(answerer, expected)


def test_the_model_defaults_to_one_the_chosen_provider_actually_serves():
    assert _settings(llm_provider="anthropic").resolved_answer_model == "claude-opus-5"
    assert _settings(llm_provider="openai").resolved_answer_model == "gpt-4o-mini"
    # An explicit choice always wins.
    assert (
        _settings(llm_provider="openai", answer_model="gpt-4.1").resolved_answer_model
        == "gpt-4.1"
    )


def test_only_the_selected_providers_key_counts_as_configured():
    # Holding an unused Anthropic key must not make an OpenAI deployment look
    # ready — /query would then fail at request time instead of at the gate.
    openai_selected = _settings(llm_provider="openai", anthropic_api_key="sk-ant-unused")
    assert not openai_selected.generation_configured
    assert openai_selected.generation_key_variable == "OPENAI_API_KEY"

    assert _settings(llm_provider="openai", openai_api_key="sk-test").generation_configured
    assert _settings(llm_provider="anthropic", anthropic_api_key="sk-ant-test").generation_configured


def test_an_unknown_provider_is_rejected_at_startup():
    with pytest.raises(ValueError):
        _settings(llm_provider="gemini")
