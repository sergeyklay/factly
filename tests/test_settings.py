"""
Tests for the settings module.

Tests the configuration models for API, inference, and application settings.
"""

import pytest
from pydantic import ValidationError

from factly.settings import FactlySettings, InferenceSettings, ModelSettings


def test_model_settings_defaults(isolate_settings_env):
    settings = ModelSettings()
    assert settings.api_base == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o"
    assert settings.api_key is None


def test_model_settings_create():
    """Test the create method properly sets values."""
    settings = ModelSettings.create(
        api_base="https://custom-api.example.com",
        model="llama-3",
        api_key="sk-test123",
    )
    assert settings.api_base == "https://custom-api.example.com"
    assert settings.model == "llama-3"
    assert settings.api_key == "sk-test123"


def test_model_settings_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-123456")
    monkeypatch.setattr(
        "factly.settings.ModelSettings.model_config",
        {
            "env_prefix": "OPENAI_",
            "extra": "ignore",
        },
    )
    settings = ModelSettings()
    assert settings.api_key == "sk-123456"


def test_inference_settings_defaults_and_create():
    """Test InferenceSettings defaults and create method."""
    # Default values
    defaults = InferenceSettings.create()
    assert defaults.temperature == 0.0
    assert defaults.top_p == 1.0
    assert defaults.max_tokens == 256
    assert defaults.n_shots == 0

    # Custom values
    custom = InferenceSettings.create(
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        n_shots=3,
    )
    assert custom.temperature == 0.5
    assert custom.top_p == 0.9
    assert custom.max_tokens == 100
    assert custom.n_shots == 3


def test_inference_settings_mmlu():
    """Test MMLU-specific settings factory."""
    mmlu = InferenceSettings.for_mmlu()
    assert mmlu.temperature == 0.0
    assert mmlu.top_p == 1.0
    assert mmlu.max_tokens == 1
    assert mmlu.n_shots == 0  # Default n_shots should be preserved


def test_inference_settings_validation():
    """Test validation constraints on InferenceSettings."""
    with pytest.raises(ValidationError):
        InferenceSettings(temperature=3.0)

    with pytest.raises(ValidationError):
        InferenceSettings(top_p=0.0)

    with pytest.raises(ValidationError):
        InferenceSettings(max_tokens=0)

    with pytest.raises(ValidationError):
        InferenceSettings(n_shots=-1)  # n_shots should be >= 0


def test_factly_settings_create():
    """Test FactlySettings create with nested dictionaries."""
    settings = FactlySettings.create(
        model={"api_key": "sk-abc123", "model": "gpt-4-turbo"},
        inference={"temperature": 0.7, "max_tokens": 50, "n_shots": 5},
    )

    assert settings.model.api_key == "sk-abc123"
    assert settings.model.model == "gpt-4-turbo"

    assert settings.inference.temperature == 0.7
    assert settings.inference.max_tokens == 50
    assert settings.inference.n_shots == 5


def test_factly_settings_from_cli(clean_api_env_vars):
    """Test CLI arguments override environment variables."""
    settings = FactlySettings.from_cli(
        model="gpt-4-turbo",
        api_key="sk-test-cli",
        temperature=0.7,
        max_tokens=50,
        n_shots=2,
    )

    assert settings.model.model == "gpt-4-turbo"
    assert settings.model.api_key == "sk-test-cli"
    assert settings.inference.temperature == 0.7
    assert settings.inference.max_tokens == 50
    assert settings.inference.n_shots == 2
