"""Common test fixtures, mocks and configurations."""

import os
import tempfile
from pathlib import Path
from typing import Dict
from unittest import mock

import pytest
from deepeval.benchmarks.mmlu.task import MMLUTask


@pytest.fixture
def temp_instructions():
    """Create a temporary instructions YAML file for testing."""
    content = """
instructions:
  - name: Basic
    prompt: Answer the multiple-choice question with A, B, C, or D.
  - name: Detailed
    prompt: Analyze the question carefully and select the most accurate answer (A, B, C, or D).
"""  # noqa: E501
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w+") as f:
        f.write(content)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def mock_tasks():
    """Return a list of mock MMLUTask objects."""
    task1 = mock.MagicMock(spec=MMLUTask)
    task1.name = "mathematics"
    task2 = mock.MagicMock(spec=MMLUTask)
    task2.name = "physics"
    return [task1, task2]


@pytest.fixture
def clean_api_env_vars(monkeypatch):
    """Fixture to provide a clean environment for API-related variables.

    Clears common API-related environment variables to ensure tests
    start with a clean slate.

    Returns:
        dict: An empty dictionary that can be used to track applied env vars.
    """
    # Clear all environment variables related to APIs
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    # Return a dict that tests can use to track what they set
    return {}


@pytest.fixture
def setup_api_env(monkeypatch):
    """Helper function to setup API environment variables for tests.

    Works with any provider's environment variables by taking a
    dictionary of variable names and values.

    Returns a function that can be used to set env vars for any API provider.
    """

    def _setup(env_vars: Dict[str, str]):
        """Set environment variables for API testing.

        Args:
            env_vars: Dictionary of environment variables to set
        """
        for key, value in env_vars.items():
            monkeypatch.setenv(key, value)

    return _setup


@pytest.fixture
def mock_openai_module(monkeypatch):
    """Mock the OpenAI module with proper cleanup.

    Returns a Mock object for the OpenAI module that tests can configure.
    """
    import importlib

    import openai

    # Save original values
    old_api_key = openai.api_key
    old_base_url = openai.base_url

    # Create a mock for the entire openai module
    mock_openai = mock.MagicMock()
    monkeypatch.setattr("factly.benchmarks.openai", mock_openai)

    # Yield the mock for test configuration
    yield mock_openai

    # Restore original values
    openai.api_key = old_api_key
    openai.base_url = old_base_url
    importlib.reload(openai)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Create a fixture to mock environment variables.

    Returns:
        callable: A function that can be used to mock os.getenv.
    """
    env_vars = {}

    def _mock_getenv(key: str, default=None):
        """Mock implementation of os.getenv."""
        return env_vars.get(key, default)

    def _set_env_vars(vars_dict: Dict[str, str]):
        """Set mock environment variables.

        Args:
            vars_dict: Dictionary of environment variables to set
        """
        env_vars.update(vars_dict)
        monkeypatch.setattr("os.getenv", _mock_getenv)
        return env_vars

    return _set_env_vars


@pytest.fixture
def clean_environment(monkeypatch):
    """Provide a completely clean environment with no preset variables.

    This fixture can be used in tests that need to start with a completely
    clean slate, ensuring no environment variables interfere with the test.
    """
    with monkeypatch.context() as m:
        # Clear all environment variables
        for key in list(os.environ.keys()):
            m.delenv(key, raising=False)
        yield


@pytest.fixture
def isolate_settings_env(monkeypatch):
    """Isolate settings from environment variables.

    Prevents the settings module from loading values from environment
    variables, ensuring tests have predictable starting conditions.
    """
    monkeypatch.setattr("factly.settings.ModelSettings.model_config", {})
    yield


@pytest.fixture
def mock_resolve_tasks():
    """Mock the resolve_tasks function."""
    with mock.patch("factly.cli.mmlu"):
        with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
            task1 = mock.MagicMock(spec=MMLUTask)
            task1.name = "mathematics"
            task2 = mock.MagicMock(spec=MMLUTask)
            task2.name = "physics"
            mock_resolve.return_value = [task1, task2]
            yield mock_resolve
