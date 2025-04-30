"""Common test fixtures, mocks and configurations."""

import os
import tempfile
from pathlib import Path
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


@pytest.fixture(autouse=True)
def isolated_env():
    """Provide an isolated environment for testing environment variables.

    This fixture automatically applies to all tests and ensures proper
    isolation of environment variables and the OpenAI module.
    """
    import importlib

    import openai

    # Save original values
    old_env = os.environ.copy()
    old_api_key = openai.api_key
    old_base_url = openai.base_url

    # Clear environment
    os.environ.clear()
    openai.api_key = None
    openai.base_url = None

    # Force module reload to ensure clean state
    importlib.reload(openai)

    yield

    # Restore original values
    os.environ.clear()
    os.environ.update(old_env)
    openai.api_key = old_api_key
    openai.base_url = old_base_url


@pytest.fixture
def mock_resolve_tasks():
    """Mock the resolve_tasks function."""
    with mock.patch("factly.cli.evaluate"):
        with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
            task1 = mock.MagicMock(spec=MMLUTask)
            task1.name = "mathematics"
            task2 = mock.MagicMock(spec=MMLUTask)
            task2.name = "physics"
            mock_resolve.return_value = [task1, task2]
            yield mock_resolve
