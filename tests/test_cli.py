"""Unit tests for the CLI module."""

import os
import subprocess
import sys
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from factly import __copyright__, __version__
from factly.cli import cli, main


@pytest.fixture(autouse=True)
def env_reset():
    """Reset environment variables before each test."""
    with mock.patch.dict(os.environ, {}, clear=True):
        yield


@pytest.fixture
def runner():
    """Fixture for CliRunner."""
    return CliRunner()


def test_version_option(runner):
    """Test that the --version option prints the correct version and exits."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"factly {__version__}" in result.output
    assert __copyright__ in result.output
    assert "This is free software" in result.output
    assert "warranty" in result.output


@pytest.mark.skipif(
    sys.platform == "win32", reason="Command execution differs on Windows"
)
def test_version_option_module():
    """Test that the --version option works when running as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "factly", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert f"factly {__version__}" in result.stdout
    assert __copyright__ in result.stdout
    assert "This is free software" in result.stdout
    assert "warranty" in result.stdout


def test_help_option(runner):
    """Test that the --help option prints help text and exits."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "evaluate" in result.output
    assert "list-tasks" in result.output


def test_list_tasks_command(runner):
    """Test that the list-tasks command prints available tasks."""
    with mock.patch("factly.cli.list_available_tasks") as mock_list:
        mock_list.return_value = "Mock task list"
        result = runner.invoke(cli, ["list-tasks"])

        assert result.exit_code == 0
        assert "Mock task list" in result.output
        mock_list.assert_called_once()


def test_list_tasks_help(runner):
    """Test that the list-tasks --help option prints help text."""
    result = runner.invoke(cli, ["list-tasks", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "List all available MMLU tasks" in result.output


def test_evaluate_help(runner):
    """Test that the evaluate --help option prints help text."""
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "-m, --model" in result.output
    assert "-u, --url" in result.output
    assert "-a, --api-key" in result.output
    assert "--n-shots" in result.output
    assert "--tasks" in result.output
    assert "--verbose" in result.output
    assert "-j, --workers" in result.output
    assert "--plot" in result.output
    assert "--plot-path" in result.output


@pytest.mark.parametrize(
    "args,model_arg,expected_model",
    [
        (["evaluate"], None, "gpt-4o"),
        (["evaluate", "--model", "gpt-4"], "gpt-4", "gpt-4"),
        (["evaluate", "-m", "gpt-3.5-turbo"], "gpt-3.5-turbo", "gpt-3.5-turbo"),
    ],
)
def test_evaluate_model_option(
    runner, args, model_arg, expected_model, mock_resolve_tasks
):
    """Test model option parsing in evaluate command."""
    with mock.patch("os.getenv") as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "OPENAI_MODEL" and model_arg is None:
                return "gpt-4o"
            return default

        mock_getenv.side_effect = getenv_side_effect

        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(cli, args)

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()
            assert mock_evaluate.call_args.kwargs["model"] == expected_model


def test_evaluate_sets_api_values(runner, mock_resolve_tasks):
    """Test that api-key and url options set the openai module values."""
    with mock.patch("factly.benchmarks.evaluate"):
        with mock.patch("factly.cli.openai") as mock_openai:
            result = runner.invoke(
                cli, ["evaluate", "-a", "test-api-key", "-u", "https://test-url.com/v1"]
            )

            assert result.exit_code == 0
            assert mock_openai.api_key == "test-api-key"
            assert mock_openai.base_url == "https://test-url.com/v1"


def test_evaluate_task_resolution(runner, mock_tasks):
    """Test task resolution in evaluate command."""
    task_names = ["mathematics", "physics"]

    with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = mock_tasks
        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(
                cli, ["evaluate", "--tasks", task_names[0], "--tasks", task_names[1]]
            )

            assert result.exit_code == 0
            mock_resolve.assert_called_once()
            assert set(mock_resolve.call_args.args[0]) == set(task_names)
            assert mock_evaluate.call_args.kwargs["tasks"] == mock_tasks


def test_evaluate_with_mock_tasks(runner, mock_tasks):
    """Test evaluation with mock tasks."""
    with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = mock_tasks
        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(cli, ["evaluate"])

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()
            assert mock_evaluate.call_args.kwargs["tasks"] == mock_tasks


def test_evaluate_task_resolution_error(runner):
    """Test error handling when task resolution fails."""
    with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
        mock_resolve.side_effect = ValueError("Invalid task")
        with mock.patch("factly.cli.logger") as mock_logger:
            result = runner.invoke(cli, ["evaluate", "--tasks", "invalid_task"])

            assert result.exit_code == 1
            mock_logger.error.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Use 'factly list-tasks' to see available tasks"
            )


def test_evaluate_default_tasks(runner, mock_tasks):
    """Test that no task argument resolves to all tasks."""
    with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = mock_tasks
        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(cli, ["evaluate"])

            assert result.exit_code == 0
            assert len(mock_resolve.call_args.args[0]) == 0
            assert mock_evaluate.call_args.kwargs["tasks"] == mock_tasks


def test_evaluate_with_custom_instructions(
    runner, temp_instructions, mock_resolve_tasks
):
    """Test using custom instructions file."""
    with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
        result = runner.invoke(
            cli, ["evaluate", "--instructions", str(temp_instructions)]
        )

        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        assert mock_evaluate.call_args.kwargs["instructions"] == temp_instructions


@pytest.mark.parametrize(
    "env_model,cli_model,expected_suffix",
    [
        (None, None, "gpt-4o"),
        (None, "gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo"),
    ],
)
def test_evaluate_model_precedence(
    runner, mock_resolve_tasks, env_model, cli_model, expected_suffix
):
    """Test precedence: CLI arg > env var > default for --model/-m."""
    with (
        mock.patch("os.getenv") as mock_os_getenv,
        mock.patch("factly.cli.os.getenv") as mock_cli_getenv,
    ):

        def getenv_side_effect(key, default=None):
            if key == "OPENAI_MODEL":
                return env_model
            return default

        mock_os_getenv.side_effect = getenv_side_effect
        mock_cli_getenv.side_effect = getenv_side_effect

        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            args = ["evaluate"]
            if cli_model:
                args += ["-m", cli_model]
            result = runner.invoke(cli, args)
            assert result.exit_code == 0

            actual_model = mock_evaluate.call_args.kwargs["model"]

            assert actual_model.endswith(expected_suffix), (
                f"Model '{actual_model}' doesn't end with '{expected_suffix}'"
            )


def test_evaluate_api_env_vars(runner):
    """Test API environment variables are used correctly."""
    api_key = "env-api-key"
    api_base = "https://env-api-base.com/v1"

    with mock.patch("os.getenv") as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "OPENAI_API_KEY":
                return api_key
            if key == "OPENAI_API_BASE":
                return api_base
            return default

        mock_getenv.side_effect = getenv_side_effect

        with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
            mock_resolve.return_value = []

            with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
                result = runner.invoke(cli, ["evaluate"])

                assert result.exit_code == 0

                call_kwargs = mock_evaluate.call_args[1]
                assert "instructions" in call_kwargs
                assert "model" in call_kwargs
                assert "tasks" in call_kwargs


def test_evaluate_api_cli_overrides_env(runner):
    """Test that CLI options override environment variables for API settings."""
    cli_api_key = "cli-api-key"
    cli_api_base = "https://cli-api-base.com/v1"

    env_api_key = "env-api-key"
    env_api_base = "https://env-api-base.com/v1"

    with mock.patch("factly.cli.openai") as mock_openai:
        mock_openai.api_key = env_api_key
        mock_openai.base_url = env_api_base

        with mock.patch("factly.cli.resolve_tasks") as mock_resolve:
            mock_resolve.return_value = []

            with mock.patch("factly.benchmarks.evaluate"):
                result = runner.invoke(
                    cli, ["evaluate", "-a", cli_api_key, "-u", cli_api_base]
                )

                assert result.exit_code == 0

                assert mock_openai.api_key == cli_api_key
                assert mock_openai.base_url == cli_api_base


def test_main_function_success():
    """Test that main function returns 0 on success."""
    with mock.patch("factly.cli.cli.main") as mock_main:
        assert main(["--version"]) == 0
        mock_main.assert_called_once_with(args=["--version"], standalone_mode=False)


def test_main_function_no_such_option():
    """Test that main function handles NoSuchOption error."""
    with mock.patch("factly.cli.cli.main") as mock_main:
        mock_main.side_effect = mock.Mock(
            side_effect=click.exceptions.NoSuchOption("--invalid")
        )
        with mock.patch("factly.cli.click.echo") as mock_echo:
            assert main(["--invalid"]) == 2
            mock_echo.assert_called_once()
            assert "No such option" in mock_echo.call_args.args[0]


def test_main_function_keyboard_interrupt():
    """Test that main function handles keyboard interrupt."""
    with mock.patch("factly.cli.cli.main") as mock_main:
        mock_main.side_effect = mock.Mock(side_effect=click.exceptions.Abort())
        with mock.patch("factly.cli.click.echo") as mock_echo:
            assert main([]) == 130
            mock_echo.assert_called_once_with("Operation aborted by user")


def test_main_function_exit():
    """Test that main function handles Exit error with proper code."""
    with mock.patch("factly.cli.cli.main") as mock_main:
        mock_main.side_effect = mock.Mock(side_effect=click.exceptions.Exit(42))
        assert main([]) == 42


def test_main_function_unexpected_error():
    """Test that main function handles unexpected errors."""
    with mock.patch("factly.cli.cli.main") as mock_main:
        mock_main.side_effect = mock.Mock(side_effect=RuntimeError("Unexpected"))
        with mock.patch("factly.cli.logger") as mock_logger:
            assert main([]) == 1
            mock_logger.error.assert_called_once()
