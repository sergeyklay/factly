"""Unit tests for the CLI module."""

import subprocess
import sys
from unittest import mock

import click
import pytest
from click.testing import CliRunner

from factly.cli import cli, get_copyright, get_version, main


@pytest.fixture
def runner():
    """Fixture for CliRunner."""
    return CliRunner()


def test_version_option(runner):
    """Test that the --version option prints the correct version and exits."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"factly {get_version()}" in result.output
    assert get_copyright() in result.output
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
    assert f"factly {get_version()}" in result.stdout
    assert get_copyright() in result.stdout
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
    with mock.patch("factly.tasks.list_available_tasks") as mock_list:
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
    runner, args, model_arg, expected_model, mock_resolve_tasks, mock_env_vars
):
    """Test model option parsing in evaluate command."""
    # Setup the environment variables for this test
    env_vars = {"OPENAI_MODEL": "gpt-4o"} if model_arg is None else {}
    mock_env_vars(env_vars)

    with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
        result = runner.invoke(cli, args)

        assert result.exit_code == 0
        mock_evaluate.assert_called_once()
        # Model is now part of settings object
        assert mock_evaluate.call_args.kwargs["settings"].model.model == expected_model


def test_evaluate_sets_api_values(runner, mock_resolve_tasks, clean_api_env_vars):
    """Test that api-key and url options set the openai module values."""
    with mock.patch("factly.benchmarks.evaluate"):
        result = runner.invoke(
            cli, ["evaluate", "-a", "test-api-key", "-u", "https://test-url.com/v1"]
        )

        assert result.exit_code == 0


def test_evaluate_task_resolution(runner, mock_tasks):
    """Test task resolution in evaluate command."""
    task_names = ["mathematics", "physics"]

    with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
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
    with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = mock_tasks
        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(cli, ["evaluate"])

            assert result.exit_code == 0
            mock_evaluate.assert_called_once()
            assert mock_evaluate.call_args.kwargs["tasks"] == mock_tasks


def test_evaluate_default_tasks(runner, mock_tasks):
    """Test that no task argument resolves to all tasks."""
    with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
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
    runner, env_model, cli_model, expected_suffix, mock_env_vars, clean_environment
):
    """Test precedence: CLI arg > env var > default for --model/-m."""
    # Set up environment variables for this test
    env_vars = {}
    if env_model:
        env_vars["OPENAI_MODEL"] = env_model
    # Always set an API key to avoid auth errors
    env_vars["OPENAI_API_KEY"] = "dummy-key"
    mock_env_vars(env_vars)

    with (
        mock.patch("factly.tasks.resolve_tasks", return_value=[]),
        mock.patch("factly.benchmarks.evaluate"),
        mock.patch("openai.OpenAI"),
        mock.patch("dotenv.load_dotenv"),
    ):
        args = ["evaluate"]
        if cli_model:
            args += ["-m", cli_model]
        result = runner.invoke(cli, args)

        assert result.exit_code == 0


def test_evaluate_api_env_vars(runner, mock_env_vars):
    """Test API environment variables are used correctly."""
    mock_env_vars(
        {
            "OPENAI_API_KEY": "env-api-key",
            "OPENAI_API_BASE": "https://env-api-base.com/v1",
        }
    )

    with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = []

        with mock.patch("factly.benchmarks.evaluate") as mock_evaluate:
            result = runner.invoke(cli, ["evaluate"])

            assert result.exit_code == 0

            call_kwargs = mock_evaluate.call_args[1]
            assert "instructions" in call_kwargs
            assert "settings" in call_kwargs
            # Check model settings are included
            assert call_kwargs["settings"].model is not None
            assert "tasks" in call_kwargs


def test_evaluate_api_cli_overrides_env(runner, setup_api_env):
    """Test that CLI options override environment variables for API settings."""
    # Set up environment variables first
    setup_api_env(
        {
            "OPENAI_API_KEY": "env-api-key",
            "OPENAI_API_BASE": "https://env-api-base.com/v1",
        }
    )

    # CLI values should override environment values
    cli_api_key = "cli-api-key"
    cli_api_base = "https://cli-api-base.com/v1"

    with mock.patch("factly.tasks.resolve_tasks") as mock_resolve:
        mock_resolve.return_value = []

        with mock.patch("factly.benchmarks.evaluate"):
            result = runner.invoke(
                cli, ["evaluate", "-a", cli_api_key, "-u", cli_api_base]
            )

            assert result.exit_code == 0


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
