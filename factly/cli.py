"""Factly CLI entrypoint."""

import logging
import sys
from pathlib import Path

import click

from .logger import setup_logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
if BASE_DIR not in sys.path:
    sys.path.append(str(BASE_DIR))

BANNER = r"""

        ██████                      █████    ████
       ███░░███                    ░░███    ░░███
      ░███ ░░░   ██████    ██████  ███████   ░███  █████ ████
     ███████    ░░░░░███  ███░░███░░░███░    ░███ ░░███ ░███
    ░░░███░      ███████ ░███ ░░░   ░███     ░███  ░███ ░███
      ░███      ███░░███ ░███  ███  ░███ ███ ░███  ░███ ░███
      █████    ░░████████░░██████   ░░█████  █████ ░░███████
     ░░░░░      ░░░░░░░░  ░░░░░░     ░░░░░  ░░░░░   ░░░░░███
                                                    ███ ░███
                                                   ░░██████
                                                    ░░░░░░



"""


def get_version() -> str:
    """Get version info."""
    from . import __version__

    return __version__


def get_copyright() -> str:
    """Get copyright info."""
    from . import __copyright__

    return __copyright__


class RichGroup(click.Group):
    """Custom Click group that displays a banner before the help text."""

    def format_help(self, ctx, formatter):
        """Writes the help into the formatter if it exists.

        This method is called by Click when the help text is requested.
        """
        click.secho(BANNER, nl=False)
        super().format_help(ctx, formatter)


@click.group(
    cls=RichGroup,
    help="CLI tool to evaluate ChatGPT factuality on MMLU benchmark.",
)
@click.version_option(
    version=get_version(),
    prog_name="factly",
    message="%(prog)s %(version)s\n"
    + get_copyright()
    + "\n"
    + "This is free software; see the source for copying conditions.  There is NO\n"
    + "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.",
)
def cli():
    """Entrypoint for factly CLI."""


@cli.command()
@click.option(
    "--instructions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path.cwd() / "instructions.yaml",
    help=(
        "Path to YAML file with system instruction variants. "
        "[default: `instructions.yaml` in the current working directory]"
    ),
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=None,
    help="Model name to use for evaluation. [default: gpt-4o]",
)
@click.option(
    "-u",
    "--url",
    type=str,
    default=None,
    help="Model API URL to use for evaluation. [default: https://api.openai.com/v1]",
)
@click.option(
    "-a",
    "--api-key",
    type=str,
    default=None,
    help="Model API key to use for evaluation.",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    show_default=True,
    help="Sampling temperature for model inference.",
)
@click.option(
    "--top-p",
    type=float,
    default=1.0,
    show_default=True,
    help="Nucleus sampling parameter.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=256,
    show_default=True,
    help="Maximum number of tokens per response.",
)
@click.option(
    "--n-shots",
    type=int,
    default=0,
    show_default=True,
    help="Number of shots for few-shot learning.",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    multiple=True,
    help=(
        "List of tasks or categories to evaluate. "
        "Use 'factly list-tasks' to see available options."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed progress information during evaluation.",
)
@click.option(
    "-j",
    "--workers",
    type=int,
    default=None,
    help=(
        "Maximum number of concurrent question evaluations. "
        "[default: auto-determined by system resources]"
    ),
)
@click.option(
    "--plot",
    is_flag=True,
    help="Generate a plot of the results after evaluation.",
)
@click.option(
    "--plot-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to save the plot. [default: ./outputs/factuality-<model>-t<count>.png]",
)
def evaluate(
    instructions: Path,
    n_shots: int,
    verbose: bool,
    model: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 1,
    tasks: list[str] | None = None,
    workers: int | None = None,
    plot: bool = False,
    plot_path: Path | None = None,
):
    """Evaluate the model on the MMLU benchmark."""
    import openai
    from dotenv import load_dotenv
    from pydantic import ValidationError

    from factly.benchmarks import evaluate as do_evaluate
    from factly.settings import FactlySettings

    from .tasks import resolve_tasks

    load_dotenv(BASE_DIR / ".env")
    setup_logging(verbose=verbose)

    try:
        # Create settings with CLI parameters taking precedence
        settings = FactlySettings.from_cli(
            model=model,
            api_key=api_key,
            api_base=url,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Configure OpenAI client from settings
        openai.api_key = settings.model.api_key
        if settings.model.api_base:
            openai.base_url = settings.model.api_base

        # Convert None to empty list to satisfy type checking
        task_names = tasks if tasks is not None else []

        # Resolve task names to actual MMLUTask objects
        mmlu_tasks = resolve_tasks(task_names)
        task_names = [t.name for t in mmlu_tasks]

        click.echo(f"Evaluating {len(mmlu_tasks)} tasks: {', '.join(task_names)}")

        do_evaluate(
            instructions=instructions,
            model=settings.model.model,
            tasks=mmlu_tasks,
            n_shots=n_shots,
            workers=workers,
            plot=plot,
            plot_path=plot_path,
            temperature=settings.inference.temperature,
            top_p=settings.inference.top_p,
            max_tokens=settings.inference.max_tokens,
        )
    except ValidationError as e:
        errors = e.errors()
        for error in errors:
            message = f"{error['loc'][0]}: {error['msg']}"
            click.echo(message, err=True)
        sys.exit(1)
    except ValueError as e:
        click.echo(f"Error resolving tasks: {e}")
        click.echo("Use 'factly list-tasks' to see available tasks", err=True)
        sys.exit(1)


@cli.command("list-tasks")
def list_tasks():
    """List all available MMLU tasks for evaluation."""
    # Import only when needed
    from .tasks import list_available_tasks

    click.echo(list_available_tasks())


def main(args: list[str] | None = None) -> int:
    try:
        # Invoke the Click command
        cli.main(args=args, standalone_mode=False)
        return 0
    except click.exceptions.NoSuchOption:
        # Handle case where no option is provided
        click.echo("No such option. Use --help for more information.", err=True)
        return 2
    except click.exceptions.Abort:
        # Handle keyboard interrupts gracefully
        click.echo("Operation aborted by user")
        return 130  # Standard exit code for SIGINT
    except click.exceptions.Exit as e:
        # Handle normal exit
        return e.exit_code
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Handle unexpected errors
        logger.error(exc, exc_info=True)
        return 1
