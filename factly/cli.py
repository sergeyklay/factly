"""Factly CLI entrypoint."""

import logging
import os
import sys
from pathlib import Path

import click
import openai
from deepeval.benchmarks.mmlu.task import MMLUTask
from dotenv import load_dotenv

from . import __copyright__, __version__

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


load_dotenv(BASE_DIR / ".env")

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_API_BASE")


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
    version=__version__,
    prog_name="factly",
    message=f"""%(prog)s %(version)s
{__copyright__}
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.""",
)
def cli():
    """Entrypoint for factly CLI."""
    pass


@cli.command()
@click.option(
    "--instructions",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path.cwd() / "instructions.yaml",
    help=(
        "Path to YAML file with system instruction variants. "
        "Default is `instructions.yaml` in the current working directory."
    ),
)
@click.option(
    "--model",
    type=str,
    default=os.getenv("OPENAI_MODEL"),
    help="OpenAI model to use for evaluation.",
)
@click.option(
    "--n-shots",
    type=int,
    default=0,
    help="Number of shots for few-shot learning (default: 0).",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    multiple=True,
    help="Comma-separated list of tasks to evaluate. Default is all tasks.",
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
    help="Maximum number of concurrent model evaluations (auto-determined by default).",
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
    help="Path to save the plot (default: ./outputs/factuality.png).",
)
def evaluate(
    instructions: Path,
    model: str,
    n_shots: int,
    verbose: bool,
    tasks: list[str] | None,
    workers: int | None = None,
    plot: bool = False,
    plot_path: Path | None = None,
):
    """Evaluate the model on the MMLU benchmark."""
    from factly.benchmarks import evaluate as do_evaluate

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.base_url = os.getenv("OPENAI_API_BASE")

    # tasks = [MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE, MMLUTask.ASTRONOMY]
    tasks = [MMLUTask.ASTRONOMY]

    do_evaluate(
        instructions=instructions,
        model=model,
        tasks=tasks,
        n_shots=n_shots,
        workers=workers,
        verbose=verbose,
        plot=plot,
        plot_path=plot_path,
    )


def main(args: list[str] | None = None) -> int:
    load_dotenv(BASE_DIR / ".env")

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
