"""Plotting utilities for Factly benchmarks."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def generate_factuality_comparison_plot(
    results: list[tuple[float, int, str]],
    model_name: str,
    output_path: Path | None = None,
    tasks: list[str] | None = None,
) -> Path:
    """Generate a bar chart comparing factuality scores of different prompts.

    Args:
        results: List of tuples containing (score, index, prompt_name)
        model_name: Name of the LLM model used for the benchmark
        output_path: Path to save the plot (default: creates outputs dir in cwd)
        tasks: List of MMLU task names used in the benchmark

    Returns:
        Path to the saved plot file
    """
    results.sort(key=lambda x: x[1])

    scores = [score * 100 for score, _, _ in results]
    labels = [name for _, _, name in results]

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = ["#a5a5a5", "#5fb0d6", "#faa638"]
    if len(results) > len(colors):
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(results)))

    bars = ax.bar(labels, scores, color=colors[: len(results)])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    if len(results) > 1:
        baseline_score = scores[0]
        for i, score in enumerate(scores[1:], 1):
            diff = score - baseline_score
            if abs(diff) < 0.001:
                diff_text = f"Same as {labels[0]}"
            else:
                diff_text = f"{'+' if diff >= 0 else ''}{diff:.2f}% vs {labels[0]}"

            if i > 1:
                prev_diff = score - scores[i - 1]
                if abs(prev_diff) < 0.001:
                    diff_text += f"\nSame as {labels[i - 1]}"
                else:
                    sign = "+" if prev_diff >= 0 else ""
                    diff_text += f"\n{sign}{prev_diff:.2f}% vs {labels[i - 1]}"

            ax.text(
                i,
                score + 0.5,
                diff_text,
                ha="center",
                va="bottom",
                color="red" if diff > 0 else ("green" if abs(diff) < 0.001 else "blue"),
                fontsize=10,
            )

    ax.set_ylabel("Factuality (%)")

    ax.set_title("Factuality Comparison of Custom Prompts over MMLU")
    ax.set_ylim(80, 100)

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    add_metadata_footer(
        fig,
        model_name=model_name,
        tasks=tasks,
    )

    if output_path is None:
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)
        count_tasks = len(tasks) if tasks else "all"
        output_path = output_dir / f"factuality-{model_name}-t{count_tasks}.png"
    else:
        output_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path, dpi=300)

    return output_path


def add_metadata_footer(
    fig: Figure,
    model_name: str,
    tasks: list[str] | None = None,
) -> None:
    """Add a metadata footer to the plot with date, model, and tasks information.

    Args:
        fig: The matplotlib figure to add footer to
        model_name: Name of the model used for evaluation
        tasks: List of task names used in the evaluation
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    if tasks:
        if len(tasks) > 5:
            visible_tasks = []
            for task in tasks[:5]:
                visible_tasks.append(task)

            tasks_text = ", ".join(visible_tasks)
            remaining = len(tasks) - 5
            tasks_text += f" ... (+{remaining})"
        else:
            tasks_text = ", ".join(tasks)
    else:
        tasks_text = "All tasks"

    footer_text = (
        f"Date: {current_date}   |   Model: {model_name}   |   Tasks: {tasks_text}"
    )

    fig.text(
        0.5,
        0.01,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#333333",
        family="sans-serif",
        weight="normal",
    )
