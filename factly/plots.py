"""Plotting utilities for Factly benchmarks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


# TODO(serghei): Refactor this function to be more readable and maintainable.
def generate_factuality_comparison_plot(  # noqa: C901
    results: list[tuple[float, str]],
    model_name: str,
    output_path: Path | None = None,
    tasks: list[str] | None = None,
) -> Path:
    """Generate a bar chart comparing factuality scores of different prompts.

    Args:
        results: List of tuples containing (score, prompt_name)
        model_name: Name of the LLM model used for the benchmark
        output_path: Path to save the plot (default: creates outputs dir in cwd)
        tasks: List of MMLU task names used in the benchmark

    Returns:
        Path to the saved plot file
    """
    # Sort ascending by score
    results.sort(key=lambda x: x[0])

    scores = [score * 100 for score, _ in results]
    labels = [name for _, name in results]

    fig, ax = plt.subplots(figsize=(12, 9))

    min_score = min(scores) if scores else 80
    max_score = max(scores) if scores else 100
    score_range = max_score - min_score

    has_very_small_values = any(score < 5 for score in scores)
    has_zeros = any(score == 0 for score in scores)

    y_min = 0 if has_zeros or has_very_small_values else max(min_score - 10, 0)
    y_max = 100

    ax.set_ylim(y_min, y_max)

    extreme_range = score_range > 30

    colors = ["#a5a5a5", "#5fb0d6", "#faa638"]
    if len(results) > len(colors):
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(results)))

    bars = ax.bar(
        labels,
        scores,
        color=colors[: len(results)],
        zorder=3,
        width=0.6,
        edgecolor=None,
        linewidth=0,
        alpha=0.9,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
            zorder=4,
        )

    if len(results) > 1:
        high_values = any(score > 90 for score in scores)

        baseline_score = scores[0]
        baseline_label = labels[0]

        for i, score in enumerate(scores[1:], 1):
            diff = score - baseline_score

            if abs(diff) < 0.001:
                diff_text = f"Same as {baseline_label}"
            else:
                diff_text = f"{'+' if diff >= 0 else ''}{diff:.2f}% vs {baseline_label}"

            if i > 1:
                prev_score = scores[i - 1]
                prev_label = labels[i - 1]
                prev_diff = score - prev_score

                if abs(prev_diff) < 0.001:
                    diff_text += f"\nSame as {prev_label}"
                else:
                    sign = "+" if prev_diff >= 0 else ""
                    diff_text += f"\n{sign}{prev_diff:.2f}% vs {prev_label}"

            if has_zeros or has_very_small_values:
                if score == 0:
                    y_pos = 5
                elif score < 5:
                    y_pos = score + 5
                elif score <= 40:
                    y_pos = score + 8
                else:
                    y_pos = min(score + 6, 60)
            elif extreme_range:
                chart_height = y_max - y_min

                if score > 90:
                    y_pos = 15
                elif score < 30:
                    y_pos = 40
                else:
                    y_pos = 10

                y_pos = y_min + (chart_height * y_pos / 100)
            elif high_values:
                if score > 95:
                    y_pos = y_min + (y_max - y_min) * 0.25

                    if i % 2 == 0:
                        y_pos = y_min + (y_max - y_min) * 0.15
                else:
                    y_pos = score + 8
            else:
                y_pos = score + 3

            ax.text(
                i,
                y_pos,
                diff_text,
                ha="center",
                va="bottom",
                color="red" if diff > 0 else ("green" if abs(diff) < 0.001 else "blue"),
                fontsize=10,
                fontweight="normal",
                zorder=4,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.7,
                    "edgecolor": "none",
                    "pad": 2,
                },
            )

    ax.set_ylabel("Factuality (%)", fontweight="normal", fontsize=12)
    ax.set_title(
        "Factuality Comparison of Custom Prompts over MMLU",
        fontweight="normal",
        fontsize=14,
    )

    ax.tick_params(axis="x", labelsize=11, labelrotation=0)
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.7, zorder=1)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    ax.set_facecolor("#f8f8f8")

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

    plt.savefig(output_path, dpi=300, bbox_inches="tight")

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
