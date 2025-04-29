"""Plotting utilities for Factly benchmarks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_factuality_comparison_plot(
    results: list[tuple[float, int, str]], output_path: Path | None = None
) -> Path:
    """Generate a bar chart comparing factuality scores of different prompts.

    Args:
        results: List of tuples containing (score, index, prompt_name)
        output_path: Path to save the plot (default: creates outputs dir in cwd)

    Returns:
        Path to the saved plot file
    """
    # Ensure results are sorted by index for consistent ordering
    results.sort(key=lambda x: x[1])

    # Extract data
    scores = [score * 100 for score, _, _ in results]  # Convert to percentages
    labels = [name for _, _, name in results]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for bars
    colors = ["#a5a5a5", "#5fb0d6", "#faa638"]
    if len(results) > len(colors):
        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(results)))

    # Create bars
    bars = ax.bar(labels, scores, color=colors[: len(results)])

    # Add value labels on top of bars
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

    # Calculate and add delta labels for comparison if more than one prompt
    if len(results) > 1:
        baseline_score = scores[0]  # Assuming first prompt is baseline
        for i, score in enumerate(scores[1:], 1):
            diff = score - baseline_score
            diff_text = f"+{diff:.2f}% vs {labels[0]}"

            # If there's a previous custom prompt, also show comparison
            if i > 1:
                prev_diff = score - scores[i - 1]
                diff_text += f"\n+{prev_diff:.2f}% vs {labels[i - 1]}"

            ax.text(
                i,
                score + 0.5,
                diff_text,
                ha="center",
                va="bottom",
                color="red" if diff > 0 else "blue",
                fontsize=10,
            )

    # Customize the plot
    ax.set_ylabel("Factuality (%)")
    ax.set_title("Factuality Comparison of Custom Prompts over MMLU")
    ax.set_ylim(80, 100)  # Set y-axis to start at 80% for better visualization

    # Add grid lines for readability
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Create output directory if it doesn't exist
    if output_path is None:
        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "factuality.png"
    else:
        output_path.parent.mkdir(exist_ok=True, parents=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    return output_path
