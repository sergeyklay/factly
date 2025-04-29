import asyncio
import logging
from pathlib import Path

import openai
import pandas as pd
import yaml
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from deepeval.benchmarks.mmlu.template import MMLUTemplate
from deepeval.benchmarks.schema import MultipleChoiceSchema
from deepeval.dataset import Golden
from tqdm import tqdm

from factly.models import FactlyGptModel
from factly.resources import ResourceManager

logger = logging.getLogger("factly.benchmarks")


class MMLUBenchmark(MMLU):
    async def a_evaluate(
        self, model: FactlyGptModel, batch_size: int | None = None
    ) -> float:
        """Evaluate a model on the MMLU benchmark with progress tracking.

        Overrides the base MMLU evaluate method to provide a cleaner evaluation
        process with a single progress bar for all questions across all tasks.
        """
        if batch_size is not None:
            raise NotImplementedError("Batch size is not supported for MMLU benchmark.")

        overall_correct_predictions = 0
        overall_total_predictions = 0
        predictions_row = []
        scores_row = []

        total_questions = 0
        all_goldens = []
        all_tasks = []

        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            if self.n_problems_per_task is not None and self.n_problems_per_task < len(
                goldens
            ):
                goldens = goldens[: self.n_problems_per_task]

            total_questions += len(goldens)
            all_goldens.extend(goldens)
            all_tasks.extend([task] * len(goldens))

        with tqdm(total=total_questions, desc=model.prompt_name) as progress_bar:
            task_results = {}

            for _, (task, golden) in enumerate(zip(all_tasks, all_goldens)):
                if task.value not in task_results:
                    task_results[task.value] = {"correct": 0, "total": 0}

                prediction_dict = await self.a_predict(model, task, golden)
                prediction, score = (
                    prediction_dict["prediction"],
                    prediction_dict["score"],
                )

                if self.verbose_mode:
                    # Add debug logs
                    logger.info("Question: %s", golden.input)
                    logger.info("Prediction: %s", prediction)
                    logger.info("Expected: %s", golden.expected_output)
                    logger.info("Score: %s", score)

                task_results[task.value]["total"] += 1
                overall_total_predictions += 1

                if score:
                    task_results[task.value]["correct"] += 1
                    overall_correct_predictions += 1

                predictions_row.append(
                    (
                        task.value,
                        golden.input,
                        prediction,
                        golden.expected_output,
                        score,
                    )
                )

                progress_bar.update(1)

            for task_name, results in task_results.items():
                task_accuracy = results["correct"] / results["total"]
                scores_row.append((task_name, task_accuracy))

        overall_accuracy = overall_correct_predictions / overall_total_predictions

        self.predictions = pd.DataFrame(
            predictions_row,
            columns=[
                "Task",
                "Input",
                "Prediction",
                "Expected Output",
                "Correct",
            ],
        )
        self.task_scores = pd.DataFrame(scores_row, columns=["Task", "Score"])
        self.overall_score = overall_accuracy

        return overall_accuracy

    async def a_predict(
        self, model: FactlyGptModel, task: MMLUTask, golden: Golden
    ) -> dict:
        if self.shots_dataset is None:
            raise ValueError("Example dataset is empty. Call load_benchmark.")

        # Define prompt template
        prompt = MMLUTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
        )

        # Enforced model generation
        try:
            res, _ = await model.a_generate(prompt=prompt, schema=MultipleChoiceSchema)
            if not isinstance(res, MultipleChoiceSchema):
                raise ValueError(
                    "Response does not have expected 'answer' attribute. "
                    "Please use a better evaluation model."
                )
            prediction = res.answer
        except TypeError:
            prompt += f"\n\n{self.confinement_instructions}"
            prediction, _ = await model.a_generate(prompt)

        # For native models, shouldn't happen but just in case
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # Ensure prediction is a string
        prediction = str(prediction) if prediction is not None else ""

        # Define Metric
        score = self.scorer.exact_match_score(golden.expected_output, prediction)
        return {"prediction": prediction, "score": score}


def load_instructions(path: Path) -> list[dict]:
    """Load system instructions from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["instructions"]


async def _evaluate_model(
    factly_model: FactlyGptModel, mmlu_tasks: list[MMLUTask], n_shots: int = 0
) -> float:
    """Evaluate a single model and return its score."""
    benchmark = MMLUBenchmark(tasks=mmlu_tasks, n_shots=n_shots)
    score = await benchmark.a_evaluate(model=factly_model)
    return float(score) if score is not None else 0.0


async def _evaluate(
    instructions: Path,
    model: str,
    mmlu_tasks: list[MMLUTask],
    n_shots: int = 0,
    workers: int | None = None,
    verbose: bool = False,
    plot: bool = False,
    plot_path: Path | None = None,
) -> None:
    """Asynchronously evaluate models with different prompts on the MMLU benchmark."""
    if not mmlu_tasks:
        logger.warning("No MMLU tasks provided, terminating evaluation")
        return

    loaded_instructions = load_instructions(instructions)
    logger.info(
        "Evaluating %d prompts across %d MMLU tasks",
        len(loaded_instructions),
        len(mmlu_tasks),
    )

    workers = workers or ResourceManager.get_optimal_workers(
        min_workers=2, max_workers=30
    )
    logger.info("Using %d concurrent workers for evaluation", workers)

    factly_models = []
    model_name_map = {}

    for idx, instruction in enumerate(loaded_instructions):
        model_instance = FactlyGptModel(
            model=model,
            system_prompt=instruction["system_prompt"],
            prompt_name=instruction["name"],
            base_url=openai.base_url,
            api_key=openai.api_key,
        )
        factly_models.append(model_instance)
        model_name_map[idx] = instruction["name"]

    semaphore = asyncio.Semaphore(workers)

    async def run_evaluation(model_to_eval, tasks_to_run, idx):
        prompt_name = model_name_map[idx]
        async with semaphore:
            score = await _evaluate_model(model_to_eval, tasks_to_run, n_shots)
            return score, idx, prompt_name

    tasks = [
        run_evaluation(model, mmlu_tasks, i) for i, model in enumerate(factly_models)
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        score, idx, name = await coro
        results.append((score, idx, name))

    results.sort(key=lambda x: x[1])
    logger.info("Final Results:")
    for score, _, name in results:
        logger.info("Prompt '%s': %.4f", name, score)

    if plot:
        try:
            from factly.plots import generate_factuality_comparison_plot

            plot_file = generate_factuality_comparison_plot(results, plot_path)
            logger.info("Generated factuality comparison plot: %s", plot_file)
        except ImportError as e:
            logger.error("Failed to generate plot: %s", e)
            logger.error("Make sure matplotlib is installed: pip install matplotlib")


def _configure_logging(verbose: bool = False):
    """Configure module-specific logging without affecting third-party loggers.

    Args:
        verbose: Whether to show detailed information
    """
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    level = logging.INFO if verbose else logging.WARNING

    logger.setLevel(level)
    logger.addHandler(console_handler)

    logger.propagate = False

    while logger.handlers:
        logger.handlers.pop()

    logger.addHandler(console_handler)


def evaluate(
    instructions: Path,
    model: str,
    tasks: list[MMLUTask],
    n_shots: int = 0,
    workers: int | None = None,
    verbose: bool = False,
    plot: bool = False,
    plot_path: Path | None = None,
):
    """Evaluate models with different prompts on the MMLU benchmark.

    Args:
        instructions: Path to YAML file with system instructions
        model: The LLM model to use
        tasks: List of MMLU tasks to evaluate (defaults to CS and Astronomy)
        n_shots: Number of shots for few-shot learning (default: 0)
        workers: Number of concurrent workers for model evaluations
                (default: auto-determined based on system resources)
        verbose: Whether to print detailed progress information (default: False)
        plot: Whether to generate a plot of the results (default: False)
        plot_path: Path to save the plot (default: ./outputs/factuality_comparison.png)
    """
    _configure_logging(verbose)
    asyncio.run(
        _evaluate(
            instructions,
            model,
            tasks,
            n_shots,
            workers,
            verbose,
            plot,
            plot_path,
        )
    )
