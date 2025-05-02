from __future__ import annotations

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
from factly.settings import FactlySettings

logger = logging.getLogger(__name__)


class MMLUBenchmark(MMLU):
    def __init__(
        self,
        tasks: list[MMLUTask] | None = None,
        n_shots: int = 0,
        n_problems_per_task: int | None = None,
        **kwargs,
    ):
        constrained_prompt = "Provide ONLY the letter of your answer (A, B, C, or D)."
        super().__init__(
            tasks=tasks or list(MMLUTask),
            n_shots=n_shots,
            n_problems_per_task=n_problems_per_task,
            confinement_instructions=constrained_prompt,
            **kwargs,
        )
        # Initialize default concurrency context
        self._concurrency_semaphore = asyncio.Semaphore(10)  # Default value

    def set_concurrency(self, max_concurrent: int | None = None):
        """Set the maximum number of concurrent question evaluations.

        Args:
            max_concurrent: Maximum number of concurrent question evaluations
        """
        # A sensible default for LLM API calls
        default_concurrency = ResourceManager.get_optimal_workers(
            min_workers=5, max_workers=20
        )
        concurrency = max_concurrent or default_concurrency
        self._concurrency_semaphore = asyncio.Semaphore(concurrency)
        logger.debug("Set evaluation concurrency to %d", concurrency)
        return concurrency

    async def a_evaluate(
        self, model: FactlyGptModel, workers: int | None = None
    ) -> float:
        """Evaluate a model on the MMLU benchmark with progress tracking.

        Overrides the base MMLU evaluate method to provide a cleaner evaluation
        process with parallel question processing for better performance.

        Args:
            model: The model to evaluate
            workers: Number of concurrent question evaluations
                (default: auto-determined)

        Returns:
            The overall accuracy score
        """
        self.set_concurrency(workers)

        # Collect all questions across all tasks
        total_questions = 0
        all_goldens = []
        all_tasks = []

        # First, collect all questions across all tasks
        for task in self.tasks:
            goldens = self.load_benchmark_dataset(task)
            if self.n_problems_per_task is not None and self.n_problems_per_task < len(
                goldens
            ):
                goldens = goldens[: self.n_problems_per_task]

            total_questions += len(goldens)
            all_goldens.extend(goldens)
            all_tasks.extend([task] * len(goldens))

        # Create progress tracking
        progress_bar = tqdm(total=total_questions, desc=model.prompt_name)

        async def process_question(task, golden, idx):
            """Process a single question with semaphore-controlled concurrency."""
            async with self._concurrency_semaphore:
                prediction, score = await self.a_predict(model, task, golden)
                progress_bar.update(1)
                return {
                    "idx": idx,
                    "task_value": task.value,
                    "input": golden.input,
                    "prediction": prediction,
                    "expected": golden.expected_output,
                    "score": score,
                }

        # Launch all evaluation tasks
        tasks = [
            process_question(task, golden, i)
            for i, (task, golden) in enumerate(zip(all_tasks, all_goldens))
        ]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Close the progress bar
        progress_bar.close()

        # Process results
        task_results = {}
        predictions_row = []
        scores_row = []
        overall_correct_predictions = 0
        overall_total_predictions = 0

        for result in results:
            task_value = result["task_value"]

            if task_value not in task_results:
                task_results[task_value] = {"correct": 0, "total": 0}

            task_results[task_value]["total"] += 1
            overall_total_predictions += 1

            if result["score"]:
                task_results[task_value]["correct"] += 1
                overall_correct_predictions += 1

            predictions_row.append(
                (
                    task_value,
                    result["input"],
                    result["prediction"],
                    result["expected"],
                    result["score"],
                )
            )

            logger.debug("Question: %s", result["input"])
            logger.debug("Prediction: %s", result["prediction"])
            logger.debug("Expected: %s", result["expected"])
            logger.debug("Score: %s", result["score"])

        # Calculate scores by task
        for task_name, results in task_results.items():
            task_accuracy = results["correct"] / results["total"]
            scores_row.append((task_name, task_accuracy))

        overall_accuracy = overall_correct_predictions / overall_total_predictions

        # Store results
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

    async def _get_structured_prediction(
        self, model: FactlyGptModel, prompt: str
    ) -> str:
        """Get a structured prediction from the model, with fallback to text.

        Returns a normalized string prediction.
        """
        # Idiomatically, MMLU tests use max_tokens=1. In that case we need to
        # add the confinement instructions to the prompt and invoke the model
        # without a schema.
        if model.max_tokens == 1:
            constrained_prompt = f"{prompt}\n\n{self.confinement_instructions}"
            response = await model.ainvoke(constrained_prompt)
            return self._normalize_text_response(response)

        try:
            # Attempt to get a structured response
            response = await model.ainvoke(
                prompt=prompt,
                schema=MultipleChoiceSchema,  # type: ignore[arg-type]
            )
            return self._extract_answer(response)
        except TypeError as e:
            # Fall back to unstructured text completion
            logger.warning("Structured output failed (%s), falling back to text", e)
            constrained_prompt = f"{prompt}\n\n{self.confinement_instructions}"
            response = await model.ainvoke(constrained_prompt)
            return self._normalize_text_response(response)

    def _extract_answer(self, response) -> str:
        """Extract the answer from a structured response of various possible types."""
        if response is None:
            return ""
        if isinstance(response, str):
            return response
        if isinstance(response, dict) and "answer" in response:
            return response["answer"]
        if isinstance(response, MultipleChoiceSchema):
            return response.answer

        raise ValueError(
            f"Unexpected response type: {type(response)}. "
            "Cannot extract answer from response."
        )

    def _normalize_text_response(self, response) -> str:
        """Normalize any text response to a valid string."""
        if response is None:
            return ""
        return str(response)

    def _create_prompt(self, task: MMLUTask, golden: Golden) -> str:
        """Create a prompt from the template and question."""
        return MMLUTemplate.generate_output(
            train_set=self.shots_dataset,
            input=golden.input,
            task=task,
            n_shots=self.n_shots,
        )

    async def a_predict(
        self, model: FactlyGptModel, task: MMLUTask, golden: Golden
    ) -> tuple[str, float]:
        if self.shots_dataset is None:
            raise RuntimeError("Example dataset is empty")

        # Generate prompt from template
        prompt = self._create_prompt(task, golden)
        prediction = await self._get_structured_prediction(model, prompt)

        # Score the prediction against the expected answer
        score = self.scorer.exact_match_score(
            str(golden.expected_output), str(prediction)
        )

        return prediction, score


def load_instructions(path: Path) -> list[dict]:
    """Load system instructions from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["instructions"]


async def _evaluate_model(
    factly_model: FactlyGptModel,
    mmlu_tasks: list[MMLUTask] | None = None,
    n_shots: int = 0,
    workers: int | None = None,
) -> float:
    """Evaluate a single model and return its score."""
    benchmark = MMLUBenchmark(tasks=mmlu_tasks, n_shots=n_shots)
    score = await benchmark.a_evaluate(model=factly_model, workers=workers)
    return float(score) if score is not None else 0.0


async def _evaluate(
    instructions: Path,
    settings: FactlySettings,
    mmlu_tasks: list[MMLUTask] | None = None,
    workers: int | None = None,
    plot: bool = False,
    plot_path: Path | None = None,
) -> None:
    """Asynchronously evaluate models with different prompts on the MMLU benchmark."""
    loaded_instructions = load_instructions(instructions)
    logger.info(
        "Evaluating %d prompts across %s MMLU tasks",
        len(loaded_instructions),
        len(mmlu_tasks) if mmlu_tasks else "all",
    )

    # Determine optimal concurrency for question evaluation based on system resources
    concurrency = workers or ResourceManager.get_optimal_workers(
        min_workers=5, max_workers=20
    )

    logger.info("Concurrency: %d concurrent question evaluations", concurrency)
    logger.info("Model name: %s", settings.model.model)
    logger.info("Temperature: %.1f", settings.inference.temperature)
    logger.info("Top-p: %.1f", settings.inference.top_p)
    logger.info("Max tokens: %d", settings.inference.max_tokens)

    # Initialize models with different instructions
    factly_models = []
    prompt_names = []

    for instruction in loaded_instructions:
        model_instance = FactlyGptModel(
            model=settings.model.model,
            system_prompt=instruction["system_prompt"],
            prompt_name=instruction["name"],
            base_url=openai.base_url,
            api_key=openai.api_key,
            temperature=settings.inference.temperature,
            top_p=settings.inference.top_p,
            max_tokens=settings.inference.max_tokens,
        )
        factly_models.append(model_instance)
        prompt_names.append(instruction["name"])

    results = []
    for i, model_instance in enumerate(factly_models):
        logger.info(
            "Evaluating prompt '%s' (%d/%d)...",
            model_instance.prompt_name,
            i + 1,
            len(factly_models),
        )

        score = await _evaluate_model(
            model_instance,
            mmlu_tasks,
            settings.inference.n_shots,
            workers,
        )
        results.append((score, model_instance.prompt_name))
        logger.info(
            "Completed evaluation for prompt '%s': %.4f",
            model_instance.prompt_name,
            score,
        )

    logger.info("\nFinal Results:")
    for score, name in sorted(results, key=lambda x: x[1]):
        logger.info("Prompt '%s': %.4f", name, score)

    if plot and len(results) > 0:
        from factly.plots import generate_factuality_comparison_plot

        # Get task names for the plot footer
        task_names = [task.name for task in mmlu_tasks] if mmlu_tasks else []

        # Generate the plot with metadata, using the display model name
        plot_file = generate_factuality_comparison_plot(
            results=results,
            model_name=factly_models[0].get_display_model_name(),
            output_path=plot_path,
            tasks=task_names,
        )
        logger.info("Generated factuality comparison plot: %s", plot_file)


def evaluate(
    instructions: Path,
    settings: FactlySettings,
    tasks: list[MMLUTask] | None = None,
    workers: int | None = None,
    plot: bool = False,
    plot_path: Path | None = None,
):
    """Evaluate models with different prompts on the MMLU benchmark.

    Args:
        instructions: Path to YAML file with system instructions
        settings: FactlySettings object containing model and inference settings
        tasks: List of MMLU tasks to evaluate (defaults to CS and Astronomy)
        workers: Number of concurrent workers for model evaluations
                (default: auto-determined based on system resources)
        plot: Whether to generate a plot of the results (default: False)
        plot_path: Path to save the plot
            (default: ./outputs/factuality-<model>-t<count>.png)
    """
    asyncio.run(_evaluate(instructions, settings, tasks, workers, plot, plot_path))
