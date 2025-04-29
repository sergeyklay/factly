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

logger = logging.getLogger(__name__)


class MMLUBenchmark(MMLU):
    def __init__(
        self,
        tasks: list[MMLUTask] | None = None,
        n_shots: int = 0,
        n_problems_per_task: int | None = None,
        verbose_mode: bool = False,
        confinement_instructions: str | None = None,
        **kwargs,
    ):
        super().__init__(
            tasks=tasks or list(MMLUTask),
            n_shots=n_shots,
            n_problems_per_task=n_problems_per_task,
            verbose_mode=verbose_mode,
            confinement_instructions=confinement_instructions,
            **kwargs,
        )

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

                logger.debug("Question: %s", golden.input)
                logger.debug("Prediction: %s", prediction)
                logger.debug("Expected: %s", golden.expected_output)
                logger.debug("Score: %s", score)

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

    async def _get_structured_prediction(
        self, model: FactlyGptModel, prompt: str
    ) -> str:
        """Get a structured prediction from the model, with fallback to text.

        Returns a normalized string prediction.
        """
        try:
            # Attempt to get a structured response
            response, _ = await model.a_generate(
                prompt=prompt,
                schema=MultipleChoiceSchema,
            )
            return self._extract_answer(response)
        except TypeError as e:
            # Fall back to unstructured text completion
            logger.warning("Structured output failed (%s), falling back to text", e)
            constrained_prompt = f"{prompt}\n\n{self.confinement_instructions}"
            text_response, _ = await model.a_generate(constrained_prompt)
            return self._normalize_text_response(text_response)

    def _extract_answer(self, response) -> str:
        """Extract the answer from a structured response of various possible types."""
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
    ) -> dict:
        if self.shots_dataset is None:
            raise RuntimeError("Example dataset is empty")

        # Generate prompt from template
        prompt = self._create_prompt(task, golden)

        # Get prediction using the most appropriate method for this model
        try:
            prediction = await self._get_structured_prediction(model, prompt)
        except TypeError:
            prompt += f"\n\n{self.confinement_instructions}"
            prediction, _ = self._normalize_text_response(
                await model.a_generate(prompt)
            )

        # Score the prediction against the expected answer
        score = self.scorer.exact_match_score(golden.expected_output, prediction)

        return {"prediction": prediction, "score": score}


def load_instructions(path: Path) -> list[dict]:
    """Load system instructions from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["instructions"]


async def _evaluate_model(
    factly_model: FactlyGptModel,
    mmlu_tasks: list[MMLUTask] | None = None,
    n_shots: int = 0,
    verbose: bool = False,
) -> float:
    """Evaluate a single model and return its score."""
    benchmark = MMLUBenchmark(
        tasks=mmlu_tasks,
        n_shots=n_shots,
        verbose_mode=verbose,
    )
    score = await benchmark.a_evaluate(model=factly_model)
    return float(score) if score is not None else 0.0


async def _evaluate(
    instructions: Path,
    model: str,
    mmlu_tasks: list[MMLUTask] | None = None,
    n_shots: int = 0,
    workers: int | None = None,
    verbose: bool = False,
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

    workers = workers or ResourceManager.get_optimal_workers(
        min_workers=2, max_workers=30
    )
    logger.info("Using %d concurrent workers for evaluation", workers)
    logger.info("Model name: %s\n", model)

    factly_models = []
    prompt_versions = {}

    for idx, instruction in enumerate(loaded_instructions):
        model_instance = FactlyGptModel(
            model=model,
            system_prompt=instruction["system_prompt"],
            prompt_name=instruction["name"],
            base_url=openai.base_url,
            api_key=openai.api_key,
        )
        factly_models.append(model_instance)
        prompt_versions[idx] = instruction["name"]

    semaphore = asyncio.Semaphore(workers)

    async def run_evaluation(model_to_eval, tasks_to_run, idx):
        async with semaphore:
            score = await _evaluate_model(
                model_to_eval,
                tasks_to_run,
                n_shots,
                verbose,
            )
            return score, idx, prompt_versions[idx]

    tasks = [
        run_evaluation(model, mmlu_tasks, i) for i, model in enumerate(factly_models)
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        score, idx, name = await coro
        results.append((score, idx, name))

    results.sort(key=lambda x: x[1])
    logger.info("\nFinal Results:")
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
    tasks: list[MMLUTask] | None = None,
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
