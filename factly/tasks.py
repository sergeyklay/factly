"""MMLU task registry and management for Factly."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from deepeval.benchmarks.mmlu.task import MMLUTask

logger = logging.getLogger(__name__)


# Task categories for organization
class TaskCategory(str, Enum):
    """Categories for organizing MMLU tasks."""

    STEM = "STEM"
    HUMANITIES = "Humanities"
    SOCIAL_SCIENCES = "Social Sciences"
    MEDICAL = "Medical"
    BUSINESS = "Business & Law"
    COMPUTER_SCIENCE = "Computer Science"
    OTHER = "Other"


# Task metadata for better discoverability and documentation
TASK_METADATA: dict[MMLUTask, dict] = {
    # STEM
    MMLUTask.ELEMENTARY_MATHEMATICS: {
        "category": TaskCategory.STEM,
        "description": "Basic arithmetic, fractions, and elementary algebra",
    },
    MMLUTask.HIGH_SCHOOL_MATHEMATICS: {
        "category": TaskCategory.STEM,
        "description": (
            "High school mathematics problems covering algebra, "
            "geometry, and basic calculus"
        ),
    },
    MMLUTask.COLLEGE_MATHEMATICS: {
        "category": TaskCategory.STEM,
        "description": (
            "College-level mathematics including calculus, "
            "linear algebra, and numerical methods"
        ),
    },
    MMLUTask.ABSTRACT_ALGEBRA: {
        "category": TaskCategory.STEM,
        "description": (
            "Advanced mathematical concepts including groups, rings, and fields"
        ),
    },
    MMLUTask.HIGH_SCHOOL_PHYSICS: {
        "category": TaskCategory.STEM,
        "description": (
            "High school physics problems covering mechanics, "
            "thermodynamics, and basic electricity"
        ),
    },
    MMLUTask.COLLEGE_PHYSICS: {
        "category": TaskCategory.STEM,
        "description": (
            "College-level physics including advanced mechanics, "
            "electromagnetism, and quantum physics"
        ),
    },
    MMLUTask.CONCEPTUAL_PHYSICS: {
        "category": TaskCategory.STEM,
        "description": (
            "Conceptual understanding of physics phenomena without heavy mathematics"
        ),
    },
    MMLUTask.HIGH_SCHOOL_CHEMISTRY: {
        "category": TaskCategory.STEM,
        "description": (
            "High school chemistry covering atomic structure, "
            "reactions, and basic organic chemistry"
        ),
    },
    MMLUTask.COLLEGE_CHEMISTRY: {
        "category": TaskCategory.STEM,
        "description": (
            "College-level chemistry including thermodynamics, "
            "kinetics, and spectroscopy"
        ),
    },
    MMLUTask.HIGH_SCHOOL_BIOLOGY: {
        "category": TaskCategory.STEM,
        "description": (
            "High school biology covering cells, genetics, ecology, and evolution"
        ),
    },
    MMLUTask.COLLEGE_BIOLOGY: {
        "category": TaskCategory.STEM,
        "description": (
            "College-level biology including molecular biology, "
            "physiology, and biotechnology"
        ),
    },
    MMLUTask.ASTRONOMY: {
        "category": TaskCategory.STEM,
        "description": (
            "Astronomy concepts including celestial bodies, "
            "cosmology, and space exploration"
        ),
    },
    MMLUTask.HIGH_SCHOOL_STATISTICS: {
        "category": TaskCategory.STEM,
        "description": (
            "High school statistics covering probability, "
            "distributions, and hypothesis testing"
        ),
    },
    MMLUTask.ELECTRICAL_ENGINEERING: {
        "category": TaskCategory.STEM,
        "description": "Electrical engineering principles, circuits, and systems",
    },
    # Computer Science
    MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE: {
        "category": TaskCategory.COMPUTER_SCIENCE,
        "description": "High school computer science concepts and algorithms",
    },
    MMLUTask.COLLEGE_COMPUTER_SCIENCE: {
        "category": TaskCategory.COMPUTER_SCIENCE,
        "description": (
            "College-level computer science including data structures, "
            "algorithms, and theory"
        ),
    },
    MMLUTask.MACHINE_LEARNING: {
        "category": TaskCategory.COMPUTER_SCIENCE,
        "description": "Machine learning concepts, algorithms, and applications",
    },
    MMLUTask.COMPUTER_SECURITY: {
        "category": TaskCategory.COMPUTER_SCIENCE,
        "description": "Computer security principles, threats, and protections",
    },
    # Social Sciences
    MMLUTask.HIGH_SCHOOL_MICROECONOMICS: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": (
            "High school microeconomics concepts including supply/demand "
            "and market structures"
        ),
    },
    MMLUTask.HIGH_SCHOOL_MACROECONOMICS: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": (
            "High school macroeconomics covering GDP, inflation, "
            "and fiscal/monetary policy"
        ),
    },
    MMLUTask.ECONOMETRICS: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "Statistical methods applied to economic data",
    },
    MMLUTask.HIGH_SCHOOL_PSYCHOLOGY: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": (
            "High school psychology concepts covering cognition, "
            "development, and behavior"
        ),
    },
    MMLUTask.PROFESSIONAL_PSYCHOLOGY: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": (
            "Professional-level psychology theories, methods, and applications"
        ),
    },
    MMLUTask.SOCIOLOGY: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "Sociological theories, methods, and social phenomena",
    },
    MMLUTask.HUMAN_SEXUALITY: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "Human sexuality, sexual development, and related topics",
    },
    MMLUTask.HIGH_SCHOOL_GOVERNMENT_AND_POLITICS: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "High school government and politics concepts",
    },
    MMLUTask.SECURITY_STUDIES: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "International security studies, conflicts, and strategy",
    },
    MMLUTask.US_FOREIGN_POLICY: {
        "category": TaskCategory.SOCIAL_SCIENCES,
        "description": "US foreign policy history, principles, and applications",
    },
    # Humanities
    MMLUTask.HIGH_SCHOOL_EUROPEAN_HISTORY: {
        "category": TaskCategory.HUMANITIES,
        "description": (
            "High school European history covering major events, figures, and movements"
        ),
    },
    MMLUTask.HIGH_SCHOOL_WORLD_HISTORY: {
        "category": TaskCategory.HUMANITIES,
        "description": (
            "High school world history covering global historical events "
            "and civilizations"
        ),
    },
    MMLUTask.HIGH_SCHOOL_US_HISTORY: {
        "category": TaskCategory.HUMANITIES,
        "description": (
            "High school US history covering American historical events and figures"
        ),
    },
    MMLUTask.PREHISTORY: {
        "category": TaskCategory.HUMANITIES,
        "description": "Human prehistory before written records",
    },
    MMLUTask.HIGH_SCHOOL_GEOGRAPHY: {
        "category": TaskCategory.HUMANITIES,
        "description": "High school geography covering physical and human geography",
    },
    MMLUTask.WORLD_RELIGIONS: {
        "category": TaskCategory.HUMANITIES,
        "description": "Major world religions, beliefs, practices, and history",
    },
    MMLUTask.PHILOSOPHY: {
        "category": TaskCategory.HUMANITIES,
        "description": "Philosophical theories, concepts, and historical figures",
    },
    MMLUTask.MORAL_SCENARIOS: {
        "category": TaskCategory.HUMANITIES,
        "description": "Ethical decision-making in hypothetical scenarios",
    },
    MMLUTask.MORAL_DISPUTES: {
        "category": TaskCategory.HUMANITIES,
        "description": "Arguments on both sides of contentious moral issues",
    },
    MMLUTask.FORMAL_LOGIC: {
        "category": TaskCategory.HUMANITIES,
        "description": "Formal logic principles, rules, and proofs",
    },
    MMLUTask.LOGICAL_FALLACIES: {
        "category": TaskCategory.HUMANITIES,
        "description": "Common logical fallacies and errors in reasoning",
    },
    # Medical
    MMLUTask.CLINICAL_KNOWLEDGE: {
        "category": TaskCategory.MEDICAL,
        "description": "Clinical medical knowledge and patient care",
    },
    MMLUTask.MEDICAL_GENETICS: {
        "category": TaskCategory.MEDICAL,
        "description": "Medical applications of genetics and genomics",
    },
    MMLUTask.PROFESSIONAL_MEDICINE: {
        "category": TaskCategory.MEDICAL,
        "description": "Professional medical practice, diagnosis, and treatment",
    },
    MMLUTask.COLLEGE_MEDICINE: {
        "category": TaskCategory.MEDICAL,
        "description": "College-level medicine and healthcare concepts",
    },
    MMLUTask.ANATOMY: {
        "category": TaskCategory.MEDICAL,
        "description": "Human anatomical structures and systems",
    },
    MMLUTask.VIROLOGY: {
        "category": TaskCategory.MEDICAL,
        "description": "Viruses, viral diseases, and treatments",
    },
    MMLUTask.NUTRITION: {
        "category": TaskCategory.MEDICAL,
        "description": "Nutritional science, diet, and health",
    },
    MMLUTask.HUMAN_AGING: {
        "category": TaskCategory.MEDICAL,
        "description": "Biological and psychological aspects of human aging",
    },
    # Business & Law
    MMLUTask.BUSINESS_ETHICS: {
        "category": TaskCategory.BUSINESS,
        "description": "Ethical principles and dilemmas in business contexts",
    },
    MMLUTask.PROFESSIONAL_ACCOUNTING: {
        "category": TaskCategory.BUSINESS,
        "description": "Professional accounting principles and practices",
    },
    MMLUTask.PROFESSIONAL_LAW: {
        "category": TaskCategory.BUSINESS,
        "description": "Professional legal concepts, cases, and principles",
    },
    MMLUTask.INTERNATIONAL_LAW: {
        "category": TaskCategory.BUSINESS,
        "description": "International legal frameworks, treaties, and cases",
    },
    MMLUTask.JURISPRUDENCE: {
        "category": TaskCategory.BUSINESS,
        "description": "Philosophy and theory of law",
    },
    MMLUTask.MARKETING: {
        "category": TaskCategory.BUSINESS,
        "description": "Marketing principles, strategies, and analysis",
    },
    MMLUTask.MANAGEMENT: {
        "category": TaskCategory.BUSINESS,
        "description": "Management principles, leadership, and organizational behavior",
    },
    MMLUTask.PUBLIC_RELATIONS: {
        "category": TaskCategory.BUSINESS,
        "description": "Public relations practices and strategies",
    },
    # Other
    MMLUTask.GLOBAL_FACTS: {
        "category": TaskCategory.OTHER,
        "description": "Facts about countries, populations, and global phenomena",
    },
    MMLUTask.MISCELLANEOUS: {
        "category": TaskCategory.OTHER,
        "description": "Various topics that don't fit into other categories",
    },
}


def get_all_tasks() -> list[MMLUTask]:
    """Get all supported MMLU tasks.

    Returns:
        List of all MMLU tasks supported by Factly
    """
    return list(TASK_METADATA.keys())


def get_tasks_by_category(category: TaskCategory) -> list[MMLUTask]:
    """Get all tasks belonging to a specific category.

    Args:
        category: The category to filter by

    Returns:
        List of MMLU tasks in the specified category
    """
    return [
        task
        for task, metadata in TASK_METADATA.items()
        if metadata["category"] == category
    ]


def get_task_by_name(name: str) -> Optional[MMLUTask]:
    """Get an MMLU task by its name (case-insensitive).

    Args:
        name: The name of the task, can be partial match

    Returns:
        The matching MMLU task or None if not found
    """
    name_lower = name.lower().replace("-", "_")

    # First try exact match
    for task in TASK_METADATA:
        if task.name.lower() == name_lower:
            return task

    # Then try partial match
    for task in TASK_METADATA:
        if name_lower in task.name.lower():
            return task

    return None


def resolve_tasks(task_names: list[str]) -> list[MMLUTask]:
    """Resolve a list of task names to actual MMLU tasks.

    Args:
        task_names: List of task names provided by the user

    Returns:
        List of resolved MMLU tasks

    Raises:
        ValueError: If any task name cannot be resolved
    """
    if not task_names:
        return get_all_tasks()

    resolved_tasks = []
    unresolved = []

    for name in task_names:
        # Check if it's a category name
        try:
            category = TaskCategory(name)
            category_tasks = get_tasks_by_category(category)
            resolved_tasks.extend(category_tasks)
            continue
        except ValueError:
            pass

        # Try to resolve as a task name
        task = get_task_by_name(name)
        if task:
            resolved_tasks.append(task)
        else:
            unresolved.append(name)

    if unresolved:
        raise ValueError(f"Could not resolve task names: {', '.join(unresolved)}")

    return list(set(resolved_tasks))  # Deduplicate


def list_available_tasks() -> str:
    """Generate a formatted string listing all available tasks.

    Returns:
        Formatted string with all available tasks grouped by category
    """
    result = []
    result.append("Available MMLU Tasks:")

    for category in TaskCategory:
        tasks = get_tasks_by_category(category)
        if not tasks:
            continue

        result.append(f"\n{category.value}:")
        for task in tasks:
            metadata = TASK_METADATA[task]
            result.append(f"  - {task.name}")
            result.append(f"    {metadata['description']}")

    result.append("\n\nUsage examples:")
    result.append("  factly evaluate --tasks STEM  # Run all STEM tasks")
    result.append(
        "  factly evaluate --tasks mathematics --tasks physics  # Run specific tasks"
    )

    return "\n".join(result)
