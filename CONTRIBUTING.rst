======================
Contributing to Factly
======================

Thank you for considering contributing to Factly! This document outlines the process for contributing to the project and the coding standards we follow.

Code of Conduct
---------------

Please be respectful and considerate when participating in this project. We strive to maintain a welcoming and inclusive environment for everyone.

Getting Started
---------------

1. Fork the repository
2. Clone your fork: ``git clone https://github.com/your-username/factly.git``
3. Set up the development environment:

   .. code-block:: bash

      cd factly
      uv pip install -e ".[dev]"

4. Create a branch for your changes: ``git checkout -b feature/your-feature-name``

Development Environment
-----------------------

Factly requires Python 3.12+ and uses `uv <https://github.com/astral-sh/uv>`_ for dependency management.

Useful Commands
^^^^^^^^^^^^^^^

- Install dependencies: ``uv pip install -r pyproject.toml``
- Add a new dependency: ``uv add <dependency>``
- Run a Python script: ``uv run script.py``
- List installed packages: ``uv pip list``

Coding Standards
----------------

Python Version
^^^^^^^^^^^^^^

All code must be compatible with **Python 3.12+**.

Code Style
^^^^^^^^^^

We use `ruff <https://docs.astral.sh/ruff>`_ for linting and formatting. Before submitting a pull request, please ensure your code passes the linter:

.. code-block:: bash

   ruff check .
   ruff format .

The linter configuration is in ``pyproject.toml``.

Naming Conventions
^^^^^^^^^^^^^^^^^^

Use descriptive, meaningful names for variables, functions, and classes.

.. code-block:: python

   # ✅ DO:
   def calculate_factuality_score(responses: list[str]) -> float:
       ...

   # ❌ DON'T:
   def cfs(r):
       ...

Docstrings
^^^^^^^^^^

Document all functions, classes, and modules using Google-style docstrings:

.. code-block:: python

   def run_evaluation(config: dict) -> None:
       """Run the factuality evaluation pipeline.

       Args:
           config (dict): Configuration for the evaluation run.
       """
       ...

Type Hints
^^^^^^^^^^

Use type hints for function parameters and return values:

.. code-block:: python

   def process_results(
       scores: dict[str, float],
       threshold: float = 0.7
   ) -> list[str]:
       """Process evaluation results.

       Args:
           scores: Dictionary mapping prompt names to scores.
           threshold: Minimum score to be considered successful.

       Returns:
           List of prompt names that exceeded the threshold.
       """
       ...

Error Handling
^^^^^^^^^^^^^^

Use try-except blocks to handle exceptions gracefully and log errors:

.. code-block:: python

   try:
       result = call_openai_api(prompt)
   except Exception as exc:
       logger.error("API call failed: %s", exc)

Logging
^^^^^^^

Use Python's built-in logging module instead of print statements:

.. code-block:: python

   import logging
   logger = logging.getLogger(__name__)

   # ✅ DO:
   logger.info("Processing task %s", task_name)

   # ❌ DON'T:
   print(f"Processing task {task_name}")

Code Structure
^^^^^^^^^^^^^^

- Limit line length to 88 characters
- Use 4 spaces for indentation
- Follow the DRY principle (Don't Repeat Yourself)
- Prefer list comprehensions over loops when appropriate
- Avoid global variables
- Keep functions and methods small and focused

Testing
-------

Write unit tests using pytest for all new functionality. Place tests in the ``tests/`` directory with a structure mirroring the main package:

::

   factly/
     └── module.py
   tests/
     └── test_module.py

Run tests with:

.. code-block:: bash

   pytest

Pull Request Process
--------------------

1. Ensure your code passes all linting and tests
2. Update documentation if necessary
3. Include a clear description of the changes in your pull request
4. Reference any related issues in your pull request description

Project Structure
-----------------

- Core functionality goes in the ``factly/`` package
- CLI entrypoint is ``factly/cli.py``
- Configuration and instructions should be loaded from YAML or JSON files
- Output should be both machine-readable (CSV/JSON) and human-friendly (charts, tables, text)

Questions?
----------

If you have questions about contributing, please open an issue or reach out to the maintainers.

References
----------

- `Python PEP 8 Style Guide <https://peps.python.org/pep-0008/>`_
- `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_
- `Project Structure Documentation <https://github.com/yourusername/factly/blob/main/.cursor/rules/project-structure.mdc>`_
