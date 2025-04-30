==================
Maintainers' Guide
==================

This document outlines essential guidelines for maintaining the Factly project.
It provides instructions for testing, building, and deploying the package, as well
as managing CI workflows.

Overview
========

The Factly project is a CLI tool for evaluating the factuality of LLMs on the MMLU benchmark.
This guide assumes familiarity with GitHub Actions, uv, and common Python development workflows.

Key configurations:

* **Python Versions Supported:** >= 3.12 (tested on 3.12 and 3.13)
* **Dependency Management:** ``uv`` version 6.x
* **Primary Dependencies:** ``click``, ``datasets``, ``deepeval``, ``pandas``, ``matplotlib``
* **Documentation Tool:** ``sphinx`` with Read the Docs theme
* **Testing Tools:** ``pytest``, ``coverage``
* **Linting Tools:** ``ruff`` (for linting and formatting)

Development Environment
=======================

Prerequisites
-------------

To work on the Factly project, you need:

* Python 3.12 or higher
* uv (for dependency management)
* Git

Setting Up
----------

Clone the repository and install dependencies:

.. code-block:: bash

   git clone https://github.com/sergeyklay/factly.git
   cd factly

   # Copy and configure environment
   cp .env.example .env
   # Edit .env with your API keys and configuration

   # Install dependencies with development tools
   uv sync --locked

Testing the Project
===================

Unit tests and coverage reporting are managed using ``pytest`` and ``coverage``.

Running Tests Locally
---------------------

Run tests using pytest:

.. code-block:: bash

   # Run all tests
   uv run coverage erase
   uv run coverage run -m pytest

Generate Coverage Reports
-------------------------

Generate HTML, XML, and LCOV coverage reports:

.. code-block:: bash

   uv run coverage combine
   uv run coverage report
   uv run coverage html
   uv run coverage xml

This will create reports in the ``coverage/`` directory with subdirectories for each format, as configured in ``pyproject.toml``.

CI Workflow
-----------

Tests are executed automatically on supported platforms and Python versions (3.12 and 3.13) via GitHub Actions.

The CI workflow includes:

* Code formatting verification
* Linting checks
* Unit tests with coverage reporting
* Coverage report upload to Codecov

Building the Package
====================

The ``factly`` package is built using ``hatchling`` as specified in ``pyproject.toml``.

Local Build
-----------

Build the package:

.. code-block:: bash

   # Build the package
   uv build

Verify the built package:

.. code-block:: bash

   uv pip install dist/*.whl
   factly --version

Documentation Management
========================

Documentation is written using ``sphinx`` with the Read the Docs theme.

Building Documentation Locally
------------------------------

Install documentation dependencies:

.. code-block:: bash

   uv sync --locked --no-default-groups --group docs

Build the documentation:

.. code-block:: bash

   # Navigate to docs directory
   cd docs

   # Build HTML documentation
   make html

Or build directly with sphinx:

.. code-block:: bash

   # Build HTML documentation
   python -m sphinx \
      --jobs auto \
      --builder html \
      --nitpicky \
      --show-traceback \
      --fail-on-warning \
      --doctree-dir docs/build/doctrees \
      docs/source docs/build/html

View the documentation:

.. code-block:: bash

   # On macOS
   open docs/build/html/index.html

   # On Linux
   xdg-open docs/build/html/index.html

   # On Windows
   start docs/build/html/index.html

Other Documentation Formats
---------------------------

The docs ``Makefile`` supports various output formats:

.. code-block:: bash

   cd docs
   make epub      # Build EPUB documentation
   make man       # Build man pages
   make clean     # Clean build directory

Without ``make``, use these sphinx-build commands:

.. code-block:: bash

   cd docs

   # Build EPUB documentation
   sphinx-build -b epub source build/epub

   # Build man pages
   sphinx-build -b man source build/man

   # Clean build directory
   rm -rf build/

CI Workflow
-----------

The docs workflow automatically builds and validates documentation on pushes and pull requests. See ``.github/workflows/docs.yml``.

Linting and Code Quality Checks
===============================

Code quality is enforced using ``ruff``, which handles both linting and formatting.

Running Locally
---------------

Lint and format code:

.. code-block:: bash

   # Lint and format code
   uv run ruff check --select I --fix ./
   uv run ruff format --target-version py312 ./

   # Check formatting without making changes
   uv run ruff format --diff --target-version py312 ./

   # Run linter without making changes
   uv run ruff check --target-version py312 --preview ./

Pre-commit Hooks
----------------

The project uses pre-commit hooks to ensure code quality before commits:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install

   # Run pre-commit hooks on all files
   pre-commit run --all-files

CI Workflow
-----------

The CI workflow in ``.github/workflows/ci.yml`` includes formatting and linting checks. Pull requests with formatting issues will show the diff of improperly formatted files.

Release Process
===============

Steps for Release
-----------------

1. Ensure all tests pass and documentation builds successfully
2. Update version in ``pyproject.toml`` and ``__init__.py``
3. Update ``CHANGELOG.rst`` with the changes in the new versio
4. Tag the version using git and push tag to GitHub:

   .. code-block:: bash

      git tag -a v1.x.y -m "Release v1.x.y"
      git push origin v1.x.y

5. Build and publish the package:

   .. code-block:: bash

      uv build
      uv publish

CI Workflow
-----------

The release workflow is triggered when a new tag matching the pattern ``v*`` is pushed to GitHub. It builds the package and publishes it to PyPI.

Continuous Integration and Deployment
=====================================

CI/CD is managed via GitHub Actions, with workflows for:

* **Testing:** Ensures functionality and compatibility across Python 3.12, and 3.13 on Ubuntu
* **Linting:** Maintains code quality with ruff
* **Documentation:** Validates and builds project documentation
* **Building:** Verifies the package's integrity
* **Release:** Publishes the package to PyPI

The CI workflow includes:

* Caching of dependencies to speed up builds
* Automatic code formatting verification
* Coverage reporting to Codecov
* JUnit XML test results

Development Guidelines
======================

Code Style
----------

The project follows the style enforced by ruff. Key style points:

* Line length: 88 characters
* Target Python version: 3.12
* Use 4 spaces for indentation
* Follow PEP 8 with some customizations in pyproject.toml

Type Annotations
----------------

Use type annotations for all function parameters and return values:

.. code-block:: python

   def process_results(
       scores: dict[str, float],
       threshold: float = 0.7
   ) -> list[str]:
       """Process evaluation results."""
       # Implementation

Documentation Standards
-----------------------

* Use Google-style docstrings for all public functions, classes, and methods
* Include examples in docstrings where appropriate
* Keep the documentation up-to-date with code changes

Example docstring:

.. code-block:: python

   def calculate_factuality_score(responses: list[str]) -> float:
       """Calculate the factuality score based on responses.

       Args:
           responses: List of model responses to evaluate

       Returns:
           A float between 0 and 1 representing factuality score
       """
       # Implementation

Troubleshooting
===============

Common Development Issues
-------------------------

1. **uv environment issues:**

   .. code-block:: bash

      # Recreate the virtual environment
      rm -rf .venv
      uv venv
      uv sync

2. **Pre-commit hook failures:**

   .. code-block:: bash

      # Update pre-commit hooks
      uv run pre-commit autoupdate

      # Run hooks manually
      uv run pre-commit run --all-files

3. **Documentation build errors:**

   .. code-block:: bash

      # Clean build directory
      cd docs
      make clean

      # Rebuild with verbose output
      uv run sphinx-build -v --nitpicky --show-traceback --fail-on-warning --builder html docs/source docs/build/html

4. **Test failures:**

   .. code-block:: bash

      # Run tests with verbose output
      uv run pytest -vvv ./factly ./tests

      # Run a specific test
      uv run pytest -vvv ./tests/test_specific_file.py::test_specific_function

5. **Cleaning build artifacts without make:**

   .. code-block:: bash

      # Remove Python cache files
      find ./ -name '__pycache__' -delete -o -name '*.pyc' -delete

      # Remove pytest cache
      rm -rf ./.pytest_cache

      # Remove coverage reports
      rm -rf ./coverage
