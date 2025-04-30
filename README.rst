======
Factly
======

|ci| |codecov| |docs|

.. -teaser-begin-

Factly is a modern CLI tool designed to evaluate the factuality of Large Language Models (LLMs) on the MMLU (Massive Multitask Language Understanding) benchmark. It provides a robust framework for prompt engineering experiments and factual accuracy assessment.

.. -teaser-end-

.. -overview-begin-

Features
--------

- Evaluate LLM factuality on the MMLU benchmark with detailed results
- Support for various prompt engineering experiments via configurable system instructions
- Generate comparative visualizations of factuality scores across models and prompts
- Structured output for easy analysis and comparison
- Built with modern Python tooling (Python 3.12, uv, click, pydantic)
- Extensible and reproducible evaluation workflows

.. note::

   Currently, only OpenAI models are supported.

Prerequisites
-------------

- Python 3.12+
- `uv <https://github.com/astral-sh/uv>`_ for dependency management

Installation
------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/factly.git
   cd factly

   # Install dependencies
   uv pip install -r pyproject.toml

   # Set up environment configuration
   cp .env.example .env
   # Edit .env with your API keys and configuration

Configuration
-------------

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Configure your API keys and settings in the ``.env`` file:

.. code-block:: bash

   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o

   # Optional, defaults to OpenAI's API base.
   # Set if you are using a proxy like LiteLLM.
   OPENAI_API_BASE=your_api_base_url

System Instructions
^^^^^^^^^^^^^^^^^^^

Prompts and instruction variants for evaluation are defined in ``instructions.yaml``. You can modify existing instructions or add new ones following the established structure.

Usage
-----

The primary entrypoint is the ``factly`` command-line interface.

Basic Evaluation
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run factuality evaluation with default settings
   factly evaluate

   # Run evaluation and generate plots
   factly evaluate --plot

   # Get help on all available options
   factly evaluate --help

Advanced Options
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Evaluate specific model on selected MMLU tasks
   factly evaluate --model gpt-4o --tasks mathematics --tasks high_school_us_history --plot

   # Specify number of shots for few-shot learning
   factly evaluate --n-shots 3 --verbose

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

Factly uses asynchronous concurrent processing to maximize evaluation throughput.
It evaluates multiple questions concurrently for each model, significantly reducing
total evaluation time. You can control the concurrency level with the ``--workers``
parameter, which defaults to an automatically determined optimal value.

Usage Examples:

.. code-block:: bash

   # Basic usage (auto-determines optimal concurrency)
   factly evaluate --tasks STEM --tasks BUSINESS

   # Set concurrency level explicitly (process 20 questions in parallel)
   factly evaluate --tasks STEM --workers 20

   # Compare performance with different concurrency levels
   factly evaluate --tasks STEM --workers 5
   factly evaluate --tasks STEM --workers 30

The implementation uses ``asyncio`` and semaphores for controlled concurrency with automatic
resource detection for optimal performance across different environments.

Project Structure
-----------------

- ``factly/`` - Main package directory containing core functionality
- ``instructions.yaml`` - System prompts/instructions for LLM evaluation
- ``outputs/`` - Generated plots and evaluation results
- ``.env`` - Local configuration (API keys, settings)

Development
-----------

Setting up the development environment:

.. code-block:: bash

   # Install development dependencies
   uv pip install -e ".[dev]"

   # Run linter
   ruff check .
   ruff format .

   # Run tests
   pytest

Contributing
-----------

Contributions are welcome! Please read our `Contributing Guide <CONTRIBUTING.rst>`_ for details on our code of conduct and the process for submitting pull requests.

License
-------

This project is licensed under the MIT License - see the `LICENSE <LICENSE>`_ file for details.

Acknowledgements
----------------

- `MMLU benchmark <https://github.com/hendrycks/test>`_ by Dan Hendrycks
- `DeepEval <https://github.com/confident-ai/deepeval>`_ for evaluation framework
- OpenAI, Anthropic, and other LLM providers

.. -acknowledgements-end-

.. |ci| image:: https://github.com/sergeyklay/factly/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/sergeyklay/factly/actions/workflows/ci.yml
   :alt: CI

.. |codecov| image:: https://codecov.io/gh/sergeyklay/factly/branch/main/graph/badge.svg?token=K2guigF0CX
   :target: https://codecov.io/gh/sergeyklay/factly
   :alt: Coverage

.. |docs| image:: https://readthedocs.org/projects/factly/badge/?version=latest
   :target: https://factly.readthedocs.io/en/latest/?badge=latest
   :alt: Docs
