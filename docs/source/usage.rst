=====
Usage
=====

This document provides detailed instructions for using the ``factly`` command-line tool.

Command Line Interface
======================

The primary entrypoint for Factly is the ``factly`` command-line interface, which provides tools to evaluate the factuality of LLMs on the MMLU benchmark.

Basic Usage
-----------

.. code-block:: bash

   # Run factuality evaluation with default settings
   factly evaluate

   # Run evaluation and generate plots
   factly evaluate --plot

   # Get help on all available options
   factly evaluate --help

   # List available MMLU tasks
   factly list-tasks

Command Structure
-----------------

Factly provides the following commands:

.. code-block:: text

   factly [OPTIONS] COMMAND [ARGS]...

Main Commands:

* ``evaluate``: Run factuality evaluation on MMLU benchmark
* ``list-tasks``: List all available MMLU tasks

Common Global Options:

* ``--help``: Show help message and exit
* ``--version``: Show version and exit

Command Line Options for ``evaluate``
-------------------------------------


--instructions file    Path to YAML file with system instruction variants.
                       Default: ``./instructions.yaml``
-m name, --model name  Model name to use for evaluation. Default: ``OPENAI_MODEL``
                       from environment variables, ``.env`` file, or ``gpt-4o`` as a
                       fallback.
                       and long descriptions
-u host, --url host    Model API URL to use for evaluation. Default: ``OPENAI_API_BASE``
                       from environment variables, ``.env`` file, or unset.
-a key, --api-key key  Model API key to use for evaluation. Default: ``OPENAI_API_KEY``
                       from environment variables, ``.env`` file, or unset.
-t float, --temperature float  Controls randomness in token selection. For MMLU benchmarking,
                       ``0.0`` (deterministic) is the canonical setting to ensure
                       reproducible results and measure raw model knowledge.
                       Higher values introduce randomness, which isn't suitable for
                       standard benchmarking. Default: ``0.0``.
--top-p float, --top-p float  Nucleus sampling parameter that controls how much of the
                       probability mass the model samples from. For benchmarking,
                       keep at ``1.0`` to disable nucleus sampling. Only modify
                       when exploring controlled randomness in outputs.
                       Default: ``1.0``.
--max-tokens int, --max-tokens int  Maximum tokens per response. For standard MMLU with
                       single-letter answers (A/B/C/D), use ``1``. For structured
                       outputs or when using system prompts that encourage reasoning,
                       use higher values (``256`` or more) and post-process to extract
                       final answers. Default: ``256``.
--n-shots int, --n-shots int  Number of examples for few-shot learning. Default set to 0
                       for zero-shot evaluation. Increasing this value provides
                       more demonstration examples in prompts to help the model
                       understand the task format. Default: ``0``.
--tasks name, --tasks name  MMLU task categories to evaluate (can be repeated).
                       Default: All tasks.
-j int, --workers int  Maximum number of concurrent question evaluations. Default:
                       Auto-detected based on system resources.
--plot                 Generate visualization plots. Default: ``False``.
--plot-path path       Path to save the plot.
                       Default: ``./outputs/factuality-<model>-t<count>.png``.
--verbose              Show detailed progress information during evaluation.
                       Default: ``False``.
--help                 Show help message and exit.

.. note::

   * For canonical MMLU evaluation (comparable to published benchmarks), use ``--n-shots 0`` with ``--max-tokens 1``
   * When using few-shot learning (``--n-shots >0``), consider setting ``--max-tokens`` to a higher value (≥256) to allow the model to follow reasoning patterns from the examples
   * If you must use ``--n-shots >0`` with ``--max-tokens 1``, ensure your few-shot examples only demonstrate single-token answers without reasoning
   * For structured output (like JSON) or when using system prompts that encourage reasoning, always use ``--max-tokens >1`` regardless of the ``--n-shots`` value


Command Line Options for ``list-tasks``
---------------------------------------

--help                 Show help message and exit.

Advanced Usage
==============

Task Selection
--------------

You can select specific MMLU tasks to evaluate:

.. code-block:: bash

   # Evaluate specific model on selected MMLU tasks
   factly evaluate --model gpt-4o --tasks mathematics --tasks high_school_us_history

   # Evaluate on STEM tasks only
   factly evaluate --tasks STEM

   # Evaluate on business-related tasks
   factly evaluate --tasks BUSINESS

Few-Shot Learning
-----------------

Configure the number of examples provided for few-shot learning:

.. code-block:: bash

   # Zero-shot evaluation (default)
   factly evaluate --n-shots 0

   # 3-shot evaluation
   factly evaluate --n-shots 3

   # 5-shot evaluation
   factly evaluate --n-shots 5

Performance Optimization
------------------------

Factly uses asynchronous concurrent processing to maximize evaluation throughput.
It evaluates multiple questions concurrently for each model, significantly reducing
total evaluation time. You can control the concurrency level with the ``--workers``
parameter:

.. code-block:: bash

   # Auto-determine optimal concurrency (default)
   factly evaluate --tasks STEM

   # Set concurrency level explicitly (process 20 questions in parallel)
   factly evaluate --tasks STEM --workers 20

The implementation uses ``asyncio`` and semaphores for controlled concurrency with automatic
resource detection for optimal performance across different environments.

System Instructions
-------------------

Factly supports different system instructions for prompt engineering experiments:

.. code-block:: bash

   # Use the default instruction from instructions.yaml in current directory
   factly evaluate

   # Use a custom instructions defined in ~/path/to/instructions.yaml file
   factly evaluate --instructions ~/path/to/instructions.yaml

By default instructions should be defined in the ``instructions.yaml`` file in current directory.
Each instruction should provide a different way to guide the model's behavior when responding to questions.

Examples
========

Basic Evaluation
----------------

.. code-block:: bash

   # Run basic evaluation with default settings
   factly evaluate

   # Run evaluation and generate plots
   factly evaluate --plot

   # Run verbose evaluation with plots
   factly evaluate --verbose --plot

Subject-Specific Evaluation
---------------------------

.. code-block:: bash

   # Evaluate mathematics knowledge
   factly evaluate --tasks mathematics --n-shots 3 --plot

   # Evaluate humanities subjects
   factly evaluate --tasks high_school_european_history --tasks high_school_us_history --plot

   # Evaluate computer science knowledge
   factly evaluate --tasks computer_science --verbose --plot

Customized Evaluation
---------------------

.. code-block:: bash

   # Customize API settings and system instruction
   factly evaluate \
     -m gpt-4o-mini \
     -u https://your-proxy.example.com/v1 \
     -a your_api_key_here \
     --instructions ~/path/to/instructions.yaml

   # Customize model inference parameters
   factly evaluate \
     --model gpt-4o \
     --temperature 0.7 \
     --top-p 0.95 \
     --max-tokens 512 \
     --tasks mathematics \
     --plot

Environment Variables
=====================

Instead of specifying command-line arguments each time, you can set environment variables in the ``.env`` file:

.. code-block:: bash

   # API Configuration
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o
   OPENAI_API_BASE=your_api_base_url  # Optional
