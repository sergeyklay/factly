======
Factly
======

|ci| |codecov| |docs|

.. -teaser-begin-

Factly is a modern CLI tool designed to evaluate the factuality of Large Language Models (LLMs) on the Massive Multitask Language Understanding (MMLU) benchmark. It provides a robust framework for prompt engineering experiments and factual accuracy assessment.

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

Quick Start
-----------

.. code-block:: bash

   # Run factuality evaluation with default settings
   factly evaluate

   # Run evaluation and generate plots
   factly evaluate --plot

   # Get help on all available options
   factly evaluate --help


That's it! The tool uses optimized default parameters and saves all outputs to the ``output`` directory.

.. note::

   For detailed installation instructions, please see the `Installation Guide <https://factly-eval.readthedocs.io/en/latest/installation.html>`_. And for usage instructions, use cases, examples, and advanced configuration options, please see the `Usage Guide <https://factly-eval.readthedocs.io/en/latest/usage.html>`_.

.. -overview-end-

.. -project-information-begin-

Project Information
===================

Factly is released under the `MIT License <https://choosealicense.com/licenses/mit/>`_, its documentation lives at `Read the Docs <https://factly-eval.readthedocs.io/>`_, the code on `GitHub <https://github.com/sergeyklay/factly>`_, and the latest release on `PyPI <https://pypi.org/project/factly-eval/>`_. It's rigorously tested on Python 3.12+.

If you'd like to contribute to Factly you're most welcome!

.. -project-information-end-

.. -support-begin-

Support
=======

Should you have any question, any remark, or if you find a bug, or if there is something you can't do with the Factly, please `open an issue <https://github.com/sergeyklay/factly/issues>`_.

.. -support-end-

.. |ci| image:: https://github.com/sergeyklay/factly/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/sergeyklay/factly/actions/workflows/ci.yml
   :alt: CI

.. |codecov| image:: https://codecov.io/gh/sergeyklay/factly/branch/main/graph/badge.svg?token=K2guigF0CX
   :target: https://codecov.io/gh/sergeyklay/factly
   :alt: Coverage

.. |docs| image:: https://readthedocs.org/projects/factly/badge/?version=latest
   :target: https://factly.readthedocs.io/en/latest/?badge=latest
   :alt: Docs
