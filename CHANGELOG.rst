Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.

Versions follow `Semantic Versioning <https://semver.org/>`_ (``<major>.<minor>.<patch>``).

Backward incompatible (breaking) changes will only be introduced in major versions with advance notice in the **Deprecations** section of releases.

1.1.0 - 2025-05-??
------------------

Features
^^^^^^^^

* Added short ``-m`` flags for ``--model`` CLI option.
* Introduced new ``--url`` (``-u``) and ``--api-key`` (``-a``) options for specifying the model API URL and API key directly via the CLI or environment variables (``OPENAI_API_BASE``, ``OPENAI_API_KEY``).
* CLI now prioritizes values from command-line options, falling back to environment variables or ``.env`` file for ``model``, ``url``, and ``api-key``.
* Implemented ``FactlySettings`` class to manage model and inference settings, including support for CLI arguments, environment variables, and ``.env`` file.
* Provided ability to set ``temperature``, ``top_p``, and ``max_tokens`` for the model via CLI arguments (``--temperature``, ``--top-p``, ``--max-tokens``).

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Removed deprecated ``license`` key in favor of ``license-files`` key in ``pyproject.toml``, aligning with `PEP 639 <https://peps.python.org/pep-0639/#add-string-value-to-license-key>`_
* Improved CLI performance by reorganizing imports and inplemented lazy loading of dependencies.
* Reworked logging to use ``factly.logger`` module, with ``DEBUG`` and ``INFO`` log levels, and ``logging.NullHandler`` for silent operation when disabled.
* Moved ``FactlyGptModel`` class from ``factly.models`` to ``factly.llms`` module.

Improved Documentation
^^^^^^^^^^^^^^^^^^^^^^

* Updated usage documentation to match current CLI options and flags, including new short/long flags (``-m/--model``, ``-u/--url``, ``-a/--api-key``), environment variable defaults, and corrected option names and descriptions in ``docs/source/usage.rst``.

1.0.1 - 2025-04-30
------------------

Improved documentation
^^^^^^^^^^^^^^^^^^^^^^

* Updated installation instructions to use ``uv sync --locked`` instead of ``uv pip install -r pyproject.toml``
* Added more project URLs in ``pyproject.toml``
* Fixed command to build docs at ReadTheDocs

1.0.0 - 2025-04-30
------------------

* Initial release.
