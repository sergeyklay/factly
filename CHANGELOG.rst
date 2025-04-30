Changelog
=========

This file contains a brief summary of new features and dependency changes or
releases, in reverse chronological order.

Versions follow `Semantic Versioning <https://semver.org/>`_ (``<major>.<minor>.<patch>``).

Backward incompatible (breaking) changes will only be introduced in major versions with advance notice in the **Deprecations** section of releases.

1.0.2 - 2025-??-??
------------------

Trivial/Internal Changes
^^^^^^^^^^^^^^^^^^^^^^^^

* Removed deprecated ``license`` key in favor of ``license-files`` key in ``pyproject.toml``, aligning with `PEP 639 <https://peps.python.org/pep-0639/#add-string-value-to-license-key>`_

1.0.1 - 2025-04-30
------------------

Improved documentation
^^^^^^^^^^^^^^^^^^^^^^

* Updated installation instructions to use ``uv sync --locked`` instead of ``uv pip install -r pyproject.toml``
* Added more project URLs in ``pyproject.toml``
* Fix command to build docs at ReadTheDocs

1.0.0 - 2025-04-30
------------------

* Initial release.
