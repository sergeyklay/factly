============
Installation
============

Overview
========

The Factly project provides command-line utility ``factly`` that provides direct access to the package's functionality without writing Python code.

Requirements
============

Before installing ``factly``, ensure you have the following prerequisites:

* Python 3.12 or higher
* `pip <https://pip.pypa.io/en/stable/>`_ (for PyPI installation)

Python Version Compatibility
----------------------------

``factly`` requires Python 3.12 or higher. This requirement ensures access to the latest language features and optimizations. The project is tested with Python 3.12, and 3.13.

If you're using an older version of Python, you'll need to upgrade before installing ``factly``:

.. code-block:: bash

   # Check your current Python version
   python --version

.. warning::
   **Python Command Differences Across Operating Systems**

   The ``python`` and ``pip`` commands may behave differently across operating systems:

   * **macOS**: By default, ``python`` often points to Python 2.x. You should use ``python3`` and ``pip3`` commands instead.
   * **Linux**: Many distributions now default ``python`` to Python 3.x, but some still maintain ``python`` as Python 2.x. Use ``python3`` and ``pip3`` to ensure you're using the correct version.
   * **Windows**: Recent installations typically have ``python`` pointing to Python 3.x, but it's best to verify with ``python --version``.

   To ensure you're using the correct Python version, always check:

   .. code-block:: bash

      # Check Python version
      python --version  # or python3 --version

      # Check pip version
      pip --version  # or pip3 --version

   If ``python`` points to Python 2.x on your system, replace all ``python`` commands in this guide with ``python3`` and all ``pip`` commands with ``pip3``.

   **Finding Your Python Installations**

   To check where Python is installed and how many Python versions you have on your system:

   **On Unix-based systems (Linux, macOS):**

   .. code-block:: bash

      # Find the location of the python/python3 executable
      which python
      which python3

      # Alternative command to find executable location
      command -v python
      command -v python3

      # List all instances of python in your PATH
      type -a python
      type -a python3

      # Check if you have multiple Python installations
      ls -l /usr/bin/python*
      ls -l /usr/local/bin/python*

   **On Windows:**

   .. code-block:: bash

      # Find the location of Python executable
      where python
      where python3

      # Check Python version and installation path
      py -0

Installation Methods
====================

There are several ways to install ``factly`` depending on your needs:

Installing from PyPI (Recommended)
----------------------------------

``factly`` is a Python package `hosted on PyPI <https://pypi.org/project/factly-eval/>`_.
The recommended installation method is using `pip <https://pip.pypa.io/en/stable/>`_ to install into a virtual environment:

.. code-block:: bash

   # Create and activate a virtual environment (recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install factly-eval
   python -m pip install factly-eval

   # Alternative commands if python points to Python 2.x on your system
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   python3 -m pip install factly-eval
   # or
   pip3 install factly-eval

After installation, the ``factly`` command will be available from the command line:

.. code-block:: bash

   # Verify installation
   factly --version

More information about ``pip`` and PyPI can be found here:

* `Install pip <https://pip.pypa.io/en/latest/installation/>`_
* `Python Packaging User Guide <https://packaging.python.org/>`_

Installing from GitHub Releases
-------------------------------

Another way to install package is to download it from GitHub Releases page:

1. Visit the `GitHub Releases page <https://github.com/sergeyklay/factly/releases>`_
2. Download the desired release artifacts (both ``.whl`` and/or ``.tar.gz`` files)
3. Download the corresponding checksum files (``SHA256SUMS``, ``SHA512SUMS``, or ``MD5SUMS``)
4. Verify the integrity of the downloaded files:

   .. code-block:: bash

      # Verify with SHA256 (recommended)
      sha256sum -c SHA256SUMS

5. Install the verified package:

   .. code-block:: bash

      # Create a directory for the download
      mkdir factly-download && cd factly-download

      # Download the latest release artifacts and checksums (replace X.Y.Z with the actual version)
      # You can use wget or curl
      wget https://github.com/sergeyklay/factly/releases/download/X.Y.Z/factly-eval-X.Y.Z-py3-none-any.whl
      wget https://github.com/sergeyklay/factly/releases/download/X.Y.Z/factly-eval-X.Y.Z.tar.gz
      wget https://github.com/sergeyklay/factly/releases/download/X.Y.Z/SHA256SUMS

      # Verify the integrity of the downloaded files
      sha256sum -c SHA256SUMS

      # Create and activate a virtual environment
      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # Install the verified package (choose one)
      pip install factly-eval-X.Y.Z-py3-none-any.whl  # Wheel file (recommended)
      # OR
      pip install factly-eval-X.Y.Z.tar.gz  # Source distribution

      # If python points to Python 2.x on your system
      pip3 install factly-eval-X.Y.Z-py3-none-any.whl
      # Or
      pip3 install factly-eval-X.Y.Z.tar.gz

      # Verify the installation
      factly --version

Installing the Development Version
----------------------------------

If you need the latest unreleased features, you can install directly from the GitHub repository:

.. code-block:: bash

   # Install the latest development version
   python -m pip install -e git+https://github.com/sergeyklay/factly.git#egg=factly-eval

   # If python points to Python 2.x on your system
   python3 -m pip install -e git+https://github.com/sergeyklay/factly.git#egg=factly-eval

.. note::
   The ``main`` branch will always contain the latest unstable version, so the experience
   might not be as smooth. If you wish to use a stable version, consider installing from PyPI
   or switching to a specific `tag <https://github.com/sergeyklay/factly/tags>`_.

Installing for Development
--------------------------

If you plan to contribute to the project or need to modify the code, follow these steps:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sergeyklay/factly.git
      cd factly

2. Create and activate a virtual environment:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

      # If python points to Python 2.x on your system
      python3 -m venv .venv
      source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install with uv:

   .. code-block:: bash

      # Set up environment configuration
      cp .env.example .env
      # Edit .env with your API keys and configuration

      # Install uv if you haven't already
      # See https://github.com/astral-sh/uv

      # Install dependencies
      uv sync --locked

4. Verifying Installation:

   .. code-block:: bash

      factly --version

      # Or using the Python module:

      python -m factly --version
      # If python points to Python 2.x on your system
      python3 -m factly --version

You should see the version information and a brief copyright notice.

Dependencies
============

Core Dependencies
-----------------

These dependencies are installed by default and are required for the basic functionality:

* ``click``: Command-line interface framework
* ``datasets``: Dataset loading and manipulation (from Hugging Face)
* ``deepeval``: Framework for evaluating LLM factuality
* ``matplotlib``: Data visualization and plotting library
* ``pandas``: Data manipulation and analysis
* ``psutil``: Process and system utilities for resource monitoring
* ``python-dotenv``: Environment variable management
* ``pyyaml``: YAML file parsing and generation
* ``transformers``: Hugging Face's transformers library for NLP models

Optional Dependency Groups
--------------------------

Factly organizes dependencies into groups that can be installed separately:

Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

Tools for development and code quality:

* ``ruff``: Fast Python linter and code formatter

Testing Dependencies
^^^^^^^^^^^^^^^^^^^^

Tools for testing the codebase:

* ``coverage``: Code coverage tool and reporting
* ``pytest``: Testing framework
* ``pytest-mock``: Mocking support for pytest
* ``pytest-asyncio``: Async testing support for pytest

Documentation Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Tools for building documentation:

* ``sphinx``: Documentation generator
* ``sphinx-rtd-theme``: Read the Docs theme for Sphinx

Installing Specific Dependency Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can install specific dependency groups:

.. code-block:: bash

   # Install only development tools
   uv sync --locked --no-default-groups --group dev

   # Install only testing tools
   uv sync --locked --no-default-groups --group testing

   # Install only documentation tools
   uv sync --locked --no-default-groups --group docs

   # Install development and testing but not documentation
   uv sync --locked --no-default-groups --group dev --group testing

Adding Dependencies
-------------------

To add a new dependency:

.. code-block:: bash

   # Add a core dependency
   uv add <package-name>

   # Add a development dependency
   uv add --group dev <package-name>

   # Add a testing dependency
   uv add --group testing <package-name>

   # Add a documentation dependency
   uv add --group docs <package-name>

Troubleshooting
===============

Common Issues
-------------

If you encounter any issues during installation:

1. Ensure you have the correct Python version (3.12+)
2. Make sure you're using the latest version of uv
3. Check for any error messages during the installation process
