# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path("..", "..", "factly").resolve()))

from factly import __author__, __copyright__, __version__

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Factly"
author = __author__
copyright = __copyright__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for nitpick -----------------------------------------------------

# In nitpick mode (-n), still ignore any of the following "broken" references
# to non-types.
nitpick_ignore = [
    # External library classes
    ("py:class", "click.core.Command"),
    ("py:class", "deepeval.benchmarks.mmlu.task.MMLUTask"),
    ("py:class", "pydantic.main.BaseModel"),
    ("py:class", "pydantic_settings.BaseSettings"),
    ("py:class", "openai.OpenAI"),
    ("py:class", "openai.AsyncOpenAI"),
    ("py:class", "matplotlib.figure.Figure"),
    ("py:func", "factly.cli.main"),
    # Pydantic and pydantic_settings types
    ("py:class", "SettingsConfigDict"),
    ("py:class", "DotenvType"),
    ("py:class", "CliSettingsSource"),
    ("py:class", "PathType"),
    # Annotation types
    ("py:class", "annotated_types.Ge"),
    ("py:class", "annotated_types.Le"),
    ("py:class", "annotated_types.Gt"),
]

# Ignore all references in the settings module
nitpick_ignore_regex = [
    # Ignore numeric literals in factly.settings that are misidentified as classes
    (r"py:class", r"[0-9]+\.[0-9]+"),
    # Ignore all constraint-related references in the settings module
    (r".*", r"factly\.settings\..*"),
]

# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# Output file base name for HTML help builder.
htmlhelp_basename = "factly-doc"
