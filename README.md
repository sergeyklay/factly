# Factly

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Factly** is a modern CLI tool designed to evaluate the factuality of Large Language Models (LLMs) like ChatGPT on the MMLU (Massive Multitask Language Understanding) benchmark. It provides a robust framework for prompt engineering experiments and factual accuracy assessment.

## Features

- Evaluate LLM factuality on the MMLU benchmark with detailed results
- Support for various prompt engineering experiments via configurable system instructions
- Generate comparative visualizations of factuality scores across models and prompts
- Structured output for easy analysis and comparison
- Built with modern Python tooling (Python 3.12, uv, click, pydantic)
- Support for major LLM providers (OpenAI, Anthropic, local models via Ollama)
- Extensible and reproducible evaluation workflows

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

```bash
# Clone the repository
git clone https://github.com/yourusername/factly.git
cd factly

# Install dependencies
uv pip install -r pyproject.toml

# Set up environment configuration
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration

1. **Environment Variables:** Configure your API keys and settings in the `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   OPENAI_MODEL=gpt-4o
   OPENAI_API_BASE=your_api_base_url # Optional, defaults to OpenAI's API base
   ```

2. **System Instructions:** Prompts and instruction variants for evaluation are defined in [`instructions.yaml`](instructions.yaml). You can modify existing instructions or add new ones following the established structure.

## Usage

The primary entrypoint is the `factly` command-line interface.

### Basic Evaluation

```bash
# Run factuality evaluation with default settings
factly evaluate

# Run evaluation and generate plots
factly evaluate --plot

# Get help on all available options
factly evaluate --help
```

### Advanced Options

```bash
# Evaluate specific model on selected MMLU tasks
factly evaluate --model gpt-4o --tasks mathematics --tasks high_school_us_history --plot

# Specify number of shots for few-shot learning
factly evaluate --n-shots 3 --verbose
```

## Project Structure

- `factly/` - Main package directory containing core functionality
- `instructions.yaml` - System prompts/instructions for LLM evaluation
- `outputs/` - Generated plots and evaluation results
- `.env` - Local configuration (API keys, settings)

## Development

Setting up the development environment:

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run linter
ruff check .
ruff format .

# Run tests
pytest
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MMLU benchmark](https://github.com/hendrycks/test) by Dan Hendrycks
- [DeepEval](https://github.com/confident-ai/deepeval) for evaluation framework
- OpenAI, Anthropic, and other LLM providers
