.PHONY: lint
lint:
	uv run ruff check --target-version py312 --preview ./

.PHONY: format
format:
	uv run ruff check --select I --fix ./
	uv run ruff format --target-version py312 ./

.PHONY: format-check
format-check:
	uv run ruff format --diff --target-version py312 ./

.PHONY: test
test:
	uv run coverage erase
	uv run coverage run -m pytest ./tests

.PHONY: ccov
ccov:
	uv run coverage combine
	uv run coverage report
	uv run coverage html
	uv run coverage xml

.PHONY: docs
docs: CONTRIBUTING.rst README.rst
	@$(MAKE) -C docs clean
	uv run python -m doctest CONTRIBUTING.rst README.rst
	uv run sphinx-build --jobs auto --builder html --nitpicky --show-traceback --fail-on-warning --doctree-dir docs/build/doctrees docs/source docs/build/html
