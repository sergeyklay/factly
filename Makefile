.PHONY: test
test:
	uv run coverage erase
	uv run coverage run -m pytest ./tests
	@echo
