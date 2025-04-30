.PHONY: test
test:
	uv run coverage erase
	uv run coverage run -m pytest ./tests
	@echo

.PHONY: ccov
ccov:
	uv run coverage combine
	uv run coverage report
	uv run coverage html
	uv run coverage xml
	@echo
