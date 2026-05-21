.PHONY: install test lint format clean help

help:
	@echo "Available targets:"
	@echo "  install  - install deps with uv (includes dev extras)"
	@echo "  test     - run pytest"
	@echo "  lint     - ruff check"
	@echo "  format   - ruff format + autofix"
	@echo "  clean    - remove caches"

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
