.PHONY: help install install-dev install-hooks format format-check lint typecheck check fix clean

help:
	@echo "Available targets:"
	@echo "  install        Install package (editable)"
	@echo "  install-dev    Install package with dev dependencies"
	@echo "  install-hooks  Install pre-commit git hooks"
	@echo "  format         Auto-format code with Ruff"
	@echo "  format-check   Verify formatting (no changes)"
	@echo "  lint           Run Ruff lint checks"
	@echo "  typecheck      Run mypy"
	@echo "  check          Run full validation suite (format-check + lint + typecheck)"
	@echo "  fix            Apply Ruff fixes, then rerun full checks"
	@echo "  clean          Remove caches and build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,logging,plotting]"

install-hooks:
	pre-commit install

format:
	ruff format scripts/ crack_segmentation/ examples/

format-check:
	ruff format --check scripts/ crack_segmentation/ examples/

lint:
	ruff check scripts/ crack_segmentation/ examples/

typecheck:
	mypy scripts/

check: format-check lint typecheck

fix:
	ruff check --fix scripts/ crack_segmentation/ examples/
	ruff format scripts/ crack_segmentation/ examples/
	$(MAKE) check

clean:
	rm -rf __pycache__ scripts/__pycache__ crack_segmentation/__pycache__ examples/__pycache__
	rm -rf .mypy_cache .ruff_cache build dist *.egg-info
