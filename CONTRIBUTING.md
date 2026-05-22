# Contributing to FisherAdapTune

We welcome bug reports, documentation improvements, and feature suggestions.

## Reporting Issues

Open a [GitHub Issue](https://github.com/Mahdi-S-Hosseini/FisherAdapTune/issues) and include:

- A clear description of the problem
- A minimal reproducible example (ideally using `examples/minimal_image_classifier.py`)
- Your environment: Python version, PyTorch version, GPU, OS

## Development Setup

```bash
git clone https://github.com/Mahdi-S-Hosseini/FisherAdapTune.git
cd FisherAdapTune
pip install -e ".[dev]"
pre-commit install
```

## Code Style

- Formatting and linting: [Ruff](https://docs.astral.sh/ruff/) (line length 100)
- Type checking: mypy on annotated modules
- Pre-commit hooks run automatically on `git commit`

To run checks manually:

```bash
make check
```

## Pull Requests

1. Fork the repo and create a feature branch from `main`.
2. Run `make check` before pushing.
3. Open a PR with a clear description of what changed and why.

All contributions must be for non-commercial purposes in line with the [CC BY-NC-SA 4.0 license](LICENSE).
