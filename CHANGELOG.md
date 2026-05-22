# Changelog

All notable changes to FisherAdapTune will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-05-22

### Added

- `FisherAdapTuneTrainer`: model-agnostic plug-and-play trainer with Fisher-guided iterative chunk freezing
- `AdaFisher`: diagonal Fisher Information Matrix optimizer used internally for Fisher statistic collection
- `fisher_core`: chunking, Jensen-Shannon divergence tracking, masking, and iterative freeze logic
- `EarlyStopping`, `save_checkpoint`, `plot_js_history` utilities
- SAM2 crack segmentation application (`crack_segmentation/`) with YAML config and full CLI
- Minimal runnable example using synthetic data (`examples/minimal_image_classifier.py`)
- Support for `Linear`, `Conv2d`, `BatchNorm2d`, and `LayerNorm` layers
- Decoupled weight decay applied only to active (unfrozen) parameter entries
- Optional W&B logging and matplotlib JS-distance grid plots
- `pyproject.toml` packaging, `Makefile` automation, pre-commit hooks
