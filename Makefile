# HyperFlow-H3 developer Makefile.
#
#   make install      create .venv, install the package with the dev tools and the example dependencies
#   make test         unit tests (CPU, no checkpoint needed)
#   make lint/format  ruff

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
# hf/ (the Hub staging area) is export-ignored from the public snapshot; lint it only where it exists.
LINT_PATHS := src tests examples $(wildcard hf)

.PHONY: help venv install test lint format clean

help: ## list targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

venv: $(VENV)/bin/python ## create the virtualenv

install: venv ## editable install + dev tools + example dependencies
	$(PIP) install -e ".[dev,examples]"
	$(PY) -c "import diffusers.modular_pipelines.minimax_h3, hyperflow_h3; print('ok', hyperflow_h3.__version__)"

test: ## unit tests on CPU
	$(PY) -m pytest -q

lint: ## ruff check + format check
	$(VENV)/bin/ruff check $(LINT_PATHS)
	$(VENV)/bin/ruff format --check $(LINT_PATHS)

format: ## ruff format + autofix
	$(VENV)/bin/ruff check --fix $(LINT_PATHS)
	$(VENV)/bin/ruff format $(LINT_PATHS)

clean: ## remove caches and build artefacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
