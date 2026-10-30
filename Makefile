.PHONY: data test lint smoke reproduce clean help

PYTHON ?= python3
PIP ?= pip3

help:
	@echo "Targets: data test lint smoke reproduce clean"

data:  ## Download and verify datasets (requires network)
	$(PYTHON) -m scripts.download_data

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

lint:  ## Run ruff and mypy
	$(PYTHON) -m ruff check src tests scripts benchmarks
	$(PYTHON) -m ruff format --check src tests scripts benchmarks
	$(PYTHON) -m mypy src

smoke:  ## Quick end-to-end pipeline check (uses --quick everywhere)
	$(PYTHON) -m diffekf data --quick
	$(PYTHON) -m diffekf split --quick
	$(PYTHON) -m diffekf tune --quick
	$(PYTHON) -m diffekf train --quick
	$(PYTHON) -m diffekf eval --quick --force
	$(PYTHON) -m diffekf ablate --quick
	$(PYTHON) -m diffekf plot --quick
	$(PYTHON) -m diffekf bench --quick
	$(PYTHON) -m diffekf reproduce --quick

reproduce:  ## Full end-to-end run on a machine that has the data
	$(PYTHON) -m diffekf data
	$(PYTHON) -m diffekf split
	$(PYTHON) -m diffekf tune
	$(PYTHON) -m diffekf train
	$(PYTHON) -m diffekf eval --force
	$(PYTHON) -m diffekf ablate
	$(PYTHON) -m diffekf plot
	$(PYTHON) -m diffekf bench
	$(PYTHON) -m scripts.fill_docs

clean:
	rm -rf results/splits.json results/results.json results/checkpoints results/logs results/figures results/smoke_do_not_cite tests/fixtures/__pycache__
