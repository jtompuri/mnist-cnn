PYTHON := .venv/bin/python

.PHONY: setup train eval viz features test lint format typecheck check

setup:
	.venv/bin/pip install -e ".[dev]"

train:
	$(PYTHON) train.py

eval:
	$(PYTHON) evaluate.py

viz:
	$(PYTHON) visualize_predictions.py

features:
	$(PYTHON) visualize_features.py

test:
	$(PYTHON) -m pytest tests/

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff format .

typecheck:
	pyright .

check: lint typecheck test
