.PHONY: install install-dev lint format typecheck test smoke clean help

# ── Variables ──────────────────────────────────────────────────────────────
PYTHON   ?= python3
PIP      ?= pip
BS       ?= 64
EPOCHS   ?= 1
SAMPLES  ?= 128

# ── Setup ──────────────────────────────────────────────────────────────────
install:
	$(PIP) install -U pip
	$(PIP) install -e .

install-dev:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	ruff check benchmark/

format:
	ruff format benchmark/

typecheck:
	mypy benchmark/ --ignore-missing-imports

check: lint typecheck

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── Smoke run (fast end-to-end sanity check) ──────────────────────────────
smoke:
	$(PYTHON) -m benchmark.main \
		--task image_classification \
		--dataset cifar10 \
		--model cnn_scratch \
		--optimizers adam,adamw,sgd_momentum \
		--epochs $(EPOCHS) \
		--batch-size $(BS) \
		--max-samples $(SAMPLES) \
		--output-dir ./results

# ── Run all combinations ────────────────────────────────────────────────────
run-all:
	bash all_instructions/run_all.sh

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache .pytest_cache

clean-results:
	rm -rf results/

# ── Help ───────────────────────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  install        Install runtime dependencies (editable)"
	@echo "  install-dev    Install runtime + dev dependencies"
	@echo "  lint           Run ruff linter"
	@echo "  format         Auto-format with ruff"
	@echo "  typecheck      Run mypy type checker"
	@echo "  check          Run lint + typecheck"
	@echo "  test           Run pytest test suite"
	@echo "  smoke          Quick end-to-end smoke run (cifar10 + 3 optimizers)"
	@echo "  run-all        Run all task/model/dataset combinations"
	@echo "  clean          Remove Python cache files"
	@echo "  clean-results  Remove results/ directory"
