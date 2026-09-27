VENV   := .venv
PYTHON := $(abspath $(VENV)/bin/python)
TESTS  ?=
JOBS   ?= 8

.DEFAULT_GOAL := help
.PHONY: help venv compile lint typecheck build test check install clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

# Dependencies only: tests exercise installing from a source checkout, which
# breaks when the package itself is importable from the venv.
$(PYTHON): pyproject.toml
	uv venv --allow-existing $(VENV)
	uv pip install --python $(PYTHON) -r pyproject.toml --extra dev
	@touch $(PYTHON)

venv: $(PYTHON) ## Create the dev virtualenv with runtime, test and lint dependencies

compile: $(PYTHON) ## Byte-compile all sources to catch syntax errors
	$(PYTHON) -m compileall -q src tests

lint: $(PYTHON) ## Lint with ruff
	$(PYTHON) -m ruff check src tests

typecheck: $(PYTHON) ## Type-check with mypy
	$(PYTHON) -m mypy

build: ## Build sdist and wheel into dist/
	uv build -o dist

test: $(PYTHON) ## Run tests in 8 processes (~1.5 min; JOBS=N to change); a subset: make test TESTS=test_phase2
ifeq ($(strip $(TESTS)),)
	cd tests && $(PYTHON) run_parallel.py -j $(JOBS)
else
	cd tests && $(PYTHON) -m unittest $(TESTS)
endif

check: compile lint typecheck test ## Compile, lint, type-check and test, as CI would

install: ## Install the repohone CLI for this user (editable)
	uv tool install --editable .

clean: ## Remove build output, caches and the venv
	rm -rf dist build $(VENV) .mypy_cache .ruff_cache
	find src tests -name '*.egg-info' -prune -exec rm -rf {} +
	find src tests -name __pycache__ -prune -exec rm -rf {} +
