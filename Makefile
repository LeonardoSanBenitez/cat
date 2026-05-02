# CAT Pipeline Makefile
#
# Targets:
#   make status    — Show pipeline status (what was run, what is stale)
#   make scrape    — Scrape meditation scripts from all sources
#   make score     — Score scraped data via local Ollama (qwen3.5:latest)
#   make analyze   — Run aggregate analysis on scored data
#   make all       — scrape + score + analyze
#   make test      — Run unit tests and mypy
#   make check     — Check if scoring is up-to-date with data and prompt
#
# Variables (override on command line):
#   OLLAMA_MODEL   — Ollama model to use (default: qwen3.5:latest)
#   OLLAMA_URL     — Ollama API base URL (default: http://localhost:11434/v1)
#   SCORED_OUT     — Scored output file (default: data/scored/full.jsonl)
#   ANALYSIS_DIR   — Analysis output directory (default: analysis/results)

OLLAMA_MODEL ?= qwen3.5:latest
OLLAMA_URL   ?= http://localhost:11434/v1
SCORED_OUT   ?= data/scored/full.jsonl
ANALYSIS_DIR ?= analysis/results
RAW_DATA     ?= data/raw/all_scripts.json

# Experiment target variables (override on command line):
#   EXP         — experiment name (e.g. v0_baseline)
#   PROMPT_FILE — path to prompt file (e.g. scoring/experiments/v0_baseline_prompt.txt)
EXP         ?= unnamed
PROMPT_FILE ?= scoring/experiments/$(EXP)_prompt.txt

.PHONY: all scrape score analyze status check test help experiment

all: scrape score analyze

help:
	@echo "CAT Pipeline — available targets:"
	@echo "  make status   Show pipeline state (what ran, what is stale)"
	@echo "  make scrape   Scrape meditation scripts from configured sources"
	@echo "  make score    Score scraped data via local Ollama"
	@echo "  make analyze  Run aggregate analysis on scored output"
	@echo "  make all      scrape + score + analyze"
	@echo "  make check    Exit 1 if scoring is stale"
	@echo "  make test     Run pytest + mypy"
	@echo ""
	@echo "Variables (set on command line to override):"
	@echo "  OLLAMA_MODEL=$(OLLAMA_MODEL)"
	@echo "  OLLAMA_URL=$(OLLAMA_URL)"
	@echo "  SCORED_OUT=$(SCORED_OUT)"
	@echo "  ANALYSIS_DIR=$(ANALYSIS_DIR)"
	@echo "  RAW_DATA=$(RAW_DATA)"

status:
	python -m scoring.track_state status

check:
	python -m scoring.track_state check-stale \
		--input $(RAW_DATA) \
		--scored $(SCORED_OUT)

scrape:
	@echo "=== Scraping meditation scripts ==="
	python -m scraping.scrape_scripts
	python -m scoring.track_state record-scrape --input $(RAW_DATA)

score:
	@echo "=== Scoring via Ollama (model=$(OLLAMA_MODEL)) ==="
	@echo "This may take a long time (minutes to hours depending on dataset size)."
	python -m scoring.pipeline $(RAW_DATA) \
		-o $(SCORED_OUT) \
		--pass2 \
		--backend ollama \
		--model $(OLLAMA_MODEL) \
		--ollama-url $(OLLAMA_URL) \
		--verbose
	python -m scoring.track_state record-score \
		--input $(RAW_DATA) \
		--output $(SCORED_OUT) \
		--model $(OLLAMA_MODEL) \
		--backend ollama

analyze:
	@echo "=== Running aggregate analysis ==="
	python -m analysis.scripts.aggregate_report $(SCORED_OUT) -o $(ANALYSIS_DIR)
	python -m scoring.track_state record-analysis \
		--input $(SCORED_OUT) \
		--output-dir $(ANALYSIS_DIR)

test:
	mypy scrape_meditations.py scoring/ scraping/ analysis/
	pytest tests/ -v

experiment:
	@echo "=== Running experiment: $(EXP) ==="
	@echo "  prompt file : $(PROMPT_FILE)"
	@echo "  output dir  : data/experiments/$(EXP)/"
	python -m scoring.pipeline $(RAW_DATA) \
		-o data/experiments/$(EXP)/scored.jsonl \
		--pass2 \
		--backend ollama \
		--model $(OLLAMA_MODEL) \
		--ollama-url $(OLLAMA_URL) \
		--prompt-file $(PROMPT_FILE) \
		--verbose
	python -m scoring.track_state record-score \
		--input $(RAW_DATA) \
		--output data/experiments/$(EXP)/scored.jsonl \
		--model $(OLLAMA_MODEL) \
		--backend ollama
	python -m analysis.scripts.aggregate_report \
		data/experiments/$(EXP)/scored.jsonl \
		-o data/experiments/$(EXP)/analysis/
	python -m scoring.track_state record-analysis \
		--input data/experiments/$(EXP)/scored.jsonl \
		--output-dir data/experiments/$(EXP)/analysis/
