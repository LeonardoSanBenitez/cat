# CAT — Contemplative Attention Taxonomy

Data pipeline for scraping, processing, and scoring meditation transcripts using the
Contemplative Attention Taxonomy (CAT) — a 10-dimensional framework characterizing
meditation practices based on their instruction text.

This is a scientific project. The codebase is one component. The full project also
includes data (scraped and scored), infrastructure (local inference engine), and human
tasks (prompt refinement, validation, publication). This document covers all of them.

---

## Team and Responsibilities

| Who | Role |
|-----|------|
| Priya | Engineering lead — pipeline architecture, infrastructure, code review |
| Maria | Research lead — CAT taxonomy, prompt design, dimension validation |
| Mark | Analysis and communication — scoring QA, results interpretation, writeup |
| Cidral | Research review, identify gaps |

---

## Quick Reference: What Script Produces What

| Goal | Command |
|------|---------|
| Check pipeline status | `make status` |
| Scrape meditation texts | `make scrape` |
| Score scraped texts (local Ollama) | `make score` |
| Run aggregate analysis | `make analyze` |
| Run all steps in sequence | `make all` |
| Run tests + mypy | `make test` |
| Check if scoring is stale | `make check` |

For details on each step, see the sections below.

---

## Infrastructure Requirements

### Local Inference Engine (required for scoring)
- Ollama must be running natively before scoring
- Default model: `qwen3:4b`
- Start: run `ollama serve`
- Health check: `curl http://localhost:11434/api/tags`
- The Makefile uses `http://localhost:11434/v1` by default

Scoring is offloaded entirely to the local LLM. Claude is used only to write and
maintain control code, prompts, and orchestration logic. This keeps inference costs
at zero for bulk scoring runs.

### Python Environment
```
pip install ".[dev]"
```

Requires Python 3.12+. Key dependencies: `openai>=1.30` (Ollama client),
`anthropic>=0.28` (fallback), `beautifulsoup4`, `requests`, `pytest`, `mypy`.

---

## Step 1 — Scrape

Collects meditation texts from configured web sources and stores them in `data/raw/`.

```bash
make scrape
# or manually:
python -m scraping.scrape_scripts
```

**Output:** `data/raw/all_scripts.json` (JSON array, one object per meditation)

**Record format:**
```json
{
  "source": "the-guided-meditation-site.com",
  "title": "The Forest Speaks",
  "url": "https://...",
  "text": "...",
  "word_count": 2626
}
```

**Current data sources:**
- `the-guided-meditation-site.com` — 26 scripts (scraped, committed)
- `inner_health_studio` — see `data/raw/inner_health_studio/`
- Manual entries — `data/raw/manual_entries.json`

After scraping, state is recorded in `data/pipeline_state.json`.

---

## Step 2 — Score

Runs the CAT scoring pipeline on the scraped data. Two passes:

1. **Pass 1** — Keyword/pattern detection (fast, no LLM, deterministic). Produces
   `pass1_scores` with estimated numeric scores for all 10 dimensions.

2. **Pass 2** — LLM holistic scoring. Reads the full transcript plus Pass 1 indicators.
   Produces `llm_scores`, `llm_confidence`, `llm_justifications`. Pass 1 estimates
   are overridden by the LLM score when available.

```bash
make score
# or manually:
python -m scoring.pipeline data/raw/all_scripts.json \
    -o data/scored/full.jsonl \
    --pass2 \
    --backend ollama \
    --model qwen3:4b \
    --verbose
```

**Options:**
- `--backend ollama` — local Ollama, free, no API key (default)
- `--backend anthropic` — Anthropic Claude API, costs tokens, requires `ANTHROPIC_API_KEY`
- `--model qwen3:4b` — model to use (any model loaded in Ollama)
- `--id TITLE` — score only one record (for testing/debugging)

**Output:** `data/scored/full.jsonl` (JSONL, one scored record per line)

After scoring completes, state is recorded in `data/pipeline_state.json`.

---

## Step 3 — Analyze

Runs aggregate analysis on the scored data to produce the statistics used in the paper.

```bash
make analyze
# or manually:
python -m analysis.scripts.aggregate_report data/scored/full.jsonl -o analysis/results/
```

**Output files in `analysis/results/`:**
- `aggregate_report.txt` — full text report (score distributions, correlations, PCA)
- `dimension_stats.csv` — per-dimension summary statistics
- `correlation_matrix.csv` — Pearson correlations between all dimensions
- `all_scores.csv` — flat CSV of all scores, suitable for R or pandas

---

## Pipeline State Tracking

The file `data/pipeline_state.json` records what was run, when, and with what inputs.
This lets you instantly know whether scoring results are up-to-date.

```bash
make status           # human-readable status summary
make check            # exit 1 if scoring is stale, 0 if fresh
```

Scoring is considered stale if:
- The raw data file (`data/raw/all_scripts.json`) has changed since last scoring run
- The LLM prompt (`scoring/pass2_llm_prompt.txt`) has changed since last scoring run

When the prompt changes, a full re-score is required because prior results reflect
the old prompt's interpretation of the taxonomy.

---

## LLM Prompt Management

The scoring prompt lives at `scoring/pass2_llm_prompt.txt`.

This file defines:
- The 10 CAT dimension definitions and anchor points
- The expected JSON output schema
- Instructions for handling edge cases

**Prompt changes require re-scoring.** The `track_state` system detects prompt hash
changes and flags results as stale.

**Maria is the primary owner of this prompt.** Any dimension definition changes,
new anchor examples, or pre-check rules (e.g. D9 cap, D6 normalization) go here.
After changing the prompt, run `make check` to confirm stale status, then `make score`
to re-score.

---

## Code Structure

```
cat/
├── scraping/            Web scrapers
│   ├── scrape_scripts.py    Script site scraper (guided_meditation_site, inner_health_studio)
│   ├── scrape_guided_meditation_site.py
│   └── scrape_inner_health_studio.py
├── scoring/             CAT scoring pipeline
│   ├── pipeline.py          Main entry point: Pass 1 + Pass 2, supports Ollama and Anthropic
│   ├── pass1_indicators.py  Keyword/pattern detection for all 10 dimensions
│   ├── pass2_llm_prompt.txt LLM scoring prompt (owns the dimension definitions)
│   └── track_state.py       Pipeline state tracker (staleness detection)
├── analysis/
│   └── scripts/
│       └── aggregate_report.py  Score distributions, correlations, PCA, CSV export
├── tests/               Unit tests (pytest)
│   ├── test_pipeline.py
│   └── test_scrape.py
├── data/
│   ├── raw/             Scraped data (committed)
│   └── scored/          Scored output (committed when complete)
├── Makefile             Workflow automation
├── pyproject.toml       Package and dependency config
└── Dockerfile           Dev container (Python + ffmpeg)
```

---

## CI

GitHub Actions runs on every push to `main`:
- `mypy` — static type checking (strict)
- `pytest` — unit tests

See `.github/workflows/mypy.yml`.

---

## Pending Work (as of 2026-05-01)

### Code
- [ ] Improve Pass 1 keyword sets based on Maria's feedback (D8 vs D2 conflation,
      D1 absence problem, D9 pre-check, D6 normalization by duration, D4 pre-filter)
- [ ] Add `scrape_meditations.py` (YouTube) back into the pipeline once scoring
      at scale is validated

### Data
- [ ] Run scoring on current `data/raw/all_scripts.json` (183 records) via Ollama
- [ ] Validate 10-20 scored records manually (Maria + Mark task)
- [ ] Extend dataset: more sources beyond the current 2 script sites

### Prompt
- [ ] Add 3-5 annotated exemplar passages per dimension anchor (Maria task)
- [ ] Add D9 pre-check (no second-order language -> cap at 40)
- [ ] Add D6 normalization instruction (transitions/minute, not raw count)
- [ ] Add meditation onset pre-filter guidance

### Infrastructure
- [ ] Ollama: consider upgrading to `qwen3:14b` once 4b quality is validated
      (requires user to assess VRAM headroom)

### Publication
- [ ] Analyze results once full scoring is done (Mark task)
- [ ] Write methodology section referencing CAT framework and qwen3:4b scoring
