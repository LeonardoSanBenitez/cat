# CAT — Cognitive Appraisal Theory Study

Meditation transcript scraping and scoring for the CAT study.

## Structure

- `scraping/` — web scrapers for meditation script sites
- `scoring/` — CAT dimension scoring pipeline (pass 1: keyword detection, pass 2: LLM)
- `scrape_meditations.py` — YouTube meditation transcript scraper
- `analysis/` — post-scoring analysis scripts
- `tests/` — unit tests

## Usage

```bash
pip install ".[dev]"

# Scrape YouTube transcripts
python scrape_meditations.py queries.txt -o data/raw/meditations.jsonl

# Run scoring pipeline (pass 1 only)
python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/pass1.jsonl

# Run scoring pipeline (pass 1 + LLM pass 2)
ANTHROPIC_API_KEY=... python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/full.jsonl --pass2
```
