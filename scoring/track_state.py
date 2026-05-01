#!/usr/bin/env python3
"""Pipeline state tracker for the CAT scoring pipeline.

Maintains data/pipeline_state.json with a record of what was done:
- When each data source was scraped (file hash + timestamp)
- When scoring was last run (prompt hash + model + backend + timestamp)
- Whether scoring results are up-to-date with current data and prompt

Usage:
    # Show current pipeline status:
    python -m scoring.track_state status

    # Record that scraping completed:
    python -m scoring.track_state record-scrape --input data/raw/all_scripts.json

    # Record that scoring completed:
    python -m scoring.track_state record-score \\
        --input data/raw/all_scripts.json \\
        --output data/scored/full.jsonl \\
        --model qwen3:4b --backend ollama

    # Check if scoring is stale (exits 0 if fresh, 1 if stale):
    python -m scoring.track_state check-stale \\
        --input data/raw/all_scripts.json \\
        --scored data/scored/full.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "pipeline_state.json"
PROMPT_FILE = Path(__file__).resolve().parent / "pass2_llm_prompt.txt"


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file (first 1 MB only for speed on large files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(1_000_000))
    return h.hexdigest()[:16]  # First 16 hex chars is enough for change detection


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict[str, Any]:
    """Load pipeline state from disk, returning empty dict if not found."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {STATE_FILE}, starting fresh.")
    return {}


def save_state(state: dict[str, Any]) -> None:
    """Persist pipeline state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def cmd_status(state: dict[str, Any]) -> None:
    """Print a human-readable pipeline status summary."""
    print("=" * 55)
    print("  CAT Pipeline Status")
    print("=" * 55)

    scrape = state.get("last_scrape", {})
    if scrape:
        print(f"\n[SCRAPE]")
        print(f"  at:    {scrape.get('at', '?')}")
        print(f"  input: {scrape.get('input_path', '?')}")
        print(f"  hash:  {scrape.get('input_hash', '?')}")
        print(f"  records: {scrape.get('record_count', '?')}")
    else:
        print("\n[SCRAPE]  -- no record --")

    scoring_runs = state.get("scoring_runs", [])
    if scoring_runs:
        latest = scoring_runs[-1]
        print(f"\n[SCORE] (latest of {len(scoring_runs)} run(s))")
        print(f"  at:      {latest.get('at', '?')}")
        print(f"  backend: {latest.get('backend', '?')}")
        print(f"  model:   {latest.get('model', '?')}")
        print(f"  input:   {latest.get('input_path', '?')}")
        print(f"  output:  {latest.get('output_path', '?')}")
        print(f"  input hash:  {latest.get('input_hash', '?')}")
        print(f"  prompt hash: {latest.get('prompt_hash', '?')}")
        print(f"  records: {latest.get('record_count', '?')} scored, {latest.get('error_count', '?')} errors")
    else:
        print("\n[SCORE]  -- no record --")

    # Staleness check
    if scrape and scoring_runs:
        latest = scoring_runs[-1]
        stale_reasons = []
        if scrape.get("input_hash") != latest.get("input_hash"):
            stale_reasons.append("input data has changed since last scoring run")
        current_prompt_hash = _file_sha256(PROMPT_FILE) if PROMPT_FILE.exists() else None
        if current_prompt_hash and current_prompt_hash != latest.get("prompt_hash"):
            stale_reasons.append("LLM prompt has changed since last scoring run")

        print()
        if stale_reasons:
            print("[WARNING] Scoring is STALE:")
            for r in stale_reasons:
                print(f"  - {r}")
            print("  Run: make score")
        else:
            print("[OK] Scoring is up-to-date with current data and prompt.")

    analysis_runs = state.get("analysis_runs", [])
    if analysis_runs:
        latest_a = analysis_runs[-1]
        print(f"\n[ANALYSIS] (latest of {len(analysis_runs)} run(s))")
        print(f"  at:     {latest_a.get('at', '?')}")
        print(f"  input:  {latest_a.get('input_path', '?')}")
        print(f"  output: {latest_a.get('output_dir', '?')}")
    else:
        print("\n[ANALYSIS]  -- no record --")

    print()


def cmd_record_scrape(state: dict[str, Any], input_path: Path) -> None:
    """Record that a scrape run completed."""
    record_count = 0
    if input_path.exists():
        raw = input_path.read_text().lstrip()
        if raw.startswith("["):
            try:
                record_count = len(json.loads(raw))
            except json.JSONDecodeError:
                pass
        else:
            record_count = sum(1 for line in raw.splitlines() if line.strip())

    state["last_scrape"] = {
        "at": _now_iso(),
        "input_path": str(input_path),
        "input_hash": _file_sha256(input_path) if input_path.exists() else None,
        "record_count": record_count,
    }
    save_state(state)
    print(f"Recorded scrape: {record_count} records from {input_path}")


def cmd_record_score(
    state: dict[str, Any],
    input_path: Path,
    output_path: Path,
    model: str,
    backend: str,
) -> None:
    """Record that a scoring run completed."""
    record_count = 0
    error_count = 0
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                record_count += 1
                if rec.get("pass2_error"):
                    error_count += 1
            except json.JSONDecodeError:
                pass

    prompt_hash = _file_sha256(PROMPT_FILE) if PROMPT_FILE.exists() else None
    input_hash = _file_sha256(input_path) if input_path.exists() else None

    run_record = {
        "at": _now_iso(),
        "backend": backend,
        "model": model,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_hash": input_hash,
        "prompt_hash": prompt_hash,
        "record_count": record_count,
        "error_count": error_count,
    }

    if "scoring_runs" not in state:
        state["scoring_runs"] = []
    state["scoring_runs"].append(run_record)
    save_state(state)
    print(f"Recorded scoring run: {record_count} records, {error_count} errors -> {output_path}")


def cmd_record_analysis(
    state: dict[str, Any],
    input_path: Path,
    output_dir: Path,
) -> None:
    """Record that an analysis run completed."""
    if "analysis_runs" not in state:
        state["analysis_runs"] = []
    state["analysis_runs"].append({
        "at": _now_iso(),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
    })
    save_state(state)
    print(f"Recorded analysis run: {input_path} -> {output_dir}")


def cmd_check_stale(
    state: dict[str, Any],
    input_path: Path,
    scored_path: Path,
) -> None:
    """Exit 1 if scoring is stale, 0 if up-to-date."""
    scoring_runs = state.get("scoring_runs", [])
    if not scoring_runs:
        print("STALE: no scoring run recorded yet")
        sys.exit(1)

    latest = scoring_runs[-1]
    stale = False

    current_input_hash = _file_sha256(input_path) if input_path.exists() else None
    if current_input_hash != latest.get("input_hash"):
        print(f"STALE: input data changed (current={current_input_hash}, last_scored={latest.get('input_hash')})")
        stale = True

    current_prompt_hash = _file_sha256(PROMPT_FILE) if PROMPT_FILE.exists() else None
    if current_prompt_hash and current_prompt_hash != latest.get("prompt_hash"):
        print(f"STALE: prompt changed (current={current_prompt_hash}, last_scored={latest.get('prompt_hash')})")
        stale = True

    if not scored_path.exists():
        print(f"STALE: scored output does not exist at {scored_path}")
        stale = True

    if stale:
        sys.exit(1)
    else:
        print("UP-TO-DATE")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT Pipeline State Tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    subparsers.add_parser("status", help="Print pipeline status")

    # record-scrape
    p_scrape = subparsers.add_parser("record-scrape", help="Record a completed scrape run")
    p_scrape.add_argument("--input", type=Path, required=True, help="Scraped data file")

    # record-score
    p_score = subparsers.add_parser("record-score", help="Record a completed scoring run")
    p_score.add_argument("--input", type=Path, required=True, help="Input data file that was scored")
    p_score.add_argument("--output", type=Path, required=True, help="Scored output JSONL file")
    p_score.add_argument("--model", required=True, help="Model used for scoring")
    p_score.add_argument("--backend", required=True, choices=["ollama", "anthropic"], help="Backend used")

    # record-analysis
    p_analysis = subparsers.add_parser("record-analysis", help="Record a completed analysis run")
    p_analysis.add_argument("--input", type=Path, required=True, help="Scored input file")
    p_analysis.add_argument("--output-dir", type=Path, required=True, help="Analysis output directory")

    # check-stale
    p_check = subparsers.add_parser("check-stale", help="Exit 1 if scoring is stale, 0 if fresh")
    p_check.add_argument("--input", type=Path, required=True, help="Raw data input file")
    p_check.add_argument("--scored", type=Path, required=True, help="Scored output file to check")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    state = load_state()

    if args.command == "status":
        cmd_status(state)
    elif args.command == "record-scrape":
        cmd_record_scrape(state, args.input)
    elif args.command == "record-score":
        cmd_record_score(state, args.input, args.output, args.model, args.backend)
    elif args.command == "record-analysis":
        cmd_record_analysis(state, args.input, args.output_dir)
    elif args.command == "check-stale":
        cmd_check_stale(state, args.input, args.scored)


if __name__ == "__main__":
    main()
