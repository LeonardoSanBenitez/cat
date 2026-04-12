#!/usr/bin/env python3
"""CAT Scoring Pipeline: Pass 1 (keyword detection) + Pass 2 (LLM scoring).

Reads scraped meditation JSONL, runs Pass 1 indicator detection, optionally
runs Pass 2 LLM scoring via the Anthropic API, and writes scored output.

Usage:
    # Pass 1 only (no API key needed):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/pass1.jsonl

    # Pass 1 + Pass 2 (requires ANTHROPIC_API_KEY):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/full.jsonl --pass2

    # Score a single meditation (for testing):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/test.jsonl --id VIDEO_ID
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as module from cat-study root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.pass1_indicators import format_pass1_summary, score_transcript

logger = logging.getLogger(__name__)

PASS2_PROMPT_PATH = Path(__file__).parent / "pass2_llm_prompt.txt"
DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TRANSCRIPT_CHARS = 30000  # truncate very long transcripts for LLM


def load_pass2_prompt() -> str:
    """Load the Pass 2 LLM prompt template."""
    return PASS2_PROMPT_PATH.read_text()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file, returning list of records."""
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def run_pass1(record: dict[str, Any]) -> dict[str, Any]:
    """Run Pass 1 keyword detection on a single record.

    Adds 'pass1_scores' to the record dict and returns it.
    """
    text = record.get("transcript_text", "")
    if not text or len(text) < 100:
        logger.warning(
            f"Skipping {record.get('video_id', '?')}: transcript too short "
            f"({len(text)} chars)"
        )
        record["pass1_scores"] = None
        record["pass1_skipped"] = True
        return record

    # Parse timestamps if available
    timestamps = None
    ts_raw = record.get("transcript_with_timestamps")
    if isinstance(ts_raw, str):
        try:
            timestamps = json.loads(ts_raw)
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(ts_raw, list):
        timestamps = ts_raw

    pass1 = score_transcript(text, timestamps)
    record["pass1_scores"] = pass1
    record["pass1_skipped"] = False
    return record


def run_pass2(
    record: dict[str, Any],
    prompt_template: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run Pass 2 LLM scoring on a single record.

    Requires the anthropic package and ANTHROPIC_API_KEY env var.
    Adds 'llm_scores', 'llm_confidence', 'llm_justifications' to the record.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        record["pass2_error"] = "anthropic package not installed"
        return record

    client = anthropic.Anthropic()

    text = record.get("transcript_text", "")
    if not text:
        record["pass2_error"] = "no transcript"
        return record

    # Truncate if needed
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n\n[TRANSCRIPT TRUNCATED]"

    # Build Pass 1 summary
    pass1 = record.get("pass1_scores")
    if pass1:
        pass1_summary = format_pass1_summary(pass1)
    else:
        pass1_summary = "(Pass 1 not available)"

    # Duration in minutes
    duration_sec = record.get("duration_seconds", 0)
    duration_min = round(duration_sec / 60, 1) if duration_sec else "unknown"

    # Fill prompt
    prompt = prompt_template.format(
        title=record.get("title", "Unknown"),
        channel=record.get("channel", "Unknown"),
        duration_minutes=duration_min,
        pass1_summary=pass1_summary,
        transcript=text,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text content
        response_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        # Parse JSON from response
        # Handle potential markdown code blocks
        json_text = response_text.strip()
        if json_text.startswith("```"):
            # Remove code fence
            lines = json_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            json_text = "\n".join(lines)

        result = json.loads(json_text)

        record["llm_scores"] = result.get("scores", {})
        record["llm_confidence"] = result.get("confidence", {})
        record["llm_justifications"] = result.get("justifications", {})
        record["llm_notes"] = result.get("notes")
        record["llm_model"] = model

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse LLM response for {record.get('video_id', '?')}: {e}"
        )
        record["pass2_error"] = f"JSON parse error: {e}"
        record["llm_raw_response"] = response_text[:1000]

    except Exception as e:
        logger.error(
            f"LLM API error for {record.get('video_id', '?')}: {e}"
        )
        record["pass2_error"] = str(e)

    return record


def merge_scores(record: dict[str, Any]) -> dict[str, Any]:
    """Create final merged scores from Pass 1 and Pass 2.

    If Pass 2 is available, use LLM scores as primary.
    Otherwise fall back to Pass 1 estimated scores.
    """
    final_scores: dict[str, Any] = {}

    llm_scores = record.get("llm_scores", {})
    pass1 = record.get("pass1_scores", {})

    # Continuous dimensions: prefer LLM, fall back to Pass 1
    continuous_dims = [
        "D1_attentional_constraint",
        "D2_somatic_engagement",
        "D5_object_density",
        "D6_temporal_dynamics",
        "D7_affective_cultivation",
        "D8_interoceptive_demand",
        "D9_metacognitive_load",
        "D10_relational_orientation",
    ]

    for dim in continuous_dims:
        if dim in llm_scores and llm_scores[dim] is not None:
            final_scores[dim] = llm_scores[dim]
        elif dim in pass1 and pass1[dim].get("estimated_score") is not None:
            final_scores[dim] = pass1[dim]["estimated_score"]
        else:
            final_scores[dim] = None

    # Categorical dimensions: prefer LLM
    if "D3_startup_modality" in llm_scores:
        final_scores["D3_startup_modality"] = llm_scores["D3_startup_modality"]
    elif "D3_startup_modality" in pass1:
        final_scores["D3_startup_modality"] = pass1["D3_startup_modality"].get(
            "suggested", "none"
        )

    if "D4_object_nature" in llm_scores:
        final_scores["D4_object_nature"] = llm_scores["D4_object_nature"]
    elif "D4_object_nature" in pass1:
        final_scores["D4_object_nature"] = pass1["D4_object_nature"].get(
            "suggested", []
        )

    record["final_scores"] = final_scores
    return record


def score_dataset(
    input_path: Path,
    output_path: Path,
    run_llm: bool = False,
    model: str = DEFAULT_MODEL,
    filter_id: str | None = None,
    rate_limit_delay: float = 1.0,
) -> None:
    """Score all meditations in a JSONL file."""
    records = read_jsonl(input_path)
    logger.info(f"Loaded {len(records)} records from {input_path}")

    if filter_id:
        records = [r for r in records if r.get("video_id") == filter_id]
        if not records:
            logger.error(f"No record found with video_id={filter_id}")
            return
        logger.info(f"Filtered to 1 record: {filter_id}")

    prompt_template = load_pass2_prompt() if run_llm else ""

    scored = []
    for i, record in enumerate(records):
        vid = record.get("video_id", f"record_{i}")
        logger.info(f"[{i+1}/{len(records)}] Scoring {vid}...")

        # Pass 1
        record = run_pass1(record)

        if record.get("pass1_skipped"):
            scored.append(record)
            continue

        # Pass 2 (LLM)
        if run_llm:
            record = run_pass2(record, prompt_template, model)
            if i < len(records) - 1:
                time.sleep(rate_limit_delay)

        # Merge
        record = merge_scores(record)
        scored.append(record)

    write_jsonl(scored, output_path)

    # Summary
    scored_count = sum(1 for r in scored if not r.get("pass1_skipped"))
    llm_count = sum(1 for r in scored if "llm_scores" in r and r["llm_scores"])
    errors = sum(1 for r in scored if "pass2_error" in r)

    logger.info(f"Done. {scored_count} scored via Pass 1, {llm_count} via Pass 2, {errors} errors.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT Scoring Pipeline")
    parser.add_argument("input", type=Path, help="Input JSONL (scraped meditations)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument(
        "--pass2", action="store_true", help="Run Pass 2 LLM scoring (needs ANTHROPIC_API_KEY)"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--id", dest="filter_id", help="Score only this video_id")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Seconds between LLM calls")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.pass2 and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set. Required for Pass 2.")
        sys.exit(1)

    score_dataset(
        input_path=args.input,
        output_path=args.output,
        run_llm=args.pass2,
        model=args.model,
        filter_id=args.filter_id,
        rate_limit_delay=args.rate_limit,
    )


if __name__ == "__main__":
    main()
