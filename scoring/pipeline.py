#!/usr/bin/env python3
"""CAT Scoring Pipeline: Pass 1 (keyword detection) + Pass 2 (LLM scoring).

Reads scraped meditation data, runs Pass 1 indicator detection, optionally
runs Pass 2 LLM scoring, and writes scored output.

Pass 2 supports two backends:
  - ollama (default): local Ollama instance via native /api/chat (http://localhost:11434).
                      Free to run, no API key required. Requires Ollama running.
  - anthropic: Anthropic Claude API. Requires ANTHROPIC_API_KEY env var.

The Ollama backend is the preferred backend. Use Claude only when you need a
specific Anthropic model or are running in a context where local inference is
unavailable.

Usage:
    # Pass 1 only (no LLM needed):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/pass1.jsonl

    # Pass 1 + Pass 2 via Ollama (default — local, free):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/full.jsonl --pass2

    # Pass 1 + Pass 2 via Ollama with a specific model:
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/full.jsonl --pass2 --model qwen3:4b

    # Pass 1 + Pass 2 via Anthropic API:
    ANTHROPIC_API_KEY=... python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/full.jsonl --pass2 --backend anthropic

    # Score a single record (for testing):
    python -m scoring.pipeline data/raw/meditations.jsonl -o data/scored/test.jsonl --id VIDEO_ID
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Allow running as module from cat-study root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring.pass1_indicators import format_pass1_summary, score_transcript

logger = logging.getLogger(__name__)

PASS2_PROMPT_PATH = Path(__file__).parent / "pass2_llm_prompt.txt"

# Ollama (local, default)
DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"

# Anthropic (cloud, explicit opt-in)
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

MAX_TRANSCRIPT_CHARS = 4000  # truncate very long transcripts for LLM


def load_pass2_prompt() -> str:
    """Load the Pass 2 LLM prompt template."""
    return PASS2_PROMPT_PATH.read_text()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON or JSONL file, returning list of records.

    Auto-detects format:
    - If the file starts with '[', treats it as a JSON array.
    - Otherwise treats each non-empty line as a separate JSON object (JSONL).
    """
    raw = path.read_text(encoding="utf-8").lstrip()
    if raw.startswith("["):
        # JSON array
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        return data

    # JSONL: one JSON object per line
    records: list[dict[str, Any]] = []
    for line_num, line in enumerate(raw.splitlines(), 1):
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


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize field names across data sources.

    Different scrapers use different field names for the transcript text:
      - YouTube scraper: "transcript_text"
      - Script scrapers (guided_meditation_site, inner_health_studio): "text"

    This function unifies them into "transcript_text" without duplicating the
    content. Mutates and returns the record.
    """
    if "transcript_text" not in record and "text" in record:
        record["transcript_text"] = record["text"]
    return record


def run_pass1(record: dict[str, Any]) -> dict[str, Any]:
    """Run Pass 1 keyword detection on a single record.

    Adds 'pass1_scores' to the record dict and returns it.
    """
    text = record.get("transcript_text", "")
    if not text or len(text) < 100:
        logger.warning(
            f"Skipping {record.get('video_id', record.get('title', '?'))}: transcript too short "
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


def _build_pass2_prompt(
    record: dict[str, Any],
    prompt_template: str,
) -> str:
    """Build the filled prompt string for Pass 2 from a record and template."""
    text = record.get("transcript_text", "")
    if len(text) > MAX_TRANSCRIPT_CHARS:
        logger.debug(
            f"Truncating transcript for {record.get('video_id', record.get('title', '?'))} "
            f"from {len(text)} to {MAX_TRANSCRIPT_CHARS} chars"
        )
        text = text[:MAX_TRANSCRIPT_CHARS] + "\n\n[TRANSCRIPT TRUNCATED]"

    pass1 = record.get("pass1_scores")
    pass1_summary = format_pass1_summary(pass1) if pass1 else "(Pass 1 not available)"

    duration_sec = record.get("duration_seconds", 0)
    duration_min = round(duration_sec / 60, 1) if duration_sec else "unknown"

    filled = prompt_template.format(
        title=record.get("title", "Unknown"),
        channel=record.get("channel", "Unknown"),
        duration_minutes=duration_min,
        pass1_summary=pass1_summary,
        transcript=text,
    )
    return filled


def _parse_llm_response(response_text: str) -> dict[str, Any]:
    """Parse JSON from LLM response.

    Handles three common LLM output formats:
    1. Plain JSON: ``{"scores": ...}``
    2. Markdown-fenced JSON: ````` ```json\\n{...}\\n``` `````
    3. Thinking-mode responses: reasoning text followed by JSON, e.g.
       ``<think>...</think>\\n\\n{"scores": ...}`` or just prose followed
       by a JSON block.  We locate the *first* ``{`` in the content and
       attempt ``json.loads`` from that position.  If that fails (e.g. the
       ``{`` is inside prose), we fall back to a regex that finds the last
       outermost JSON object in the string.
    """
    json_text = response_text.strip()

    # Strip markdown code fences (``` or ```json ... ```)
    if json_text.startswith("```"):
        lines = json_text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        json_text = "\n".join(lines).strip()

    # Fast path: the entire text is valid JSON
    try:
        return json.loads(json_text)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        pass

    # Thinking-mode path: find the first '{' and try to parse from there
    idx = json_text.find("{")
    if idx != -1:
        try:
            return json.loads(json_text[idx:])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Last-resort: find the last outermost JSON object via regex
    # This handles cases where early '{' characters appear in prose.
    matches = list(re.finditer(r"\{", json_text))
    for m in reversed(matches):
        try:
            return json.loads(json_text[m.start():])  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            continue

    # Nothing worked — re-raise a clean error
    raise json.JSONDecodeError(
        "No valid JSON object found in LLM response",
        json_text,
        0,
    )


def run_pass2_ollama(
    record: dict[str, Any],
    prompt_template: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Run Pass 2 LLM scoring using a local Ollama instance (native /api/chat endpoint).

    Uses httpx directly against Ollama's native API rather than the OpenAI-compatible
    /v1 shim.  The OpenAI-compat shim silently discards qwen3 thinking tokens: when
    max_tokens is exhausted during the think phase the shim returns an empty
    choices[0].message.content.  The native endpoint always returns thinking text
    (wrapped in <think>...</think>) plus the answer in message.content, which
    _parse_llm_response() already handles.

    The chat URL is derived from base_url by stripping any trailing /v1 suffix, then
    appending /api/chat.  With the default base_url of http://localhost:11434/v1 this
    resolves to http://localhost:11434/api/chat.

    Adds 'llm_scores', 'llm_confidence', 'llm_justifications' to the record.
    """
    text = record.get("transcript_text", "")
    if not text:
        record["pass2_error"] = "no transcript"
        return record

    # Derive the native API base (strip /v1 if present) and build the chat URL.
    native_base = base_url.rstrip("/")
    if native_base.endswith("/v1"):
        native_base = native_base[:-3]
    chat_url = f"{native_base}/api/chat"

    prompt = _build_pass2_prompt(record, prompt_template)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"num_predict": 4000},
    }

    # Use a long read timeout (30 min) so large transcripts processed by a
    # local model (e.g. qwen3:4b) are never aborted mid-generation.
    # connect timeout stays short (10 s) to catch a missing Ollama daemon fast.
    ollama_timeout = httpx.Timeout(1800.0, connect=10.0)

    response_text = ""
    t0 = time.monotonic()
    try:
        http_response = httpx.post(chat_url, json=payload, timeout=ollama_timeout)
        http_response.raise_for_status()
        elapsed = time.monotonic() - t0

        response_data = http_response.json()
        response_text = response_data.get("message", {}).get("content", "")
        logger.info(
            f"Ollama response for {record.get('video_id', record.get('title', '?'))}: "
            f"{elapsed:.1f}s ({len(response_text)} chars)"
        )

        result = _parse_llm_response(response_text)

        record["llm_scores"] = result.get("scores", {})
        record["llm_confidence"] = result.get("confidence", {})
        record["llm_justifications"] = result.get("justifications", {})
        record["llm_notes"] = result.get("notes")
        record["llm_model"] = f"ollama:{model}"
        record["llm_backend"] = "ollama"

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse Ollama response for {record.get('video_id', record.get('title', '?'))}: {e}"
        )
        record["pass2_error"] = f"JSON parse error: {e}"
        record["llm_raw_response"] = response_text[:1000]

    except Exception as e:
        logger.error(
            f"Ollama API error for {record.get('video_id', record.get('title', '?'))}: {e}"
        )
        record["pass2_error"] = str(e)

    return record


def run_pass2_anthropic(
    record: dict[str, Any],
    prompt_template: str,
    model: str = DEFAULT_ANTHROPIC_MODEL,
) -> dict[str, Any]:
    """Run Pass 2 LLM scoring via the Anthropic Claude API.

    Requires the anthropic package and ANTHROPIC_API_KEY env var.
    This backend consumes paid API tokens. Prefer run_pass2_ollama for bulk scoring.

    Adds 'llm_scores', 'llm_confidence', 'llm_justifications' to the record.
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed. Run: pip install anthropic")
        record["pass2_error"] = "anthropic package not installed"
        return record

    text = record.get("transcript_text", "")
    if not text:
        record["pass2_error"] = "no transcript"
        return record

    client = anthropic.Anthropic()

    prompt = _build_pass2_prompt(record, prompt_template)

    response_text = ""
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        for block in response.content:
            if hasattr(block, "text"):
                response_text += block.text

        result = _parse_llm_response(response_text)

        record["llm_scores"] = result.get("scores", {})
        record["llm_confidence"] = result.get("confidence", {})
        record["llm_justifications"] = result.get("justifications", {})
        record["llm_notes"] = result.get("notes")
        record["llm_model"] = model
        record["llm_backend"] = "anthropic"

    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse Anthropic response for {record.get('video_id', record.get('title', '?'))}: {e}"
        )
        record["pass2_error"] = f"JSON parse error: {e}"
        record["llm_raw_response"] = response_text[:1000]

    except Exception as e:
        logger.error(
            f"Anthropic API error for {record.get('video_id', record.get('title', '?'))}: {e}"
        )
        record["pass2_error"] = str(e)

    return record


def run_pass2(
    record: dict[str, Any],
    prompt_template: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    backend: str = DEFAULT_BACKEND,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
) -> dict[str, Any]:
    """Run Pass 2 LLM scoring on a single record, dispatching to the chosen backend.

    Args:
        record: Meditation record dict, modified in place.
        prompt_template: Filled prompt template string.
        model: Model name. For ollama: e.g. "qwen3:4b". For anthropic: Anthropic model ID.
        backend: "ollama" (default) or "anthropic".
        ollama_base_url: Ollama API base URL (only used when backend="ollama").

    Returns:
        The record dict with llm_scores, llm_confidence, llm_justifications added.
    """
    if backend == "ollama":
        return run_pass2_ollama(record, prompt_template, model=model, base_url=ollama_base_url)
    elif backend == "anthropic":
        return run_pass2_anthropic(record, prompt_template, model=model)
    else:
        record["pass2_error"] = f"Unknown backend: {backend!r}. Choose 'ollama' or 'anthropic'."
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
    model: str = DEFAULT_OLLAMA_MODEL,
    backend: str = DEFAULT_BACKEND,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    filter_id: str | None = None,
    rate_limit_delay: float = 1.0,
) -> None:
    """Score all meditations in a JSONL file.

    Args:
        input_path: Input JSONL file (one record per line).
        output_path: Output JSONL path.
        run_llm: If True, run Pass 2 LLM scoring in addition to Pass 1.
        model: LLM model name. Defaults to qwen3:4b (Ollama).
        backend: LLM backend: "ollama" (default) or "anthropic".
        ollama_base_url: Ollama API base URL (ignored when backend=anthropic).
        filter_id: If set, score only the record with this video_id.
        rate_limit_delay: Seconds to wait between LLM calls (for Anthropic rate limits).
                          Ollama is local so delay can safely be 0, but default is kept
                          for safety.
    """
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
        # Normalize field names (transcript_text vs text)
        record = normalize_record(record)

        vid = record.get("video_id", record.get("title", f"record_{i}"))
        logger.info(f"[{i+1}/{len(records)}] Scoring {vid}...")

        try:
            # Pass 1
            record = run_pass1(record)

            if record.get("pass1_skipped"):
                scored.append(record)
                continue

            # Pass 2 (LLM)
            if run_llm:
                record = run_pass2(
                    record,
                    prompt_template,
                    model=model,
                    backend=backend,
                    ollama_base_url=ollama_base_url,
                )
                if rate_limit_delay > 0 and i < len(records) - 1:
                    time.sleep(rate_limit_delay)

            # Merge
            record = merge_scores(record)

        except Exception as e:
            logger.warning(
                f"Record-level error for {vid} (record {i+1}/{len(records)}): {e}. "
                f"Writing null-scores record and continuing."
            )
            record["pass2_error"] = f"pipeline error: {e}"
            record["llm_scores"] = None
            record["llm_confidence"] = None
            record["llm_justifications"] = None
            record["final_scores"] = None

        scored.append(record)

    write_jsonl(scored, output_path)

    # Summary
    scored_count = sum(1 for r in scored if not r.get("pass1_skipped"))
    llm_count = sum(1 for r in scored if "llm_scores" in r and r["llm_scores"])
    errors = sum(1 for r in scored if "pass2_error" in r)

    logger.info(
        f"Done. {scored_count} scored via Pass 1, {llm_count} via Pass 2 "
        f"(backend={backend}, model={model}), {errors} errors."
    )


_LOCKFILE_PATH = Path("data/pipeline.lock")


def _acquire_lockfile(lock_path: Path = _LOCKFILE_PATH) -> None:
    """Acquire an exclusive lockfile to prevent multiple concurrent pipeline instances.

    Writes the current PID to *lock_path*.  If the file already exists, logs
    the PID of the competing process and exits with a non-zero status so the
    caller does not accidentally hammer Ollama with parallel requests.

    The lock is released automatically when the process exits (via atexit).

    Args:
        lock_path: Path to the lockfile.  Defaults to ``data/pipeline.lock``.

    Raises:
        SystemExit: If a lockfile already exists (another instance is running).
    """
    if lock_path.exists():
        try:
            existing_pid = lock_path.read_text().strip()
        except OSError:
            existing_pid = "<unknown>"
        logger.error(
            f"Pipeline is already running (PID {existing_pid}). "
            f"Lockfile: {lock_path}. "
            "If no other pipeline is running, delete the lockfile and retry."
        )
        sys.exit(1)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))

    def _release() -> None:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_release)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CAT Scoring Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pass 1 only (fast, no LLM):
  python -m scoring.pipeline data/raw/all_scripts.json -o data/scored/pass1.jsonl

  # Pass 1 + Pass 2 via local Ollama (preferred for bulk runs):
  python -m scoring.pipeline data/raw/all_scripts.json -o data/scored/full.jsonl --pass2

  # Pass 1 + Pass 2 via Anthropic (use sparingly — costs tokens):
  ANTHROPIC_API_KEY=... python -m scoring.pipeline data/raw/all_scripts.json \\
      -o data/scored/full.jsonl --pass2 --backend anthropic

  # Test on a single record:
  python -m scoring.pipeline data/raw/all_scripts.json -o data/scored/test.jsonl --id my_id
""",
    )
    parser.add_argument("input", type=Path, help="Input JSON or JSONL (scraped meditations)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument(
        "--pass2", action="store_true", help="Run Pass 2 LLM scoring"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "anthropic"],
        default=DEFAULT_BACKEND,
        help=(
            f"LLM backend for Pass 2. 'ollama' (default) uses the local Ollama instance "
            f"(free, no API key). 'anthropic' uses the Anthropic API (requires ANTHROPIC_API_KEY)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"LLM model name. For ollama: default is '{DEFAULT_OLLAMA_MODEL}'. "
            f"For anthropic: default is '{DEFAULT_ANTHROPIC_MODEL}'."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help=f"Ollama API base URL (default: {DEFAULT_OLLAMA_BASE_URL})",
    )
    parser.add_argument("--id", dest="filter_id", help="Score only the record with this video_id")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.0,
        help="Seconds to wait between LLM calls. Default 0 (no delay) for Ollama; set to 1.0+ for Anthropic.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Prevent multiple concurrent pipeline instances from hammering Ollama.
    _acquire_lockfile()

    # Determine effective model
    if args.model is not None:
        effective_model = args.model
    elif args.backend == "anthropic":
        effective_model = DEFAULT_ANTHROPIC_MODEL
    else:
        effective_model = DEFAULT_OLLAMA_MODEL

    # Validate backend prerequisites
    if args.pass2 and args.backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set. Required when --backend anthropic.")
        sys.exit(1)

    score_dataset(
        input_path=args.input,
        output_path=args.output,
        run_llm=args.pass2,
        model=effective_model,
        backend=args.backend,
        ollama_base_url=args.ollama_url,
        filter_id=args.filter_id,
        rate_limit_delay=args.rate_limit,
    )


if __name__ == "__main__":
    main()
