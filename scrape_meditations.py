#!/usr/bin/env python3
"""YouTube meditation transcript scraper for the CAT study.

Searches YouTube for meditation videos, extracts metadata and transcripts,
and outputs structured JSONL.

Usage:
    python scrape_meditations.py queries.txt -o output.jsonl
    python scrape_meditations.py queries.txt -o output.jsonl --max-results 10
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

MIN_DURATION = 8 * 60    # 8 minutes in seconds
MAX_DURATION = 45 * 60   # 45 minutes in seconds
DEFAULT_MAX_RESULTS = 20
SLEEP_BETWEEN_QUERIES = 2.0  # be polite to YouTube


@dataclass
class VideoRecord:
    """A single scraped video with metadata and transcript."""

    video_id: str
    title: str
    channel: str
    duration_seconds: int
    view_count: int
    description: str
    transcript_text: str
    transcript_with_timestamps: list[dict[str, Any]]
    search_query_source: str
    url: str = ""

    def __post_init__(self) -> None:
        if not self.url:
            self.url = f"https://www.youtube.com/watch?v={self.video_id}"


def search_youtube(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Search YouTube using yt-dlp and return metadata for matching videos.

    Returns a list of dicts with video metadata. Filters to the duration
    range [MIN_DURATION, MAX_DURATION].
    """
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        "--no-warnings",
        "--quiet",
    ]
    logger.info("Searching: %s (max %d results)", query, max_results)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Search timed out for query: %s", query)
        return []

    if result.returncode != 0:
        logger.warning("yt-dlp search failed for query '%s': %s", query, result.stderr[:200])
        return []

    videos: list[dict[str, Any]] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        videos.append(entry)

    return videos


def get_full_metadata(video_id: str) -> dict[str, Any] | None:
    """Fetch full metadata for a single video using yt-dlp."""
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "--dump-json",
        "--no-download",
        "--no-warnings",
        "--quiet",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Metadata fetch timed out for %s", video_id)
        return None

    if result.returncode != 0:
        logger.warning("Failed to get metadata for %s", video_id)
        return None

    try:
        return json.loads(result.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None


_yt_api = YouTubeTranscriptApi()


def fetch_transcript(video_id: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Fetch transcript for a video. Returns (full_text, timestamped_segments) or None.

    Uses youtube-transcript-api >= 1.0 instance-based API.
    Tries to fetch English transcript first, falls back to listing available
    transcripts and picking the best English option.
    """
    # Try direct fetch with English language preference
    try:
        segments = _yt_api.fetch(video_id, languages=["en"])
    except Exception:
        logger.debug("No English transcript for %s via direct fetch", video_id)
        # Try listing transcripts and finding English
        try:
            transcript_list = _yt_api.list(video_id)
            # Look for any English transcript
            found = None
            for t in transcript_list:
                if t.language_code.startswith("en"):
                    found = t
                    break
            if found is None:
                logger.debug("No English transcript available for %s", video_id)
                return None
            segments = found.fetch()
        except Exception:
            logger.debug("No transcripts available for %s", video_id)
            return None

    # Build timestamped list and full text
    timestamped: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for seg in segments:
        text = seg.text
        start = seg.start
        duration = seg.duration
        timestamped.append({
            "start": start,
            "duration": duration,
            "text": text,
        })
        text_parts.append(text)

    full_text = " ".join(text_parts)
    return full_text, timestamped


def extract_metadata_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields we care about from yt-dlp metadata."""
    return {
        "video_id": meta.get("id", ""),
        "title": meta.get("title", ""),
        "channel": meta.get("channel", meta.get("uploader", "")),
        "duration_seconds": int(meta.get("duration", 0) or 0),
        "view_count": int(meta.get("view_count", 0) or 0),
        "description": meta.get("description", ""),
    }


def process_query(
    query: str,
    seen_ids: set[str],
    max_results: int,
) -> list[VideoRecord]:
    """Process a single search query and return new VideoRecords."""
    search_results = search_youtube(query, max_results=max_results)
    records: list[VideoRecord] = []

    for entry in search_results:
        video_id = entry.get("id", entry.get("url", ""))
        if not video_id or video_id in seen_ids:
            continue

        # Check duration from search results first (flat playlist gives us this)
        duration = int(entry.get("duration", 0) or 0)
        if duration > 0 and (duration < MIN_DURATION or duration > MAX_DURATION):
            logger.debug("Skipping %s: duration %ds out of range", video_id, duration)
            continue

        seen_ids.add(video_id)

        # Get full metadata if search result was flat
        if "description" not in entry or entry.get("description") is None:
            full_meta = get_full_metadata(video_id)
            if full_meta is None:
                logger.warning("Could not fetch full metadata for %s, skipping", video_id)
                continue
            meta = extract_metadata_fields(full_meta)
        else:
            meta = extract_metadata_fields(entry)

        # Re-check duration with full metadata
        if meta["duration_seconds"] < MIN_DURATION or meta["duration_seconds"] > MAX_DURATION:
            logger.debug("Skipping %s: duration %ds out of range", video_id, meta["duration_seconds"])
            continue

        # Fetch transcript
        transcript_result = fetch_transcript(video_id)
        if transcript_result is None:
            logger.info("No transcript for %s (%s), skipping", video_id, meta["title"][:60])
            continue

        full_text, timestamped = transcript_result

        record = VideoRecord(
            video_id=meta["video_id"],
            title=meta["title"],
            channel=meta["channel"],
            duration_seconds=meta["duration_seconds"],
            view_count=meta["view_count"],
            description=meta["description"],
            transcript_text=full_text,
            transcript_with_timestamps=timestamped,
            search_query_source=query,
        )
        records.append(record)
        logger.info(
            "Scraped: %s (%s) [%dm%ds] - %d transcript segments",
            video_id,
            meta["title"][:50],
            meta["duration_seconds"] // 60,
            meta["duration_seconds"] % 60,
            len(timestamped),
        )

    return records


def load_existing_ids(output_path: Path) -> set[str]:
    """Load video IDs from an existing JSONL file for deduplication across runs."""
    ids: set[str] = set()
    if not output_path.exists():
        return ids
    with open(output_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                ids.add(record["video_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def run_pipeline(
    queries: list[str],
    output_path: Path,
    max_results: int = DEFAULT_MAX_RESULTS,
    append: bool = True,
) -> list[VideoRecord]:
    """Run the full scraping pipeline.

    Args:
        queries: list of YouTube search queries
        output_path: path to output JSONL file
        max_results: max results per query from YouTube search
        append: if True, append to existing file and skip already-seen IDs

    Returns:
        list of all new VideoRecords scraped
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    if append:
        seen_ids = load_existing_ids(output_path)
        if seen_ids:
            logger.info("Loaded %d existing video IDs for deduplication", len(seen_ids))

    all_records: list[VideoRecord] = []
    mode = "a" if append else "w"

    with open(output_path, mode) as f:
        for i, query in enumerate(queries):
            query = query.strip()
            if not query or query.startswith("#"):
                continue

            logger.info("--- Query %d/%d: %s ---", i + 1, len(queries), query)
            records = process_query(query, seen_ids, max_results)

            for record in records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

            all_records.extend(records)
            logger.info("Got %d new records for query: %s", len(records), query)

            # Be polite
            if i < len(queries) - 1:
                time.sleep(SLEEP_BETWEEN_QUERIES)

    return all_records


def load_queries(path: Path) -> list[str]:
    """Load search queries from a text file (one per line, # for comments)."""
    lines = path.read_text().strip().split("\n")
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scrape YouTube meditation transcripts for the CAT study",
    )
    parser.add_argument(
        "queries_file",
        type=Path,
        help="Path to text file with search queries (one per line)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("meditations.jsonl"),
        help="Output JSONL file path (default: meditations.jsonl)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Max results per search query (default: {DEFAULT_MAX_RESULTS})",
    )
    parser.add_argument(
        "--no-append",
        action="store_true",
        help="Overwrite output file instead of appending",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.queries_file.exists():
        logger.error("Queries file not found: %s", args.queries_file)
        return 1

    queries = load_queries(args.queries_file)
    if not queries:
        logger.error("No queries found in %s", args.queries_file)
        return 1

    logger.info("Loaded %d queries from %s", len(queries), args.queries_file)
    logger.info("Output: %s (append=%s)", args.output, not args.no_append)

    records = run_pipeline(
        queries=queries,
        output_path=args.output,
        max_results=args.max_results,
        append=not args.no_append,
    )

    logger.info("Done. Total new records: %d", len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
