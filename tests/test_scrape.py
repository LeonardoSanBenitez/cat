"""Tests for scrape_meditations.py -- all external calls are mocked."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from scrape_meditations import (
    DEFAULT_MAX_RESULTS,
    MAX_DURATION,
    MIN_DURATION,
    VideoRecord,
    extract_metadata_fields,
    load_existing_ids,
    load_queries,
    process_query,
    run_pipeline,
    search_youtube,
)


# ---------- Fixtures ----------

def _make_yt_meta(
    video_id: str = "abc123",
    title: str = "Test Meditation",
    channel: str = "MeditationChannel",
    duration: int = 900,  # 15 minutes
    view_count: int = 50000,
    description: str = "A guided meditation",
) -> dict[str, Any]:
    return {
        "id": video_id,
        "title": title,
        "channel": channel,
        "duration": duration,
        "view_count": view_count,
        "description": description,
    }


def _make_transcript_segments() -> list[Any]:
    """Simulate transcript segments as returned by youtube_transcript_api."""

    class FakeSegment:
        def __init__(self, start: float, duration: float, text: str) -> None:
            self.start = start
            self.duration = duration
            self.text = text

        def get(self, key: str, default: Any = None) -> Any:
            return getattr(self, key, default)

    return [
        FakeSegment(0.0, 5.0, "Welcome to this meditation"),
        FakeSegment(5.0, 5.0, "Find a comfortable position"),
        FakeSegment(10.0, 5.0, "Close your eyes"),
    ]


# ---------- Tests: extract_metadata_fields ----------

class TestExtractMetadataFields:
    def test_basic(self) -> None:
        meta = _make_yt_meta()
        result = extract_metadata_fields(meta)
        assert result["video_id"] == "abc123"
        assert result["title"] == "Test Meditation"
        assert result["channel"] == "MeditationChannel"
        assert result["duration_seconds"] == 900
        assert result["view_count"] == 50000

    def test_missing_fields(self) -> None:
        result = extract_metadata_fields({})
        assert result["video_id"] == ""
        assert result["duration_seconds"] == 0
        assert result["view_count"] == 0

    def test_none_duration(self) -> None:
        meta = _make_yt_meta()
        meta["duration"] = None
        result = extract_metadata_fields(meta)
        assert result["duration_seconds"] == 0

    def test_uploader_fallback(self) -> None:
        meta = {"id": "x", "uploader": "FallbackChannel"}
        result = extract_metadata_fields(meta)
        assert result["channel"] == "FallbackChannel"


# ---------- Tests: VideoRecord ----------

class TestVideoRecord:
    def test_url_auto_generated(self) -> None:
        r = VideoRecord(
            video_id="xyz",
            title="Test",
            channel="Ch",
            duration_seconds=600,
            view_count=100,
            description="",
            transcript_text="hello",
            transcript_with_timestamps=[],
            search_query_source="test query",
        )
        assert r.url == "https://www.youtube.com/watch?v=xyz"

    def test_url_explicit(self) -> None:
        r = VideoRecord(
            video_id="xyz",
            title="Test",
            channel="Ch",
            duration_seconds=600,
            view_count=100,
            description="",
            transcript_text="hello",
            transcript_with_timestamps=[],
            search_query_source="test query",
            url="https://custom.url",
        )
        assert r.url == "https://custom.url"


# ---------- Tests: load_queries ----------

class TestLoadQueries:
    def test_loads_queries(self, tmp_path: Path) -> None:
        f = tmp_path / "queries.txt"
        f.write_text("# comment\nquery one\n\nquery two\n# another comment\n")
        result = load_queries(f)
        assert result == ["query one", "query two"]

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("# only comments\n")
        result = load_queries(f)
        assert result == []


# ---------- Tests: load_existing_ids ----------

class TestLoadExistingIds:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = load_existing_ids(tmp_path / "nope.jsonl")
        assert result == set()

    def test_loads_ids(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"video_id": "aaa"}) + "\n"
            + json.dumps({"video_id": "bbb"}) + "\n"
        )
        result = load_existing_ids(f)
        assert result == {"aaa", "bbb"}

    def test_skips_bad_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "out.jsonl"
        f.write_text("not json\n" + json.dumps({"video_id": "ok"}) + "\n")
        result = load_existing_ids(f)
        assert result == {"ok"}


# ---------- Tests: search_youtube ----------

class TestSearchYoutube:
    @mock.patch("scrape_meditations.subprocess.run")
    def test_basic_search(self, mock_run: mock.Mock) -> None:
        meta1 = _make_yt_meta(video_id="v1")
        meta2 = _make_yt_meta(video_id="v2")
        mock_run.return_value = mock.Mock(
            returncode=0,
            stdout=json.dumps(meta1) + "\n" + json.dumps(meta2) + "\n",
            stderr="",
        )
        results = search_youtube("test query", max_results=5)
        assert len(results) == 2
        assert results[0]["id"] == "v1"

    @mock.patch("scrape_meditations.subprocess.run")
    def test_timeout(self, mock_run: mock.Mock) -> None:
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("yt-dlp", 120)
        results = search_youtube("test")
        assert results == []

    @mock.patch("scrape_meditations.subprocess.run")
    def test_failure(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = mock.Mock(returncode=1, stdout="", stderr="error")
        results = search_youtube("test")
        assert results == []


# ---------- Tests: process_query ----------

class TestProcessQuery:
    @mock.patch("scrape_meditations.fetch_transcript")
    @mock.patch("scrape_meditations.get_full_metadata")
    @mock.patch("scrape_meditations.search_youtube")
    def test_full_pipeline(
        self,
        mock_search: mock.Mock,
        mock_meta: mock.Mock,
        mock_transcript: mock.Mock,
    ) -> None:
        mock_search.return_value = [
            _make_yt_meta(video_id="v1", duration=900),
        ]
        mock_meta.return_value = _make_yt_meta(video_id="v1", duration=900)
        mock_transcript.return_value = (
            "Welcome Find Close",
            [{"start": 0, "duration": 5, "text": "Welcome"}],
        )

        seen: set[str] = set()
        records = process_query("test", seen, max_results=5)
        assert len(records) == 1
        assert records[0].video_id == "v1"
        assert "v1" in seen

    @mock.patch("scrape_meditations.fetch_transcript")
    @mock.patch("scrape_meditations.search_youtube")
    def test_skips_duplicate(
        self,
        mock_search: mock.Mock,
        mock_transcript: mock.Mock,
    ) -> None:
        mock_search.return_value = [_make_yt_meta(video_id="v1", duration=900)]
        seen: set[str] = {"v1"}
        records = process_query("test", seen, max_results=5)
        assert len(records) == 0
        mock_transcript.assert_not_called()

    @mock.patch("scrape_meditations.search_youtube")
    def test_skips_too_short(self, mock_search: mock.Mock) -> None:
        mock_search.return_value = [_make_yt_meta(video_id="v1", duration=60)]
        records = process_query("test", set(), max_results=5)
        assert len(records) == 0

    @mock.patch("scrape_meditations.search_youtube")
    def test_skips_too_long(self, mock_search: mock.Mock) -> None:
        mock_search.return_value = [_make_yt_meta(video_id="v1", duration=3600)]
        records = process_query("test", set(), max_results=5)
        assert len(records) == 0

    @mock.patch("scrape_meditations.fetch_transcript")
    @mock.patch("scrape_meditations.get_full_metadata")
    @mock.patch("scrape_meditations.search_youtube")
    def test_skips_no_transcript(
        self,
        mock_search: mock.Mock,
        mock_meta: mock.Mock,
        mock_transcript: mock.Mock,
    ) -> None:
        mock_search.return_value = [_make_yt_meta(video_id="v1", duration=900)]
        mock_meta.return_value = _make_yt_meta(video_id="v1", duration=900)
        mock_transcript.return_value = None

        records = process_query("test", set(), max_results=5)
        assert len(records) == 0


# ---------- Tests: run_pipeline ----------

class TestRunPipeline:
    @mock.patch("scrape_meditations.process_query")
    def test_writes_jsonl(self, mock_pq: mock.Mock, tmp_path: Path) -> None:
        record = VideoRecord(
            video_id="v1",
            title="Med",
            channel="Ch",
            duration_seconds=900,
            view_count=100,
            description="desc",
            transcript_text="hello world",
            transcript_with_timestamps=[{"start": 0, "duration": 5, "text": "hello world"}],
            search_query_source="test",
        )
        mock_pq.return_value = [record]

        out = tmp_path / "out.jsonl"
        results = run_pipeline(["test query"], out, max_results=5)

        assert len(results) == 1
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["video_id"] == "v1"
        assert parsed["transcript_text"] == "hello world"

    @mock.patch("scrape_meditations.process_query")
    def test_append_mode(self, mock_pq: mock.Mock, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        out.write_text(json.dumps({"video_id": "existing"}) + "\n")

        mock_pq.return_value = []
        run_pipeline(["test"], out, append=True)

        # The mock receives the seen_ids set -- check it was called with existing IDs
        call_args = mock_pq.call_args
        seen_ids = call_args[0][1]  # second positional arg
        assert "existing" in seen_ids

    @mock.patch("scrape_meditations.process_query")
    def test_skips_comments(self, mock_pq: mock.Mock, tmp_path: Path) -> None:
        mock_pq.return_value = []
        out = tmp_path / "out.jsonl"
        run_pipeline(["# comment", "", "real query"], out)
        # Only "real query" should trigger process_query
        assert mock_pq.call_count == 1


# ---------- Tests: CLI (main) ----------

class TestMain:
    @mock.patch("scrape_meditations.run_pipeline")
    def test_cli_basic(self, mock_pipeline: mock.Mock, tmp_path: Path) -> None:
        from scrape_meditations import main

        mock_pipeline.return_value = []
        queries_file = tmp_path / "q.txt"
        queries_file.write_text("test query\n")
        out_file = tmp_path / "out.jsonl"

        result = main([str(queries_file), "-o", str(out_file)])
        assert result == 0
        mock_pipeline.assert_called_once()

    def test_cli_missing_file(self, tmp_path: Path) -> None:
        from scrape_meditations import main
        result = main([str(tmp_path / "nope.txt")])
        assert result == 1
