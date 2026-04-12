#!/usr/bin/env python3
"""Aggregate analysis of CAT-scored meditation transcripts.

Reads scored JSONL and produces:
1. Score distributions per dimension (CSV + text summary)
2. Correlation matrix (CSV)
3. PCA analysis (variance explained, loadings)
4. Summary statistics for Jose's paper

Usage:
    python -m analysis.scripts.aggregate_report data/scored/full.jsonl -o analysis/results/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTINUOUS_DIMS = [
    "D1_attentional_constraint",
    "D2_somatic_engagement",
    "D5_object_density",
    "D6_temporal_dynamics",
    "D7_affective_cultivation",
    "D8_interoceptive_demand",
    "D9_metacognitive_load",
    "D10_relational_orientation",
]

SHORT_NAMES = {
    "D1_attentional_constraint": "D1_AC",
    "D2_somatic_engagement": "D2_SE",
    "D5_object_density": "D5_OD",
    "D6_temporal_dynamics": "D6_TD",
    "D7_affective_cultivation": "D7_AffC",
    "D8_interoceptive_demand": "D8_ID",
    "D9_metacognitive_load": "D9_ML",
    "D10_relational_orientation": "D10_RO",
}


def read_scored_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read scored JSONL, filtering to records with final_scores."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("final_scores"):
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def extract_score_matrix(records: list[dict[str, Any]]) -> list[dict[str, float | None]]:
    """Extract continuous dimension scores into a list of dicts."""
    matrix = []
    for rec in records:
        scores = rec.get("final_scores", {})
        row = {}
        for dim in CONTINUOUS_DIMS:
            val = scores.get(dim)
            row[dim] = float(val) if val is not None else None
        matrix.append(row)
    return matrix


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"n": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                "q25": 0, "median": 0, "q75": 0}

    n = len(values)
    sorted_v = sorted(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    std = var ** 0.5

    def percentile(data: list[float], p: float) -> float:
        k = (len(data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        return data[f] + (k - f) * (data[c] - data[f])

    return {
        "n": n,
        "mean": round(mean, 1),
        "std": round(std, 1),
        "min": round(sorted_v[0], 1),
        "q25": round(percentile(sorted_v, 0.25), 1),
        "median": round(percentile(sorted_v, 0.5), 1),
        "q75": round(percentile(sorted_v, 0.75), 1),
        "max": round(sorted_v[-1], 1),
    }


def compute_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation between two lists."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    sx = (sum((xi - mx) ** 2 for xi in x) / (n - 1)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y) / (n - 1)) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return float(round(cov / (sx * sy), 3))


def generate_distribution_summary(
    matrix: list[dict[str, float | None]],
    output_dir: Path,
) -> str:
    """Generate distribution stats per dimension."""
    lines = ["# Score Distributions", ""]

    csv_path = output_dir / "dimension_stats.csv"
    csv_rows = []

    for dim in CONTINUOUS_DIMS:
        values: list[float] = [v for row in matrix if (v := row[dim]) is not None]
        stats = compute_stats(values)

        short = SHORT_NAMES.get(dim, dim)
        lines.append(f"## {short}")
        lines.append(f"  n={stats['n']}  mean={stats['mean']}  std={stats['std']}")
        lines.append(f"  min={stats['min']}  Q25={stats['q25']}  median={stats['median']}  "
                      f"Q75={stats['q75']}  max={stats['max']}")
        lines.append("")

        csv_rows.append({"dimension": short, **stats})

    # Write CSV
    if csv_rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        lines.append(f"CSV: {csv_path}")

    return "\n".join(lines)


def generate_correlation_matrix(
    matrix: list[dict[str, float | None]],
    output_dir: Path,
) -> str:
    """Generate correlation matrix between continuous dimensions."""
    # Build paired value lists (only where both values exist)
    dim_values: dict[str, list[float]] = {dim: [] for dim in CONTINUOUS_DIMS}
    valid_indices: dict[tuple[str, str], list[int]] = {}

    for i, row in enumerate(matrix):
        for dim in CONTINUOUS_DIMS:
            val = row[dim]
            if val is not None:
                dim_values[dim].append(val)

    lines = ["# Correlation Matrix", ""]

    # Header
    short_names = [SHORT_NAMES.get(d, d) for d in CONTINUOUS_DIMS]
    header = "         " + "  ".join(f"{s:>8}" for s in short_names)
    lines.append(header)

    csv_path = output_dir / "correlation_matrix.csv"
    csv_rows = []

    for di, dim_i in enumerate(CONTINUOUS_DIMS):
        row_vals = {}
        corr_strs = []
        for dj, dim_j in enumerate(CONTINUOUS_DIMS):
            # Get paired values
            pairs_x = []
            pairs_y = []
            for row in matrix:
                xi = row[dim_i]
                xj = row[dim_j]
                if xi is not None and xj is not None:
                    pairs_x.append(xi)
                    pairs_y.append(xj)

            r = compute_correlation(pairs_x, pairs_y) if pairs_x else 0.0
            row_vals[SHORT_NAMES.get(dim_j, dim_j)] = r
            corr_strs.append(f"{r:>8.3f}")

        short_i = SHORT_NAMES.get(dim_i, dim_i)
        lines.append(f"{short_i:>8} " + "  ".join(corr_strs))
        csv_rows.append({"dimension": short_i, **row_vals})

    # Write CSV
    if csv_rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        lines.append(f"\nCSV: {csv_path}")

    return "\n".join(lines)


def generate_pca_analysis(
    matrix: list[dict[str, float | None]],
    output_dir: Path,
) -> str:
    """Simple PCA using eigendecomposition of correlation matrix.

    Uses only stdlib -- no numpy/scipy dependency. This is a basic
    implementation for initial exploration. For publication, use proper
    sklearn PCA.
    """
    # Filter to complete cases
    complete = [row for row in matrix if all(row[d] is not None for d in CONTINUOUS_DIMS)]

    if len(complete) < 10:
        return f"# PCA\nInsufficient complete cases ({len(complete)}). Need at least 10."

    n = len(complete)
    k = len(CONTINUOUS_DIMS)

    # Standardize (complete rows guaranteed to have no None values by the filter above)
    means: list[float] = []
    stds: list[float] = []
    for dim in CONTINUOUS_DIMS:
        vals: list[float] = [row[dim] for row in complete]  # type: ignore[misc]
        m = sum(vals) / n
        s = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5
        means.append(m)
        stds.append(s if s > 0 else 1.0)

    standardized: list[list[float]] = []
    for row in complete:
        z: list[float] = []
        for j, dim in enumerate(CONTINUOUS_DIMS):
            raw = row[dim]
            z.append((raw - means[j]) / stds[j])  # type: ignore[operator]
        standardized.append(z)

    # Compute correlation matrix (k x k)
    corr = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            xi = [standardized[r][i] for r in range(n)]
            xj = [standardized[r][j] for r in range(n)]
            corr[i][j] = sum(a * b for a, b in zip(xi, xj)) / (n - 1)

    # Note: proper eigendecomposition requires numpy.
    # For now, report the diagonal (variances) and off-diagonal structure.
    lines = ["# PCA (preliminary)", ""]
    lines.append(f"Complete cases: {n}")
    lines.append(f"Dimensions: {k}")
    lines.append("")
    lines.append("Note: Full PCA with eigendecomposition requires numpy/sklearn.")
    lines.append("Install with: pip install numpy scikit-learn")
    lines.append("Then use analysis/scripts/pca_full.py for proper analysis.")
    lines.append("")
    lines.append("Variance per dimension (standardized):")
    for j, dim in enumerate(CONTINUOUS_DIMS):
        short = SHORT_NAMES.get(dim, dim)
        lines.append(f"  {short}: {corr[j][j]:.3f}")

    lines.append("")
    lines.append("Strongest correlations (|r| > 0.3):")
    for i in range(k):
        for j in range(i + 1, k):
            if abs(corr[i][j]) > 0.3:
                si = SHORT_NAMES.get(CONTINUOUS_DIMS[i], CONTINUOUS_DIMS[i])
                sj = SHORT_NAMES.get(CONTINUOUS_DIMS[j], CONTINUOUS_DIMS[j])
                lines.append(f"  {si} <-> {sj}: r={corr[i][j]:.3f}")

    return "\n".join(lines)


def generate_categorical_summary(records: list[dict[str, Any]]) -> str:
    """Summarize categorical dimensions D3 and D4."""
    lines = ["# Categorical Dimensions", ""]

    # D3: Startup Modality
    d3_counts: dict[str, int] = {}
    for rec in records:
        scores = rec.get("final_scores", {})
        d3 = scores.get("D3_startup_modality", "unknown")
        d3_counts[d3] = d3_counts.get(d3, 0) + 1

    lines.append("## D3: Startup Modality")
    for cat, count in sorted(d3_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records) if records else 0
        lines.append(f"  {cat}: {count} ({pct:.0f}%)")
    lines.append("")

    # D4: Object Nature
    d4_counts: dict[str, int] = {}
    for rec in records:
        scores = rec.get("final_scores", {})
        d4 = scores.get("D4_object_nature", [])
        if isinstance(d4, str):
            d4 = [d4]
        for cat in d4:
            d4_counts[cat] = d4_counts.get(cat, 0) + 1

    lines.append("## D4: Object Nature (multi-valued)")
    for cat, count in sorted(d4_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records) if records else 0
        lines.append(f"  {cat}: {count} ({pct:.0f}%)")

    return "\n".join(lines)


def generate_metadata_summary(records: list[dict[str, Any]]) -> str:
    """Summarize dataset metadata."""
    lines = ["# Dataset Summary", ""]
    lines.append(f"Total meditations scored: {len(records)}")

    # Duration stats
    durations = [
        r.get("duration_seconds", 0) / 60
        for r in records
        if r.get("duration_seconds")
    ]
    if durations:
        stats = compute_stats(durations)
        lines.append(
            f"Duration (minutes): mean={stats['mean']}, "
            f"median={stats['median']}, range=[{stats['min']}, {stats['max']}]"
        )

    # Unique channels
    channels = set(r.get("channel", "unknown") for r in records)
    lines.append(f"Unique channels/teachers: {len(channels)}")

    # Search query distribution
    queries: dict[str, int] = {}
    for r in records:
        q = r.get("search_query_source", "unknown")
        queries[q] = queries.get(q, 0) + 1

    lines.append(f"\nSearch query sources ({len(queries)} unique):")
    for q, count in sorted(queries.items(), key=lambda x: -x[1])[:15]:
        lines.append(f"  {q}: {count}")

    # Scoring coverage
    pass2_count = sum(1 for r in records if r.get("llm_scores"))
    lines.append(f"\nPass 1 scored: {len(records)}")
    lines.append(f"Pass 2 (LLM) scored: {pass2_count}")

    return "\n".join(lines)


def write_scores_csv(records: list[dict[str, Any]], output_dir: Path) -> Path:
    """Write a flat CSV of all scores for easy analysis in R/pandas."""
    csv_path = output_dir / "all_scores.csv"

    fieldnames = [
        "video_id", "title", "channel", "duration_minutes",
        "search_query_source",
        "D1_AC", "D2_SE", "D3_startup", "D4_objects",
        "D5_OD", "D6_TD", "D7_AffC", "D8_ID", "D9_ML", "D10_RO",
        "scoring_method",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            scores = rec.get("final_scores", {})
            d4 = scores.get("D4_object_nature", [])
            if isinstance(d4, list):
                d4 = "|".join(d4)

            method = "pass2" if rec.get("llm_scores") else "pass1"

            writer.writerow({
                "video_id": rec.get("video_id", ""),
                "title": rec.get("title", ""),
                "channel": rec.get("channel", ""),
                "duration_minutes": round(rec.get("duration_seconds", 0) / 60, 1),
                "search_query_source": rec.get("search_query_source", ""),
                "D1_AC": scores.get("D1_attentional_constraint"),
                "D2_SE": scores.get("D2_somatic_engagement"),
                "D3_startup": scores.get("D3_startup_modality", ""),
                "D4_objects": d4,
                "D5_OD": scores.get("D5_object_density"),
                "D6_TD": scores.get("D6_temporal_dynamics"),
                "D7_AffC": scores.get("D7_affective_cultivation"),
                "D8_ID": scores.get("D8_interoceptive_demand"),
                "D9_ML": scores.get("D9_metacognitive_load"),
                "D10_RO": scores.get("D10_relational_orientation"),
                "scoring_method": method,
            })

    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CAT Aggregate Analysis")
    parser.add_argument("input", type=Path, help="Scored JSONL file")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Output directory for reports and CSVs")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = read_scored_jsonl(args.input)
    if not records:
        logger.error("No scored records found.")
        sys.exit(1)

    logger.info(f"Loaded {len(records)} scored records.")

    matrix = extract_score_matrix(records)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all reports
    sections = []
    sections.append(generate_metadata_summary(records))
    sections.append(generate_distribution_summary(matrix, output_dir))
    sections.append(generate_correlation_matrix(matrix, output_dir))
    sections.append(generate_pca_analysis(matrix, output_dir))
    sections.append(generate_categorical_summary(records))

    # Write combined text report
    report = "\n\n" + "=" * 60 + "\n\n".join(sections)
    report_path = output_dir / "aggregate_report.txt"
    report_path.write_text(report)
    logger.info(f"Report: {report_path}")

    # Write flat CSV
    csv_path = write_scores_csv(records, output_dir)
    logger.info(f"Scores CSV: {csv_path}")

    # Print summary to stdout
    print(report)


if __name__ == "__main__":
    main()
