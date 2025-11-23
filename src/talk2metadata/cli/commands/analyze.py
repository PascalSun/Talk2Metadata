"""Analyze command for latency log analysis."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogEntry:
    """Parsed log entry."""

    timestamp: datetime | None
    level: str
    message: str
    operation: str | None = None
    duration_ms: float | None = None


@click.command(name="analyze")
@click.argument("log_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for detailed JSON report (optional)",
)
@click.option(
    "--histogram",
    "-h",
    help="Generate ASCII histogram for specific operation (e.g., 'query_encoding')",
)
@click.option(
    "--bins",
    type=int,
    default=20,
    help="Number of histogram bins (default: 20)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format (default: text)",
)
def analyze_cmd(log_file, output, histogram, bins, output_format):
    """Analyze latency from log files.

    LOG_FILE: Path to the log file to analyze

    \b
    Examples:
        # Basic analysis
        talk2metadata analyze logs/mcp_server.log

        # Generate histogram
        talk2metadata analyze logs/mcp_server.log --histogram query_encoding

        # Save detailed report
        talk2metadata analyze logs/mcp_server.log --output report.json

        # JSON output
        talk2metadata analyze logs/mcp_server.log --format json
    """
    log_path = Path(log_file)

    # Parse logs
    latencies, slow_queries, total_entries = _parse_log_file(log_path)

    if output_format == "text" and not histogram:
        click.echo()
        click.echo("=" * 80)
        click.echo("LATENCY ANALYSIS REPORT")
        click.echo("=" * 80)
        click.echo(f"\nLog File: {log_path}")
        click.echo(f"Total Entries: {total_entries}")
        click.echo(f"Analysis Time: {datetime.now().isoformat()}")

    # Generate report
    report = {
        "log_file": str(log_path),
        "total_entries": total_entries,
        "analysis_timestamp": datetime.now().isoformat(),
        "latency_stats": _analyze_latencies(latencies),
        "slow_queries": _analyze_slow_queries(slow_queries),
    }

    # Display results
    if histogram:
        # Generate histogram for specific operation
        if histogram not in latencies or not latencies[histogram]:
            click.echo(f"❌ No data found for operation: {histogram}", err=True)
            raise click.Abort()

        _print_histogram(histogram, latencies[histogram], bins)

    elif output_format == "json":
        # JSON output
        click.echo(json.dumps(report, indent=2))

    else:
        # Text output
        _print_report(report)

    # Save detailed report if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        if output_format == "text":
            click.echo(f"\n💾 Detailed report saved to: {output_path}")


def _parse_log_file(log_path: Path):
    """Parse log file and extract latency information."""
    latencies: Dict[str, List[float]] = defaultdict(list)
    slow_queries: List[Dict[str, Any]] = []
    total_entries = 0

    # Pattern for timing logs
    timing_pattern = re.compile(r"(\S+) completed in ([\d.]+)ms", re.IGNORECASE)

    # Pattern for structured JSON logs
    json_pattern = re.compile(r"\{.*\}")

    with open(log_path, "r") as f:
        for line in f:
            total_entries += 1

            # Check for timing information
            timing_match = timing_pattern.search(line)
            if timing_match:
                operation = timing_match.group(1)
                duration_ms = float(timing_match.group(2))
                latencies[operation].append(duration_ms)

            # Check for structured JSON
            json_match = json_pattern.search(line)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))

                    # Check for slow query event
                    if data.get("event") == "slow_query":
                        slow_queries.append(data)

                    # Extract timing from JSON
                    if "duration_ms" in data:
                        operation = data.get("event", "unknown")
                        duration_ms = data["duration_ms"]
                        latencies[operation].append(duration_ms)

                except json.JSONDecodeError:
                    pass

    return latencies, slow_queries, total_entries


def _analyze_latencies(latencies: Dict[str, List[float]]) -> Dict[str, Any]:
    """Analyze latency statistics."""
    analysis = {}

    for operation, timings in latencies.items():
        if not timings:
            continue

        sorted_timings = sorted(timings)
        n = len(sorted_timings)

        stats = {
            "count": n,
            "mean_ms": round(statistics.mean(timings), 3),
            "median_ms": round(statistics.median(timings), 3),
            "min_ms": round(min(timings), 3),
            "max_ms": round(max(timings), 3),
            "stddev_ms": round(statistics.stdev(timings), 3) if n > 1 else 0.0,
            "p95_ms": round(sorted_timings[int(n * 0.95)], 3) if n >= 20 else None,
            "p99_ms": round(sorted_timings[int(n * 0.99)], 3) if n >= 100 else None,
        }

        analysis[operation] = stats

    return analysis


def _analyze_slow_queries(slow_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze slow queries."""
    if not slow_queries:
        return {"count": 0, "queries": []}

    # Sort by duration
    sorted_queries = sorted(
        slow_queries, key=lambda x: x.get("duration_ms", 0), reverse=True
    )

    return {
        "count": len(slow_queries),
        "mean_duration_ms": round(
            statistics.mean(q.get("duration_ms", 0) for q in slow_queries), 3
        ),
        "queries": sorted_queries[:20],  # Top 20 slowest
    }


def _print_report(report: Dict[str, Any]) -> None:
    """Print formatted report."""
    click.echo()
    click.echo("-" * 80)
    click.echo("LATENCY STATISTICS BY OPERATION")
    click.echo("-" * 80)

    stats = report["latency_stats"]
    if stats:
        # Sort by mean latency
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]["mean_ms"], reverse=True)

        click.echo(
            f"\n{'Operation':<30} {'Count':>8} {'Mean':>10} {'Median':>10} {'P95':>10} {'P99':>10}"
        )
        click.echo("-" * 80)

        for op, data in sorted_ops:
            p95 = f"{data['p95_ms']:.2f}" if data["p95_ms"] else "N/A"
            p99 = f"{data['p99_ms']:.2f}" if data["p99_ms"] else "N/A"

            click.echo(
                f"{op:<30} {data['count']:>8} "
                f"{data['mean_ms']:>9.2f}ms {data['median_ms']:>9.2f}ms "
                f"{p95:>9}ms {p99:>9}ms"
            )
    else:
        click.echo("\nNo latency statistics found.")

    click.echo()
    click.echo("-" * 80)
    click.echo("SLOW QUERIES")
    click.echo("-" * 80)

    slow = report["slow_queries"]
    if slow["count"] > 0:
        click.echo(f"\nTotal slow queries: {slow['count']}")
        click.echo(f"Average duration: {slow['mean_duration_ms']:.2f}ms")
        click.echo(f"\nTop {min(10, len(slow['queries']))} slowest queries:")

        for i, q in enumerate(slow["queries"][:10], 1):
            query = q.get("query", "N/A")
            if len(query) > 50:
                query = query[:47] + "..."

            click.echo(
                f"\n  {i}. {query}\n"
                f"     Duration: {q.get('duration_ms', 0):.2f}ms\n"
                f"     Threshold: {q.get('threshold_ms', 0):.2f}ms"
            )
            if "details" in q:
                details = q["details"]
                click.echo(
                    f"     Top-k: {details.get('top_k', 'N/A')}, "
                    f"Hybrid: {details.get('hybrid', 'N/A')}, "
                    f"Results: {details.get('results_count', 'N/A')}"
                )
    else:
        click.echo("\nNo slow queries detected.")

    click.echo()
    click.echo("=" * 80)


def _print_histogram(operation: str, timings: List[float], bins: int) -> None:
    """Generate ASCII histogram for operation latency."""
    min_val = min(timings)
    max_val = max(timings)
    bin_width = (max_val - min_val) / bins if bins > 0 else 1

    # Create bins
    histogram = [0] * bins
    for t in timings:
        bin_idx = min(int((t - min_val) / bin_width), bins - 1)
        histogram[bin_idx] += 1

    # Print histogram
    max_count = max(histogram) if histogram else 1
    scale = 50 / max_count if max_count > 0 else 1

    click.echo()
    click.echo("=" * 80)
    click.echo(f"LATENCY DISTRIBUTION FOR '{operation}'")
    click.echo("=" * 80)
    click.echo(
        f"Total: {len(timings)}, Min: {min_val:.2f}ms, Max: {max_val:.2f}ms, "
        f"Mean: {statistics.mean(timings):.2f}ms"
    )
    click.echo("-" * 80)

    for i, count in enumerate(histogram):
        bin_start = min_val + i * bin_width
        bin_end = bin_start + bin_width
        bar = "#" * int(count * scale)
        click.echo(f"{bin_start:7.2f}-{bin_end:7.2f}ms [{count:4d}] {bar}")

    click.echo("=" * 80)
