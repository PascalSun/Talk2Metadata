"""Benchmark command for performance testing."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import click

from talk2metadata.core.modes import RecordVoter
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_indexes_dir
from talk2metadata.utils.timing import get_latency_tracker

logger = get_logger(__name__)

# Default benchmark queries
DEFAULT_QUERIES = [
    "customer information",
    "order details",
    "product catalog",
    "shipping address",
    "payment method",
    "user account",
    "sales data",
    "inventory status",
    "contact information",
    "transaction history",
]


@click.command(name="benchmark")
@click.option(
    "--queries",
    "-q",
    multiple=True,
    help="Queries to benchmark (can specify multiple times). Uses defaults if not provided.",
)
@click.option(
    "--num-runs",
    "-n",
    type=int,
    default=10,
    help="Number of runs per query (default: 10)",
)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for JSON results (optional)",
)
@click.option(
    "--include-batch",
    is_flag=True,
    help="Include batch query benchmarks",
)
@click.option(
    "--include-cold-start",
    is_flag=True,
    help="Include cold start benchmarks (requires reloading)",
)
@click.pass_context
def benchmark_cmd(
    ctx,
    queries,
    num_runs,
    top_k,
    output,
    include_batch,
    include_cold_start,
):
    """Run performance benchmarks for the retrieval system.

    \b
    Examples:
        # Quick benchmark with defaults
        talk2metadata benchmark

        # Custom queries
        talk2metadata benchmark -q "customer search" -q "order history"

        # Full benchmark with all tests
        talk2metadata benchmark --num-runs 20 --include-hybrid --include-batch

        # Save results to file
        talk2metadata benchmark --output benchmark_results.json
    """
    config = get_config()
    run_id = config.get("run_id")
    index_dir = get_indexes_dir(run_id, config)

    # Validate index exists (multi-table index for RecordVoter)
    schema_path = index_dir / "schema_metadata.json"
    if not schema_path.exists():
        from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

        metadata_dir = get_metadata_dir(run_id, config)
        schema_path = find_schema_file(metadata_dir)
        if not schema_path or not Path(schema_path).exists():
            click.echo(
                f"❌ Index not found at {index_dir}\n"
                "   Please run 'talk2metadata index' first.",
                err=True,
            )
            raise click.Abort()

    # Use provided queries or defaults
    query_list = list(queries) if queries else DEFAULT_QUERIES

    click.echo("=" * 60)
    click.echo("PERFORMANCE BENCHMARK")
    click.echo("=" * 60)
    click.echo(f"Queries: {len(query_list)}")
    click.echo(f"Runs per query: {num_runs}")
    click.echo(f"Top-K: {top_k}")
    click.echo()

    # Reset tracker for clean measurements
    get_latency_tracker().reset()

    results = {
        "timestamp": time.time(),
        "config": {
            "num_queries": len(query_list),
            "num_runs": num_runs,
            "top_k": top_k,
        },
        "benchmarks": {},
    }

    # 1. Cold Start Benchmark (optional)
    if include_cold_start:
        click.echo("📊 Running cold start benchmark...")
        cold_start_results = _benchmark_cold_start(config, index_dir, schema_path)
        results["benchmarks"]["cold_start"] = cold_start_results
        click.echo(f"   RecordVoter: {cold_start_results['load_ms']:.1f}ms")
        click.echo()

    # 2. Single Query Benchmark
    click.echo("📊 Running single query benchmark...")
    single_results = _benchmark_single_queries(
        index_dir, schema_path, query_list, top_k, num_runs
    )
    results["benchmarks"]["single_queries"] = single_results
    click.echo(f"   Mean:   {single_results['latencies']['mean_ms']:.2f}ms")
    click.echo(f"   Median: {single_results['latencies']['median_ms']:.2f}ms")
    click.echo(f"   P95:    {single_results['latencies']['p95_ms']:.2f}ms")
    click.echo(f"   P99:    {single_results['latencies']['p99_ms']:.2f}ms")
    click.echo()

    # 3. Batch Query Benchmark (optional)
    if include_batch:
        click.echo("📊 Running batch query benchmark...")
        batch_results = _benchmark_batch_queries(
            index_dir, schema_path, query_list, top_k, max(5, num_runs // 2)
        )
        results["benchmarks"]["batch_queries"] = batch_results
        click.echo(
            f"   Batch total: {batch_results['batch_latencies']['mean_ms']:.2f}ms"
        )
        click.echo(
            f"   Per query:   {batch_results['per_query_latencies']['mean_ms']:.2f}ms"
        )
        click.echo()

    # 5. Component Breakdown
    click.echo("📊 Component breakdown:")
    component_stats = get_latency_tracker().get_stats()
    results["component_breakdown"] = component_stats

    for component, stats in component_stats.items():
        if stats["count"] > 0:
            click.echo(
                f"   {component:25s}: {stats['mean_ms']:7.2f}ms (n={stats['count']})"
            )

    click.echo()
    click.echo("=" * 60)
    click.echo("✅ Benchmark completed")
    click.echo("=" * 60)

    # Save results if output path provided
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\n💾 Results saved to: {output_path}")


def _benchmark_cold_start(config, index_dir, schema_path):
    """Benchmark cold start latency."""
    start = time.perf_counter()
    _ = RecordVoter.from_paths(index_dir, schema_path)
    load_time = (time.perf_counter() - start) * 1000

    return {
        "load_ms": round(load_time, 3),
    }


def _benchmark_single_queries(index_dir, schema_path, queries, top_k, num_runs):
    """Benchmark single query latency."""
    retriever = RecordVoter.from_paths(index_dir, schema_path)
    all_latencies = []

    for query in queries:
        for _ in range(num_runs):
            start = time.perf_counter()
            retriever.search(query, top_k=top_k)
            duration_ms = (time.perf_counter() - start) * 1000
            all_latencies.append(duration_ms)

    return {
        "num_queries": len(queries),
        "num_runs": num_runs,
        "top_k": top_k,
        "latencies": _calculate_stats(all_latencies),
    }


def _benchmark_batch_queries(index_dir, schema_path, queries, top_k, num_runs):
    """Benchmark batch query latency (sequential)."""
    retriever = RecordVoter.from_paths(index_dir, schema_path)
    batch_latencies = []

    for _ in range(num_runs):
        start = time.perf_counter()
        # RecordVoter doesn't have search_batch, so simulate it
        for query in queries:
            retriever.search(query, top_k=top_k)
        duration_ms = (time.perf_counter() - start) * 1000
        batch_latencies.append(duration_ms)

    per_query_latencies = [lat / len(queries) for lat in batch_latencies]

    return {
        "batch_size": len(queries),
        "num_runs": num_runs,
        "top_k": top_k,
        "batch_latencies": {
            "mean_ms": round(statistics.mean(batch_latencies), 3),
            "median_ms": round(statistics.median(batch_latencies), 3),
            "min_ms": round(min(batch_latencies), 3),
            "max_ms": round(max(batch_latencies), 3),
        },
        "per_query_latencies": {
            "mean_ms": round(statistics.mean(per_query_latencies), 3),
            "median_ms": round(statistics.median(per_query_latencies), 3),
        },
    }


def _calculate_stats(latencies):
    """Calculate latency statistics."""
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    return {
        "mean_ms": round(statistics.mean(latencies), 3),
        "median_ms": round(statistics.median(latencies), 3),
        "min_ms": round(min(latencies), 3),
        "max_ms": round(max(latencies), 3),
        "stddev_ms": round(statistics.stdev(latencies), 3) if n > 1 else 0.0,
        "p95_ms": round(sorted_latencies[int(n * 0.95)], 3) if n >= 20 else None,
        "p99_ms": round(sorted_latencies[int(n * 0.99)], 3) if n >= 100 else None,
    }
