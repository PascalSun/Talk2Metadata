"""Utility commands - agent, analyze, benchmark."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import click

from talk2metadata.core.modes import RecordVoter
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_indexes_dir
from talk2metadata.utils.timing import get_latency_tracker

logger = get_logger(__name__)


@click.group(name="utils")
def utils_group():
    """Utility commands for agents, analysis, and benchmarking.

    This command group provides tools for managing LLM agents,
    analyzing logs, and running performance benchmarks.
    """
    pass


# ============================================================================
# Agent commands
# ============================================================================


@utils_group.group(name="agent")
def agent_group():
    """Manage LLM agent providers and servers."""
    pass


@agent_group.command(name="vllm-server")
@click.option(
    "--model",
    "-m",
    help="Model name/identifier (overrides config.yml)",
)
@click.option(
    "--host",
    help="Host to bind the server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    help="Port to bind the server to (default: 8000)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    help="Number of tensor parallel replicas",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    help="Fraction of GPU memory to use (0.0 to 1.0)",
)
@click.option(
    "--max-model-len",
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"], case_sensitive=False),
    help="Data type for model weights",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Trust remote code when loading model",
)
@click.option(
    "--download-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Directory to download and load model",
)
@click.option(
    "--api-key",
    help="API key for authentication (optional)",
)
@click.option(
    "--served-model-name",
    help="Model name to serve (defaults to model name)",
)
def vllm_server_cmd(
    model: str | None,
    host: str | None,
    port: int | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    dtype: str | None,
    trust_remote_code: bool,
    download_dir: str | None,
    api_key: str | None,
    served_model_name: str | None,
):
    """Start a vLLM OpenAI-compatible API server.

    This command starts a vLLM server that provides an OpenAI-compatible API
    for local high-performance LLM inference.

    Model and port are read from config.yml (agent.vllm.model and agent.vllm.base_url)
    if not specified via command-line options.

    \b
    Examples:
        # Start server using model from config.yml
        talk2metadata utils agent vllm-server

        # Override model from command line
        talk2metadata utils agent vllm-server --model meta-llama/Llama-2-7b-chat-hf

        # Start server on custom port
        talk2metadata utils agent vllm-server --port 8080

        # Start server with GPU memory limit
        talk2metadata utils agent vllm-server --gpu-memory-utilization 0.9

        # Start server with tensor parallelism
        talk2metadata utils agent vllm-server --tensor-parallel-size 2
    """
    try:
        # Check if vllm is installed
        import vllm  # noqa: F401
    except ImportError:
        click.echo(
            "❌ vLLM is not installed.\n" "   Install it with: pip install vllm",
            err=True,
        )
        sys.exit(1)

    # Load config
    config = get_config()
    agent_config = config.get("agent", {})
    vllm_config = agent_config.get("vllm", {})

    # Resolve model: command-line > config.vllm.model > config.model
    if model is None:
        model = vllm_config.get("model") or agent_config.get("model")
        if model is None:
            click.echo(
                "❌ Model not specified and not found in config.yml.\n"
                "   Please specify --model or set agent.vllm.model in config.yml",
                err=True,
            )
            sys.exit(1)

    # Resolve host and port from base_url if provided
    if host is None:
        host = "0.0.0.0"  # Default host

    if port is None:
        # Try to extract port from base_url
        base_url = vllm_config.get("base_url", "")
        if base_url:
            try:
                parsed = urlparse(base_url)
                if parsed.port:
                    port = parsed.port
                elif parsed.scheme == "http":
                    port = 80
                elif parsed.scheme == "https":
                    port = 443
                else:
                    port = 8000
            except Exception:
                port = 8000
        else:
            port = 8000

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Add optional arguments
    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if download_dir:
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--download-dir", str(download_path)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    click.echo("🚀 Starting vLLM server...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Endpoint: http://{host}:{port}/v1")
    click.echo(f"\n   Command: {' '.join(cmd)}\n")

    try:
        # Run the server (this will block)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        click.echo(f"\n❌ Server failed to start: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Analyze command
# ============================================================================


@dataclass
class LogEntry:
    """Parsed log entry."""

    timestamp: datetime | None
    level: str
    message: str
    operation: str | None = None
    duration_ms: float | None = None


@utils_group.command(name="analyze")
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
        talk2metadata utils analyze logs/mcp_server.log

        # Generate histogram
        talk2metadata utils analyze logs/mcp_server.log --histogram query_encoding

        # Save detailed report
        talk2metadata utils analyze logs/mcp_server.log --output report.json

        # JSON output
        talk2metadata utils analyze logs/mcp_server.log --format json
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


# ============================================================================
# Benchmark command
# ============================================================================

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


@utils_group.command(name="benchmark")
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
        talk2metadata utils benchmark

        # Custom queries
        talk2metadata utils benchmark -q "customer search" -q "order history"

        # Full benchmark with all tests
        talk2metadata utils benchmark --num-runs 20 --include-batch

        # Save results to file
        talk2metadata utils benchmark --output benchmark_results.json
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
                "   Please run 'talk2metadata search prepare' first.",
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
