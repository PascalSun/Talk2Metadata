# Performance Monitoring and Benchmarking

This guide covers performance monitoring and benchmarking tools for Talk2Metadata.

## Quick Start

### 1. Real-Time Monitoring (MCP Server)

While the MCP server is running, check current performance:

```bash
# JSON metrics
curl http://localhost:8000/metrics

# Prometheus format
curl http://localhost:8000/metrics/prometheus
```

### 2. Run Benchmarks

Test system performance with the `benchmark` command:

```bash
# Quick benchmark
talk2metadata benchmark

# Custom queries
talk2metadata benchmark -q "customer data" -q "order history" --num-runs 20

# Full benchmark with all tests
talk2metadata benchmark --include-hybrid --include-batch --include-cold-start

# Save results
talk2metadata benchmark --output results.json
```

**Options:**
- `-q, --queries`: Custom queries to test (can use multiple times)
- `-n, --num-runs`: Number of runs per query (default: 10)
- `-k, --top-k`: Number of results to return (default: 5)
- `--include-hybrid`: Test hybrid search (requires BM25 index)
- `--include-batch`: Test batch queries
- `--include-cold-start`: Measure initialization time
- `-o, --output`: Save JSON results to file

### 3. Analyze Logs

Parse historical logs to analyze performance:

```bash
# Basic analysis
talk2metadata analyze logs/mcp_server.log

# Generate histogram for specific operation
talk2metadata analyze logs/mcp_server.log --histogram query_encoding

# Save detailed report
talk2metadata analyze logs/mcp_server.log --output report.json

# JSON output
talk2metadata analyze logs/mcp_server.log --format json
```

**Options:**
- `-h, --histogram`: Generate histogram for specific operation
- `--bins`: Number of histogram bins (default: 20)
- `-o, --output`: Save detailed report to file
- `-f, --format`: Output format (text or json)

## Metrics Available

### Real-Time Metrics (`/metrics`)

```json
{
  "uptime_seconds": 3600.0,
  "total_requests": 1250,
  "requests_per_minute": 20.83,
  "error_rate": 0.0024,
  "latency_stats": {
    "tool.search": {
      "count": 1000,
      "mean_ms": 48.2,
      "p50_ms": 42.1,
      "p95_ms": 85.3,
      "p99_ms": 120.4
    },
    "query_encoding": {
      "mean_ms": 15.3,
      "p95_ms": 18.2
    },
    "faiss_search": {
      "mean_ms": 3.2,
      "p95_ms": 5.1
    }
  }
}
```

### Benchmark Results

```
==============================================================
PERFORMANCE BENCHMARK
==============================================================
Queries: 10
Runs per query: 10
Top-K: 5

📊 Running single query benchmark...
   Mean:   45.23ms
   Median: 42.10ms
   P95:    78.34ms
   P99:    95.12ms

📊 Component breakdown:
   query_encoding            :   15.34ms (n=100)
   faiss_search              :    3.21ms (n=100)
   result_formatting         :    0.85ms (n=100)
==============================================================
✅ Benchmark completed
==============================================================
```

### Log Analysis Report

```
================================================================================
LATENCY ANALYSIS REPORT
================================================================================

Log File: logs/mcp_server.log
Total Entries: 5234

--------------------------------------------------------------------------------
LATENCY STATISTICS BY OPERATION
--------------------------------------------------------------------------------

Operation                         Count       Mean     Median        P95        P99
--------------------------------------------------------------------------------
tool.search                        1000      48.23ms   42.10ms   78.34ms   95.12ms
query_encoding                     1000      15.34ms   14.80ms   18.23ms   22.14ms
faiss_search                       1000       3.21ms    2.95ms    5.12ms    8.34ms

--------------------------------------------------------------------------------
SLOW QUERIES
--------------------------------------------------------------------------------

Total slow queries: 25
Average duration: 125.45ms

Top 10 slowest queries:
  1. complex multi-table query...
     Duration: 250.12ms
     Threshold: 100.00ms
================================================================================
```

## What Gets Measured

### Query Pipeline

```
Request → tool.search (total: ~45ms)
├─ query_encoding (~15ms)      # Embedding generation
├─ faiss_search (~3ms)          # Vector search
└─ result_formatting (<1ms)     # JSON serialization
```

### Hybrid Search

```
Request → tool.search (hybrid, total: ~65ms)
├─ query_encoding (~15ms)
├─ faiss_search (~3ms)
├─ bm25_search (~12ms)
│  ├─ tokenization
│  ├─ scoring
│  └─ ranking
└─ fusion_rrf (~5ms)            # Result fusion
```

### Cold Start

```
First request
├─ index_load (2-3 seconds)     # FAISS + records
└─ model_load (included above)  # SentenceTransformer
```

## Performance Targets

| Metric | Target | Action |
|--------|--------|--------|
| P95 Latency | < 100ms | ✅ Acceptable |
| P99 Latency | < 200ms | ✅ Acceptable |
| Cold Start | < 3s | ✅ Acceptable |
| Error Rate | < 1% | Monitor |

## Best Practices

### During Development

1. Run benchmarks after code changes:
   ```bash
   talk2metadata benchmark --output before.json
   # ... make changes ...
   talk2metadata benchmark --output after.json
   # Compare results
   ```

2. Use DEBUG logging to see component timings:
   ```bash
   talk2metadata search "query" --log-level DEBUG
   ```

### In Production

1. Monitor live metrics:
   ```bash
   watch -n 2 'curl -s http://localhost:8000/metrics | jq ".latency_stats"'
   ```

2. Analyze logs periodically:
   ```bash
   talk2metadata analyze logs/production.log --output weekly-report.json
   ```

3. Set up Prometheus scraping:
   ```yaml
   scrape_configs:
     - job_name: 'talk2metadata'
       static_configs:
         - targets: ['localhost:8000']
       metrics_path: '/metrics/prometheus'
   ```

## Integration with Monitoring Tools

### Prometheus + Grafana

```promql
# Request rate
rate(talk2metadata_requests_total[5m])

# P95 latency
talk2metadata_latency_ms_tool_search{quantile="0.95"}

# Error rate
rate(talk2metadata_errors_total[5m]) / rate(talk2metadata_requests_total[5m])
```

## Troubleshooting

### High Latency

1. **Check component breakdown:**
   ```bash
   curl http://localhost:8000/metrics | jq '.latency_stats'
   ```

2. **Identify bottleneck:**
   - `query_encoding` high? → Model too large or CPU-bound
   - `faiss_search` high? → Index too large
   - `bm25_search` high? → Corpus too large

3. **Run targeted benchmark:**
   ```bash
   talk2metadata benchmark --num-runs 50 -q "slow query here"
   ```

### High Cold Start Time

1. **Measure cold start:**
   ```bash
   talk2metadata benchmark --include-cold-start
   ```

2. **Common causes:**
   - Large model download (first run)
   - Large index files
   - Slow disk I/O

## CLI Command Reference

### `talk2metadata benchmark`

Run performance benchmarks.

**Usage:**
```bash
talk2metadata benchmark [OPTIONS]
```

**Options:**
- `-q, --queries TEXT`: Queries to benchmark (multiple allowed)
- `-n, --num-runs INTEGER`: Runs per query (default: 10)
- `-k, --top-k INTEGER`: Results to return (default: 5)
- `-o, --output PATH`: Save JSON results
- `--include-hybrid`: Include hybrid search tests
- `--include-batch`: Include batch query tests
- `--include-cold-start`: Include cold start measurement

**Examples:**
```bash
# Quick benchmark
talk2metadata benchmark

# Custom test
talk2metadata benchmark -q "test1" -q "test2" --num-runs 20

# Full benchmark
talk2metadata benchmark --include-hybrid --include-batch --include-cold-start -o results.json
```

### `talk2metadata analyze`

Analyze latency from log files.

**Usage:**
```bash
talk2metadata analyze [OPTIONS] LOG_FILE
```

**Options:**
- `-o, --output PATH`: Save detailed JSON report
- `-h, --histogram TEXT`: Generate histogram for operation
- `--bins INTEGER`: Histogram bins (default: 20)
- `-f, --format [text|json]`: Output format (default: text)

**Examples:**
```bash
# Basic analysis
talk2metadata analyze logs/mcp_server.log

# Histogram
talk2metadata analyze logs/mcp_server.log --histogram query_encoding

# Save report
talk2metadata analyze logs/mcp_server.log --output report.json
```

## Legacy Scripts

The following scripts in `scripts/` directory are **deprecated**:
- ❌ `scripts/benchmark_latency.py` → Use `talk2metadata benchmark`
- ❌ `scripts/analyze_latency.py` → Use `talk2metadata analyze`

They are kept for backwards compatibility but will be removed in a future version.

---

For more information, see the main documentation at `docs/`.
