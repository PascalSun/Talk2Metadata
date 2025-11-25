# Performance Monitoring

Performance monitoring and benchmarking tools for Talk2Metadata.

## Timing Utilities

Use timing decorators and context managers to track performance:

```python
from talk2metadata.utils.timing import timed, TimingContext

# Function timing
@timed("search_operation")
def search(query: str):
    # ... search logic
    pass

# Block timing
with TimingContext("indexing"):
    # ... indexing logic
    pass
```

## Metrics Collection

Collect and export metrics:

```python
from talk2metadata.utils.metrics import get_metrics_collector

collector = get_metrics_collector()
snapshot = collector.get_snapshot()

# Get statistics
print(f"Total requests: {snapshot.total_requests}")
print(f"Error rate: {snapshot.error_rate:.2%}")
print(f"Latency stats: {snapshot.latency_stats}")
```

## MCP Server Metrics

When the MCP server is running, access metrics via HTTP:

```bash
# JSON metrics
curl http://localhost:8010/metrics

# Prometheus format
curl http://localhost:8010/metrics/prometheus
```

## Performance Targets

| Metric | Target |
|--------|--------|
| P95 Latency | < 100ms |
| P99 Latency | < 200ms |
| Cold Start | < 3s |
