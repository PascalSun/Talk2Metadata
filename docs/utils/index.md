# Utilities

Documentation for utility modules in `src/talk2metadata/utils/`.

## Configuration (`config.py`)

Configuration management with YAML support:

```python
from talk2metadata.utils.config import get_config, load_config

# Get global config (loads from config.yml or uses defaults)
config = get_config()

# Access nested config values
model = config.get("embedding.model_name")
top_k = config.get("retrieval.top_k", 5)  # with default

# Load custom config
config = load_config("path/to/config.yml")
```

## Logging (`logging.py`)

Structured logging setup:

```python
from talk2metadata.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", log_file="logs/app.log")

# Get logger for module
logger = get_logger(__name__)
logger.info("Starting process")
logger.debug("Debug details")
```

## Paths (`paths.py`)

Path utilities with run_id support:

```python
from talk2metadata.utils.paths import (
    get_metadata_dir,
    get_processed_dir,
    get_indexes_dir,
    get_qa_dir,
    get_benchmark_dir,
    get_db_dir,
    find_schema_file,
)

# Get directories (supports run_id)
metadata_dir = get_metadata_dir(run_id="wamex_run")
indexes_dir = get_indexes_dir(run_id="wamex_run")

# Find schema file
schema_path = find_schema_file(metadata_dir, target_table="orders")
```

## Timing (`timing.py`)

Performance timing utilities:

```python
from talk2metadata.utils.timing import timed, TimingContext, get_latency_tracker

# Decorator for function timing
@timed("my_function")
def process_data():
    pass

# Context manager for block timing
with TimingContext("indexing"):
    # ... do work
    pass

# Get timing statistics
tracker = get_latency_tracker()
stats = tracker.get_stats("indexing")
print(f"Mean: {stats.mean_ms}ms, P95: {stats.p95_ms}ms")
```

## Metrics (`metrics.py`)

Metrics collection and export:

```python
from talk2metadata.utils.metrics import get_metrics_collector

collector = get_metrics_collector()
snapshot = collector.get_snapshot()

# Export as JSON
print(snapshot.to_dict())
```

## CSV to Database (`csv_to_db.py`)

Convert CSV files to SQLite database:

```python
from talk2metadata.utils.csv_to_db import create_sqlite_from_csv
from pathlib import Path

# Create SQLite database from CSV files
db_path = create_sqlite_from_csv(
    csv_data_dir=Path("data/raw"),
    run_id="wamex_run",
    schema_metadata=schema  # Optional: adds FK constraints
)
```
