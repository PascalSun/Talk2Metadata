# Utilities Documentation

This section contains documentation for the utility modules in `src/talk2metadata/utils/`.

## Available Utilities

### 📊 [Performance Monitoring](monitoring.md)
Comprehensive latency monitoring, benchmarking, and performance analysis tools.

**Key Features:**
- Real-time latency tracking
- Performance benchmarking CLI (`talk2metadata benchmark`)
- Log analysis CLI (`talk2metadata analyze`)
- Prometheus metrics export
- Slow query detection

**Modules:**
- `utils/timing.py` - Timing decorators and context managers
- `utils/metrics.py` - Metrics collection and export

---

### 🔧 Configuration (`config.py`)

Configuration management for Talk2Metadata.

**Key Functions:**
```python
from talk2metadata.utils.config import get_config, load_config

# Get global config
config = get_config()

# Load custom config
config = load_config("path/to/config.yml")
```

**Features:**
- YAML configuration loading
- Environment variable support
- Default values
- Nested configuration access

**Configuration Structure:**
```yaml
data:
  raw_dir: "./data/raw"
  processed_dir: "./data/processed"
  indexes_dir: "./data/indexes"
  metadata_dir: "./data/metadata"

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: null  # auto-detect
  batch_size: 32
  normalize: true

retrieval:
  top_k: 5
  use_hybrid_search: false
  similarity_metric: "cosine"

logging:
  level: "INFO"
  file: null
```

---

### 📝 Logging (`logging.py`)

Structured logging setup and utilities.

**Key Functions:**
```python
from talk2metadata.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(level="INFO", log_file="logs/app.log")

# Get logger for module
logger = get_logger(__name__)

# Use logger
logger.info("Starting process")
logger.debug("Debug details")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

**Features:**
- Structured logging with context
- File and console output
- Configurable log levels
- Per-module loggers
- Exception stack traces

---

### 📂 Paths (`paths.py`)

Path utilities for data directories and file discovery.

**Key Functions:**
```python
from talk2metadata.utils.paths import (
    get_raw_dir,
    get_processed_dir,
    get_indexes_dir,
    get_metadata_dir,
    find_schema_file,
)

# Get standard directories
raw_dir = get_raw_dir()
processed_dir = get_processed_dir()
indexes_dir = get_indexes_dir()

# Find schema file
schema_path = find_schema_file(metadata_dir)
```

**Features:**
- Automatic directory creation
- Run-specific paths (with run_id)
- Schema file discovery
- Path validation

---

## Quick Reference

### Performance Monitoring

```bash
# Real-time metrics (MCP server running)
curl http://localhost:8000/metrics

# Run benchmarks
talk2metadata benchmark --num-runs 20

# Analyze logs
talk2metadata analyze logs/mcp_server.log
```

### Configuration

```python
# Access config
from talk2metadata.utils.config import get_config
config = get_config()
model = config.get("embedding.model_name")
```

### Logging

```python
# Setup and use
from talk2metadata.utils.logging import setup_logging, get_logger

setup_logging(level="DEBUG")
logger = get_logger(__name__)
logger.info("Message here")
```

### Paths

```python
# Get directories
from talk2metadata.utils.paths import get_indexes_dir
index_dir = get_indexes_dir()
```

---

## Module Organization

```
src/talk2metadata/utils/
├── __init__.py
├── config.py          # Configuration management
├── logging.py         # Logging utilities
├── paths.py           # Path utilities
├── timing.py          # Performance timing (NEW)
└── metrics.py         # Metrics collection (NEW)
```

---

## Related Documentation

- [Performance Monitoring Guide](monitoring.md) - Detailed monitoring documentation
- [Getting Started](../getting-started/quickstart.md) - Basic setup
- [MCP Integration](../mcp/quickstart.md) - MCP server usage

---

## Examples

### Complete Performance Monitoring Setup

```python
# setup.py
from talk2metadata.utils.logging import setup_logging, get_logger
from talk2metadata.utils.config import get_config
from talk2metadata.utils.timing import timed, TimingContext
from talk2metadata.utils.metrics import get_metrics_collector

# Setup
setup_logging(level="INFO", log_file="logs/app.log")
logger = get_logger(__name__)
config = get_config()

# Use timing decorator
@timed("my_function")
def process_data():
    with TimingContext("step_1"):
        # ... do work
        pass

    with TimingContext("step_2"):
        # ... do more work
        pass

# Run and check metrics
process_data()

collector = get_metrics_collector()
snapshot = collector.get_snapshot()
print(snapshot.to_dict())
```

### Custom Configuration

```python
# Load custom config
from talk2metadata.utils.config import load_config

config = load_config("custom_config.yml")
model = config.get("embedding.model_name", "default-model")
```

### Structured Logging

```python
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# With context
logger.info(
    "Query executed",
    extra={
        "query": "customer search",
        "results_count": 10,
        "duration_ms": 45.2
    }
)
```

---

## Best Practices

### Performance Monitoring

1. **Use decorators for functions:**
   ```python
   @timed("operation_name")
   def my_function():
       pass
   ```

2. **Use context managers for blocks:**
   ```python
   with TimingContext("block_name"):
       # code here
       pass
   ```

3. **Check metrics regularly:**
   ```bash
   curl http://localhost:8000/metrics
   ```

### Configuration

1. **Use environment variables for secrets:**
   ```yaml
   # config.yml
   database:
     password: ${DB_PASSWORD}
   ```

2. **Provide defaults:**
   ```python
   value = config.get("key.path", default_value)
   ```

### Logging

1. **Use appropriate levels:**
   - DEBUG: Detailed diagnostic info
   - INFO: General information
   - WARNING: Warning messages
   - ERROR: Error messages

2. **Include context:**
   ```python
   logger.error("Failed to process", extra={"user_id": 123})
   ```

3. **Use exc_info for exceptions:**
   ```python
   try:
       risky_operation()
   except Exception as e:
       logger.error("Operation failed", exc_info=True)
   ```

---

## Troubleshooting

### Config not loading
```python
# Check config location
from talk2metadata.utils.config import get_config
config = get_config()
print(config._config_path)  # Shows loaded config path
```

### Logging not appearing
```python
# Ensure setup is called
from talk2metadata.utils.logging import setup_logging
setup_logging(level="DEBUG")  # Make sure this is called early
```

### Metrics not tracking
```python
# Check if timing is enabled
from talk2metadata.utils.timing import get_latency_tracker
tracker = get_latency_tracker()
print(tracker.get_stats())  # Should show recorded operations
```

---

**Last Updated:** 2025-11-23
