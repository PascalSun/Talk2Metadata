# Schema Detection

## Overview

The Schema module detects and understands the structure of relational databases, including:

- **Foreign Key Detection**: Automatically identify relationships between tables
- **Primary Key Inference**: Detect primary keys when not explicitly defined
- **Schema Metadata**: Extract and store comprehensive table and column information

## Key Components

### Foreign Key Detection

Talk2Metadata uses a hybrid approach:

#### Rule-Based Detection

Heuristic methods that identify foreign keys through:

- **Naming patterns**: Columns ending with `_id`, `_key`, or matching table names
- **Inclusion dependencies**: Child table values are subset of parent table values
- **Coverage thresholds**: Configurable minimum coverage (default: 90%)

#### Agent-Based Detection

LLM-powered detection for complex relationships:

- **Semantic analysis**: Understanding column name meanings
- **Value pattern recognition**: Identifying non-obvious relationships

## Usage

### CSV File Analysis

```python
from talk2metadata.connectors import CSVLoader
from talk2metadata.core.schema import SchemaDetector

# Load CSV files
loader = CSVLoader(data_dir="./data")
tables = loader.load()

# Detect schema
detector = SchemaDetector(tables, target_table="orders")
schema = detector.detect()

print(f"Detected {len(schema.foreign_keys)} foreign keys")
```

### Database Connection

```python
from talk2metadata.connectors import DBConnector
from talk2metadata.core.schema import SchemaDetector

# Connect to database
connector = DBConnector(
    db_type="postgresql",
    host="localhost",
    database="sales_db"
)
tables = connector.load()

# Detect schema
detector = SchemaDetector(tables)
schema = detector.detect()
```

## Configuration

```yaml
schema:
  fk_detection:
    method: "hybrid"  # "rule_based", "agent_based", or "hybrid"
    coverage_threshold: 0.9
```
