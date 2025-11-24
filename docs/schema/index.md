# Schema Detection

## Overview

The Schema module is responsible for detecting and understanding the structure of relational databases, including:

- **Foreign Key Detection**: Automatically identify relationships between tables
- **Primary Key Inference**: Detect primary keys when not explicitly defined
- **Schema Metadata**: Extract and store comprehensive table and column information
- **Data Type Detection**: Analyze column types and constraints

## Key Components

### 1. Foreign Key Detection

Talk2Metadata employs a hybrid approach for foreign key detection:

#### Rule-Based Detection

Heuristic methods that identify foreign keys through:
- **Naming patterns**: Columns ending with `_id`, `_key`, or matching table names
- **Inclusion dependencies**: Child table values are subset of parent table values
- **Coverage thresholds**: Configurable minimum coverage (default: 90%)
- **Data type matching**: Foreign key and primary key must have compatible types

#### Agent-Based Detection

LLM-powered detection for complex relationships:
- **Semantic analysis**: Understanding column name meanings
- **Value pattern recognition**: Identifying non-obvious relationships
- **Star schema awareness**: Prioritizing relationships to central/target tables
- **Fuzzy matching**: Handling singular/plural variations

### 2. Schema Metadata

Comprehensive metadata extraction including:
- **Tables**: Names, row counts, primary keys
- **Columns**: Names, types, nullable, unique constraints
- **Relationships**: Foreign keys with coverage metrics
- **Sample Data**: Representative values for semantic understanding

### 3. Configuration

Schema detection can be configured through:
```yaml
schema:
  fk_detection:
    method: "hybrid"  # "rule_based", "agent_based", or "hybrid"
    coverage_threshold: 0.9
    enable_fuzzy_matching: true

  agent:
    provider: "openai"  # or "anthropic", "gemini", etc.
    model: "gpt-4"
```

## Use Cases

### 1. CSV File Analysis

Automatically detect relationships in uploaded CSV files:
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

### 2. Database Connection

Connect to existing databases and extract schema:
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

# Detect schema (including existing FK constraints)
detector = SchemaDetector(tables)
schema = detector.detect()
```

### 3. Schema Validation

Verify and augment existing schema definitions:
```python
# Detect additional relationships not in schema
schema = detector.detect()

# Compare with database constraints
db_fks = connector.get_foreign_keys()
detected_fks = schema.foreign_keys

# Find missing relationships
missing = set(detected_fks) - set(db_fks)
print(f"Found {len(missing)} additional relationships")
```

## Architecture

```
Schema Detection Pipeline:

1. Data Loading
   ├─ CSV Files → CSVLoader
   └─ Databases → DBConnector

2. Metadata Extraction
   ├─ Table schemas
   ├─ Column types
   ├─ Primary keys (if available)
   └─ Sample data

3. Foreign Key Detection
   ├─ Rule-Based Analysis
   │  ├─ Name pattern matching
   │  ├─ Inclusion dependency check
   │  └─ Coverage calculation
   │
   └─ Agent-Based Analysis (optional)
      ├─ Semantic understanding
      ├─ Value pattern analysis
      └─ Relationship scoring

4. Schema Metadata Output
   ├─ TableMetadata (columns, types, PKs)
   ├─ ForeignKey (relationships, coverage)
   └─ SchemaMetadata (complete graph)
```

## Implementation Details

### Data Structures

**ForeignKey**
```python
@dataclass
class ForeignKey:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float  # 0.0 - 1.0
    detection_method: str  # "rule_based" or "agent_based"
```

**TableMetadata**
```python
@dataclass
class TableMetadata:
    table_name: str
    columns: List[str]
    column_types: Dict[str, str]
    primary_key: Optional[str]
    sample_values: Dict[str, List[Any]]
    row_count: int
```

**SchemaMetadata**
```python
@dataclass
class SchemaMetadata:
    tables: Dict[str, TableMetadata]
    foreign_keys: List[ForeignKey]
    target_table: Optional[str]
```

### Detection Algorithms

**Inclusion Dependency Check**
```python
def check_inclusion_dependency(
    child_values: Set[Any],
    parent_values: Set[Any],
    threshold: float = 0.9
) -> bool:
    """
    Check if child values are mostly contained in parent values
    """
    if not child_values:
        return False

    overlap = child_values & parent_values
    coverage = len(overlap) / len(child_values)

    return coverage >= threshold
```

**Name Pattern Matching**
```python
def match_fk_pattern(
    child_col: str,
    parent_table: str,
    parent_col: str
) -> bool:
    """
    Check if column names suggest FK relationship
    """
    # Handle singular/plural
    patterns = [
        f"{parent_table}_id",
        f"{parent_table}_key",
        f"{singularize(parent_table)}_id",
        parent_col  # Direct column name match
    ]

    return child_col.lower() in [p.lower() for p in patterns]
```

## Performance Considerations

### Optimization Strategies

1. **Sampling**: For large tables (>100K rows), sample data for FK detection
2. **Caching**: Cache detected schemas to avoid re-computation
3. **Parallel Processing**: Detect FKs for multiple table pairs in parallel
4. **Early Termination**: Skip pairs with incompatible data types

### Scalability

- **Small datasets** (<10 tables): Full analysis in seconds
- **Medium datasets** (10-50 tables): Minutes with hybrid approach
- **Large datasets** (50+ tables): Use rule-based only or selective agent calls

## Best Practices

### 1. Choose Detection Method

- **Rule-based only**: Fast, deterministic, good for well-structured schemas
- **Agent-based only**: Handles complex relationships, requires API costs
- **Hybrid** (recommended): Best balance of accuracy and speed

### 2. Set Appropriate Thresholds

```python
# Strict: Only high-confidence relationships
detector = SchemaDetector(tables, fk_coverage_threshold=0.95)

# Lenient: Capture more potential relationships
detector = SchemaDetector(tables, fk_coverage_threshold=0.80)
```

### 3. Validate Results

Always review detected foreign keys, especially for:
- Low coverage relationships (<90%)
- Agent-detected relationships
- Unexpected table connections

### 4. Provide Hints

Help the detector by providing:
```python
# Specify target table for star schema prioritization
detector = SchemaDetector(tables, target_table="orders")

# Provide known relationships
detector = SchemaDetector(
    tables,
    known_fks=[
        ForeignKey("orders", "customer_id", "customers", "id", 1.0, "manual")
    ]
)
```

## Troubleshooting

### Common Issues

**Issue**: Foreign keys not detected
- **Cause**: Low data coverage or non-standard naming
- **Solution**: Lower `fk_coverage_threshold` or use agent-based detection

**Issue**: Too many false positive FKs
- **Cause**: Columns with overlapping values (e.g., status codes)
- **Solution**: Increase threshold or add exclusion patterns

**Issue**: Agent-based detection is slow
- **Cause**: Too many table pairs, API rate limits
- **Solution**: Use rule-based as first pass, agent only for ambiguous cases

## Future Enhancements

- [ ] Composite foreign key detection
- [ ] Constraint detection (CHECK, UNIQUE)
- [ ] Schema versioning and evolution tracking
- [ ] Visual schema diagram generation
- [ ] Automatic denormalization recommendations

## Related Documentation

- [QA Generation](../qa/index.md) - Uses detected schema for question generation
- [Retriever](../retriever/index.md) - Leverages schema for efficient indexing
- [Getting Started](../getting-started/quickstart.md) - Quick setup guide

## API Reference

See `src/talk2metadata/core/schema.py` for detailed API documentation.
