# QA Generation

## Overview

The QA (Question-Answer) Generation module creates synthetic question-answer pairs for training and evaluating record localization systems. Each QA pair consists of:

1. **Natural language question**: Human-readable query
2. **SQL answer**: Executable SQL query to retrieve records
3. **Ground truth records**: Expected result set from the database

## Key Features

- **Multi-level difficulty**: Generate questions from Easy (0E) to Expert (4iH)
- **Schema-aware**: Leverages detected foreign key relationships
- **Diverse patterns**: Supports both chain (path) and star (intersection) queries
- **Controllable generation**: Configure difficulty distribution and question types
- **Quality validation**: Automatic verification of SQL and ground truth consistency

## Architecture

```
QA Generation Pipeline:

1. Schema Analysis
   ├─ Detect foreign keys
   ├─ Identify star/chain patterns
   └─ Extract table/column metadata

2. Question Template Selection
   ├─ Choose difficulty level (0E - 4iH)
   ├─ Select pattern type (path/star)
   └─ Determine filter complexity

3. Question Generation
   ├─ Natural language generation
   ├─ SQL query construction
   └─ Parameter instantiation

4. Answer Generation
   ├─ Execute SQL query
   ├─ Collect ground truth records
   └─ Validate result consistency

5. Quality Assurance
   ├─ Check SQL validity
   ├─ Verify non-empty results
   └─ Validate difficulty classification
```

## Difficulty Classification

Talk2Metadata uses a systematic difficulty classification based on Query Graph patterns.

### Format: `{Pattern}{Difficulty}`

- **Pattern**: JOIN structure (0, 1p, 2p, 2i, 3p, 3i, 4i)
- **Difficulty**: Filter complexity (E, M, H)

### Quick Reference

| Code | Description | Example |
|------|-------------|---------|
| **0E** | Direct, Easy | "Find completed orders" |
| **1pM** | 1-hop path, Medium | "Find orders from Healthcare customers with revenue>1M" |
| **2pE** | 2-hop chain, Easy | "Find orders from US-West region customers" |
| **2iE** | 2-way star, Easy | "Find orders: Healthcare customers × Software products" |
| **3iM** | 3-way star, Medium | "Find orders: Healthcare customers × Enterprise software × Senior sales" |

**See [Difficulty Classification](./difficulty-classification.md) for complete specification.**

## Usage

### Basic QA Generation

```python
from talk2metadata.qa import QAGenerator
from talk2metadata.core.schema import SchemaDetector

# Load and detect schema
schema = SchemaDetector(tables, target_table="orders").detect()

# Initialize QA generator
qa_gen = QAGenerator(schema=schema)

# Generate questions at specific difficulty
questions = qa_gen.generate(
    difficulty="2iE",
    num_questions=10
)

for qa in questions:
    print(f"Q: {qa.question}")
    print(f"SQL: {qa.sql}")
    print(f"Records: {len(qa.ground_truth)} results")
```

### Batch Generation with Distribution

```python
# Generate balanced dataset
distribution = {
    "0E": 50,   # 50 easy direct queries
    "1pE": 100, # 100 easy 1-hop
    "1pM": 80,
    "2iE": 30,
    "2iM": 20,
}

qa_pairs = qa_gen.generate_batch(distribution)

# Save to file
qa_gen.save(qa_pairs, "qa_dataset.jsonl")
```

### Custom Templates

```python
# Define custom question templates
templates = {
    "1pE": [
        "Find {target} from {foreign_table} with {condition}",
        "Show me all {target} where {foreign_table} has {condition}",
    ],
    "2iE": [
        "Locate {target} matching {condition1} and {condition2}",
    ]
}

qa_gen = QAGenerator(schema=schema, templates=templates)
```

## Question Templates

### Template Structure

Each template is a pattern with placeholders:
```
"Find {target} from {table} where {column} {operator} {value}"
```

### Placeholder Types

- `{target}`: Target table name (e.g., "orders")
- `{table}`: Foreign table name (e.g., "customers")
- `{column}`: Column name (e.g., "industry")
- `{operator}`: Comparison operator (=, >, <, LIKE, IN)
- `{value}`: Concrete value from sample data

### Pattern-Specific Templates

**Direct (0)**
```
"Find {target} with {column} = '{value}'"
"Show me all {target} where {column} > {value}"
"List {target} having {condition1} and {condition2}"
```

**Single-hop Path (1p)**
```
"Find {target} from {foreign_table} in {value}"
"Show {target} where {foreign_table} has {column} = '{value}'"
"Get all {target} associated with {foreign_table} matching {condition}"
```

**Two-hop Path (2p)**
```
"Find {target} from {table1} in {table2} where {condition}"
"Locate {target} connected to {table2} via {table1}"
```

**Two-way Intersection (2i)**
```
"Find {target} where {table1} has {condition1} and {table2} has {condition2}"
"Show {target} matching both {table1} criteria and {table2} criteria"
```

## SQL Generation

### Pattern-Specific SQL Templates

**Direct (0)**
```sql
SELECT * FROM {target}
WHERE {column} {operator} {value}
```

**Single-hop Path (1p)**
```sql
SELECT t.*
FROM {target} t
JOIN {foreign_table} f ON t.{fk_column} = f.{pk_column}
WHERE f.{filter_column} {operator} {value}
```

**Two-hop Path (2p)**
```sql
SELECT t.*
FROM {target} t
JOIN {table1} t1 ON t.{fk1} = t1.{pk1}
JOIN {table2} t2 ON t1.{fk2} = t2.{pk2}
WHERE t2.{filter_column} {operator} {value}
```

**Two-way Intersection (2i)**
```sql
SELECT t.*
FROM {target} t
JOIN {table1} t1 ON t.{fk1} = t1.{pk1}
JOIN {table2} t2 ON t.{fk2} = t2.{pk2}
WHERE t1.{col1} {op1} {val1}
  AND t2.{col2} {op2} {val2}
```

### SQL Validation

Generated SQL is validated for:
- **Syntax correctness**: Parse and validate SQL
- **Table existence**: All referenced tables exist in schema
- **Column validity**: Columns exist and have correct types
- **JOIN correctness**: Foreign key relationships are valid
- **Non-empty results**: Query returns at least one record

## Ground Truth Generation

### Execution

```python
# Execute SQL and collect results
ground_truth = qa_gen.execute_sql(sql, max_results=100)

# Store as QA pair
qa_pair = QAPair(
    question=question,
    sql=sql,
    ground_truth=ground_truth,
    difficulty="2iE",
    schema_hash=schema.hash()
)
```

### Ground Truth Format

```json
{
  "question": "Find orders from Healthcare customers buying Software products",
  "sql": "SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id JOIN products p ON o.product_id = p.id WHERE c.industry = 'Healthcare' AND p.category = 'Software'",
  "ground_truth": [
    {
      "id": 1001,
      "customer_id": 1,
      "product_id": 101,
      "amount": 50000,
      "status": "completed"
    },
    {
      "id": 1005,
      "customer_id": 3,
      "product_id": 103,
      "amount": 75000,
      "status": "pending"
    }
  ],
  "difficulty": "2iE",
  "num_results": 2,
  "metadata": {
    "pattern": "2i",
    "tables_involved": ["orders", "customers", "products"],
    "filter_columns": ["customers.industry", "products.category"]
  }
}
```

### Result Validation

- **Consistency**: SQL execution produces expected number of results
- **Completeness**: All required fields are present
- **Correctness**: Results match filter criteria
- **Uniqueness**: No duplicate QA pairs (by SQL or question hash)

## Quality Control

### Automatic Checks

1. **SQL Validity**
   - Syntax check
   - Table/column existence
   - JOIN path validity

2. **Result Quality**
   - Non-empty results (configurable minimum)
   - Reasonable result count (not too many/few)
   - Diverse value distributions

3. **Question Quality**
   - No placeholder leakage (e.g., "{table}")
   - Natural language fluency
   - Unambiguous phrasing

4. **Difficulty Accuracy**
   - Pattern matches actual SQL structure
   - Column count matches difficulty level
   - Classification is consistent

### Manual Review

For production datasets, consider:
- **Sample review**: Manually check 10% of generated QA pairs
- **Edge cases**: Review highest/lowest difficulty questions
- **Diversity**: Ensure coverage of all tables and columns

## Dataset Distribution

### Recommended Distributions

**Balanced Dataset** (for general training)
```python
distribution = {
    "0E": 150,   # 15%
    "0M": 100,   # 10%
    "0H": 50,    # 5%
    "1pE": 200,  # 20%
    "1pM": 200,  # 20%
    "1pH": 100,  # 10%
    "2pE": 30,   # 3%
    "2pM": 30,   # 3%
    "2iE": 30,   # 3%
    "2iM": 30,   # 3%
    "2iH": 30,   # 3%
    "3iE": 20,   # 2%
    "3iM": 20,   # 2%
    "3iH": 10,   # 1%
}
# Total: 1000 questions
```

**Progressive Dataset** (for curriculum learning)
```python
# Phase 1: Easy
phase1 = {"0E": 400, "0M": 200, "1pE": 200, "1pM": 200}

# Phase 2: Medium
phase2 = {"1pM": 300, "1pH": 200, "2pE": 200, "2iE": 200, "0H": 100}

# Phase 3: Hard
phase3 = {"2pM": 200, "2iM": 200, "2iH": 200, "3iE": 200, "3iM": 100, "3iH": 100}
```

## Configuration

### Generator Configuration

```yaml
qa_generation:
  # Target table for record localization
  target_table: "orders"

  # Difficulty distribution
  difficulty_distribution:
    0E: 0.15
    0M: 0.10
    1pE: 0.20
    1pM: 0.20
    2iE: 0.10
    # ... etc

  # Generation parameters
  max_attempts_per_question: 5
  min_results_per_question: 1
  max_results_per_question: 100

  # Template settings
  use_custom_templates: false
  template_file: null

  # Validation
  validate_sql: true
  validate_results: true
  allow_empty_results: false

  # Output
  output_format: "jsonl"  # or "json", "csv"
  include_metadata: true
```

## Advanced Features

### Constraint-Based Generation

Generate questions with specific constraints:
```python
# Only generate questions involving specific tables
qa_gen.generate(
    difficulty="2iE",
    required_tables=["customers", "products"],
    num_questions=50
)

# Avoid certain columns
qa_gen.generate(
    difficulty="1pM",
    excluded_columns=["internal_id", "created_at"],
    num_questions=30
)
```

### Temporal Awareness

Generate time-based questions:
```python
# Questions involving date ranges
qa_gen.generate_temporal(
    difficulty="1pM",
    date_column="order_date",
    num_questions=20
)
# Output: "Find orders from Healthcare customers placed after 2024-01-01"
```

### Multi-Turn Questions

Generate follow-up questions:
```python
# Initial question
q1 = "Find all Healthcare customers"

# Follow-up question
q2 = qa_gen.generate_followup(
    previous_question=q1,
    difficulty="1pM"
)
# Output: "Among those Healthcare customers, which have revenue > $1M?"
```

## Best Practices

### 1. Start Simple

Begin with direct queries (0E) and gradually increase complexity:
```python
for difficulty in ["0E", "1pE", "2iE", "3iE"]:
    qa_gen.generate(difficulty=difficulty, num_questions=10)
```

### 2. Validate Generated Data

Always check generated QA pairs:
```python
qa_pairs = qa_gen.generate(difficulty="2iM", num_questions=100)

# Check for issues
issues = qa_gen.validate_batch(qa_pairs)
if issues:
    print(f"Found {len(issues)} invalid QA pairs")
```

### 3. Balance Your Dataset

Ensure coverage across difficulty levels and patterns:
```python
stats = qa_gen.analyze_distribution(qa_pairs)
print(f"Pattern distribution: {stats['patterns']}")
print(f"Difficulty distribution: {stats['difficulties']}")
```

### 4. Use Real Data Distributions

Sample filter values from actual data:
```python
# This is automatic - values come from schema.sample_values
# But you can provide custom value distributions:
qa_gen = QAGenerator(
    schema=schema,
    value_distributions={
        "customers.industry": ["Healthcare", "Finance", "Technology"],
        "products.category": ["Software", "Hardware", "Services"]
    }
)
```

## Troubleshooting

### Common Issues

**Issue**: Generated SQL fails to execute
- **Cause**: Invalid JOIN paths or missing foreign keys
- **Solution**: Verify schema detection, check FK relationships

**Issue**: Empty result sets
- **Cause**: Filter values don't exist in actual data
- **Solution**: Use `schema.sample_values` or real data distributions

**Issue**: Questions are too similar
- **Cause**: Limited template diversity
- **Solution**: Add more templates or use LLM-based generation

**Issue**: Difficulty classification is incorrect
- **Cause**: Mismatch between actual SQL and difficulty
- **Solution**: Enable `validate_difficulty` in config

## Future Enhancements

- [ ] LLM-based question generation (more natural phrasing)
- [ ] Paraphrase generation (multiple questions for same SQL)
- [ ] Negative examples (questions with no valid answers)
- [ ] Aggregation queries (COUNT, SUM, AVG)
- [ ] Subquery support
- [ ] Multi-table result joins

## Related Documentation

- **[Difficulty Classification](./difficulty-classification.md)** - Complete difficulty specification
- **[Schema Detection](../schema/index.md)** - Foreign key detection
- **[Retriever](../retriever/index.md)** - Using QA pairs for evaluation

## API Reference

See `src/talk2metadata/qa/` for detailed implementation (to be developed).
