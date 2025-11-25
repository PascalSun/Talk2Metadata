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

## Difficulty Classification

Format: `{Pattern}{Difficulty}`

- **Pattern**: JOIN structure (0, 1p, 2p, 2i, 3p, 3i, 4i)
- **Difficulty**: Filter complexity (E, M, H)

### Examples

- `0E`: Direct query, Easy - "Find completed orders"
- `1pM`: Single-hop path, Medium - "Find orders from Healthcare customers with revenue>1M"
- `2iE`: Two-way star, Easy - "Find orders: Healthcare customers × Software products"

See [Difficulty Classification](./difficulty-classification.md) for complete specification.

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
```

### Batch Generation

```python
# Generate balanced dataset
distribution = {
    "0E": 50,
    "1pE": 100,
    "1pM": 80,
    "2iE": 30,
}

qa_pairs = qa_gen.generate_batch(distribution)
qa_gen.save(qa_pairs, "qa_dataset.jsonl")
```
