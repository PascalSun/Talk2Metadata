
# Talk2Metadata

**Talk to your metadata** - An out-of-box system for semantic search over structured metadata with automatic schema detection, QA generation, and retrieval strategy optimization.

<div align="center">

<img src="docs/assets/favicon/android-chrome-512x512.png" alt="Talk2Metadata Logo" width="128" height="128">

</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

Talk2Metadata is an end-to-end system for semantic search over structured metadata. The system:

1. **Takes input**: A target table and related FK tables (CSV files or databases)
2. **Detects schema**: Automatically identifies star schema structure
3. **Generates QA**: Creates evaluation questions based on difficulty classification
4. **Evaluates strategies**: Tests different indexing and retrieval approaches
5. **Finds best solution**: Selects optimal strategy based on evaluation results

## Context Example

Given metadata with a star schema structure:

**Target table: `orders`**
```
| id  | customer_id | product_id | amount | status    |
| --- | ----------- | ---------- | ------ | --------- |
| 1   | 1           | 101        | 50000  | completed |
| 2   | 2           | 102        | 30000  | pending   |
```

**Related FK tables: `customers`, `products`**
```
customers:          products:
| id  | name     | industry    id  | name      | category |
| --- | -------- | --------------- | --------- | -------- |  |
| 1   | Acme     | Healthcare  101 | Analytics | Software |
| 2   | TechCorp | Technology  102 | Platform  | Software |
```

The system detects `orders.customer_id -> customers.id` and `orders.product_id -> products.id`, then enables queries like "Find orders from Healthcare customers buying Software products".

### Key Features

- Automatic star schema detection
- QA generation with difficulty classification
- Multiple retrieval strategies
- Automatic evaluation and optimization
- Supports CSV files and SQL databases

## Example

Given metadata with a target table `orders` and related tables `customers` and `products`:

```bash
# 1. Detect schema (finds: orders -> customers, orders -> products)
talk2metadata ingest csv data/raw --target orders

# 2. Generate QA pairs (e.g., "Find orders from Healthcare customers")
talk2metadata prepare

# 3. Build index and evaluate strategies
talk2metadata index --hybrid
talk2metadata evaluate

# 4. Search
talk2metadata search "orders from healthcare customers buying software"
```

## Quick Start

```bash
# 1. Use ./setup.sh to install dependencies and activate the virtual environment
./setup.sh
source .venv/bin/activate

# 2. Ingest metadata (CSV or database)
talk2metadata schema ingest csv data/raw --target orders

# 3. Generate QA pairs for evaluation
talk2metadata search prepare

# 4. Evaluate
talk2metadata search evaluate

# 5. Search
talk2metadata search "orders from healthcare customers buying software"

```

## Installation

**Unix/macOS/Linux/WSL:**
```bash
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata
./setup.sh
source .venv/bin/activate
```

**Windows:**
```cmd
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata
setup.bat
.venv\Scripts\activate.bat
```

**Manual Installation:**
```bash
uv sync
# or
pip install -e .
```

## Usage

### Complete Workflow

```bash
# 1. Ingest metadata (detects star schema automatically)
talk2metadata schema ingest csv data/raw --target orders

# 2. Generate QA pairs for evaluation
talk2metadata search prepare

# 3. Evaluate
talk2metadata search evaluate

# 4. Search
talk2metadata search "customers in healthcare"
```

### Python API

```python
from talk2metadata import Retriever

retriever = Retriever.from_paths(
    "data/indexes/index.faiss",
    "data/processed/tables.pkl"
)

results = retriever.search("healthcare customers", top_k=5)
for result in results:
    print(f"Rank {result.rank}: {result.data}")
```

## Documentation

Full documentation: [https://pascalsun.github.io/Talk2Metadata/](https://pascalsun.github.io/Talk2Metadata/)

## License

MIT License - see [LICENSE](LICENSE)
