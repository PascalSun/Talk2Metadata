
# Talk2Metadata

**Talk to your metadata**


<div align="center">

<img src="assets/favicon/android-chrome-512x512.png" alt="Talk2Metadata Logo" width="128" height="128">

</div>

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
| id  | name | industry    id | name | category |
| --- | ---- | -------------- | ---- | -------- ||
| 1   | Acme     | Healthcare  101 | Analytics | Software |
| 2   | TechCorp | Technology  102 | Platform  | Software |
```

The system detects `orders.customer_id -> customers.id` and `orders.product_id -> products.id`, then enables queries like "Find orders from Healthcare customers buying Software products".

## Key Features

- Automatic star schema detection
- QA generation with difficulty classification
- Multiple retrieval strategies
- Automatic evaluation and optimization

## Example

Given metadata with a target table `orders` and related tables `customers` and `products`:

```bash
# 1. Detect schema (finds: orders -> customers, orders -> products)
talk2metadata schema ingest csv data/raw --target orders

# 2. Generate QA pairs (e.g., "Find orders from Healthcare customers")
talk2metadata search prepare


# 3. Evaluate
talk2metadata search evaluate

# 3. or directly search
talk2metadata search "orders from healthcare customers buying software"
```


## Core Modules

### 🔗 [Schema Detection](schema/index.md)
Automatically detect foreign key relationships and understand database structure.

### 🚀 [Retriever](retriever/index.md)
Index and search database records semantically with hybrid search support.

### 🎯 [QA Generation](qa/index.md)
Generate training data for record localization with multi-level difficulty.

## Getting Started

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)

## Advanced Features

- [MCP Server Quick Start](mcp/quickstart.md)
- [Utilities & Tools](utils/index.md)
