# Talk2Metadata

**Question-driven multi-table record retrieval system**

Talk2Metadata enables semantic search over relational databases using natural language queries. It automatically detects table relationships, creates rich embeddings with joined data, and provides fast similarity search using FAISS.

## Key Features

- 🔍 **Semantic Search**: Query your data using natural language instead of SQL
- 🔗 **Automatic FK Detection**: Infers foreign key relationships from data patterns
- 📊 **Multi-Table Support**: Automatically joins related tables for rich context
- 🚀 **Production-Ready**: Clean architecture, comprehensive testing, structured logging
- 🎯 **Lightweight**: No GPU required, runs efficiently on CPU
- 🔌 **Flexible Connectors**: Supports CSV files and SQL databases

## Quick Example

```bash
# 1. Ingest data from CSV files
talk2metadata ingest csv ./data/csv --target orders

# 2. Build search index
talk2metadata index

# 3. Search using natural language
talk2metadata search "healthcare customers with high revenue"
```

## How It Works

```mermaid
graph LR
    A[CSV/Database] --> B[Schema Detection]
    B --> C[FK Inference]
    C --> D[Denormalized Text]
    D --> E[Embeddings]
    E --> F[FAISS Index]
    F --> G[Semantic Search]
```

1. **Ingest**: Load data from CSV files or databases
2. **Detect**: Automatically detect schema and foreign key relationships
3. **Index**: Generate embeddings with joined table data
4. **Search**: Query using natural language to find relevant records

## Use Cases

- **Customer Support**: "Find all high-value customers in healthcare who had issues last month"
- **Sales Analysis**: "Show me recent large orders from technology companies"
- **Data Exploration**: "What products did customers in Europe purchase?"
- **Compliance**: "Find all transactions above $100k requiring approval"

## Architecture

Talk2Metadata is built with a modular architecture:

- **Core**: Schema detection, embedding generation, retrieval
- **Connectors**: Pluggable data sources (CSV, PostgreSQL, MySQL, SQLite)
- **CLI**: User-friendly command-line interface
- **API**: RESTful API for integration (optional)

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [REST API Reference](api-reference/rest-api.md)
