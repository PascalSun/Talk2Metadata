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

Talk2Metadata is built with a modular architecture organized around three core components:

### 🔗 [Schema Detection](schema/index.md)

Automatically understand your database structure:
- **Foreign Key Detection**: Rule-based and AI-powered relationship discovery
- **Primary Key Inference**: Identify primary keys from data patterns
- **Schema Metadata**: Extract comprehensive table and column information
- **Hybrid Approach**: Combine heuristics with LLM-based semantic analysis

### 🎯 [QA Generation](qa/index.md)

Generate training data for record localization:
- **Multi-Level Difficulty**: From Easy (0E) to Expert (4iH)
- **Pattern Types**: Support both chain (path) and star (intersection) queries
- **SQL + Ground Truth**: Generate questions with executable SQL and expected results
- **[Difficulty Classification](qa/difficulty-classification.md)**: Systematic framework based on Query Graph patterns

### 🚀 [Retriever](retriever/index.md)

Index and search database records semantically:
- **Denormalized Indexing**: Join related tables for rich context
- **Hybrid Search**: Combine semantic (vector) and keyword (BM25) retrieval
- **FAISS Integration**: Fast similarity search at scale
- **Multiple Strategies**: Semantic-only, keyword-only, or hybrid fusion

### 🔌 Additional Components

- **Connectors**: Pluggable data sources (CSV, PostgreSQL, MySQL, SQLite)
- **CLI**: User-friendly command-line interface
- **MCP Server**: Model Context Protocol server for AI agent integration

## Next Steps

### Getting Started
- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)

### Core Modules
- [Schema Detection Guide](schema/index.md) - Detect foreign keys and understand your schema
- [QA Generation Guide](qa/index.md) - Generate training datasets
- [Retriever Guide](retriever/index.md) - Index and search records

### Advanced Features
- [MCP Server Quick Start](mcp/quickstart.md)
- [Utilities & Tools](utils/index.md) - Configuration, logging, performance monitoring

### Monitoring & Performance
- [Performance Monitoring](utils/monitoring.md) - Benchmarking and latency analysis
