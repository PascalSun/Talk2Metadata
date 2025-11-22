# Talk2Metadata

**Question-driven multi-table record retrieval system** - Find relevant database rows using natural language queries.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

Talk2Metadata enables semantic search over relational databases. It automatically detects table relationships, creates rich embeddings with joined data, and provides fast similarity search using natural language.

### Key Features

- 🔍 **Semantic Search**: Query using natural language instead of SQL
- ⚡ **Hybrid Search**: Combines BM25 + semantic search for better results
- 🔗 **Auto FK Detection**: Infers foreign key relationships from data
- 📊 **Multi-Table Support**: Joins related tables for rich context
- 🎨 **Schema Visualization**: Interactive HTML visualization of FK relationships
- ✅ **Schema Validation**: Review and validate schemas before indexing
- 🚀 **Production-Ready**: Clean architecture, tested, documented
- 🎯 **Lightweight**: No GPU required, runs on CPU
- 🔌 **Flexible**: Supports CSV files and SQL databases

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Ingest sample data
uv run talk2metadata ingest csv data/raw --target orders

# 3. Build search index
uv run talk2metadata index

# 4. Search!
uv run talk2metadata search "healthcare customers with high revenue"
```

## Hybrid Search

Talk2Metadata supports **hybrid search** that combines two complementary approaches:

1. **BM25** (keyword-based): Finds exact term matches, good for specific entities
2. **Semantic** (embedding-based): Understands meaning, good for conceptual queries

The hybrid mode uses **Reciprocal Rank Fusion (RRF)** by default to intelligently combine results, providing better accuracy than either method alone.

```bash
# Build hybrid index
talk2metadata index --hybrid

# Use hybrid search
talk2metadata search "healthcare customers" --hybrid
```

**Benefits:**
- Better recall for specific terms (IDs, names, codes)
- Better precision for semantic queries
- More robust to query variations

## Installation

### Quick Setup (Recommended)

**Unix/macOS/Linux/WSL:**
```bash
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Run setup script - it will guide you through options
./setup.sh          # Basic installation
./setup.sh --mcp    # With MCP server support

# Activate environment
source .venv/bin/activate
```

**Windows:**
```cmd
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Run setup script
setup.bat          # Basic installation
setup.bat --mcp    # With MCP server support

# Activate environment
.venv\Scripts\activate.bat
```

The setup script automatically:
- ✅ Checks Python version (3.11+)
- ✅ Installs uv package manager
- ✅ Creates virtual environment
- ✅ Installs Talk2Metadata with dependencies
- ✅ Creates project directories (data, logs, examples)
- ✅ Copies configuration templates

### Manual Installation

```bash
# Using uv (recommended)
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata
uv sync

# Or with pip
pip install -e .

# With specific features
pip install -e ".[mcp]"        # MCP server
```

## Usage Examples

### CLI

```bash
# Ingest from CSV
talk2metadata ingest csv ./data/raw --target orders

# Ingest from database
talk2metadata ingest database "postgresql://localhost/mydb" --target orders

# Review and validate schema (before indexing)
talk2metadata schema --validate
talk2metadata schema --visualize  # Generate HTML visualization

# Build index (semantic only)
talk2metadata index

# Build hybrid index (BM25 + semantic, better quality)
talk2metadata index --hybrid

# Search (semantic)
talk2metadata search "customers in healthcare"

# Search (hybrid - better results)
talk2metadata search "customers in healthcare" --hybrid

# Advanced search options
talk2metadata search "pending orders" --top-k 10 --format json
talk2metadata search "high value orders" --hybrid --show-score
```

### Python API

```python
from talk2metadata import Retriever

# Load retriever
retriever = Retriever.from_paths(
    "data/indexes/index.faiss",
    "data/processed/tables.pkl"
)

# Search
results = retriever.search("healthcare customers", top_k=5)

for result in results:
    print(f"Rank {result.rank}: {result.data}")
```

## Architecture

```
┌─────────────────┐
│ CSV / Database  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Schema Detection│ ← Automatic FK inference
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Denormalized    │ ← Join related tables
│ Text Generation │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embeddings      │ ← sentence-transformers
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ FAISS Index     │ ← Fast similarity search
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Search │
└─────────────────┘
```

## Project Structure

```
Talk2Metadata/
├── src/Talk2Metadata/
│   ├── core/              # Schema, indexing, retrieval
│   ├── connectors/        # CSV and database connectors
│   ├── cli/               # Command-line interface
│   ├── mcp/               # MCP server
│   └── utils/             # Config, logging
├── data/                  # Sample data
│   ├── raw/              # CSV files
│   ├── processed/        # Processed data
│   └── indexes/          # FAISS indexes
├── examples/             # Usage examples
├── docs/                 # Documentation
└── tests/                # Test suite
```

## Documentation

Full documentation available at: [https://pascalsun.github.io/Talk2Metadata/](https://pascalsun.github.io/Talk2Metadata/)

## Testing

Run the complete workflow test:

```bash
./scripts/test_workflow.sh
```

Or run unit tests:

```bash
uv run pytest
```

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/

# Run tests
pytest --cov=talk2metadata
```

## Sample Data

The repository includes sample data with 3 tables:
- **customers**: 10 customers from various industries
- **products**: 10 products (software and services)
- **orders**: 20 orders linking customers to products

Try these queries:
- "healthcare customers with high revenue"
- "orders from technology companies"
- "machine learning products"
- "pending orders in US-West region"

## Performance

- Small datasets (<100K rows): <1s search latency
- Embedding generation: ~1000 rows/second (CPU)
- Index size: ~4 bytes/dimension/vector

## Roadmap

- [x] **Hybrid search (BM25 + semantic)** - Complete! Use `--hybrid` flag
- [x] **MCP server integration** - Complete! Use `./setup.sh --mcp`
- [ ] Cross-encoder reranking
- [ ] Agent-based query parsing
- [ ] Evaluation metrics
- [ ] Multi-user support with permissions

## MCP Server

Talk2Metadata includes a **Model Context Protocol (MCP) server** that exposes semantic search and schema exploration to AI agents like Claude, ChatGPT, and Cursor.

### Quick Start

```bash
# Install with MCP support
./setup.sh --mcp

# Prepare data
talk2metadata ingest csv ./data/csv --target orders
talk2metadata index --hybrid

# Start MCP server
talk2metadata-mcp sse
```

Server runs at `http://localhost:8010` with OAuth 2.0 authentication.

### Features

- 🔍 **Search Tool**: Natural language queries across all tables
- 📊 **Schema Tools**: Explore tables, columns, and foreign keys
- 🔐 **OAuth 2.0**: Secure access with OIDC authentication
- 📚 **Resources**: URI-based data access
- 💡 **Prompts**: Built-in help and search guides

### Documentation

- [Quick Start Guide](docs/mcp-quickstart.md) - Get started in 3 steps
- [Integration Guide](docs/mcp-integration.md) - Complete user documentation
- [Implementation Details](docs/mcp-implementation.md) - Technical architecture

### Example Usage

```bash
# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8010

# Or integrate with Claude Desktop (add to claude_desktop_config.json)
{
  "mcpServers": {
    "talk2metadata": {
      "url": "http://localhost:8010/mcp",
      "authorization": "Bearer YOUR_TOKEN"
    }
  }
}
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

Architecture inspired by:
- [Docs2Synth](https://github.com/AI4WA/Docs2Synth) - MCP integration patterns
- Production ML systems at Airbnb, Shopify, Netflix

Built with:
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [MCP](https://modelcontextprotocol.io/) - Model Context Protocol
- [uv](https://github.com/astral-sh/uv) - Package management
