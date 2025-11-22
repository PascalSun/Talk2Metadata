# Installation

## Requirements

- Python 3.11 or 3.12
- 4GB RAM minimum (for small datasets)
- 50MB disk space for package
- Additional space for embeddings model (~80MB)

## Installation Methods

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/talk2metadata.git
cd talk2metadata

# Install dependencies
uv sync

# Run CLI
uv run talk2metadata --help
```

### Using pip

```bash
# Install from source
pip install -e .

# Or with all features
pip install -e ".[full,dev]"
```

## Optional Dependencies

### Full Installation (API Server)

Includes FastAPI, Uvicorn, and hybrid search support:

```bash
uv sync --group full
# or
pip install -e ".[full]"
```

Features:
- FastAPI REST API server
- BM25 hybrid search
- API documentation (Swagger/ReDoc)

### Development Installation

Includes testing, linting, and documentation tools:

```bash
uv sync --all-extras
# or
pip install -e ".[dev]"
```

Tools included:
- pytest, pytest-cov
- black, isort, flake8
- mkdocs, mkdocs-material
- ipython

### Agent Support (Optional)

For future LLM-based query parsing:

```bash
pip install -e ".[agent]"
```

Includes:
- OpenAI Python client
- Anthropic Python client

## Verify Installation

```bash
# Check CLI is available
uv run talk2metadata --version

# Run tests
uv run pytest

# Check imports
uv run python -c "from talk2metadata import __version__; print(__version__)"
```

## Configuration

Create a `config.yml` file (optional):

```bash
cp config.example.yml config.yml
```

Edit `config.yml` to customize:

- Data directories
- Embedding model
- FK detection thresholds
- Retrieval settings
- Hybrid search parameters

Key configuration options:
- `data.raw_dir`, `data.indexes_dir` - Data directories
- `embedding.model_name` - Embedding model (default: `all-MiniLM-L6-v2`)
- `schema.fk_detection.min_coverage` - FK detection threshold (default: 0.9)
- `retrieval.hybrid.alpha` - Hybrid search weight (0=BM25, 1=semantic)
- `retrieval.hybrid.fusion_method` - Fusion method (`rrf` or `weighted_sum`)

See `config.example.yml` for all available options.

## Next Steps

- [Quick Start Tutorial](quickstart.md)
- [API Server Guide](../user-guide/api-server.md)
