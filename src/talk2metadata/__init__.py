"""Talk2Metadata - Question-driven multi-table record retrieval."""

__version__ = "0.1.0"

# Connectors
from talk2metadata.connectors import (
    BaseConnector,
    ConnectorFactory,
    CSVLoader,
    DBConnector,
)

# Core modules
from talk2metadata.core import (
    ForeignKey,
    Indexer,
    Retriever,
    SchemaDetector,
    SchemaMetadata,
    SearchResult,
    TableMetadata,
)

# Utils
from talk2metadata.utils.config import Config, get_config, load_config


# Lazy imports for optional dependencies
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "HybridRetriever":
        try:
            from talk2metadata.core import HybridRetriever

            return HybridRetriever
        except ImportError as e:
            raise ImportError(
                "HybridRetriever requires additional dependencies. "
                "Install with: pip install talk2metadata[mcp]"
            ) from e
    elif name == "BM25Index":
        try:
            from talk2metadata.core import BM25Index

            return BM25Index
        except ImportError as e:
            raise ImportError(
                "BM25Index requires additional dependencies. "
                "Install with: pip install talk2metadata[mcp]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Version
    "__version__",
    # Core
    "Indexer",
    "Retriever",
    "SearchResult",
    "HybridRetriever",
    "BM25Index",
    "SchemaDetector",
    "SchemaMetadata",
    "TableMetadata",
    "ForeignKey",
    # Connectors
    "BaseConnector",
    "ConnectorFactory",
    "CSVLoader",
    "DBConnector",
    # Config
    "Config",
    "get_config",
    "load_config",
]
