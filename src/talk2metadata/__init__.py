"""Talk2Metadata - Question-driven multi-table record retrieval."""

__version__ = "0.1.0"

# Core modules
from talk2metadata.core.indexer import Indexer
from talk2metadata.core.retriever import Retriever, SearchResult
from talk2metadata.core.schema import (
    ForeignKey,
    SchemaDetector,
    SchemaMetadata,
    TableMetadata,
)

# Connectors
from talk2metadata.connectors import (
    BaseConnector,
    ConnectorFactory,
    CSVLoader,
    DBConnector,
)

# Utils
from talk2metadata.utils.config import Config, get_config, load_config

# Lazy imports for optional dependencies
def __getattr__(name):
    """Lazy import for optional modules."""
    if name == "HybridRetriever":
        try:
            from talk2metadata.core.hybrid_retriever import HybridRetriever
            return HybridRetriever
        except ImportError as e:
            raise ImportError(
                f"HybridRetriever requires additional dependencies. "
                f"Install with: pip install talk2metadata[mcp]"
            ) from e
    elif name == "BM25Index":
        try:
            from talk2metadata.core.hybrid_retriever import BM25Index
            return BM25Index
        except ImportError as e:
            raise ImportError(
                f"BM25Index requires additional dependencies. "
                f"Install with: pip install talk2metadata[mcp]"
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
