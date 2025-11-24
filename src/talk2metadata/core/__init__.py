"""Core modules for Talk2Metadata."""

# Re-export all public APIs for backward compatibility
from talk2metadata.core.index import Indexer
from talk2metadata.core.retriever import (
    BM25Index,
    HybridRetriever,
    HybridSearchResult,
    Retriever,
    SearchResult,
)
from talk2metadata.core.schema import (
    ForeignKey,
    SchemaDetector,
    SchemaMetadata,
    TableMetadata,
    export_schema_for_review,
    generate_html_visualization,
    validate_schema,
)

__all__ = [
    # Schema
    "ForeignKey",
    "SchemaDetector",
    "SchemaMetadata",
    "TableMetadata",
    "export_schema_for_review",
    "generate_html_visualization",
    "validate_schema",
    # Index
    "Indexer",
    # Retriever
    "BM25Index",
    "HybridRetriever",
    "HybridSearchResult",
    "Retriever",
    "SearchResult",
]
