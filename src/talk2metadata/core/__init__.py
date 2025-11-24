"""Core modules for Talk2Metadata."""

# Re-export all public APIs
from talk2metadata.core.modes import (
    Indexer,
    RecordVoter,
    RecordVoteSearchResult,
)
from talk2metadata.core.modes.record_embedding.search_result import SearchResult
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
    # Index (from modes)
    "Indexer",
    # Retriever (from modes)
    "RecordVoteSearchResult",
    "RecordVoter",
    "SearchResult",
]
