"""Schema detection and metadata management."""

from talk2metadata.core.schema.schema import SchemaDetector, SchemaMetadata
from talk2metadata.core.schema.schema_viz import (
    export_schema_for_review,
    generate_html_visualization,
    validate_schema,
)
from talk2metadata.core.schema.types import ForeignKey, TableMetadata

__all__ = [
    "ForeignKey",
    "SchemaDetector",
    "SchemaMetadata",
    "TableMetadata",
    "export_schema_for_review",
    "generate_html_visualization",
    "validate_schema",
]
