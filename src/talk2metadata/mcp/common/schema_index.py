"""Shared schema metadata access for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Global schema instance
_schema: Optional[SchemaMetadata] = None


def get_schema(data_dir: str | Path | None = None) -> SchemaMetadata:
    """Get or load the schema metadata.

    Args:
        data_dir: Optional path to data directory

    Returns:
        SchemaMetadata instance
    """
    global _schema

    if _schema is not None:
        return _schema

    # Load configuration
    config = get_config()

    if data_dir is None:
        data_dir = Path(config.get("data.output_dir", "./data/processed"))
    else:
        data_dir = Path(data_dir)

    schema_path = data_dir / "schema.json"

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema not found at {schema_path}. Please run 'talk2metadata ingest' first."
        )

    try:
        _schema = SchemaMetadata.load(schema_path)
        logger.info(f"Loaded schema with {len(_schema.tables)} tables")
        return _schema

    except Exception as e:
        logger.error(f"Failed to load schema: {e}")
        raise
