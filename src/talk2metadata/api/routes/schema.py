"""Schema information endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from talk2metadata.api.models import ForeignKeyInfo, SchemaResponse, TableInfo
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["schema"])


@router.get("/schema", response_model=SchemaResponse)
async def get_schema() -> SchemaResponse:
    """Get schema information.

    Returns metadata about tables, columns, and foreign key relationships.

    Returns:
        SchemaResponse with complete schema information

    Raises:
        HTTPException: If schema metadata not found
    """
    config = get_config()
    metadata_dir = Path(config.get("data.metadata_dir", "./data/metadata"))
    metadata_path = metadata_dir / "schema.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Schema metadata not found at {metadata_path}. "
            "Please run 'talk2metadata ingest' first.",
        )

    try:
        schema_metadata = SchemaMetadata.load(metadata_path)

        # Convert to API models
        tables = {
            name: TableInfo(
                name=meta.name,
                columns=meta.columns,
                primary_key=meta.primary_key,
                row_count=meta.row_count,
            )
            for name, meta in schema_metadata.tables.items()
        }

        foreign_keys = [
            ForeignKeyInfo(
                child_table=fk.child_table,
                child_column=fk.child_column,
                parent_table=fk.parent_table,
                parent_column=fk.parent_column,
                coverage=fk.coverage,
            )
            for fk in schema_metadata.foreign_keys
        ]

        return SchemaResponse(
            target_table=schema_metadata.target_table,
            tables=tables,
            foreign_keys=foreign_keys,
        )

    except Exception as e:
        logger.error(f"Failed to load schema: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load schema: {str(e)}",
        )


@router.get("/schema/tables")
async def list_tables() -> dict:
    """List all tables in the schema.

    Returns:
        Dict with table names and basic info
    """
    config = get_config()
    metadata_dir = Path(config.get("data.metadata_dir", "./data/metadata"))
    metadata_path = metadata_dir / "schema.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Schema metadata not found",
        )

    try:
        schema_metadata = SchemaMetadata.load(metadata_path)

        return {
            "target_table": schema_metadata.target_table,
            "tables": [
                {
                    "name": meta.name,
                    "row_count": meta.row_count,
                    "column_count": len(meta.columns),
                }
                for meta in schema_metadata.tables.values()
            ],
        }

    except Exception as e:
        logger.error(f"Failed to list tables: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tables: {str(e)}",
        )
