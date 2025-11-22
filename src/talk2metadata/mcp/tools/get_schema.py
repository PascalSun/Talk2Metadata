"""Get schema tool with foreign key information."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from ..common.schema_index import get_schema


async def handle_get_schema(args: dict[str, Any]) -> list[TextContent]:
    """Get complete schema information including foreign key relationships.

    Args:
        args: Empty dictionary (no parameters required)

    Returns:
        List of TextContent with schema information
    """
    try:
        schema = get_schema()

        # Format tables
        tables = {}
        for name, meta in schema.tables.items():
            tables[name] = {
                "columns": meta.columns,
                "primary_key": meta.primary_key,
                "row_count": meta.row_count,
                "sample_values": meta.sample_values,
            }

        # Format foreign keys
        foreign_keys = [
            {
                "child_table": fk.child_table,
                "child_column": fk.child_column,
                "parent_table": fk.parent_table,
                "parent_column": fk.parent_column,
                "coverage": fk.coverage,
            }
            for fk in schema.foreign_keys
        ]

        output = {
            "target_table": schema.target_table,
            "table_count": len(tables),
            "foreign_key_count": len(foreign_keys),
            "tables": tables,
            "foreign_keys": foreign_keys,
        }

        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    except FileNotFoundError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Schema not found",
                        "message": str(e),
                        "hint": "Please run 'talk2metadata ingest' to load data first.",
                    },
                    indent=2,
                ),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2),
            )
        ]


TOOL_SPEC = {
    "name": "get_schema",
    "description": (
        "Get complete schema metadata including all tables, columns, data types, "
        "foreign key relationships, and sample values. Useful for understanding "
        "the database structure and relationships between tables."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
