"""List tables tool."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from ..common.schema_index import get_schema


async def handle_list_tables(args: dict[str, Any]) -> list[TextContent]:
    """List all available tables in the schema.

    Args:
        args: Empty dictionary (no parameters required)

    Returns:
        List of TextContent with table information
    """
    try:
        schema = get_schema()

        # Format table list
        tables = []
        for name, meta in schema.tables.items():
            tables.append(
                {
                    "name": name,
                    "columns": list(meta.columns.keys()),
                    "column_count": len(meta.columns),
                    "row_count": meta.row_count,
                    "primary_key": meta.primary_key,
                    "is_target": name == schema.target_table,
                }
            )

        # Sort by name
        tables.sort(key=lambda x: x["name"])

        output = {
            "table_count": len(tables),
            "target_table": schema.target_table,
            "tables": tables,
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
    "name": "list_tables",
    "description": "List all available tables in the schema with basic information (columns, row count, primary key).",
    "inputSchema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}
