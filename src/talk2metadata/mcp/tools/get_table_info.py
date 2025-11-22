"""Get detailed information about a specific table."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from ..common.schema_index import get_schema


async def handle_get_table_info(args: dict[str, Any]) -> list[TextContent]:
    """Get detailed information about a specific table.

    Args:
        args: Dictionary with 'table_name' key

    Returns:
        List of TextContent with table information
    """
    table_name = args.get("table_name", "")

    if not table_name:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": "table_name parameter is required"}, indent=2
                ),
            )
        ]

    try:
        schema = get_schema()

        # Check if table exists
        if table_name not in schema.tables:
            available_tables = list(schema.tables.keys())
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": f"Table '{table_name}' not found",
                            "available_tables": available_tables,
                        },
                        indent=2,
                    ),
                )
            ]

        # Get table metadata
        meta = schema.tables[table_name]

        # Get related tables
        related_tables = schema.get_related_tables(table_name)

        # Get foreign keys
        fks = schema.get_foreign_keys_for_table(table_name)
        foreign_keys = [
            {
                "child_table": fk.child_table,
                "child_column": fk.child_column,
                "parent_table": fk.parent_table,
                "parent_column": fk.parent_column,
                "coverage": fk.coverage,
            }
            for fk in fks
        ]

        output = {
            "name": table_name,
            "is_target": table_name == schema.target_table,
            "columns": meta.columns,
            "primary_key": meta.primary_key,
            "row_count": meta.row_count,
            "sample_values": meta.sample_values,
            "related_tables": related_tables,
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
    "name": "get_table_info",
    "description": (
        "Get detailed information about a specific table, including columns, "
        "data types, row count, sample values, related tables, and foreign key "
        "relationships."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "table_name": {
                "type": "string",
                "description": "Name of the table to get information about",
            }
        },
        "required": ["table_name"],
    },
}
