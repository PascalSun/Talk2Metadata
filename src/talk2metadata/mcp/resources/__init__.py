from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from mcp.server import Server
from mcp.types import Resource, ResourceTemplate

from talk2metadata import __version__

from ..common.schema_index import get_schema


def register_resources(server: Server) -> None:
    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="resource://talk2metadata/info",
                name="Server Information",
                description="Basic information about the Talk2Metadata MCP server.",
                mimeType="text/plain",
            ),
            Resource(
                uri="resource://talk2metadata/status",
                name="Server Status",
                description="Server status and health information.",
                mimeType="application/json",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        return [
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/table/{table_name}",
                name="Table Information",
                description=(
                    "Get detailed information about a specific table, including "
                    "columns, data types, sample values, and foreign key relationships."
                ),
                mimeType="application/json",
            ),
            ResourceTemplate(
                uriTemplate="resource://talk2metadata/schema",
                name="Complete Schema",
                description=(
                    "Get the complete schema metadata including all tables and "
                    "foreign key relationships."
                ),
                mimeType="application/json",
            ),
        ]

    def _read_info() -> str:
        return (
            "Talk2Metadata MCP Server - A Model Context Protocol server for "
            "question-driven multi-table record retrieval. Provides semantic search "
            "across relational data with schema understanding and foreign key navigation."
        )

    def _read_status() -> str:
        return json.dumps(
            {
                "status": "running",
                "version": __version__,
                "server": "Talk2Metadata MCP",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _read_table_by_name(table_name: str) -> str:
        try:
            schema = get_schema()

            if table_name not in schema.tables:
                available_tables = list(schema.tables.keys())
                return json.dumps(
                    {
                        "error": "Table not found",
                        "error_code": "TABLE_NOT_FOUND",
                        "table_name": table_name,
                        "message": f"No table found with name '{table_name}'.",
                        "available_tables": available_tables,
                    },
                    indent=2,
                )

            meta = schema.tables[table_name]
            related_tables = schema.get_related_tables(table_name)
            fks = schema.get_foreign_keys_for_table(table_name)

            return json.dumps(
                {
                    "name": table_name,
                    "is_target": table_name == schema.target_table,
                    "columns": meta.columns,
                    "primary_key": meta.primary_key,
                    "row_count": meta.row_count,
                    "sample_values": meta.sample_values,
                    "related_tables": related_tables,
                    "foreign_keys": [
                        {
                            "child_table": fk.child_table,
                            "child_column": fk.child_column,
                            "parent_table": fk.parent_table,
                            "parent_column": fk.parent_column,
                            "coverage": fk.coverage,
                        }
                        for fk in fks
                    ],
                },
                indent=2,
            )
        except FileNotFoundError as e:
            return json.dumps(
                {
                    "error": "Schema not found",
                    "error_code": "SCHEMA_NOT_FOUND",
                    "message": str(e),
                    "hint": "Please run 'talk2metadata ingest' to load data first.",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": "Failed to read table information",
                    "error_code": "READ_ERROR",
                    "message": str(e),
                },
                indent=2,
            )

    def _read_schema() -> str:
        try:
            schema = get_schema()

            tables = {}
            for name, meta in schema.tables.items():
                tables[name] = {
                    "columns": meta.columns,
                    "primary_key": meta.primary_key,
                    "row_count": meta.row_count,
                    "sample_values": meta.sample_values,
                }

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

            return json.dumps(
                {
                    "target_table": schema.target_table,
                    "table_count": len(tables),
                    "foreign_key_count": len(foreign_keys),
                    "tables": tables,
                    "foreign_keys": foreign_keys,
                },
                indent=2,
            )
        except FileNotFoundError as e:
            return json.dumps(
                {
                    "error": "Schema not found",
                    "error_code": "SCHEMA_NOT_FOUND",
                    "message": str(e),
                    "hint": "Please run 'talk2metadata ingest' to load data first.",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": "Failed to read schema",
                    "error_code": "READ_ERROR",
                    "message": str(e),
                },
                indent=2,
            )

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        uri_str = str(uri)
        if uri_str == "resource://talk2metadata/info":
            return _read_info()
        if uri_str == "resource://talk2metadata/status":
            return _read_status()
        if uri_str == "resource://talk2metadata/schema":
            return _read_schema()

        table_match = re.match(r"^resource://talk2metadata/table/(.+)$", uri_str)
        if table_match:
            return _read_table_by_name(table_match.group(1))

        return json.dumps(
            {
                "error": "Unknown resource",
                "error_code": "UNKNOWN_RESOURCE",
                "uri": uri_str,
                "message": f"Resource URI '{uri_str}' is not recognized.",
            },
            indent=2,
        )
