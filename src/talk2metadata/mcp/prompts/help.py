"""Help prompt for Talk2Metadata MCP server."""

from __future__ import annotations

from mcp.types import GetPromptResult, PromptMessage, TextContent


async def get_help_prompt() -> GetPromptResult:
    """Return help information about using Talk2Metadata MCP server."""
    return GetPromptResult(
        description="Guide for using Talk2Metadata MCP server",
        messages=[
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=(
                        "# Talk2Metadata MCP Server - Quick Start Guide\n\n"
                        "## Overview\n"
                        "Talk2Metadata provides semantic search and schema exploration "
                        "for multi-table relational data. It helps you find relevant records "
                        "using natural language queries and understand relationships between tables.\n\n"
                        "## Available Tools\n\n"
                        "### 1. search\n"
                        "Search for relevant records using natural language.\n"
                        "- `query`: Your search question (e.g., 'customers in healthcare')\n"
                        "- `top_k`: Number of results (default: 5)\n"
                        "- `hybrid`: Use BM25+semantic hybrid search for better results\n\n"
                        "### 2. list_tables\n"
                        "Get a list of all available tables with basic information.\n\n"
                        "### 3. get_schema\n"
                        "Get complete schema metadata including foreign key relationships.\n\n"
                        "### 4. get_table_info\n"
                        "Get detailed information about a specific table.\n"
                        "- `table_name`: Name of the table to inspect\n\n"
                        "## Available Resources\n\n"
                        "- `resource://talk2metadata/info` - Server information\n"
                        "- `resource://talk2metadata/status` - Health status\n"
                        "- `resource://talk2metadata/schema` - Complete schema\n"
                        "- `resource://talk2metadata/table/{name}` - Specific table info\n\n"
                        "## Typical Workflow\n\n"
                        "1. **Explore schema**: Use `list_tables` or `get_schema` to understand available data\n"
                        "2. **Inspect tables**: Use `get_table_info` to see columns and relationships\n"
                        "3. **Search records**: Use `search` to find relevant records with natural language\n"
                        "4. **Refine search**: Use `hybrid=true` for better search quality\n\n"
                        "## Example Queries\n\n"
                        "```\n"
                        "# Find customers in a specific industry\n"
                        'search(query="customers in healthcare industry", top_k=10)\n\n'
                        "# Find high-value orders\n"
                        'search(query="orders with high value", hybrid=true)\n\n'
                        "# Explore schema relationships\n"
                        "get_schema()\n"
                        "get_table_info(table_name='orders')\n"
                        "```\n\n"
                        "## Tips\n\n"
                        "- Use hybrid search for better results (combines keyword and semantic matching)\n"
                        "- Check foreign keys to understand table relationships\n"
                        "- Sample values help understand column content\n"
                        "- The target table is the main table for your queries\n"
                    ),
                ),
            )
        ],
    )


PROMPT_SPEC = {
    "name": "help",
    "description": "Get help and guidance on using Talk2Metadata MCP server",
}
