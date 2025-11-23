"""Search tool for finding relevant records."""

from __future__ import annotations

import json
from typing import Any

from mcp.types import TextContent

from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.metrics import log_slow_query
from talk2metadata.utils.timing import TimingContext

from ..common.retriever import get_retriever

logger = get_logger(__name__)


async def handle_search(args: dict[str, Any]) -> list[TextContent]:
    """Search for relevant records using natural language query.

    Args:
        args: Dictionary with 'query' and optional 'top_k', 'hybrid' keys

    Returns:
        List of TextContent with search results
    """
    import time

    query = args.get("query", "")
    top_k = args.get("top_k", 5)
    hybrid = args.get("hybrid", False)

    if not query:
        return [
            TextContent(
                type="text",
                text=json.dumps({"error": "Query parameter is required"}, indent=2),
            )
        ]

    try:
        start_time = time.perf_counter()

        # Get retriever
        with TimingContext("retriever_init"):
            retriever = get_retriever(use_hybrid=hybrid)

        # Search
        with TimingContext("search_execution"):
            results = retriever.search(query, top_k=top_k)

        # Format results
        with TimingContext("result_serialization"):
            formatted = []
            for r in results:
                result_dict = {
                    "rank": r.rank,
                    "table": r.table,
                    "row_id": r.row_id,
                    "score": r.score,
                    "data": r.data,
                }

                # Add hybrid-specific fields if available
                if hybrid and hasattr(r, "bm25_score"):
                    result_dict["bm25_score"] = r.bm25_score
                    result_dict["semantic_score"] = r.semantic_score

                formatted.append(result_dict)

            output = {
                "query": query,
                "top_k": top_k,
                "search_mode": "hybrid" if hybrid else "semantic",
                "results_count": len(formatted),
                "results": formatted,
            }

        # Check for slow query
        total_duration = (time.perf_counter() - start_time) * 1000
        slow_query_threshold = 100.0  # milliseconds
        if total_duration > slow_query_threshold:
            log_slow_query(
                query=query,
                duration_ms=total_duration,
                threshold_ms=slow_query_threshold,
                details={
                    "top_k": top_k,
                    "hybrid": hybrid,
                    "results_count": len(formatted),
                },
            )

        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    except FileNotFoundError as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": "Index not found",
                        "message": str(e),
                        "hint": "Please run 'talk2metadata index' to build the search index first.",
                    },
                    indent=2,
                ),
            )
        ]
    except Exception as e:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {"error": "Search failed", "message": str(e)}, indent=2
                ),
            )
        ]


TOOL_SPEC = {
    "name": "search",
    "description": (
        "Search for relevant records across all tables using natural language queries. "
        "Supports both semantic search (embeddings) and hybrid search (BM25 + semantic)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5)",
                "default": 5,
            },
            "hybrid": {
                "type": "boolean",
                "description": "Use hybrid search (BM25 + semantic) for better results (default: false)",
                "default": False,
            },
        },
        "required": ["query"],
    },
}
