"""Search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from talk2metadata.api.models import (
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse, responses={500: {"model": ErrorResponse}})
async def search(request: Request, search_request: SearchRequest) -> SearchResponse:
    """Search for relevant records using natural language.

    This endpoint performs semantic or hybrid search (BM25 + semantic)
    over the indexed records and returns the most relevant matches.

    Args:
        search_request: Search request with query, top_k, and optional hybrid params

    Returns:
        SearchResponse with ranked results

    Raises:
        HTTPException: If search fails or index not loaded
    """
    # Get the appropriate retriever (semantic or hybrid)
    if search_request.hybrid:
        retriever = request.app.state.hybrid_retriever
        if not retriever:
            raise HTTPException(
                status_code=500,
                detail="Hybrid search not available. Please start the server with a hybrid index or set hybrid=false.",
            )
    else:
        retriever = request.app.state.retriever
        if not retriever:
            raise HTTPException(
                status_code=500,
                detail="Search index not loaded. Please ensure the server was started with a valid index.",
            )

    try:
        mode = "hybrid" if search_request.hybrid else "semantic"
        logger.info(f"Search query [{mode}]: '{search_request.query}' (top_k={search_request.top_k})")

        # Perform search (both retrievers have the same search interface)
        results = retriever.search(
            query=search_request.query,
            top_k=search_request.top_k,
        )

        # Convert to API models
        result_items = []
        for r in results:
            item_dict = {
                "rank": r.rank,
                "table": r.table,
                "row_id": r.row_id,
                "score": r.score,
                "data": r.data,
            }
            # Add hybrid-specific fields if available
            if search_request.hybrid and hasattr(r, "bm25_score"):
                item_dict["bm25_score"] = r.bm25_score
                item_dict["semantic_score"] = r.semantic_score
                item_dict["fusion_method"] = r.fusion_method
            result_items.append(SearchResultItem(**item_dict))

        stats = retriever.get_stats()

        response_dict = {
            "query": search_request.query,
            "top_k": search_request.top_k,
            "search_mode": mode,
            "results": result_items,
            "total_records": stats["total_records"],
        }

        if search_request.hybrid:
            response_dict["fusion_method"] = search_request.fusion_method
            response_dict["alpha"] = search_request.alpha

        return SearchResponse(**response_dict)

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/search/status")
async def search_status(request: Request) -> dict:
    """Get search service status.

    Returns information about the loaded index and model.
    """
    retriever = request.app.state.retriever

    if not retriever:
        return {
            "index_loaded": False,
            "message": "Search index not loaded",
        }

    stats = retriever.get_stats()

    return {
        "index_loaded": True,
        "total_records": stats["total_records"],
        "index_size": stats["index_size"],
        "embedding_dimension": stats["embedding_dimension"],
        "model": stats["model"],
    }
