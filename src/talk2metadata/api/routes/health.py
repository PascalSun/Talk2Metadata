"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request

from talk2metadata import __version__
from talk2metadata.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint.

    Returns service status and basic information.
    """
    retriever = request.app.state.retriever

    if retriever:
        stats = retriever.get_stats()
        return HealthResponse(
            status="healthy",
            version=__version__,
            index_loaded=True,
            total_records=stats["total_records"],
        )
    else:
        return HealthResponse(
            status="degraded",
            version=__version__,
            index_loaded=False,
            total_records=None,
        )
