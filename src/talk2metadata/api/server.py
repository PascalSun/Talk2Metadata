"""FastAPI server for Talk2Metadata."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from talk2metadata import __version__
from talk2metadata.api.routes import health, schema, search
from talk2metadata.core.retriever import Retriever
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app.

    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("Starting Talk2Metadata API server")
    setup_logging(level="INFO")

    # Load search index
    try:
        config = get_config()
        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))
        index_path = index_dir / "index.faiss"
        records_path = index_dir / "records.pkl"
        bm25_path = index_dir / "bm25.pkl"

        if index_path.exists() and records_path.exists():
            logger.info(f"Loading search index from {index_dir}")
            retriever = Retriever.from_paths(index_path, records_path)
            stats = retriever.get_stats()
            logger.info(
                f"Semantic index loaded: {stats['total_records']} records, "
                f"model={stats['model']}"
            )
            app.state.retriever = retriever

            # Try to load hybrid retriever if BM25 index exists
            if bm25_path.exists():
                logger.info("BM25 index found, loading hybrid retriever")
                from talk2metadata.core.hybrid_retriever import HybridRetriever

                alpha = config.get("retrieval.hybrid.alpha", 0.5)
                fusion_method = config.get("retrieval.hybrid.fusion_method", "rrf")

                hybrid_retriever = HybridRetriever.from_paths(
                    index_path, records_path, bm25_path, alpha=alpha, fusion_method=fusion_method
                )
                hybrid_stats = hybrid_retriever.get_stats()
                logger.info(
                    f"Hybrid retriever loaded: fusion={hybrid_stats['fusion_method']}, "
                    f"alpha={hybrid_stats['alpha']}"
                )
                app.state.hybrid_retriever = hybrid_retriever
            else:
                logger.info("BM25 index not found, hybrid search unavailable")
                app.state.hybrid_retriever = None
        else:
            logger.warning(
                f"Index not found at {index_dir}. "
                "Search functionality will be unavailable. "
                "Please run 'talk2metadata index' first."
            )
            app.state.retriever = None
            app.state.hybrid_retriever = None

    except Exception as e:
        logger.error(f"Failed to load index: {e}", exc_info=True)
        app.state.retriever = None
        app.state.hybrid_retriever = None

    yield

    # Shutdown
    logger.info("Shutting down Talk2Metadata API server")
    app.state.retriever = None
    app.state.hybrid_retriever = None


def create_app(
    index_path: Optional[str | Path] = None,
    records_path: Optional[str | Path] = None,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        index_path: Optional path to FAISS index (overrides config)
        records_path: Optional path to records file (overrides config)

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Talk2Metadata API",
        description="Question-driven multi-table record retrieval",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "detail": str(exc),
            },
        )

    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Talk2Metadata API",
            "version": __version__,
            "description": "Question-driven multi-table record retrieval",
            "endpoints": {
                "health": "/health",
                "search": "/api/v1/search",
                "schema": "/api/v1/schema",
                "docs": "/docs",
            },
        }

    # Register routers
    app.include_router(health.router)
    app.include_router(search.router)
    app.include_router(schema.router)

    return app


# For uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "talk2metadata.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
