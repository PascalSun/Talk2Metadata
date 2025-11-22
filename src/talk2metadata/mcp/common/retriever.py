"""Shared retriever instance for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from talk2metadata.core.retriever import Retriever
from talk2metadata.core.hybrid_retriever import HybridRetriever
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)

# Global retriever instance
_retriever: Optional[Retriever | HybridRetriever] = None


def get_retriever(
    index_dir: str | Path | None = None, use_hybrid: bool = False
) -> Retriever | HybridRetriever:
    """Get or create the global retriever instance.

    Args:
        index_dir: Optional path to index directory
        use_hybrid: Whether to use hybrid retriever (BM25 + semantic)

    Returns:
        Retriever or HybridRetriever instance
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    # Load configuration
    config = get_config()

    if index_dir is None:
        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))
    else:
        index_dir = Path(index_dir)

    index_path = index_dir / "index.faiss"
    records_path = index_dir / "records.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. Please run 'talk2metadata index' first."
        )

    # Load retriever
    try:
        if use_hybrid:
            bm25_path = index_dir / "bm25.pkl"
            if not bm25_path.exists():
                logger.warning(
                    f"BM25 index not found at {bm25_path}. Falling back to semantic retriever."
                )
                _retriever = Retriever.from_paths(index_path, records_path)
            else:
                _retriever = HybridRetriever.from_paths(
                    index_path, records_path, bm25_path
                )
                logger.info("Loaded hybrid retriever (BM25 + semantic)")
        else:
            _retriever = Retriever.from_paths(index_path, records_path)
            logger.info("Loaded semantic retriever")

        return _retriever

    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        raise
