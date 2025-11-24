"""Shared retriever instance for MCP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from talk2metadata.core.modes import RecordVoter
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

logger = get_logger(__name__)

# Global retriever instance
_retriever: Optional[RecordVoter] = None


def get_retriever(
    index_dir: str | Path | None = None, use_hybrid: bool = False
) -> RecordVoter:
    """Get or create the global retriever instance.

    Args:
        index_dir: Optional path to index directory
        use_hybrid: Ignored (kept for compatibility, always uses RecordVoter)

    Returns:
        RecordVoter instance
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

    # Find schema metadata
    schema_path = index_dir / "schema_metadata.json"
    if not schema_path.exists():
        # Try metadata directory
        run_id = config.get("run_id")
        metadata_dir = get_metadata_dir(run_id, config)
        schema_path = find_schema_file(metadata_dir)
        if not schema_path or not Path(schema_path).exists():
            raise FileNotFoundError(
                "Schema metadata not found. Please run 'talk2metadata index' first."
            )

    # Load RecordVoter retriever
    try:
        _retriever = RecordVoter.from_paths(index_dir, schema_path)
        logger.info("Loaded RecordVoter retriever")

        return _retriever

    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        raise
