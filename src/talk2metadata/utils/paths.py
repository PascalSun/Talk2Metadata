"""Path utilities for managing run_id-based directory structure."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from talk2metadata.utils.config import get_config


def sanitize_run_id(run_id: str) -> str:
    """Sanitize run_id for use in file paths.

    Args:
        run_id: Run ID string

    Returns:
        Sanitized run ID safe for filesystem
    """
    return re.sub(r"[^\w\-_.]", "_", str(run_id))


def get_run_base_dir(
    run_id: Optional[str] = None, base_dir: Optional[Path] = None
) -> Path:
    """Get base directory for a run_id.

    If run_id is provided, returns data/{run_id}/, otherwise returns data/.

    Args:
        run_id: Optional run ID. If None, uses default data directory structure.
        base_dir: Optional base directory. Defaults to "./data"

    Returns:
        Path to run base directory
    """
    if base_dir is None:
        base_dir = Path("./data")

    if run_id:
        run_id_safe = sanitize_run_id(run_id)
        return base_dir / run_id_safe
    return base_dir


def get_metadata_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get metadata directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to metadata directory
    """
    if config is None:
        config = get_config()

    if run_id:
        run_base = get_run_base_dir(run_id)
        return run_base / "metadata"
    else:
        return Path(config.get("data.metadata_dir", "./data/metadata"))


def get_processed_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get processed directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to processed directory
    """
    if config is None:
        config = get_config()

    if run_id:
        run_base = get_run_base_dir(run_id)
        return run_base / "processed"
    else:
        return Path(config.get("data.processed_dir", "./data/processed"))


def get_indexes_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get indexes directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to indexes directory
    """
    if config is None:
        config = get_config()

    if run_id:
        run_base = get_run_base_dir(run_id)
        return run_base / "indexes"
    else:
        return Path(config.get("data.indexes_dir", "./data/indexes"))


def get_qa_dir(run_id: Optional[str] = None, config=None) -> Path:
    """Get QA directory path for a run_id.

    Args:
        run_id: Optional run ID
        config: Optional config instance. If None, uses get_config()

    Returns:
        Path to QA directory
    """
    if config is None:
        config = get_config()

    if run_id:
        run_base = get_run_base_dir(run_id)
        return run_base / "qa"
    else:
        # Default to data/qa if no run_id
        return Path("./data/qa")


def find_schema_file(metadata_dir: Path) -> Path:
    """Find schema JSON file in metadata directory.

    Looks for schema.json first, then falls back to schema_*.json files.

    Args:
        metadata_dir: Path to metadata directory

    Returns:
        Path to schema file

    Raises:
        FileNotFoundError: If no schema file is found
    """
    # First try schema.json
    schema_path = metadata_dir / "schema.json"
    if schema_path.exists():
        return schema_path

    # Fall back to schema_*.json files
    schema_files = list(metadata_dir.glob("schema_*.json"))
    if schema_files:
        # Return the first one found (or most recent if multiple)
        return sorted(schema_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    raise FileNotFoundError(
        f"No schema file found in {metadata_dir}. "
        "Expected schema.json or schema_*.json"
    )
