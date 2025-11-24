"""Mode registry for managing different indexing and retrieval strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModeInfo:
    """Information about a mode."""

    name: str
    description: str
    indexer_class: Type
    retriever_class: Type
    enabled: bool = True


class BaseIndexer(ABC):
    """Base class for mode-specific indexers."""

    @abstractmethod
    def build_index(
        self, tables: Dict, schema_metadata: SchemaMetadata, **kwargs
    ) -> Any:
        """Build index for this mode."""
        pass


class BaseRetriever(ABC):
    """Base class for mode-specific retrievers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[Any]:
        """Search using this mode."""
        pass


class ModeRegistry:
    """Registry for indexing and retrieval modes."""

    def __init__(self):
        self._modes: Dict[str, ModeInfo] = {}
        self._active_mode: Optional[str] = None

    def register(
        self,
        name: str,
        description: str,
        indexer_class: Type[BaseIndexer],
        retriever_class: Type[BaseRetriever],
        enabled: bool = True,
    ) -> None:
        """Register a new mode.

        Args:
            name: Mode name (e.g., "record_embedding")
            description: Human-readable description
            indexer_class: Indexer class for this mode
            retriever_class: Retriever class for this mode
            enabled: Whether this mode is enabled
        """
        self._modes[name] = ModeInfo(
            name=name,
            description=description,
            indexer_class=indexer_class,
            retriever_class=retriever_class,
            enabled=enabled,
        )
        logger.info(f"Registered mode: {name}")

    def get(self, name: str) -> Optional[ModeInfo]:
        """Get mode information by name.

        Args:
            name: Mode name

        Returns:
            ModeInfo or None if not found
        """
        return self._modes.get(name)

    def get_active(self) -> Optional[str]:
        """Get active mode name from config.

        Returns:
            Active mode name or None
        """
        if self._active_mode:
            return self._active_mode

        config = get_config()
        active = config.get("modes.active", "record_embedding")
        if active in self._modes and self._modes[active].enabled:
            return active
        return None

    def set_active(self, name: str) -> None:
        """Set active mode.

        Args:
            name: Mode name to activate
        """
        if name not in self._modes:
            raise ValueError(f"Mode '{name}' not registered")
        if not self._modes[name].enabled:
            raise ValueError(f"Mode '{name}' is disabled")
        self._active_mode = name
        logger.info(f"Set active mode to: {name}")

    def list_modes(self, enabled_only: bool = False) -> Dict[str, ModeInfo]:
        """List all registered modes.

        Args:
            enabled_only: If True, only return enabled modes

        Returns:
            Dict mapping mode name -> ModeInfo
        """
        if enabled_only:
            return {name: info for name, info in self._modes.items() if info.enabled}
        return self._modes.copy()

    def get_all_enabled(self) -> list[str]:
        """Get list of all enabled mode names.

        Returns:
            List of enabled mode names
        """
        return [name for name, info in self._modes.items() if info.enabled]


# Global registry instance
_registry = ModeRegistry()


def register_mode(
    name: str,
    description: str,
    indexer_class: Type[BaseIndexer],
    retriever_class: Type[BaseRetriever],
    enabled: bool = True,
) -> None:
    """Register a mode in the global registry.

    Args:
        name: Mode name
        description: Mode description
        indexer_class: Indexer class
        retriever_class: Retriever class
        enabled: Whether enabled
    """
    _registry.register(name, description, indexer_class, retriever_class, enabled)


def get_mode(name: str) -> Optional[ModeInfo]:
    """Get mode information.

    Args:
        name: Mode name

    Returns:
        ModeInfo or None
    """
    return _registry.get(name)


def get_active_mode() -> Optional[str]:
    """Get active mode name.

    Returns:
        Active mode name or None
    """
    return _registry.get_active()


def get_registry() -> ModeRegistry:
    """Get the global mode registry.

    Returns:
        ModeRegistry instance
    """
    return _registry


def get_mode_config(mode_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific mode.

    Args:
        mode_name: Mode name

    Returns:
        Mode configuration dict with 'indexer' and 'retriever' keys, or None
    """
    config = get_config()
    mode_config = config.get(f"modes.{mode_name}")
    if isinstance(mode_config, dict) and "indexer" in mode_config:
        return mode_config
    return None


def get_mode_indexer_config(mode_name: str) -> Dict[str, Any]:
    """Get indexer configuration for a specific mode.

    Args:
        mode_name: Mode name

    Returns:
        Indexer configuration dict (falls back to global embedding config)
    """
    mode_config = get_mode_config(mode_name)
    if mode_config and "indexer" in mode_config:
        return mode_config["indexer"]

    # Fall back to global embedding config
    config = get_config()
    return {
        "model_name": config.get(
            "embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "device": config.get("embedding.device"),
        "batch_size": config.get("embedding.batch_size", 32),
        "normalize": config.get("embedding.normalize", True),
    }


def get_mode_retriever_config(mode_name: str) -> Dict[str, Any]:
    """Get retriever configuration for a specific mode.

    Args:
        mode_name: Mode name

    Returns:
        Retriever configuration dict (falls back to global retrieval config)
    """
    mode_config = get_mode_config(mode_name)
    if mode_config and "retriever" in mode_config:
        return mode_config["retriever"]

    # Fall back to global retrieval config
    config = get_config()
    return {
        "top_k": config.get("retrieval.top_k", 5),
        "similarity_metric": config.get("retrieval.similarity_metric", "cosine"),
        "per_table_top_k": config.get("retrieval.per_table_top_k", 5),
        "use_reranking": config.get("retrieval.use_reranking", False),
    }
