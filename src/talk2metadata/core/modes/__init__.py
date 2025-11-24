"""Indexing and retrieval modes for Talk2Metadata."""

# Import to register modes
from talk2metadata.core.modes import record_embedding  # noqa: F401
from talk2metadata.core.modes.registry import (
    ModeRegistry,
    get_active_mode,
    get_mode,
    get_mode_config,
    get_mode_indexer_config,
    get_mode_retriever_config,
    get_registry,
    register_mode,
)

__all__ = [
    "ModeRegistry",
    "get_active_mode",
    "get_mode",
    "get_mode_config",
    "get_mode_indexer_config",
    "get_mode_retriever_config",
    "get_registry",
    "register_mode",
]

# Re-export mode-specific classes for convenience
from talk2metadata.core.modes.record_embedding import (  # noqa: E402, F401
    Indexer,
    RecordVoter,
    RecordVoteSearchResult,
)

__all__.extend(["Indexer", "RecordVoter", "RecordVoteSearchResult"])
