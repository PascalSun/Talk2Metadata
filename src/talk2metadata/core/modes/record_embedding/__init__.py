"""Record embedding mode - record-level embedding with voting-based cross-table search."""

from __future__ import annotations

from talk2metadata.core.modes.record_embedding.indexer import Indexer
from talk2metadata.core.modes.record_embedding.retriever import (
    RecordVoter,
    RecordVoteSearchResult,
)
from talk2metadata.core.modes.registry import register_mode

# Register the record_embedding mode
register_mode(
    name="record_embedding",
    description="Record-level embedding with voting-based cross-table search (RecordVoter)",
    indexer_class=Indexer,
    retriever_class=RecordVoter,
    enabled=True,
)

__all__ = ["Indexer", "RecordVoter", "RecordVoteSearchResult"]
