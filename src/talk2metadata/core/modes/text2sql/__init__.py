"""Text2SQL mode - convert natural language questions to SQL queries."""

from __future__ import annotations

from talk2metadata.core.modes.registry import register_mode
from talk2metadata.core.modes.text2sql.indexer import Indexer
from talk2metadata.core.modes.text2sql.retriever import (
    DirectText2SQLRetriever,
    Text2SQLSearchResult,
    TwoStepText2SQLRetriever,
)

# Register the text2sql mode with direct approach as default
register_mode(
    name="text2sql",
    description="Text-to-SQL: Convert natural language questions to SQL queries and execute them",
    indexer_class=Indexer,
    retriever_class=DirectText2SQLRetriever,
    enabled=True,
)

# Register a variant with two-step approach
register_mode(
    name="text2sql_two_step",
    description="Text-to-SQL (Two-step): Locate relevant columns/tables first, then generate SQL",
    indexer_class=Indexer,
    retriever_class=TwoStepText2SQLRetriever,
    enabled=True,
)

__all__ = [
    "Indexer",
    "DirectText2SQLRetriever",
    "TwoStepText2SQLRetriever",
    "Text2SQLSearchResult",
]
