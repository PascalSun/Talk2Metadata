"""Business logic for search commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from talk2metadata.core.modes import (
    RecordVoter,
    get_mode_retriever_config,
    get_registry,
)
from talk2metadata.core.modes.comparison import ModeComparator
from talk2metadata.core.modes.record_embedding.search_result import SearchResult
from talk2metadata.core.modes.text2sql import (
    DirectText2SQLRetriever,
    TwoStepText2SQLRetriever,
)
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import Config
from talk2metadata.utils.csv_to_db import get_or_create_db_connection
from talk2metadata.utils.paths import (
    find_schema_file,
    get_indexes_dir,
    get_metadata_dir,
)


class SearchHandler:
    """Handler for search operations.

    Encapsulates business logic for searching and comparing across modes.
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.registry = get_registry()

    def _get_text2sql_connection(
        self, schema_metadata: SchemaMetadata, run_id: Optional[str]
    ) -> str:
        """Resolve or build a database connection for text2sql modes."""
        ingest_config = self.config.get("ingest", {})
        return get_or_create_db_connection(
            ingest_config=ingest_config,
            schema_metadata=schema_metadata,
            run_id=run_id or self.config.get("run_id"),
        )

    def load_retriever(
        self,
        mode_name: str,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        per_table_top_k: int = 5,
    ):
        """Load retriever for a specific mode.

        Args:
            mode_name: Mode name
            index_dir: Optional index directory
            run_id: Optional run ID
            per_table_top_k: Top-k per table for record_embedding mode

        Returns:
            Retriever instance

        Raises:
            FileNotFoundError: If index not found
            NotImplementedError: If mode retriever not implemented
        """
        # Determine index directory
        if not index_dir:
            base_index_dir = get_indexes_dir(
                run_id or self.config.get("run_id"), self.config
            )
            index_dir = base_index_dir / mode_name
        else:
            index_dir = Path(index_dir)
            # If base dir provided, use mode subdirectory
            if (index_dir / mode_name).exists():
                index_dir = index_dir / mode_name

        # Find schema file
        schema_path = index_dir / "schema_metadata.json"
        if not schema_path.exists():
            metadata_dir = get_metadata_dir(
                run_id or self.config.get("run_id"), self.config
            )
            # Get target_table from config to find correct schema file
            target_table = self.config.get("ingest.target_table")
            schema_path = find_schema_file(metadata_dir, target_table=target_table)
            if not schema_path or not Path(schema_path).exists():
                raise FileNotFoundError(
                    f"Schema metadata not found for mode '{mode_name}'"
                )

        # Get mode-specific config
        mode_retriever_config = get_mode_retriever_config(mode_name)

        # Load schema metadata (needed for text2sql modes)
        schema_metadata = SchemaMetadata.load(schema_path)

        # Initialize retriever based on mode
        if mode_name == "record_embedding":
            return RecordVoter.from_paths(
                index_dir,
                schema_path,
                per_table_top_k=per_table_top_k
                or mode_retriever_config.get("per_table_top_k", 5),
            )
        elif mode_name in ("text2sql", "text2sql_two_step"):
            connection_string = self._get_text2sql_connection(schema_metadata, run_id)
            retriever_cls = (
                DirectText2SQLRetriever
                if mode_name == "text2sql"
                else TwoStepText2SQLRetriever
            )
            return retriever_cls(
                schema_metadata=schema_metadata,
                connection_string=connection_string,
                mode_name=mode_name,
            )

        raise NotImplementedError(f"Retriever for mode '{mode_name}' not implemented")

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode_name: Optional[str] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        per_table_top_k: int = 5,
    ) -> List[Union[SearchResult, Any]]:
        """Perform search using specified mode.

        Args:
            query: Search query
            top_k: Number of results to return
            mode_name: Optional mode name (uses active mode if None)
            index_dir: Optional index directory
            run_id: Optional run ID
            per_table_top_k: Top-k per table for record_embedding mode

        Returns:
            List of SearchResult objects
        """
        # Determine mode
        if not mode_name:
            from talk2metadata.core.modes import get_active_mode

            mode_name = get_active_mode() or "record_embedding"

        # Load retriever
        retriever = self.load_retriever(
            mode_name=mode_name,
            index_dir=index_dir,
            run_id=run_id,
            per_table_top_k=per_table_top_k,
        )

        # Perform search
        return retriever.search(query, top_k=top_k)

    def load_retrievers_for_comparison(
        self,
        modes_to_compare: Optional[List[str]] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load retrievers for all modes to compare.

        Args:
            modes_to_compare: Optional list of mode names
            index_dir: Optional base index directory
            run_id: Optional run ID

        Returns:
            Dictionary mapping mode_name -> retriever
        """
        # Determine modes to compare
        if not modes_to_compare:
            comparison_config = self.config.get("modes.compare", {})
            modes_to_compare = comparison_config.get("modes", [])
            if not modes_to_compare:
                modes_to_compare = self.registry.get_all_enabled()

        if not modes_to_compare:
            raise ValueError("No enabled modes found for comparison")

        # Load retrievers
        base_index_dir = (
            index_dir
            if index_dir
            else get_indexes_dir(run_id or self.config.get("run_id"), self.config)
        )
        base_index_dir = Path(base_index_dir)

        retrievers = {}
        for mode_name in modes_to_compare:
            mode_info = self.registry.get(mode_name)
            if not mode_info or not mode_info.enabled:
                continue

            try:
                retriever = self.load_retriever(
                    mode_name=mode_name,
                    index_dir=base_index_dir,
                    run_id=run_id,
                )
                retrievers[mode_name] = retriever
            except (FileNotFoundError, NotImplementedError):
                # Skip modes that can't be loaded
                continue

        return retrievers

    def compare_modes(
        self,
        query: str,
        top_k: int = 5,
        modes_to_compare: Optional[List[str]] = None,
        index_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ):
        """Compare search results across multiple modes.

        Args:
            query: Search query
            top_k: Number of results to return per mode
            modes_to_compare: Optional list of mode names
            index_dir: Optional base index directory
            run_id: Optional run ID

        Returns:
            ComparisonResult object
        """
        # Load retrievers
        retrievers = self.load_retrievers_for_comparison(
            modes_to_compare=modes_to_compare,
            index_dir=index_dir,
            run_id=run_id,
        )

        if not retrievers:
            raise ValueError("No retrievers loaded for comparison")

        # Run comparison
        comparator = ModeComparator(modes=list(retrievers.keys()))
        return comparator.compare(query, top_k=top_k, retrievers=retrievers)
