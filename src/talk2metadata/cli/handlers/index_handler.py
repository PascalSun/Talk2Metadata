"""Business logic for index building commands."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from talk2metadata.core.modes import Indexer, get_mode_indexer_config, get_registry
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import Config
from talk2metadata.utils.paths import get_indexes_dir, get_processed_dir


class IndexHandler:
    """Handler for index building operations.

    Encapsulates business logic for building and saving search indexes.
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config
        self.registry = get_registry()

    def load_tables_from_pickle(
        self,
        tables_path: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load tables from pickle file.

        Args:
            tables_path: Optional path to tables file
            run_id: Optional run ID

        Returns:
            Dictionary of DataFrames

        Raises:
            FileNotFoundError: If tables file not found
        """
        if not tables_path:
            processed_dir = get_processed_dir(
                run_id or self.config.get("run_id"), self.config
            )
            tables_path = processed_dir / "tables.pkl"

        if not Path(tables_path).exists():
            raise FileNotFoundError(f"Tables not found at {tables_path}")

        with open(tables_path, "rb") as f:
            return pickle.load(f)

    def determine_modes_to_build(
        self,
        mode: Optional[str] = None,
        all_modes: bool = False,
    ) -> List[str]:
        """Determine which modes to build indexes for.

        Args:
            mode: Specific mode name
            all_modes: Whether to build all enabled modes

        Returns:
            List of mode names to build

        Raises:
            ValueError: If no modes found or mode not registered
        """
        if all_modes:
            modes = self.registry.get_all_enabled()
            if not modes:
                raise ValueError("No enabled modes found")
            return modes

        if mode:
            active_mode = mode
        else:
            from talk2metadata.core.modes import get_active_mode

            active_mode = get_active_mode() or "record_embedding"

        if not self.registry.get(active_mode):
            available = ", ".join(self.registry.get_all_enabled())
            raise ValueError(f"Mode '{active_mode}' not found. Available: {available}")

        return [active_mode]

    def build_index_for_mode(
        self,
        mode_name: str,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> Tuple[Dict, Indexer]:
        """Build index for a specific mode.

        Args:
            mode_name: Mode name
            tables: Dictionary of DataFrames
            schema_metadata: Schema metadata
            model_name: Optional model name (overrides config)
            batch_size: Batch size for embedding generation

        Returns:
            Tuple of (table_indices, indexer)

        Raises:
            Exception: If index building fails
        """
        mode_info = self.registry.get(mode_name)
        if not mode_info or not mode_info.enabled:
            raise ValueError(f"Mode '{mode_name}' is not enabled")

        # Get mode-specific config
        mode_indexer_config = get_mode_indexer_config(mode_name)

        # Initialize indexer (CLI args override config)
        indexer = Indexer(
            model_name=model_name or mode_indexer_config.get("model_name"),
            device=mode_indexer_config.get("device"),
            batch_size=(
                batch_size
                if batch_size is not None
                else mode_indexer_config.get("batch_size", 32)
            ),
            normalize=mode_indexer_config.get("normalize", True),
        )

        # Build index
        table_indices = indexer.build_index(tables, schema_metadata)

        return table_indices, indexer

    def save_index_for_mode(
        self,
        mode_name: str,
        table_indices: Dict,
        indexer: Indexer,
        schema_metadata_path: Path,
        output_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Save index for a specific mode.

        Args:
            mode_name: Mode name
            table_indices: Table indices dictionary
            indexer: Indexer instance
            schema_metadata_path: Path to schema metadata file
            output_dir: Optional base output directory
            run_id: Optional run ID

        Returns:
            Path to saved index directory
        """
        base_index_dir = (
            output_dir
            if output_dir
            else get_indexes_dir(run_id or self.config.get("run_id"), self.config)
        )
        mode_index_dir = Path(base_index_dir) / mode_name
        mode_index_dir.mkdir(parents=True, exist_ok=True)

        # Save index
        indexer.save_multi_table_index(table_indices, mode_index_dir)

        # Copy schema metadata
        schema_copy_path = mode_index_dir / "schema_metadata.json"
        shutil.copy(schema_metadata_path, schema_copy_path)

        return mode_index_dir

    def get_index_stats(self, table_indices: Dict) -> Dict[str, Dict]:
        """Get statistics for built indexes.

        Args:
            table_indices: Table indices dictionary

        Returns:
            Dictionary mapping table_name -> stats
        """
        stats = {}
        for table_name, (idx, records) in table_indices.items():
            stats[table_name] = {
                "vectors": idx.ntotal,
                "records": len(records),
            }
        return stats
