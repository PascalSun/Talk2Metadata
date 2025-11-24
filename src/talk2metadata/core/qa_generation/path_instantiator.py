"""Path instantiator for creating concrete paths from patterns."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import pandas as pd

from talk2metadata.core.qa_generation.patterns import PathInstance, PathNode, PathPattern
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class PathInstantiator:
    """Instantiate path patterns from real data."""

    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        schema: SchemaMetadata,
        max_attempts: int = 100,
    ):
        """Initialize path instantiator.

        Args:
            tables: Dictionary mapping table names to DataFrames
            schema: Schema metadata
            max_attempts: Maximum attempts to find a valid path
        """
        self.tables = tables
        self.schema = schema
        self.target_table = schema.target_table
        self.max_attempts = max_attempts

        # Build ANumber lookup maps for efficient FK traversal
        self._build_anumber_maps()

    def _build_anumber_maps(self) -> None:
        """Build maps for efficient ANumber-based FK traversal."""
        # Map: ANumber -> List[records] for each table
        self.anumber_to_records: Dict[str, Dict[Any, List[Dict[str, Any]]]] = {}

        for table_name, df in self.tables.items():
            self.anumber_to_records[table_name] = {}
            anumber_col = None

            # Find ANumber/Anumber column
            for col in ["ANumber", "Anumber"]:
                if col in df.columns:
                    anumber_col = col
                    break

            if anumber_col is None:
                continue

            # Group records by ANumber
            for _, row in df.iterrows():
                anumber = row[anumber_col]
                if pd.isna(anumber):
                    continue

                if anumber not in self.anumber_to_records[table_name]:
                    self.anumber_to_records[table_name][anumber] = []

                # Convert row to dict
                record = {
                    "table": table_name,
                    "row_id": self._get_row_id(row, table_name),
                    "data": row.to_dict(),
                    "anumber": anumber,
                }
                self.anumber_to_records[table_name][anumber].append(record)

    def _get_row_id(self, row: pd.Series, table_name: str) -> Any:
        """Get primary key value for a row.

        Args:
            row: DataFrame row
            table_name: Table name

        Returns:
            Primary key value
        """
        meta = self.schema.tables[table_name]
        pk_col = meta.primary_key

        if pk_col and pk_col in row.index:
            return row[pk_col]
        else:
            # Fallback to first column or index
            return row.iloc[0] if len(row) > 0 else None

    def instantiate(
        self,
        pattern: PathPattern,
        num_instances: int = 1,
        require_valid: bool = True,
    ) -> List[PathInstance]:
        """Instantiate a path pattern from real data.

        Args:
            pattern: Path pattern to instantiate
            num_instances: Number of instances to generate
            require_valid: Whether to only return valid paths

        Returns:
            List of PathInstance objects
        """
        instances = []
        attempts = 0

        while len(instances) < num_instances and attempts < self.max_attempts:
            attempts += 1
            try:
                instance = self._try_instantiate(pattern)
                if instance is not None:
                    if not require_valid or self._is_valid_instance(instance):
                        instances.append(instance)
            except Exception as e:
                logger.debug(f"Failed to instantiate path (attempt {attempts}): {e}")
                continue

        if len(instances) < num_instances:
            logger.warning(
                f"Only generated {len(instances)}/{num_instances} instances "
                f"for pattern {pattern.pattern} after {attempts} attempts"
            )

        return instances

    def _try_instantiate(self, pattern: PathPattern) -> Optional[PathInstance]:
        """Try to instantiate a single path.

        Args:
            pattern: Path pattern

        Returns:
            PathInstance or None if failed
        """
        nodes = []
        target_indices = []

        # Start from first table in pattern
        start_table = pattern.pattern[0]
        if start_table not in self.tables:
            return None

        # Get all available ANumber values for starting table
        start_anumbers = list(
            self.anumber_to_records.get(start_table, {}).keys()
        )
        if not start_anumbers:
            return None

        # Randomly select starting ANumber
        current_anumber = random.choice(start_anumbers)

        # Traverse pattern
        for i, table_name in enumerate(pattern.pattern):
            if table_name not in self.tables:
                return None

            # Get records for this table and ANumber
            records = self.anumber_to_records.get(table_name, {}).get(
                current_anumber, []
            )
            if not records:
                # Path broken, cannot continue
                return None

            # Randomly select one record
            record = random.choice(records)
            node = PathNode(
                table=table_name,
                row_id=record["row_id"],
                data=record["data"],
                anumber=current_anumber,
            )
            nodes.append(node)

            # Check if this is target table
            if table_name == self.target_table:
                target_indices.append(len(nodes) - 1)

            # For next iteration, current_anumber stays the same
            # (all tables share same ANumber for FK relationship)

        if not target_indices:
            # Path doesn't reach target table
            return None

        return PathInstance(
            pattern=pattern,
            nodes=nodes,
            target_node_indices=target_indices,
            metadata={"anumber": current_anumber},
        )

    def _is_valid_instance(self, instance: PathInstance) -> bool:
        """Check if a path instance is valid.

        Args:
            instance: Path instance

        Returns:
            True if valid
        """
        # Must have at least one target node
        if not instance.target_node_indices:
            return False

        # Must have nodes
        if not instance.nodes:
            return False

        # All nodes must have valid data
        for node in instance.nodes:
            if not node.data:
                return False

        return True

    def get_all_anumbers(self) -> List[Any]:
        """Get all ANumber values that exist in target table.

        Returns:
            List of ANumber values
        """
        if self.target_table not in self.anumber_to_records:
            return []
        return list(self.anumber_to_records[self.target_table].keys())

