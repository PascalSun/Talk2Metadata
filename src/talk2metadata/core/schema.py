"""Schema detection and foreign key inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ForeignKey:
    """Foreign key relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float  # 0.0-1.0 (percentage of child values found in parent)

    def __repr__(self) -> str:
        return (
            f"FK({self.child_table}.{self.child_column} -> "
            f"{self.parent_table}.{self.parent_column}, coverage={self.coverage:.2f})"
        )


@dataclass
class TableMetadata:
    """Metadata for a single table."""

    name: str
    columns: Dict[str, str]  # column_name -> dtype
    primary_key: Optional[str] = None
    row_count: int = 0
    sample_values: Dict[str, List[str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"TableMetadata({self.name}, columns={len(self.columns)}, rows={self.row_count})"


@dataclass
class SchemaMetadata:
    """Complete schema metadata."""

    tables: Dict[str, TableMetadata]
    foreign_keys: List[ForeignKey]
    target_table: str

    def save(self, path: str | Path) -> None:
        """Save metadata to JSON file.

        Args:
            path: Path to save JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved schema metadata to {path}")

    @classmethod
    def load(cls, path: str | Path) -> SchemaMetadata:
        """Load metadata from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            SchemaMetadata instance
        """
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "tables": {
                name: {
                    "columns": meta.columns,
                    "primary_key": meta.primary_key,
                    "row_count": meta.row_count,
                    "sample_values": meta.sample_values,
                }
                for name, meta in self.tables.items()
            },
            "foreign_keys": [
                {
                    "child_table": fk.child_table,
                    "child_column": fk.child_column,
                    "parent_table": fk.parent_table,
                    "parent_column": fk.parent_column,
                    "coverage": fk.coverage,
                }
                for fk in self.foreign_keys
            ],
            "target_table": self.target_table,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> SchemaMetadata:
        """Create from dictionary.

        Args:
            data: Dictionary with schema data

        Returns:
            SchemaMetadata instance
        """
        tables = {
            name: TableMetadata(
                name=name,
                columns=meta["columns"],
                primary_key=meta.get("primary_key"),
                row_count=meta.get("row_count", 0),
                sample_values=meta.get("sample_values", {}),
            )
            for name, meta in data["tables"].items()
        }

        foreign_keys = [
            ForeignKey(
                child_table=fk["child_table"],
                child_column=fk["child_column"],
                parent_table=fk["parent_table"],
                parent_column=fk["parent_column"],
                coverage=fk["coverage"],
            )
            for fk in data["foreign_keys"]
        ]

        return cls(
            tables=tables,
            foreign_keys=foreign_keys,
            target_table=data["target_table"],
        )

    def get_related_tables(self, table_name: str) -> List[str]:
        """Get all tables related to the given table via foreign keys.

        Args:
            table_name: Table name

        Returns:
            List of related table names
        """
        related = set()

        for fk in self.foreign_keys:
            if fk.child_table == table_name:
                related.add(fk.parent_table)
            elif fk.parent_table == table_name:
                related.add(fk.child_table)

        return list(related)

    def get_foreign_keys_for_table(
        self, table_name: str, direction: str = "both"
    ) -> List[ForeignKey]:
        """Get foreign keys involving a table.

        Args:
            table_name: Table name
            direction: 'outgoing' (child), 'incoming' (parent), or 'both'

        Returns:
            List of ForeignKey objects
        """
        if direction == "outgoing":
            return [fk for fk in self.foreign_keys if fk.child_table == table_name]
        elif direction == "incoming":
            return [fk for fk in self.foreign_keys if fk.parent_table == table_name]
        else:  # both
            return [
                fk
                for fk in self.foreign_keys
                if fk.child_table == table_name or fk.parent_table == table_name
            ]

    def __repr__(self) -> str:
        return (
            f"SchemaMetadata(tables={len(self.tables)}, "
            f"fks={len(self.foreign_keys)}, target={self.target_table})"
        )


class SchemaDetector:
    """Schema detection with foreign key inference."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize schema detector.

        Args:
            config: Configuration dict (uses global config if None)
        """
        self.config = config or get_config().get("schema", {})
        self.fk_config = self.config.get("fk_detection", {})
        self.min_coverage = self.fk_config.get("min_coverage", 0.9)
        self.tolerance = self.fk_config.get("inclusion_tolerance", 0.1)

    def detect(
        self,
        tables: Dict[str, pd.DataFrame],
        target_table: str,
        provided_schema: Optional[Dict] = None,
    ) -> SchemaMetadata:
        """Detect schema and infer foreign keys.

        Args:
            tables: Dict mapping table_name -> DataFrame
            target_table: Name of the target table
            provided_schema: Optional schema dict with FK information

        Returns:
            SchemaMetadata object

        Example:
            >>> tables = {"orders": orders_df, "customers": customers_df}
            >>> detector = SchemaDetector()
            >>> metadata = detector.detect(tables, target_table="orders")
        """
        logger.info(f"Detecting schema for {len(tables)} tables")

        if target_table not in tables:
            raise ValueError(
                f"Target table '{target_table}' not found in tables: {list(tables.keys())}"
            )

        # 1. Extract table metadata
        table_metadata = self._extract_table_metadata(tables)

        # 2. Detect foreign keys
        if provided_schema and "foreign_keys" in provided_schema:
            logger.info("Using provided schema for foreign keys")
            fks = self._parse_provided_fks(provided_schema["foreign_keys"], tables)
        else:
            logger.info("Inferring foreign keys from data")
            fks = self._detect_foreign_keys(tables, table_metadata)

        logger.info(f"Detected {len(fks)} foreign key relationships")
        for fk in fks:
            logger.debug(f"  {fk}")

        return SchemaMetadata(
            tables=table_metadata,
            foreign_keys=fks,
            target_table=target_table,
        )

    def _extract_table_metadata(
        self, tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, TableMetadata]:
        """Extract metadata from tables.

        Args:
            tables: Dict of DataFrames

        Returns:
            Dict of TableMetadata objects
        """
        metadata = {}

        for name, df in tables.items():
            # Collect sample values for each column (first 3 non-null values)
            sample_values = {}
            for col in df.columns:
                non_null = df[col].dropna().head(3).astype(str).tolist()
                if non_null:
                    sample_values[col] = non_null

            metadata[name] = TableMetadata(
                name=name,
                columns={col: str(dtype) for col, dtype in df.dtypes.items()},
                primary_key=self._infer_primary_key(df),
                row_count=len(df),
                sample_values=sample_values,
            )

            logger.debug(
                f"Table {name}: {len(df.columns)} columns, {len(df)} rows, "
                f"PK={metadata[name].primary_key}"
            )

        return metadata

    def _infer_primary_key(self, df: pd.DataFrame) -> Optional[str]:
        """Infer primary key column.

        Args:
            df: DataFrame

        Returns:
            Primary key column name or None
        """
        # Priority 1: Column named 'id'
        if "id" in df.columns and df["id"].is_unique and not df["id"].isna().any():
            return "id"

        # Priority 2: Column ending with '_id' that is unique
        for col in df.columns:
            if (
                col.endswith("_id")
                and df[col].is_unique
                and not df[col].isna().any()
            ):
                return col

        # Priority 3: Any unique column without nulls
        for col in df.columns:
            if df[col].is_unique and not df[col].isna().any():
                return col

        return None

    def _detect_foreign_keys(
        self,
        tables: Dict[str, pd.DataFrame],
        table_metadata: Dict[str, TableMetadata],
    ) -> List[ForeignKey]:
        """Detect foreign key relationships.

        Args:
            tables: Dict of DataFrames
            table_metadata: Dict of TableMetadata

        Returns:
            List of ForeignKey objects
        """
        fks = []

        for child_name, child_df in tables.items():
            for child_col in child_df.columns:
                # Heuristic 1: Column name ends with _id or _key
                if not (child_col.endswith("_id") or child_col.endswith("_key")):
                    continue

                # Skip if this is the table's own primary key
                if child_col == table_metadata[child_name].primary_key:
                    continue

                # Heuristic 2: Extract potential parent table name
                parent_candidates = self._get_parent_candidates(
                    child_col, list(tables.keys())
                )

                for parent_name in parent_candidates:
                    if parent_name == child_name:  # Skip self-references
                        continue

                    parent_df = tables[parent_name]
                    parent_pk = table_metadata[parent_name].primary_key

                    if parent_pk is None or parent_pk not in parent_df.columns:
                        continue

                    # Check inclusion dependency
                    coverage = self._check_inclusion(
                        child_df[child_col],
                        parent_df[parent_pk],
                    )

                    if coverage >= self.min_coverage:
                        fks.append(
                            ForeignKey(
                                child_table=child_name,
                                child_column=child_col,
                                parent_table=parent_name,
                                parent_column=parent_pk,
                                coverage=coverage,
                            )
                        )
                        break  # Found FK, stop searching for this column

        return fks

    def _get_parent_candidates(
        self, column_name: str, table_names: List[str]
    ) -> List[str]:
        """Get potential parent table names from column name.

        Args:
            column_name: Column name (e.g., 'customer_id')
            table_names: Available table names

        Returns:
            List of candidate parent table names
        """
        # Remove suffixes
        base_name = (
            column_name.replace("_id", "")
            .replace("_key", "")
            .replace("_fk", "")
        )

        candidates = []

        # Exact match
        if base_name in table_names:
            candidates.append(base_name)

        # Plural forms
        if base_name + "s" in table_names:
            candidates.append(base_name + "s")

        # Try removing common prefixes
        for table_name in table_names:
            if table_name.lower().startswith(base_name.lower()):
                candidates.append(table_name)

        return candidates

    def _check_inclusion(
        self,
        child_values: pd.Series,
        parent_values: pd.Series,
    ) -> float:
        """Check inclusion dependency (child ⊆ parent).

        Args:
            child_values: Child column values
            parent_values: Parent column values

        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        child_set = set(child_values.dropna())
        parent_set = set(parent_values.dropna())

        if not child_set:
            return 0.0

        overlap = child_set & parent_set
        return len(overlap) / len(child_set)

    def _parse_provided_fks(
        self, fk_list: List[Dict], tables: Dict[str, pd.DataFrame]
    ) -> List[ForeignKey]:
        """Parse foreign keys from provided schema.

        Args:
            fk_list: List of FK dicts
            tables: Dict of DataFrames (for validation)

        Returns:
            List of ForeignKey objects
        """
        fks = []

        for fk_dict in fk_list:
            child_table = fk_dict["child_table"]
            child_column = fk_dict["child_column"]
            parent_table = fk_dict["parent_table"]
            parent_column = fk_dict["parent_column"]

            # Validate
            if child_table not in tables:
                logger.warning(f"Child table {child_table} not found, skipping FK")
                continue

            if parent_table not in tables:
                logger.warning(f"Parent table {parent_table} not found, skipping FK")
                continue

            # Calculate coverage
            coverage = self._check_inclusion(
                tables[child_table][child_column],
                tables[parent_table][parent_column],
            )

            fks.append(
                ForeignKey(
                    child_table=child_table,
                    child_column=child_column,
                    parent_table=parent_table,
                    parent_column=parent_column,
                    coverage=coverage,
                )
            )

        return fks
