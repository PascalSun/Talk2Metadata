"""Schema data types and models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    description: Optional[str] = None  # Human-readable description of the table
    column_descriptions: Dict[str, str] = field(
        default_factory=dict
    )  # column_name -> description

    def __repr__(self) -> str:
        return f"TableMetadata({self.name}, columns={len(self.columns)}, rows={self.row_count})"
