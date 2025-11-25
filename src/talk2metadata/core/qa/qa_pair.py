"""QA pair definition with comprehensive metadata.

Contains question, answer SQL, answer record IDs, strategy, and related metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QAPair:
    """A question-answer pair for evaluation."""

    # Core QA data
    question: str  # Natural language question
    answer_row_ids: List[Any]  # List of target table row IDs (the answer)
    sql: str  # SQL query that produces the answer

    # Strategy and difficulty
    strategy: str  # Difficulty code (e.g., "2iM")
    difficulty_score: float  # Numeric difficulty score

    # Related tables and columns
    involved_tables: List[str]  # All tables involved in the query
    involved_columns: List[str]  # All columns used in filters (table.column format)

    # Validation status
    is_valid: Optional[bool] = None  # Whether this QA pair passed validation
    validation_errors: List[str] = field(default_factory=list)

    # SQL validation status
    sql_valid: Optional[bool] = None  # Whether SQL executed successfully
    sql_validation_error: Optional[str] = None  # Error message if SQL execution failed

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    @property
    def answer_count(self) -> int:
        """Number of answer records."""
        return len(self.answer_row_ids)

    @property
    def tier(self) -> str:
        """Difficulty tier (easy/medium/hard/expert)."""
        if self.difficulty_score < 1.0:
            return "easy"
        elif self.difficulty_score < 2.0:
            return "medium"
        elif self.difficulty_score < 3.0:
            return "hard"
        else:
            return "expert"

    def __repr__(self) -> str:
        return (
            f"QAPair(question='{self.question[:50]}...', "
            f"answers={len(self.answer_row_ids)}, "
            f"strategy={self.strategy}, "
            f"valid={self.is_valid})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answer_row_ids": self.answer_row_ids,
            "answer_count": self.answer_count,
            "sql": self.sql,
            "strategy": self.strategy,
            "difficulty_score": self.difficulty_score,
            "tier": self.tier,
            "involved_tables": self.involved_tables,
            "involved_columns": self.involved_columns,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "sql_valid": self.sql_valid,
            "sql_validation_error": self.sql_validation_error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QAPair:
        """Create QAPair from dictionary.

        Args:
            data: Dictionary with QA pair data

        Returns:
            QAPair instance
        """
        return cls(
            question=data["question"],
            answer_row_ids=data["answer_row_ids"],
            sql=data["sql"],
            strategy=data["strategy"],
            difficulty_score=data["difficulty_score"],
            involved_tables=data["involved_tables"],
            involved_columns=data["involved_columns"],
            is_valid=data.get("is_valid"),
            validation_errors=data.get("validation_errors", []),
            sql_valid=data.get("sql_valid"),
            sql_validation_error=data.get("sql_validation_error"),
            metadata=data.get("metadata", {}),
        )
