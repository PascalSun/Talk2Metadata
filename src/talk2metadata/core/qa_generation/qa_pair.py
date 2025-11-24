"""QA pair definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from talk2metadata.core.qa_generation.patterns import PathInstance


@dataclass
class QAPair:
    """A question-answer pair for evaluation."""

    question: str  # Natural language question
    answer_row_ids: List[Any]  # List of target table row IDs (the answer)
    path_instance: Optional[PathInstance] = None  # The path instance that generated this QA
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    # Quality metrics
    is_valid: Optional[bool] = None  # Whether this QA pair passed validation
    validation_errors: List[str] = field(default_factory=list)
    difficulty: Optional[str] = None  # "easy" | "medium" | "hard"
    answer_count: int = 0  # Number of answers (for statistics)

    def __post_init__(self):
        """Set answer_count after initialization."""
        if self.answer_count == 0:
            self.answer_count = len(self.answer_row_ids)

    def __repr__(self) -> str:
        return f"QAPair(question='{self.question[:50]}...', answers={len(self.answer_row_ids)}, valid={self.is_valid})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "answer_row_ids": self.answer_row_ids,
            "answer_count": self.answer_count,
            "difficulty": self.difficulty,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
        }

