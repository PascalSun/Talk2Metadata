"""Path pattern and instance definitions for QA generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PathPattern:
    """A path pattern template for generating queries.

    A path pattern defines a sequence of tables connected by foreign keys,
    representing a meaningful query pattern that users might ask.
    """

    pattern: List[
        str
    ]  # Sequence of table names, e.g., ["historic_titles", "wamex_reports"]
    semantic: str  # Semantic description of what this path represents
    question_template: str  # Template for generating questions, e.g., "哪些报告的标题包含'{historic_title}'？"
    answer_type: str  # "single" | "multiple" | "aggregate"
    difficulty: str  # "easy" | "medium" | "hard"
    description: Optional[str] = None  # Additional description

    def __repr__(self) -> str:
        return f"PathPattern(pattern={self.pattern}, difficulty={self.difficulty})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern": self.pattern,
            "semantic": self.semantic,
            "question_template": self.question_template,
            "answer_type": self.answer_type,
            "difficulty": self.difficulty,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PathPattern":
        """Create from dictionary."""
        return cls(
            pattern=data["pattern"],
            semantic=data["semantic"],
            question_template=data["question_template"],
            answer_type=data.get("answer_type", "multiple"),
            difficulty=data.get("difficulty", "medium"),
            description=data.get("description"),
        )


@dataclass
class PathNode:
    """A node in an instantiated path."""

    table: str
    row_id: Any  # Primary key value
    data: Dict[str, Any]  # Full row data
    anumber: Optional[Any] = None  # ANumber/Anumber value for FK linking

    def __repr__(self) -> str:
        return f"PathNode(table={self.table}, row_id={self.row_id})"


@dataclass
class PathInstance:
    """An instantiated path from real data.

    This represents a concrete path through the knowledge graph,
    following a PathPattern but using actual data records.
    """

    pattern: PathPattern
    nodes: List[PathNode]  # Sequence of nodes in the path
    target_node_indices: List[int] = field(
        default_factory=list
    )  # Indices of nodes that are in target table
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata

    def get_target_nodes(self) -> List[PathNode]:
        """Get all nodes that are in the target table."""
        return [self.nodes[i] for i in self.target_node_indices]

    def get_answer_row_ids(self) -> List[Any]:
        """Get row IDs of target table nodes (the answer)."""
        return [node.row_id for node in self.get_target_nodes()]

    def __repr__(self) -> str:
        return f"PathInstance(pattern={self.pattern.pattern}, nodes={len(self.nodes)}, answers={len(self.target_node_indices)})"
