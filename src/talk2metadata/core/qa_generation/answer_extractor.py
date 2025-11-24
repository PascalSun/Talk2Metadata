"""Answer extractor for extracting target table row IDs from paths."""

from __future__ import annotations

from typing import Any, List

from talk2metadata.core.qa_generation.patterns import PathInstance
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class AnswerExtractor:
    """Extract answers (target table row IDs) from path instances."""

    def __init__(self, target_table: str):
        """Initialize answer extractor.

        Args:
            target_table: Name of the target table
        """
        self.target_table = target_table

    def extract(self, path_instance: PathInstance) -> List[Any]:
        """Extract answer row IDs from a path instance.

        Args:
            path_instance: Path instance

        Returns:
            List of target table row IDs (the answer)
        """
        return path_instance.get_answer_row_ids()

    def extract_all(
        self, path_instances: List[PathInstance]
    ) -> List[List[Any]]:
        """Extract answers from multiple path instances.

        Args:
            path_instances: List of path instances

        Returns:
            List of answer lists (one per instance)
        """
        return [self.extract(pi) for pi in path_instances]

