"""Business logic for QA generation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from talk2metadata.core.qa_generation import QAGenerator
from talk2metadata.core.qa_generation.patterns import PathPattern
from talk2metadata.core.qa_generation.qa_pair import QAPair
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import Config
from talk2metadata.utils.paths import get_qa_dir


class QAHandler:
    """Handler for QA generation operations.

    Encapsulates business logic for QA generation commands,
    keeping CLI commands thin and focused on user interaction.

    Example:
        >>> handler = QAHandler(config)
        >>> patterns, path = handler.generate_patterns(schema, tables)
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config

    def generate_patterns(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        num_patterns: Optional[int] = None,
        output_file: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[PathPattern], Path]:
        """Generate path patterns for QA generation.

        Args:
            schema: Schema metadata
            tables: Dictionary of DataFrames
            num_patterns: Number of patterns to generate (uses config default if None)
            output_file: Optional output file path
            run_id: Optional run ID

        Returns:
            Tuple of (patterns, output_path)
        """
        qa_config = self.config.get("qa_generation", {})
        agent_config = self.config.get("agent", {})

        num_patterns = num_patterns or qa_config.get("num_patterns", 15)
        provider = agent_config.get("provider")
        model = agent_config.get("model")

        # Determine output path
        if output_file:
            output_path = Path(output_file)
        else:
            qa_dir = get_qa_dir(run_id or self.config.get("run_id"), self.config)
            qa_dir.mkdir(parents=True, exist_ok=True)
            output_path = qa_dir / "kg_paths.json"

        # Generate patterns
        generator = QAGenerator(
            schema=schema,
            tables=tables,
            provider=provider,
            model=model,
        )

        patterns = generator.generate_patterns(
            num_patterns=num_patterns,
            save_path=output_path,
        )

        return patterns, output_path

    def load_patterns(
        self,
        schema: SchemaMetadata,
        patterns_file: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[PathPattern], Path]:
        """Load path patterns from file.

        Args:
            schema: Schema metadata
            patterns_file: Optional path to patterns file
            run_id: Optional run ID

        Returns:
            Tuple of (patterns, patterns_path)

        Raises:
            FileNotFoundError: If patterns file not found
        """
        # Determine patterns file path
        if patterns_file:
            patterns_path = Path(patterns_file)
        else:
            qa_dir = get_qa_dir(run_id or self.config.get("run_id"), self.config)
            patterns_path = qa_dir / "kg_paths.json"

        if not patterns_path.exists():
            raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

        # Load patterns using generator
        generator = QAGenerator(
            schema=schema,
            tables={},  # Not needed for loading patterns
        )
        patterns = generator.load_patterns(patterns_path)

        return patterns, patterns_path

    def generate_qa_pairs(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        patterns: List[PathPattern],
        instances_per_pattern: Optional[int] = None,
        validate: Optional[bool] = None,
        filter_valid: Optional[bool] = None,
        output_file: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[List[QAPair], Optional[Path]]:
        """Generate QA pairs from path patterns.

        Args:
            schema: Schema metadata
            tables: Dictionary of DataFrames
            patterns: List of path patterns to instantiate
            instances_per_pattern: Number of instances per pattern (uses config if None)
            validate: Whether to validate QA pairs (uses config if None)
            filter_valid: Whether to filter invalid pairs (uses config if None)
            output_file: Optional output file path
            run_id: Optional run ID

        Returns:
            Tuple of (qa_pairs, output_path)
        """
        qa_config = self.config.get("qa_generation", {})
        agent_config = self.config.get("agent", {})

        instances_per_pattern = instances_per_pattern or qa_config.get(
            "instances_per_pattern", 5
        )
        validate = validate if validate is not None else qa_config.get("validate", True)
        filter_valid = (
            filter_valid
            if filter_valid is not None
            else qa_config.get("filter_valid", True)
        )
        auto_save = qa_config.get("auto_save", True)

        provider = agent_config.get("provider")
        model = agent_config.get("model")

        # Create generator
        generator = QAGenerator(
            schema=schema,
            tables=tables,
            provider=provider,
            model=model,
        )

        # Generate QA pairs
        qa_pairs = generator.generate_qa_from_patterns(
            patterns=patterns,
            instances_per_pattern=instances_per_pattern,
            validate=validate,
            filter_valid=filter_valid,
        )

        # Determine output path
        output_path = None
        if output_file:
            output_path = Path(output_file)
        elif auto_save:
            qa_dir = get_qa_dir(run_id or self.config.get("run_id"), self.config)
            qa_dir.mkdir(parents=True, exist_ok=True)
            output_path = qa_dir / "qa_pairs.json"

        # Save if output path specified
        if output_path:
            saved_path = generator.save(
                qa_pairs,
                output_path=output_path,
                auto_save=auto_save,
                run_id=run_id,
            )
            return qa_pairs, saved_path

        return qa_pairs, None

    def get_qa_statistics(self, qa_pairs: List[QAPair]) -> Dict[str, any]:
        """Calculate statistics for QA pairs.

        Args:
            qa_pairs: List of QA pairs

        Returns:
            Dictionary with statistics
        """
        if not qa_pairs:
            return {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "difficulties": {},
            }

        valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
        difficulty_stats = {}
        for qa in qa_pairs:
            diff = qa.difficulty or "unknown"
            difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1

        return {
            "total": len(qa_pairs),
            "valid": valid_count,
            "invalid": len(qa_pairs) - valid_count,
            "difficulties": difficulty_stats,
        }
