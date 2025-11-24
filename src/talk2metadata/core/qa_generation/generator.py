"""Main QA generator class that coordinates all components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa_generation.answer_extractor import AnswerExtractor
from talk2metadata.core.qa_generation.path_instantiator import PathInstantiator
from talk2metadata.core.qa_generation.pattern_generator import PathPatternGenerator
from talk2metadata.core.qa_generation.patterns import PathPattern
from talk2metadata.core.qa_generation.qa_pair import QAPair
from talk2metadata.core.qa_generation.question_generator import QuestionGenerator
from talk2metadata.core.qa_generation.validator import QAValidator
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class QAGenerator:
    """Main class for generating QA pairs from database schema and data."""

    def __init__(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize QA generator.

        Args:
            schema: Schema metadata
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance (shared across components)
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
        """
        self.schema = schema
        self.tables = tables
        self.target_table = schema.target_table

        # Initialize components
        if agent is None:
            agent = AgentWrapper(provider=provider, model=model)

        self.pattern_generator = PathPatternGenerator(agent=agent)
        self.path_instantiator = PathInstantiator(tables, schema)
        self.question_generator = QuestionGenerator(agent=agent)
        self.answer_extractor = AnswerExtractor(schema.target_table)
        self.validator = QAValidator(agent=agent)

    def generate_patterns(
        self,
        num_patterns: int = 15,
        save_path: Optional[Path | str] = None,
    ) -> List[PathPattern]:
        """Generate path patterns.

        Args:
            num_patterns: Number of path patterns to generate
            save_path: Optional path to save patterns

        Returns:
            List of PathPattern objects
        """
        logger.info(f"Generating {num_patterns} path patterns...")
        patterns = self.pattern_generator.generate_patterns(
            self.schema, num_patterns=num_patterns
        )
        logger.info(f"Generated {len(patterns)} path patterns")

        # Save patterns if requested
        if save_path:
            self.save_patterns(patterns, save_path)
            logger.info(f"Saved patterns to {save_path}")

        return patterns

    def generate(
        self,
        num_patterns: int = 15,
        instances_per_pattern: int = 5,
        validate: bool = True,
        filter_valid: bool = True,
        patterns: Optional[List[PathPattern]] = None,
    ) -> List[QAPair]:
        """Generate QA pairs.

        Args:
            num_patterns: Number of path patterns to generate (if patterns not provided)
            instances_per_pattern: Number of path instances per pattern
            validate: Whether to validate QA pairs
            filter_valid: Whether to filter out invalid QA pairs
            patterns: Optional pre-generated patterns to use

        Returns:
            List of QAPair objects
        """
        logger.info(
            f"Generating QA pairs: {num_patterns if patterns is None else len(patterns)} patterns, "
            f"{instances_per_pattern} instances per pattern"
        )

        # Step 1: Generate path patterns (if not provided)
        if patterns is None:
            logger.info("Step 1: Generating path patterns...")
            patterns = self.pattern_generator.generate_patterns(
                self.schema, num_patterns=num_patterns
            )
            logger.info(f"Generated {len(patterns)} path patterns")
        else:
            logger.info(f"Using {len(patterns)} provided path patterns")

        # Step 2: Instantiate paths
        logger.info("Step 2: Instantiating paths from data...")
        all_instances = []
        for pattern in patterns:
            instances = self.path_instantiator.instantiate(
                pattern, num_instances=instances_per_pattern
            )
            all_instances.extend(instances)
        logger.info(f"Generated {len(all_instances)} path instances")

        # Step 3: Generate questions and answers
        logger.info("Step 3: Generating questions and extracting answers...")
        qa_pairs = []
        for instance in all_instances:
            try:
                question = self.question_generator.generate(instance)
                answer_row_ids = self.answer_extractor.extract(instance)

                qa_pair = QAPair(
                    question=question,
                    answer_row_ids=answer_row_ids,
                    path_instance=instance,
                    difficulty=instance.pattern.difficulty,
                    metadata={
                        "pattern": instance.pattern.pattern,
                        "semantic": instance.pattern.semantic,
                    },
                )

                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.warning(f"Failed to generate QA for instance: {e}")
                continue

        logger.info(f"Generated {len(qa_pairs)} QA pairs")

        # Step 4: Validate QA pairs
        if validate:
            logger.info("Step 4: Validating QA pairs...")
            self.validator.validate_batch(qa_pairs)
            valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
            logger.info(f"Validated: {valid_count}/{len(qa_pairs)} valid")

            if filter_valid:
                qa_pairs = [qa for qa in qa_pairs if qa.is_valid]
                logger.info(f"Filtered to {len(qa_pairs)} valid QA pairs")

        return qa_pairs

    def generate_qa_from_patterns(
        self,
        patterns: List[PathPattern],
        instances_per_pattern: int = 5,
        validate: bool = True,
        filter_valid: bool = True,
    ) -> List[QAPair]:
        """Generate QA pairs from provided path patterns.

        This is a separate method from generate() to allow two-step workflow:
        1. Generate patterns (with review)
        2. Generate QA pairs from patterns

        Args:
            patterns: List of path patterns to use
            instances_per_pattern: Number of path instances per pattern
            validate: Whether to validate QA pairs
            filter_valid: Whether to filter out invalid QA pairs

        Returns:
            List of QAPair objects
        """
        logger.info(
            f"Generating QA pairs from {len(patterns)} patterns, "
            f"{instances_per_pattern} instances per pattern"
        )

        # Step 1: Instantiate paths
        logger.info("Step 1: Instantiating paths from data...")
        all_instances = []
        for pattern in patterns:
            instances = self.path_instantiator.instantiate(
                pattern, num_instances=instances_per_pattern
            )
            all_instances.extend(instances)
        logger.info(f"Generated {len(all_instances)} path instances")

        # Step 2: Generate questions and answers
        logger.info("Step 2: Generating questions and extracting answers...")
        qa_pairs = []
        for instance in all_instances:
            try:
                question = self.question_generator.generate(instance)
                answer_row_ids = self.answer_extractor.extract(instance)

                qa_pair = QAPair(
                    question=question,
                    answer_row_ids=answer_row_ids,
                    path_instance=instance,
                    difficulty=instance.pattern.difficulty,
                    metadata={
                        "pattern": instance.pattern.pattern,
                        "semantic": instance.pattern.semantic,
                    },
                )

                qa_pairs.append(qa_pair)
            except Exception as e:
                logger.warning(f"Failed to generate QA for instance: {e}")
                continue

        logger.info(f"Generated {len(qa_pairs)} QA pairs")

        # Step 3: Validate QA pairs
        if validate:
            logger.info("Step 3: Validating QA pairs...")
            self.validator.validate_batch(qa_pairs)
            valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
            logger.info(f"Validated: {valid_count}/{len(qa_pairs)} valid")

            if filter_valid:
                qa_pairs = [qa for qa in qa_pairs if qa.is_valid]
                logger.info(f"Filtered to {len(qa_pairs)} valid QA pairs")

        return qa_pairs

    def save(
        self,
        qa_pairs: List[QAPair],
        output_path: Optional[Path | str] = None,
        auto_save: bool = True,
        run_id: Optional[str] = None,
    ) -> Path:
        """Save QA pairs to file.

        Args:
            qa_pairs: List of QA pairs to save
            output_path: Optional explicit path to save QA pairs
            auto_save: If True and output_path is None, auto-save to qa/qa_pairs.json
            run_id: Optional run ID for auto-save path

        Returns:
            Path where QA pairs were saved
        """
        if output_path is None and auto_save:
            # Auto-save to qa/qa_pairs.json in run directory
            from talk2metadata.utils.paths import get_qa_dir

            qa_dir = get_qa_dir(run_id)
            qa_dir.mkdir(parents=True, exist_ok=True)
            output_path = qa_dir / "qa_pairs.json"
        elif output_path is None:
            raise ValueError(
                "Either output_path must be provided or auto_save must be True"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "target_table": self.target_table,
            "total_qa_pairs": len(qa_pairs),
            "valid_qa_pairs": sum(1 for qa in qa_pairs if qa.is_valid),
            "qa_pairs": [qa.to_dict() for qa in qa_pairs],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
        return output_path

    def save_patterns(
        self, patterns: List[PathPattern], output_path: Path | str
    ) -> None:
        """Save path patterns to file for review.

        Args:
            patterns: List of PathPattern objects to save
            output_path: Path to output file (JSON)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "target_table": self.target_table,
            "total_patterns": len(patterns),
            "patterns": [p.to_dict() for p in patterns],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(patterns)} path patterns to {output_path}")

    def load_patterns(self, patterns_path: Path | str) -> List[PathPattern]:
        """Load path patterns from file.

        Args:
            patterns_path: Path to patterns JSON file

        Returns:
            List of PathPattern objects
        """
        patterns_path = Path(patterns_path)
        with open(patterns_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        patterns = [PathPattern.from_dict(p) for p in data["patterns"]]
        logger.info(f"Loaded {len(patterns)} path patterns from {patterns_path}")
        return patterns

    @classmethod
    def from_schema_file(
        cls,
        schema_path: Path | str,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> QAGenerator:
        """Create QAGenerator from schema file.

        Args:
            schema_path: Path to schema JSON file
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)

        Returns:
            QAGenerator instance
        """
        schema = SchemaMetadata.load(schema_path)
        return cls(schema, tables, agent=agent, provider=provider, model=model)
