"""Main QA generator class that coordinates all components.

Generates QA pairs based on difficulty strategies by:
1. Selecting strategies based on configured weights
2. Building SQL queries with appropriate JOINs and filters
3. Generating natural language questions using LLM
4. Extracting answer record IDs
5. Validating QA pairs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from talk2metadata.agent import AgentWrapper
from talk2metadata.core.qa.difficulty_classifier import DifficultyClassifier
from talk2metadata.core.qa.qa_pair import QAPair
from talk2metadata.core.qa.query_builder import QueryBuilder
from talk2metadata.core.qa.question_generator import QuestionGenerator
from talk2metadata.core.qa.strategy_selector import StrategySelector
from talk2metadata.core.qa.verifier import QAVerifier
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
        strategy_weights: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
        max_answer_records: int = 10,
    ):
        """Initialize QA generator.

        Args:
            schema: Schema metadata
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance (shared across components)
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            strategy_weights: Optional weights for specific strategies
            tier_weights: Optional weights for tiers
            max_answer_records: Maximum number of answer records per question (default: 10)
                                Questions with more records are considered too general
        """
        self.schema = schema
        self.tables = tables
        self.target_table = schema.target_table

        # Initialize agent
        if agent is None:
            agent = AgentWrapper(provider=provider, model=model)
        self.agent = agent

        # Initialize components
        self.classifier = DifficultyClassifier()
        self.strategy_selector = StrategySelector(
            strategy_weights=strategy_weights, tier_weights=tier_weights
        )
        self.query_builder = QueryBuilder(schema, tables)
        self.question_generator = QuestionGenerator(agent, schema)
        self.verifier = QAVerifier(agent, max_answer_records=max_answer_records)

    def generate(
        self,
        total_qa_pairs: int = 100,
        pairs_per_strategy: Optional[int] = None,
        validate: bool = True,
        filter_valid: bool = True,
    ) -> List[QAPair]:
        """Generate QA pairs based on difficulty strategies.

        Args:
            total_qa_pairs: Total number of QA pairs to generate
            pairs_per_strategy: If specified, generate this many pairs per strategy
                              (overrides total_qa_pairs and weights)
            validate: Whether to validate QA pairs
            filter_valid: Whether to filter out invalid QA pairs

        Returns:
            List of QAPair objects
        """
        logger.info(
            f"Generating QA pairs: total={total_qa_pairs}, "
            f"pairs_per_strategy={pairs_per_strategy}"
        )

        # Step 1: Select strategies
        logger.info("Step 1: Selecting difficulty strategies...")
        strategies = self.strategy_selector.select_strategies(
            total_count=total_qa_pairs, pairs_per_strategy=pairs_per_strategy
        )
        logger.info(f"Selected {len(strategies)} strategy instances")

        # Step 2: Generate queries for each strategy
        logger.info("Step 2: Generating SQL queries...")
        queries = []
        for i, strategy in enumerate(strategies):
            try:
                query_spec = self.query_builder.build_query(strategy)
                if query_spec:
                    queries.append(query_spec)
                    if (i + 1) % 10 == 0:
                        logger.debug(f"Generated {i + 1}/{len(strategies)} queries")
            except Exception as e:
                logger.warning(f"Failed to generate query for strategy {strategy}: {e}")
                continue

        logger.info(f"Successfully generated {len(queries)} queries")

        # Step 3: Generate natural language questions
        logger.info("Step 3: Generating natural language questions...")
        qa_pairs = []
        for i, query_spec in enumerate(queries):
            try:
                question = self.question_generator.generate(query_spec)

                # Create QAPair
                qa_pair = QAPair(
                    question=question,
                    answer_row_ids=query_spec.answer_row_ids,
                    sql=query_spec.sql,
                    strategy=query_spec.strategy,
                    difficulty_score=self.classifier.get_score(query_spec.strategy),
                    involved_tables=query_spec.involved_tables,
                    involved_columns=query_spec.involved_columns,
                    metadata={
                        "target_table": self.target_table,
                    },
                )

                qa_pairs.append(qa_pair)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Generated {i + 1}/{len(queries)} questions")

            except Exception as e:
                logger.warning(f"Failed to generate question for query: {e}")
                continue

        logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")

        # Step 4: Validate QA pairs
        if validate:
            logger.info("Step 4: Validating QA pairs...")
            self.verifier.verify_batch(qa_pairs)

            valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
            logger.info(f"Validation complete: {valid_count}/{len(qa_pairs)} valid")

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

        # Compute statistics
        total_qa_pairs = len(qa_pairs)
        valid_qa_pairs = sum(1 for qa in qa_pairs if qa.is_valid)

        # Group by strategy
        strategy_distribution = {}
        for qa in qa_pairs:
            strategy_distribution[qa.strategy] = (
                strategy_distribution.get(qa.strategy, 0) + 1
            )

        # Group by tier
        tier_distribution = {}
        for qa in qa_pairs:
            tier = qa.tier
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        data = {
            "target_table": self.target_table,
            "total_qa_pairs": total_qa_pairs,
            "valid_qa_pairs": valid_qa_pairs,
            "strategy_distribution": strategy_distribution,
            "tier_distribution": tier_distribution,
            "qa_pairs": [qa.to_dict() for qa in qa_pairs],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(qa_pairs)} QA pairs to {output_path}")
        return output_path

    @classmethod
    def load(cls, qa_pairs_path: Path | str) -> List[QAPair]:
        """Load QA pairs from file.

        Args:
            qa_pairs_path: Path to QA pairs JSON file

        Returns:
            List of QAPair objects
        """
        qa_pairs_path = Path(qa_pairs_path)
        with open(qa_pairs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        qa_pairs = [QAPair.from_dict(qa) for qa in data["qa_pairs"]]
        logger.info(f"Loaded {len(qa_pairs)} QA pairs from {qa_pairs_path}")
        return qa_pairs

    @classmethod
    def from_schema_file(
        cls,
        schema_path: Path | str,
        tables: Dict[str, pd.DataFrame],
        agent: Optional[AgentWrapper] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        strategy_weights: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
        max_answer_records: int = 10,
    ) -> QAGenerator:
        """Create QAGenerator from schema file.

        Args:
            schema_path: Path to schema JSON file
            tables: Dictionary mapping table names to DataFrames
            agent: Optional AgentWrapper instance
            provider: LLM provider name (if agent is None)
            model: LLM model name (if agent is None)
            strategy_weights: Optional weights for specific strategies
            tier_weights: Optional weights for tiers
            max_answer_records: Maximum number of answer records per question (default: 10)

        Returns:
            QAGenerator instance
        """
        schema = SchemaMetadata.load(schema_path)
        return cls(
            schema,
            tables,
            agent=agent,
            provider=provider,
            model=model,
            strategy_weights=strategy_weights,
            tier_weights=tier_weights,
            max_answer_records=max_answer_records,
        )
