"""Business logic for QA generation commands."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from talk2metadata.core.qa import QAGenerator, QAPair
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import Config
from talk2metadata.utils.paths import get_qa_dir


class QAHandler:
    """Handler for QA generation operations.

    Encapsulates business logic for QA generation commands,
    keeping CLI commands thin and focused on user interaction.

    Example:
        >>> handler = QAHandler(config)
        >>> qa_pairs, path = handler.generate_qa_pairs_strategy_based(schema, tables)
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config

    def generate_qa_pairs_strategy_based(
        self,
        schema: SchemaMetadata,
        tables: Dict[str, pd.DataFrame],
        total_qa_pairs: Optional[int] = None,
        pairs_per_strategy: Optional[int] = None,
        validate: Optional[bool] = None,
        filter_valid: Optional[bool] = None,
        output_file: Optional[str] = None,
        run_id: Optional[str] = None,
        max_answer_records: Optional[int] = None,
    ) -> Tuple[List[QAPair], Optional[Path]]:
        """Generate QA pairs based on difficulty strategies.

        Args:
            schema: Schema metadata
            tables: Dictionary of DataFrames
            total_qa_pairs: Total number of QA pairs to generate (uses config if None)
            pairs_per_strategy: If specified, generate this many pairs per strategy
            validate: Whether to validate QA pairs (uses config if None)
            filter_valid: Whether to filter invalid pairs (uses config if None)
            output_file: Optional output file path
            run_id: Optional run ID
            max_answer_records: Maximum answer records per question (uses config if None)

        Returns:
            Tuple of (qa_pairs, output_path)
        """
        qa_config = self.config.get("qa_generation", {})
        agent_config = self.config.get("agent", {})

        total_qa_pairs = total_qa_pairs or qa_config.get("total_qa_pairs", 100)
        validate = validate if validate is not None else qa_config.get("validate", True)
        filter_valid = (
            filter_valid
            if filter_valid is not None
            else qa_config.get("filter_valid", True)
        )
        auto_save = qa_config.get("auto_save", True)

        provider = agent_config.get("provider")
        model = agent_config.get("model")

        # Get strategy or tier weights
        strategy_weights = qa_config.get("strategy_weights")
        tier_weights = qa_config.get("tier_weights")
        max_answer_records = (
            max_answer_records
            if max_answer_records is not None
            else qa_config.get("max_answer_records", 10)
        )

        # Create generator
        generator = QAGenerator(
            schema=schema,
            tables=tables,
            provider=provider,
            model=model,
            strategy_weights=strategy_weights,
            tier_weights=tier_weights,
            max_answer_records=max_answer_records,
        )

        # Generate QA pairs
        qa_pairs = generator.generate(
            total_qa_pairs=total_qa_pairs,
            pairs_per_strategy=pairs_per_strategy,
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
            # Reload merged QA pairs from file to get the complete list
            # (including any existing QA pairs that were merged)
            merged_qa_pairs = QAGenerator.load(saved_path)
            return merged_qa_pairs, saved_path

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
                "strategies": {},
                "tiers": {},
            }

        valid_count = sum(1 for qa in qa_pairs if qa.is_valid)

        # Strategy distribution
        strategy_stats = {}
        for qa in qa_pairs:
            strategy = qa.strategy or "unknown"
            strategy_stats[strategy] = strategy_stats.get(strategy, 0) + 1

        # Tier distribution
        tier_stats = {}
        for qa in qa_pairs:
            tier = qa.tier
            tier_stats[tier] = tier_stats.get(tier, 0) + 1

        return {
            "total": len(qa_pairs),
            "valid": valid_count,
            "invalid": len(qa_pairs) - valid_count,
            "strategies": strategy_stats,
            "tiers": tier_stats,
        }
