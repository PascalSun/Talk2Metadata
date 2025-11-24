"""Strategy selector for QA generation.

Selects difficulty strategies based on configured weights.
"""

import random
from typing import Dict, List, Optional

from talk2metadata.core.qa.difficulty_classifier import DifficultyClassifier
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class StrategySelector:
    """Selects difficulty strategies based on weights."""

    def __init__(
        self,
        strategy_weights: Optional[Dict[str, int]] = None,
        tier_weights: Optional[Dict[str, int]] = None,
    ):
        """Initialize strategy selector.

        Args:
            strategy_weights: Optional weights for specific strategies
                             (e.g., {"0E": 10, "1pM": 5})
            tier_weights: Optional weights for tiers
                         (e.g., {"easy": 30, "medium": 50, "hard": 15, "expert": 5})
        """
        self.classifier = DifficultyClassifier()
        self.strategy_weights = strategy_weights
        self.tier_weights = tier_weights

        # Compute final weights
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute final strategy weights from tier or strategy weights."""
        all_strategies = self.classifier.get_all_strategies()

        if self.strategy_weights:
            # Use explicit strategy weights
            self.final_weights = {
                s: self.strategy_weights.get(s, 0) for s in all_strategies
            }
        elif self.tier_weights:
            # Use tier weights - distribute evenly within each tier
            self.final_weights = {}
            for tier, weight in self.tier_weights.items():
                strategies_in_tier = self.classifier.get_strategies_by_tier(tier)
                if strategies_in_tier:
                    weight_per_strategy = weight / len(strategies_in_tier)
                    for strategy in strategies_in_tier:
                        self.final_weights[strategy] = weight_per_strategy
        else:
            # Equal distribution
            self.final_weights = {s: 1.0 for s in all_strategies}

        # Normalize weights
        total = sum(self.final_weights.values())
        if total > 0:
            self.final_weights = {s: w / total for s, w in self.final_weights.items()}

        logger.debug(f"Computed strategy weights: {self.final_weights}")

    def select_strategies(
        self,
        total_count: int,
        pairs_per_strategy: Optional[int] = None,
    ) -> List[str]:
        """Select strategies for QA generation.

        Args:
            total_count: Total number of QA pairs to generate
            pairs_per_strategy: If specified, generate this many pairs per strategy
                              (overrides total_count and weights)

        Returns:
            List of selected strategies (with duplicates for multiple instances)
        """
        if pairs_per_strategy is not None:
            # Generate fixed number per strategy
            strategies = []
            for strategy in self.classifier.get_all_strategies():
                if self.final_weights.get(strategy, 0) > 0:
                    strategies.extend([strategy] * pairs_per_strategy)
            return strategies

        # Select strategies based on weights
        strategy_list = list(self.final_weights.keys())
        weights = list(self.final_weights.values())

        # Remove strategies with zero weight
        strategy_list = [s for s, w in zip(strategy_list, weights) if w > 0]
        weights = [w for w in weights if w > 0]

        if not strategy_list:
            logger.warning("No strategies with non-zero weights")
            return []

        # Use random.choices to select strategies based on weights
        strategies = random.choices(strategy_list, weights=weights, k=total_count)

        # Log distribution
        from collections import Counter

        distribution = Counter(strategies)
        logger.info(f"Selected strategy distribution: {dict(distribution)}")

        return strategies

    def get_strategy_info(self, strategy: str) -> Dict[str, any]:
        """Get information about a strategy.

        Args:
            strategy: Difficulty code (e.g., "2iM")

        Returns:
            Dictionary with strategy information
        """
        return {
            "code": strategy,
            "score": self.classifier.get_score(strategy),
            "tier": self.classifier.get_tier(strategy),
            "weight": self.final_weights.get(strategy, 0),
        }
