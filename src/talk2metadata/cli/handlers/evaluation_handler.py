"""Business logic for search evaluation commands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from talk2metadata.core.qa.qa_pair import QAPair
from talk2metadata.utils.config import Config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_benchmark_dir, get_qa_dir

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Evaluation result for a single QA pair."""

    question: str
    expected_row_ids: Set[Union[int, str]]
    predicted_row_ids: Set[Union[int, str]]
    correct: bool
    precision: float
    recall: float
    f1: float
    mode_name: str
    strategy: Optional[str] = None
    difficulty_score: Optional[float] = None

    @property
    def is_exact_match(self) -> bool:
        """Whether prediction exactly matches expected."""
        return self.expected_row_ids == self.predicted_row_ids

    @property
    def intersection_size(self) -> int:
        """Number of correctly predicted row IDs."""
        return len(self.expected_row_ids & self.predicted_row_ids)


@dataclass
class ModeEvaluationSummary:
    """Summary statistics for a mode's evaluation."""

    mode_name: str
    total_questions: int
    exact_matches: int
    correct_predictions: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    avg_precision: float
    avg_recall: float
    avg_f1: float

    # Per-strategy breakdown
    strategy_stats: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        """Initialize strategy_stats if None."""
        if self.strategy_stats is None:
            self.strategy_stats = {}


class EvaluationHandler:
    """Handler for search evaluation operations.

    Evaluates search modes using QA pairs and computes metrics.
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config

    def load_qa_pairs(
        self, qa_path: Optional[Path] = None, run_id: Optional[str] = None
    ) -> List[QAPair]:
        """Load QA pairs from JSON file.

        Args:
            qa_path: Optional path to QA pairs JSON file
            run_id: Optional run ID

        Returns:
            List of QAPair objects

        Raises:
            FileNotFoundError: If QA pairs file not found
        """
        if not qa_path:
            qa_dir = get_qa_dir(run_id or self.config.get("run_id"), self.config)
            qa_path = qa_dir / "qa_pairs.json"

        qa_path = Path(qa_path)
        if not qa_path.exists():
            raise FileNotFoundError(f"QA pairs file not found at {qa_path}")

        logger.info(f"Loading QA pairs from {qa_path}")
        with open(qa_path, "r") as f:
            data = json.load(f)

        # Handle both old format (list) and new format (dict with qa_pairs key)
        if isinstance(data, list):
            qa_pairs_data = data
        else:
            qa_pairs_data = data.get("qa_pairs", [])

        qa_pairs = [QAPair.from_dict(qa_dict) for qa_dict in qa_pairs_data]

        logger.info(f"Loaded {len(qa_pairs)} QA pairs")
        return qa_pairs

    def evaluate_qa_pair(
        self,
        qa_pair: QAPair,
        retriever: Any,
        top_k: int = 10,
        mode_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate a single QA pair against a retriever.

        Args:
            qa_pair: QA pair to evaluate
            retriever: Retriever instance (must have search method)
            top_k: Number of results to retrieve
            mode_name: Name of the mode being evaluated

        Returns:
            EvaluationResult object
        """
        # Run search
        try:
            results = retriever.search(qa_pair.question, top_k=top_k)
        except Exception as e:
            logger.warning(
                f"Search failed for question '{qa_pair.question[:50]}...': {e}"
            )
            results = []

        # Extract predicted row IDs
        predicted_row_ids = set()
        for result in results:
            # Handle different result types
            if hasattr(result, "row_id"):
                # Standard SearchResult
                predicted_row_ids.add(result.row_id)
            elif hasattr(result, "data") and isinstance(result.data, list):
                # Text2SQL result (list of dicts)
                # Try to extract row IDs from data
                for row in result.data:
                    # Look for common ID fields
                    for id_field in ["id", "row_id", "ANumber", "Number"]:
                        if id_field in row:
                            predicted_row_ids.add(row[id_field])
                            break
            elif hasattr(result, "data") and isinstance(result.data, dict):
                # Single row result
                row = result.data
                for id_field in ["id", "row_id", "ANumber", "Number"]:
                    if id_field in row:
                        predicted_row_ids.add(row[id_field])
                        break

        # Convert expected to set
        expected_row_ids = set(qa_pair.answer_row_ids)

        # Calculate metrics
        intersection = expected_row_ids & predicted_row_ids
        precision = (
            len(intersection) / len(predicted_row_ids) if predicted_row_ids else 0.0
        )
        recall = len(intersection) / len(expected_row_ids) if expected_row_ids else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationResult(
            question=qa_pair.question,
            expected_row_ids=expected_row_ids,
            predicted_row_ids=predicted_row_ids,
            correct=len(intersection) > 0,
            precision=precision,
            recall=recall,
            f1=f1,
            mode_name=mode_name,
            strategy=qa_pair.strategy,
            difficulty_score=qa_pair.difficulty_score,
        )

    def evaluate_mode(
        self,
        mode_name: str,
        qa_pairs: List[QAPair],
        retriever: Any,
        top_k: int = 10,
    ) -> ModeEvaluationSummary:
        """Evaluate a single mode on all QA pairs.

        Args:
            mode_name: Name of the mode
            qa_pairs: List of QA pairs to evaluate
            retriever: Retriever instance
            top_k: Number of results to retrieve per query

        Returns:
            ModeEvaluationSummary object
        """
        logger.info(f"Evaluating mode '{mode_name}' on {len(qa_pairs)} QA pairs")

        results = []
        # Use click progress bar for individual QA pairs
        try:
            import click

            with click.progressbar(
                qa_pairs,
                label=f"  Evaluating {mode_name}",
                length=len(qa_pairs),
                show_percent=True,
                show_pos=True,
                item_show_func=lambda x: (
                    f"Question: {x.question[:60]}..." if x else None
                ),
            ) as bar:
                for qa_pair in bar:
                    result = self.evaluate_qa_pair(qa_pair, retriever, top_k, mode_name)
                    results.append(result)
        except ImportError:
            # Fallback if click is not available
            for i, qa_pair in enumerate(qa_pairs):
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processing {i + 1}/{len(qa_pairs)}...")
                result = self.evaluate_qa_pair(qa_pair, retriever, top_k, mode_name)
                results.append(result)

        # Calculate summary statistics
        total = len(results)
        exact_matches = sum(1 for r in results if r.is_exact_match)
        correct_predictions = sum(1 for r in results if r.correct)

        # Overall metrics (micro-averaged)
        total_expected = sum(len(r.expected_row_ids) for r in results)
        total_predicted = sum(len(r.predicted_row_ids) for r in results)
        total_intersection = sum(r.intersection_size for r in results)

        precision = total_intersection / total_predicted if total_predicted > 0 else 0.0
        recall = total_intersection / total_expected if total_expected > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Accuracy (at least one correct prediction)
        accuracy = correct_predictions / total if total > 0 else 0.0

        # Macro-averaged metrics
        avg_precision = sum(r.precision for r in results) / total if total > 0 else 0.0
        avg_recall = sum(r.recall for r in results) / total if total > 0 else 0.0
        avg_f1 = sum(r.f1 for r in results) / total if total > 0 else 0.0

        # Per-strategy breakdown
        strategy_stats = {}
        strategies = set(r.strategy for r in results if r.strategy)
        for strategy in strategies:
            strategy_results = [r for r in results if r.strategy == strategy]
            if not strategy_results:
                continue

            strategy_total = len(strategy_results)
            strategy_correct = sum(1 for r in strategy_results if r.correct)
            strategy_precision = (
                sum(r.precision for r in strategy_results) / strategy_total
            )
            strategy_recall = sum(r.recall for r in strategy_results) / strategy_total
            strategy_f1 = sum(r.f1 for r in strategy_results) / strategy_total

            strategy_stats[strategy] = {
                "total": strategy_total,
                "correct": strategy_correct,
                "accuracy": (
                    strategy_correct / strategy_total if strategy_total > 0 else 0.0
                ),
                "precision": strategy_precision,
                "recall": strategy_recall,
                "f1": strategy_f1,
            }

        return ModeEvaluationSummary(
            mode_name=mode_name,
            total_questions=total,
            exact_matches=exact_matches,
            correct_predictions=correct_predictions,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            strategy_stats=strategy_stats,
        )

    def evaluate_all_modes(
        self,
        mode_names: List[str],
        retrievers: Dict[str, Any],
        qa_pairs: List[QAPair],
        top_k: int = 10,
    ) -> Dict[str, ModeEvaluationSummary]:
        """Evaluate multiple modes and return summary.

        Args:
            mode_names: List of mode names to evaluate
            retrievers: Dict mapping mode_name -> retriever instance
            qa_pairs: List of QA pairs to evaluate
            top_k: Number of results to retrieve per query

        Returns:
            Dict mapping mode_name -> ModeEvaluationSummary
        """
        summaries = {}

        # Evaluate each mode (progress bar is shown within evaluate_mode for QA pairs)
        for mode_name in mode_names:
            if mode_name not in retrievers:
                logger.warning(f"Retriever not found for mode '{mode_name}', skipping")
                continue

            retriever = retrievers[mode_name]
            summary = self.evaluate_mode(mode_name, qa_pairs, retriever, top_k)
            summaries[mode_name] = summary

        return summaries

    def format_evaluation_results_txt(
        self,
        summaries: Dict[str, ModeEvaluationSummary],
        qa_count: int,
    ) -> str:
        """Format evaluation results as text.

        Args:
            summaries: Dict mapping mode_name -> ModeEvaluationSummary
            qa_count: Total number of QA pairs evaluated

        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 100)
        lines.append("Evaluation Results Summary")
        lines.append("=" * 100)
        lines.append(f"\nTotal QA Pairs Evaluated: {qa_count}")
        lines.append(f"Modes Evaluated: {', '.join(sorted(summaries.keys()))}")
        lines.append("")

        # Overall metrics table
        lines.append("Overall Metrics:")
        lines.append("-" * 100)
        header = (
            f"{'Mode':<20} {'Total':<8} {'Exact':<8} {'Correct':<10} "
            f"{'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}"
        )
        lines.append(header)
        lines.append("-" * 100)

        for mode_name, summary in sorted(summaries.items()):
            row = (
                f"{mode_name:<20} "
                f"{summary.total_questions:<8} "
                f"{summary.exact_matches:<8} "
                f"{summary.correct_predictions:<10} "
                f"{summary.precision:<12.4f} "
                f"{summary.recall:<12.4f} "
                f"{summary.f1:<12.4f} "
                f"{summary.accuracy:<12.4f}"
            )
            lines.append(row)

        lines.append("-" * 100)
        lines.append("")

        # Per-strategy breakdown for each mode
        for mode_name, summary in sorted(summaries.items()):
            if summary.strategy_stats:
                lines.append(f"\n{mode_name} - Per-Strategy Breakdown:")
                lines.append("-" * 80)
                strategy_header = (
                    f"{'Strategy':<12} {'Total':<8} {'Correct':<10} "
                    f"{'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}"
                )
                lines.append(strategy_header)
                lines.append("-" * 80)

                for strategy, stats in sorted(summary.strategy_stats.items()):
                    row = (
                        f"{strategy:<12} "
                        f"{stats['total']:<8} "
                        f"{stats['correct']:<10} "
                        f"{stats['accuracy']:<12.4f} "
                        f"{stats['precision']:<12.4f} "
                        f"{stats['recall']:<12.4f} "
                        f"{stats['f1']:<12.4f}"
                    )
                    lines.append(row)
                lines.append("-" * 80)
                lines.append("")

        lines.append("=" * 100)

        return "\n".join(lines)

    def save_evaluation_results(
        self,
        summaries: Dict[str, ModeEvaluationSummary],
        qa_pairs: List[QAPair],
        output_path: Optional[Path] = None,
        run_id: Optional[str] = None,
        auto_save: bool = True,
        format: str = "json",
    ) -> Path:
        """Save evaluation results to benchmark directory.

        Args:
            summaries: Dict mapping mode_name -> ModeEvaluationSummary
            qa_pairs: List of QA pairs that were evaluated
            output_path: Optional explicit path to save results
            run_id: Optional run ID for auto-save path
            auto_save: If True and output_path is None, auto-save to benchmark directory
            format: Output format ("json" or "txt")

        Returns:
            Path where results were saved
        """
        if output_path is None and auto_save:
            # Auto-save to benchmark directory with timestamp
            # Use run_id from config if not provided
            if run_id is None:
                run_id = self.config.get("run_id")
            benchmark_dir = get_benchmark_dir(run_id, self.config)
            benchmark_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_names = "_".join(sorted(summaries.keys()))
            if len(mode_names) > 50:
                mode_names = f"{len(summaries)}_modes"
            extension = "txt" if format == "txt" else "json"
            filename = f"evaluation_{timestamp}_{mode_names}.{extension}"
            output_path = benchmark_dir / filename
        elif output_path is None:
            raise ValueError(
                "Either output_path must be provided or auto_save must be True"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if format == "txt":
            # Save as text
            text_content = self.format_evaluation_results_txt(summaries, len(qa_pairs))
            output_path.write_text(text_content, encoding="utf-8")
        else:
            # Save as JSON
            results = {
                "timestamp": datetime.now().isoformat(),
                "total_qa_pairs": len(qa_pairs),
                "modes_evaluated": list(summaries.keys()),
                "modes": {
                    mode_name: {
                        "total_questions": summary.total_questions,
                        "exact_matches": summary.exact_matches,
                        "correct_predictions": summary.correct_predictions,
                        "precision": summary.precision,
                        "recall": summary.recall,
                        "f1": summary.f1,
                        "accuracy": summary.accuracy,
                        "avg_precision": summary.avg_precision,
                        "avg_recall": summary.avg_recall,
                        "avg_f1": summary.avg_f1,
                        "strategy_stats": summary.strategy_stats,
                    }
                    for mode_name, summary in summaries.items()
                },
                "qa_pairs_count": len(qa_pairs),
            }
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        logger.info(f"Evaluation results saved to {output_path}")
        return output_path
