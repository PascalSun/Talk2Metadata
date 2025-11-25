"""Search management commands - prepare and retrieve."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import click

from talk2metadata.cli.decorators import handle_errors, with_run_id
from talk2metadata.cli.handlers import (
    EvaluationHandler,
    PrepareHandler,
    SearchHandler,
)
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.core.modes import get_active_mode, get_registry
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.group(name="search")
def search_group():
    """Search management commands for preparation and retrieval.

    This command group provides tools for preparing modes (building indexes or
    loading databases) and retrieving records using natural language queries.
    """
    pass


@search_group.command(name="prepare")
@click.option(
    "--metadata",
    "-m",
    "metadata_path",
    type=click.Path(exists=True),
    help="Path to schema metadata JSON (default: data/metadata/schema.json)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Prepare specific mode (default: prepare all enabled modes)",
)
@click.option(
    "--all-modes",
    is_flag=True,
    default=False,
    help="Prepare all enabled modes (default: prepare active mode only)",
)
@with_run_id
@handle_errors
@click.pass_context
def prepare_cmd(
    ctx,
    metadata_path,
    mode,
    all_modes,
    run_id,
):
    """Prepare modes for use (build indexes or load CSV to database).

    This is a unified preparation command that handles different preparation steps:
    - Index-based modes (e.g., record_embedding): Builds indexes automatically using config settings
    - Database-based modes (e.g., text2sql): Loads CSV data to database automatically

    All settings (model, batch_size, etc.) are read from config.yml.
    Configure mode-specific settings under 'modes.{mode_name}.indexer' in config.yml.

    \b
    Examples:
        # Prepare all enabled modes (builds indexes and loads databases)
        talk2metadata search prepare

        # Prepare specific mode
        talk2metadata search prepare --mode text2sql

        # Prepare text2sql mode (loads CSV to database)
        talk2metadata search prepare --mode text2sql

        # Prepare record_embedding mode (builds index if not exists, uses config for model)
        talk2metadata search prepare --mode record_embedding
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Initialize handlers
    prepare_handler = PrepareHandler(config)
    loader = CLIDataLoader(config)

    # Load schema metadata
    out.section("📄 Loading schema metadata...")
    schema_metadata = loader.load_schema(schema_file=metadata_path, run_id=run_id)

    # Determine modes to prepare
    registry = get_registry()
    if mode:
        if not registry.get(mode):
            available = ", ".join(registry.get_all_enabled())
            out.error(f"Mode '{mode}' not found. Available: {available}")
            raise click.Abort()
        modes_to_prepare = [mode]
    elif all_modes:
        modes_to_prepare = registry.get_all_enabled()
        if not modes_to_prepare:
            out.error("No enabled modes found")
            raise click.Abort()
    else:
        # Default: prepare active mode
        active_mode = get_active_mode() or "record_embedding"
        modes_to_prepare = [active_mode]

    out.section(f"🔧 Preparing {len(modes_to_prepare)} mode(s)...")
    out.stats({"Modes": ", ".join(modes_to_prepare)})

    # Prepare each mode (automatically builds indexes or loads databases as needed)
    # Force rebuild when called from prepare command
    results = prepare_handler.prepare_all_modes(
        mode_names=modes_to_prepare,
        schema_metadata=schema_metadata,
        run_id=run_id,
        force=True,
    )

    # Display results
    out.section(f"\n{'='*60}")
    for mode_name, result in results.items():
        status = result.get("status", "unknown")
        message = result.get("message", "")

        if status == "success":
            out.success(f"✅ {mode_name}: {message}")
        elif status == "info":
            out.info(f"ℹ️  {mode_name}: {message}")
        else:
            out.error(f"❌ {mode_name}: {message}")

    out.section(f"{'='*60}")

    # Check if any modes need index building
    modes_needing_index = [
        name for name, result in results.items() if result.get("requires_index", False)
    ]

    if modes_needing_index:
        out.next_steps(
            "Next step:",
            [
                f"Run 'talk2metadata search prepare --mode {mode}' to build index"
                for mode in modes_needing_index
            ],
        )


def _display_comparison_results(
    comparison_result, query, top_k, output_format, retrievers
):
    """Display comparison mode results."""
    if output_format == "json":
        output = {
            "query": query,
            "top_k": top_k,
            "mode": "comparison",
            "modes_compared": list(retrievers.keys()),
            "comparison": {
                "common_results_count": len(comparison_result.common_results),
                "overlap_stats": comparison_result.overlap_stats,
                "mode_results": {
                    mode: [
                        {
                            "rank": r.rank,
                            "table": r.table,
                            "row_id": r.row_id,
                            "score": r.score,
                        }
                        for r in results
                    ]
                    for mode, results in comparison_result.mode_results.items()
                },
            },
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        out.section("=" * 80)
        out.section("COMPARISON RESULTS")
        out.section("=" * 80)

        click.echo(
            f"\nCommon Results (appear in all modes): {len(comparison_result.common_results)}"
        )
        if comparison_result.common_results:
            for r in comparison_result.common_results[:top_k]:
                click.echo(f"  - {r.table}.{r.row_id} (score: {r.score:.4f})")

        out.section("\nOverlap Statistics:")
        for mode_name, overlap in comparison_result.overlap_stats.items():
            click.echo(f"  - {mode_name}: {overlap}% overlap")

        out.section("\nMode-specific Results:")
        for mode_name, results in comparison_result.mode_results.items():
            click.echo(f"\n  {mode_name} ({len(results)} results):")
            unique = comparison_result.unique_results.get(mode_name, [])
            click.echo(f"    - Unique: {len(unique)}")
            for r in results[:5]:
                is_unique = any(
                    ur.row_id == r.row_id and ur.table == r.table for ur in unique
                )
                marker = "★" if is_unique else " "
                click.echo(
                    f"    {marker} Rank {r.rank}: {r.table}.{r.row_id} (score: {r.score:.4f})"
                )


def _display_search_results(
    results, query, top_k, mode_name, output_format, show_score, per_table_top_k=None
):
    """Display single mode search results."""
    if output_format == "json":
        result_dicts = []
        for r in results:
            # Handle text2sql results (data is a list of dicts)
            if hasattr(r, "sql_query"):
                result_dict = {
                    "rank": r.rank,
                    "table": r.table,
                    "sql_query": r.sql_query,
                    "row_count": r.row_count,
                    "score": r.score,
                    "data": r.data,  # List of dicts
                }
            else:
                result_dict = {
                    "rank": r.rank,
                    "table": r.table,
                    "row_id": r.row_id,
                    "score": r.score,
                    "data": r.data,
                }
                # Add RecordVoter fields if available
                if hasattr(r, "match_count"):
                    result_dict["match_count"] = r.match_count
                    result_dict["matched_tables"] = r.matched_tables
            result_dicts.append(result_dict)

        output = {
            "query": query,
            "top_k": top_k,
            "search_mode": mode_name,
            "per_table_top_k": (
                per_table_top_k if mode_name == "record_embedding" else None
            ),
            "results": result_dicts,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        if not results:
            out.error("No results found")
            return

        out.success(f"Found {len(results)} results:\n")

        for result in results:
            out.section("=" * 80)
            click.echo(f"Rank #{result.rank}")
            if show_score:
                click.echo(f"Score: {result.score:.4f}")

            # Handle text2sql results
            if hasattr(result, "sql_query"):
                click.echo(f"SQL Query: {result.sql_query}")
                click.echo(f"Table: {result.table}")
                click.echo(f"Rows Returned: {result.row_count}")
                click.echo("\nData:")

                # Display each row
                for idx, row_data in enumerate(result.data, 1):
                    click.echo(f"\n  Row {idx}:")
                    for key, value in row_data.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:97] + "..."
                        click.echo(f"    {key}: {value_str}")
            else:
                # Handle regular results (record_embedding, etc.)
                if hasattr(result, "match_count"):
                    click.echo(f"Votes: {result.match_count}")
                    click.echo(f"Voter Tables: {', '.join(result.matched_tables)}")
                click.echo(f"Table: {result.table}")
                click.echo(f"Row ID: {result.row_id}")
                click.echo("\nData:")

                for key, value in result.data.items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    click.echo(f"  {key}: {value_str}")

            click.echo()

        out.section("=" * 80)
        out.success(f"Retrieved {len(results)} records")


@search_group.command(name="retrieve")
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=None,
    help="Number of results to return (default: from config.yml modes.{mode}.retriever.top_k)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default=None,
    help="Output format (default: from config.yml search.output_format or 'text')",
)
@click.option(
    "--show-score",
    is_flag=True,
    default=False,
    help="Show similarity scores (default: from config.yml search.show_score or False)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Search mode (default: from config.yml modes.active)",
)
@click.option(
    "--compare",
    is_flag=True,
    default=None,
    help="Compare results from all enabled modes (default: from config.yml modes.compare.enabled)",
)
@with_run_id
@handle_errors
@click.pass_context
def retrieve_cmd(
    ctx,
    query,
    top_k,
    output_format,
    show_score,
    mode,
    compare,
    run_id,
):
    """Retrieve relevant records using natural language query.

    QUERY: Natural language search query

    Most settings are read from config.yml:
    - top_k: modes.{mode}.retriever.top_k
    - per_table_top_k: modes.{mode}.retriever.per_table_top_k
    - output_format: search.output_format (or 'text' as default)
    - show_score: search.show_score (or False as default)
    - mode: modes.active
    - compare: modes.compare.enabled

    \b
    Examples:
        # Simple search (uses config.yml settings)
        talk2metadata search retrieve "customers in healthcare industry"

        # Override top_k
        talk2metadata search retrieve "high value orders" --top-k 10

        # Override mode
        talk2metadata search retrieve "recent orders" --mode text2sql

        # JSON output
        talk2metadata search retrieve "recent orders" --format json

        # Compare all modes
        talk2metadata search retrieve "recent orders" --compare
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Determine mode (CLI override > config > default)
    mode_name = mode or get_active_mode() or "record_embedding"

    # Get retriever config for the mode
    from talk2metadata.core.modes import get_mode_retriever_config

    mode_retriever_config = get_mode_retriever_config(mode_name)

    # Read values from config (CLI override > config > default)
    if top_k is None:
        top_k = mode_retriever_config.get("top_k", 5)

    if output_format is None:
        output_format = config.get("search.output_format", "text")

    # If show_score is False (default), try to read from config
    # If user explicitly set --show-score, use that value
    if not show_score:
        show_score = config.get("search.show_score", False)

    if compare is None:
        compare = config.get("modes.compare.enabled", False)

    per_table_top_k = mode_retriever_config.get("per_table_top_k", 5)

    # Initialize handler
    handler = SearchHandler(config)

    # Handle comparison mode
    if compare:
        if output_format == "text":
            out.section(f'🔍 Comparison Mode: "{query}"')
            out.stats({"Top-K": top_k})

        try:
            # Load retrievers and run comparison
            retrievers = handler.load_retrievers_for_comparison(
                index_dir=None,  # Auto-determined from run_id
                run_id=run_id,
            )

            if not retrievers:
                out.error("No retrievers loaded for comparison")
                raise click.Abort()

            if output_format == "text":
                out.section(
                    f"📊 Comparing {len(retrievers)} mode(s): {', '.join(retrievers.keys())}\n"
                )

            comparison_result = handler.compare_modes(
                query=query,
                top_k=top_k,
                index_dir=None,  # Auto-determined from run_id
                run_id=run_id,
            )

            # Display results
            _display_comparison_results(
                comparison_result, query, top_k, output_format, retrievers
            )

        except Exception as e:
            out.error(f"Comparison failed: {e}")
            logger.exception("Comparison search failed")
            raise click.Abort()

        return

    # Single mode search
    if output_format == "text":
        out.section(f'🔍 Searching: "{query}" [Mode: {mode_name}]')
        out.stats({"Top-K": top_k})

    try:
        results = handler.search(
            query=query,
            top_k=top_k,
            mode_name=mode_name,
            index_dir=None,  # Auto-determined from run_id
            run_id=run_id,
            per_table_top_k=per_table_top_k,
        )

        # Display results
        _display_search_results(
            results, query, top_k, mode_name, output_format, show_score, per_table_top_k
        )

    except (FileNotFoundError, NotImplementedError) as e:
        out.error(str(e))
        out.info("Please run 'talk2metadata search prepare' first.")
        raise click.Abort()
    except Exception as e:
        out.error(f"Search failed: {e}")
        logger.exception("Search failed")
        raise click.Abort()


def _display_evaluation_results(
    summaries: Dict[str, Any],
    output_format: str,
    qa_count: int,
):
    """Display evaluation results.

    Args:
        summaries: Dict mapping mode_name -> ModeEvaluationSummary
        output_format: Output format ("text" or "json")
        qa_count: Total number of QA pairs evaluated
    """
    if output_format == "json":
        result_dict = {
            "total_qa_pairs": qa_count,
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
        }
        click.echo(json.dumps(result_dict, indent=2))
    else:
        # Text output - create a table
        out.section("=" * 100)
        out.success("Evaluation Results Summary")
        out.section("=" * 100)

        # Overall metrics table
        click.echo("\nOverall Metrics:")
        click.echo("-" * 100)
        header = f"{'Mode':<20} {'Total':<8} {'Exact':<8} {'Correct':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}"
        click.echo(header)
        click.echo("-" * 100)

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
            click.echo(row)

        click.echo("-" * 100)
        click.echo()

        # Per-strategy breakdown for each mode
        for mode_name, summary in sorted(summaries.items()):
            if summary.strategy_stats:
                click.echo(f"\n{mode_name} - Per-Strategy Breakdown:")
                click.echo("-" * 80)
                strategy_header = (
                    f"{'Strategy':<12} {'Total':<8} {'Correct':<10} "
                    f"{'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}"
                )
                click.echo(strategy_header)
                click.echo("-" * 80)

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
                    click.echo(row)
                click.echo("-" * 80)
                click.echo()

        out.section("=" * 100)


@search_group.command(name="evaluate")
@click.option(
    "--qa-pairs",
    "-q",
    "qa_path",
    type=click.Path(exists=True),
    help="Path to QA pairs JSON file (overrides config, default: data/{run_id}/qa/qa_pairs.json)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Evaluate specific mode (overrides config, default: active mode or all enabled modes)",
)
@click.option(
    "--all-modes",
    is_flag=True,
    default=None,
    help="Evaluate all enabled modes (overrides config)",
)
@with_run_id
@handle_errors
@click.pass_context
def evaluate_cmd(
    ctx,
    qa_path,
    mode,
    all_modes,
    run_id,
):
    """Evaluate search modes using QA pairs.

    This command evaluates search modes by running queries from QA pairs and
    comparing results with ground truth answers. All settings are read from
    config.yml under the 'evaluation' section.

    \b
    Examples:
        # Evaluate using config settings
        talk2metadata search evaluate

        # Evaluate specific mode (overrides config)
        talk2metadata search evaluate --mode record_embedding

        # Evaluate all enabled modes (overrides config)
        talk2metadata search evaluate --all-modes

        # Use custom QA pairs file (overrides config)
        talk2metadata search evaluate --qa-pairs ./custom_qa.json
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Get evaluation settings from config
    eval_config = config.get("evaluation", {})
    top_k = eval_config.get("top_k", 10)
    output_format = eval_config.get("output_format", "text")
    save_format = eval_config.get("save_format", "both")
    auto_save = eval_config.get("auto_save", True)
    evaluate_all_modes = eval_config.get("evaluate_all_modes", False)

    # Initialize handlers
    eval_handler = EvaluationHandler(config)
    search_handler = SearchHandler(config)

    # Load QA pairs
    try:
        qa_pairs = eval_handler.load_qa_pairs(qa_path=qa_path, run_id=run_id)
    except FileNotFoundError as e:
        out.error(str(e))
        out.info("Please run 'talk2metadata qa generate' first to create QA pairs.")
        raise click.Abort()

    if not qa_pairs:
        out.error("No QA pairs found in file")
        raise click.Abort()

    # Determine modes to evaluate (CLI options override config)
    registry = get_registry()
    if all_modes is True or (all_modes is None and evaluate_all_modes):
        modes_to_evaluate = registry.get_all_enabled()
        if not modes_to_evaluate:
            out.error("No enabled modes found")
            raise click.Abort()
    elif mode:
        if not registry.get(mode):
            available = ", ".join(registry.get_all_enabled())
            out.error(f"Mode '{mode}' not found. Available: {available}")
            raise click.Abort()
        modes_to_evaluate = [mode]
    else:
        # Default: evaluate active mode
        active_mode = get_active_mode() or "record_embedding"
        modes_to_evaluate = [active_mode]

    if output_format == "text":
        out.section(
            f"📊 Evaluating {len(modes_to_evaluate)} mode(s) on {len(qa_pairs)} QA pairs"
        )
        out.stats({"Modes": ", ".join(modes_to_evaluate), "Top-K": top_k})

    # Load retrievers for all modes
    try:
        retrievers = {}
        for mode_name in modes_to_evaluate:
            try:
                retriever = search_handler.load_retriever(
                    mode_name=mode_name,
                    index_dir=None,  # Use default from config
                    run_id=run_id,
                )
                retrievers[mode_name] = retriever
            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Failed to load retriever for mode '{mode_name}': {error_msg}"
                )
                if output_format == "text":
                    out.warning(f"⚠️  Skipping mode '{mode_name}': {error_msg}")
                continue

        if not retrievers:
            out.error(
                "No retrievers loaded. Please run 'talk2metadata search prepare' first."
            )
            raise click.Abort()

        # Run evaluation
        summaries = eval_handler.evaluate_all_modes(
            mode_names=list(retrievers.keys()),
            retrievers=retrievers,
            qa_pairs=qa_pairs,
            top_k=top_k,
        )

        # Display results
        _display_evaluation_results(summaries, output_format, len(qa_pairs))

        # Save results to benchmark directory (if auto_save is enabled)
        if auto_save:
            try:
                # Use run_id from config if not provided via CLI
                save_run_id = run_id or config.get("run_id")

                # Save based on save_format
                saved_paths = []
                if save_format in ("json", "both"):
                    saved_path = eval_handler.save_evaluation_results(
                        summaries=summaries,
                        qa_pairs=qa_pairs,
                        output_path=None,  # Auto-save to benchmark directory
                        run_id=save_run_id,
                        auto_save=True,
                        format="json",
                    )
                    saved_paths.append(saved_path)

                if save_format in ("txt", "both"):
                    saved_path = eval_handler.save_evaluation_results(
                        summaries=summaries,
                        qa_pairs=qa_pairs,
                        output_path=None,  # Auto-save to benchmark directory
                        run_id=save_run_id,
                        auto_save=True,
                        format="txt",
                    )
                    saved_paths.append(saved_path)

                if output_format == "text":
                    if len(saved_paths) == 1:
                        out.success(
                            f"\n💾 Evaluation results saved to: {saved_paths[0]}"
                        )
                    else:
                        out.success("\n💾 Evaluation results saved to:")
                        for path in saved_paths:
                            out.success(f"   - {path}")
            except Exception as e:
                logger.warning(f"Failed to save evaluation results: {e}")
                if output_format == "text":
                    out.warning(f"Could not save results: {e}")

    except Exception as e:
        out.error(f"Evaluation failed: {e}")
        logger.exception("Evaluation failed")
        raise click.Abort()
