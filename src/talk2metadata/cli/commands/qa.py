"""QA generation commands - strategy-based QA generation."""

from __future__ import annotations

import click

from talk2metadata.cli.decorators import (
    handle_errors,
    with_agent_config,
    with_standard_options,
)
from talk2metadata.cli.handlers import QAHandler
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.group(name="qa")
def qa_group():
    """QA generation commands for creating evaluation datasets.

    This command group provides tools for generating question-answer pairs
    from your database schema and data based on difficulty strategies,
    useful for creating evaluation datasets.
    """
    pass


@qa_group.command(name="generate")
@with_standard_options
@with_agent_config
@click.option(
    "--total",
    "-t",
    type=int,
    help="Total number of QA pairs to generate (default: from config)",
)
@click.option(
    "--pairs-per-strategy",
    "-p",
    type=int,
    help="Generate this many pairs per strategy (overrides --total and weights)",
)
@click.option(
    "--validate/--no-validate",
    default=None,
    help="Validate generated QA pairs (default: from config)",
)
@click.option(
    "--filter-valid/--no-filter-valid",
    default=None,
    help="Filter out invalid QA pairs (default: from config)",
)
@click.option(
    "--max-answer-records",
    type=int,
    help="Maximum number of answer records per question (default: 10, questions with more are too general)",
)
@handle_errors
@click.pass_context
def qa_generate_cmd(
    ctx,
    run_id,
    schema_file,
    output,
    provider,
    model,
    total,
    pairs_per_strategy,
    validate,
    filter_valid,
    max_answer_records,
):
    """Generate QA pairs based on difficulty strategies.

    This command generates QA pairs by:
    1. Selecting difficulty strategies based on configured weights
    2. Building SQL queries with appropriate JOINs and filters
    3. Generating natural language questions using LLM
    4. Extracting answer record IDs
    5. Validating QA pairs

    \b
    Examples:
        # Generate QA pairs with settings from config.yml
        talk2metadata qa generate

        # Generate 50 QA pairs
        talk2metadata qa generate --total 50

        # Generate 5 pairs per strategy
        talk2metadata qa generate --pairs-per-strategy 5

        # Save to custom location
        talk2metadata qa generate --output custom_qa_pairs.json

        # Use specific model
        talk2metadata qa generate --provider openai --model gpt-4o

        # Disable validation
        talk2metadata qa generate --no-validate
    """
    config = get_config()

    # Update config with CLI options
    if provider:
        config.set("agent.provider", provider)
    if model:
        config.set("agent.model", model)
    if run_id:
        config.set("run_id", run_id)

    # Read target_table from config for schema file naming
    target_table = config.get("ingest.target_table")

    # Load data
    loader = CLIDataLoader(config)
    schema, tables, config, run_id = loader.load_schema_and_tables(
        schema_file=schema_file, run_id=run_id, target_table=target_table
    )

    # Generate QA pairs
    handler = QAHandler(config)

    # Analyze strategy capabilities
    out.section("📋 Analyzing schema capabilities...")
    analysis = handler.analyze_strategy_capabilities(schema)
    capabilities = analysis["capabilities"]
    config_check = analysis["config_check"]

    # Show schema info
    schema_info = capabilities["schema_info"]
    out.stats(
        {
            "Target table": schema_info["target_table"],
            "Target table columns": schema_info["target_table_columns"],
            "Target table foreign keys": schema_info["target_table_fks"],
            "Total tables": schema_info["total_tables"],
            "Total foreign keys": schema_info["total_foreign_keys"],
        }
    )

    # Show supported strategies
    supported = capabilities["supported_strategies"]
    unsupported = capabilities["unsupported_strategies"]
    out.section("✅ Supported strategies:")
    if supported:
        # Group by tier for better display
        from talk2metadata.core.qa import DifficultyClassifier

        classifier = DifficultyClassifier()
        by_tier = {}
        for strategy in supported:
            tier = classifier.get_tier(strategy)
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(strategy)

        for tier in ["easy", "medium", "hard", "expert"]:
            if tier in by_tier:
                strategies_str = ", ".join(sorted(by_tier[tier]))
                out.info(f"   {tier.capitalize()}: {strategies_str}")
    else:
        out.warning("   No strategies are supported!")

    # Show unsupported strategies if any
    if unsupported:
        out.section("❌ Unsupported strategies:")
        reasons = capabilities["reasons"]
        for strategy in sorted(unsupported):
            reason = reasons.get(strategy, "Unknown reason")
            out.warning(f"   {strategy}: {reason}")

    # Check configured strategies
    if config_check["configured_strategies"]:
        out.section("⚙️  Configuration check:")
        feasible = config_check["feasible_strategies"]
        infeasible = config_check["infeasible_strategies"]

        if feasible:
            out.success(f"   ✅ {len(feasible)} configured strategies are feasible")
            if len(feasible) <= 10:
                out.info(f"      {', '.join(feasible)}")

        if infeasible:
            out.warning(
                f"   ⚠️  {len(infeasible)} configured strategies are NOT feasible:"
            )
            reasons = config_check["reasons"]
            for strategy in sorted(infeasible):
                reason = reasons.get(strategy, "Unknown reason")
                out.warning(f"      {strategy}: {reason}")

    # Get parameters
    total_qa_pairs = total or config.get("qa_generation.total_qa_pairs", 100)
    do_validate = (
        validate if validate is not None else config.get("qa_generation.validate", True)
    )
    do_filter_valid = (
        filter_valid
        if filter_valid is not None
        else config.get("qa_generation.filter_valid", True)
    )
    max_answer_records = (
        max_answer_records
        if max_answer_records is not None
        else config.get("qa_generation.max_answer_records", 10)
    )

    out.section("🔍 Generating QA pairs...")
    out.stats(
        {
            "Total QA pairs": total_qa_pairs if not pairs_per_strategy else "N/A",
            "Pairs per strategy": pairs_per_strategy or "Based on weights",
            "Validation": "enabled" if do_validate else "disabled",
            "Filter invalid": "yes" if do_filter_valid else "no",
            "Max answer records": max_answer_records,
        }
    )

    qa_pairs, output_path = handler.generate_qa_pairs_strategy_based(
        schema=schema,
        tables=tables,
        total_qa_pairs=total_qa_pairs,
        pairs_per_strategy=pairs_per_strategy,
        validate=do_validate,
        filter_valid=do_filter_valid,
        output_file=output,
        run_id=run_id,
        max_answer_records=max_answer_records,
    )

    out.success(f"Generated {len(qa_pairs)} QA pairs")

    # Show statistics
    if qa_pairs:
        stats = handler.get_qa_statistics(qa_pairs)
        out.section("📊 Statistics:")
        out.stats(
            {
                "Total": stats["total"],
                "Valid": f"{stats['valid']}/{stats['total']}",
                "Invalid": stats["invalid"],
            }
        )

        if stats.get("strategies"):
            out.section("   Strategy distribution:")
            out.stats(stats["strategies"], indent="     ")

        if stats.get("tiers"):
            out.section("   Tier distribution:")
            out.stats(stats["tiers"], indent="     ")

    if output_path:
        out.success(f"Saved QA pairs to {output_path}")

    # Show sample QA pairs
    if qa_pairs:
        out.section("📝 Sample QA pairs:")
        for i, qa in enumerate(qa_pairs[:3], 1):
            click.echo(f"\n   {i}. Question: {qa.question[:100]}...")
            click.echo(f"      Answer: {len(qa.answer_row_ids)} row ID(s)")
            click.echo(f"      Strategy: {qa.strategy} (tier: {qa.tier})")
            if qa.is_valid is False:
                out.warning(f"      Invalid: {', '.join(qa.validation_errors)}")

    out.section("✅ QA generation complete!")
