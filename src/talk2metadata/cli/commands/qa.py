"""QA generation commands - refactored with modular structure."""

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
from talk2metadata.core.qa_generation.pattern_review import review_patterns_interactive
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.group(name="qa")
def qa_group():
    """QA generation commands for creating evaluation datasets.

    This command group provides tools for generating question-answer pairs
    from your database schema and data, useful for creating evaluation datasets.
    """
    pass


@qa_group.command(name="path-generate")
@with_standard_options
@with_agent_config
@click.option(
    "--num-patterns",
    "-n",
    type=int,
    help="Number of path patterns to generate (default: from config)",
)
@handle_errors
@click.pass_context
def qa_path_generate_cmd(
    ctx, run_id, schema_file, output, provider, model, num_patterns
):
    """Generate path patterns for QA generation.

    This command uses an LLM to analyze your database schema and generate
    meaningful path patterns that represent query patterns users might ask.

    \b
    Examples:
        # Generate patterns with settings from config.yml
        talk2metadata qa path-generate

        # Generate with specific run ID
        talk2metadata qa path-generate --run-id wamex_run

        # Save to custom location
        talk2metadata qa path-generate --output custom_patterns.json

        # Use specific model
        talk2metadata qa path-generate --provider openai --model gpt-4o
    """
    config = get_config()

    # Update config with CLI options
    if provider:
        config.set("agent.provider", provider)
    if model:
        config.set("agent.model", model)
    if run_id:
        config.set("run_id", run_id)

    # Load data
    loader = CLIDataLoader(config)
    schema, tables, config, run_id = loader.load_schema_and_tables(
        schema_file=schema_file, run_id=run_id
    )

    # Generate patterns
    handler = QAHandler(config)
    num_patterns = num_patterns or config.get("qa_generation.num_patterns", 15)

    out.progress_start(f"Generating {num_patterns} path patterns...")
    patterns, output_path = handler.generate_patterns(
        schema=schema,
        tables=tables,
        num_patterns=num_patterns,
        output_file=output,
        run_id=run_id,
    )

    out.success(f"Generated and saved {len(patterns)} patterns to {output_path}")
    out.next_steps(
        "📝 Next steps:",
        [
            "Review patterns: talk2metadata qa review",
            "Generate QA pairs: talk2metadata qa generate",
        ],
    )


@qa_group.command(name="review")
@with_standard_options
@click.option(
    "--patterns-file",
    "-p",
    type=click.Path(exists=True),
    help="Path to patterns file (default: qa/kg_paths.json in run directory)",
)
@handle_errors
@click.pass_context
def qa_review_cmd(ctx, run_id, schema_file, output, patterns_file):
    """Review and edit path patterns in web browser.

    Opens a web-based interface for reviewing and editing path patterns.
    Changes are saved directly to the patterns file.

    \b
    Examples:
        # Review patterns from default location
        talk2metadata qa review

        # Review patterns from custom file
        talk2metadata qa review --patterns-file custom_patterns.json

        # Use specific run ID
        talk2metadata qa review --run-id wamex_run
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Load schema
    loader = CLIDataLoader(config)
    schema = loader.load_schema(schema_file=schema_file, run_id=run_id)

    # Load patterns
    handler = QAHandler(config)
    try:
        patterns, patterns_path = handler.load_patterns(
            schema=schema,
            patterns_file=patterns_file,
            run_id=run_id,
        )
        out.success(f"Loaded {len(patterns)} patterns from {patterns_path}")
    except FileNotFoundError as e:
        out.error(f"Patterns file not found: {e}")
        out.info("Run 'talk2metadata qa path-generate' first.")
        raise click.Abort()

    # Review patterns
    out.section("📝 Opening pattern review interface...")
    try:
        reviewed_patterns = review_patterns_interactive(patterns, patterns_path)
        out.success(f"Review complete, {len(reviewed_patterns)} patterns saved")
    except Exception as e:
        out.error(f"Review failed: {e}")
        logger.exception("Pattern review failed")
        raise click.Abort()


@qa_group.command(name="generate")
@with_standard_options
@with_agent_config
@click.option(
    "--patterns-file",
    "-p",
    type=click.Path(exists=True),
    help="Path to patterns file (default: qa/kg_paths.json in run directory)",
)
@click.option(
    "--instances",
    "-i",
    type=int,
    help="Number of instances per pattern (default: from config)",
)
@click.option(
    "--validate/--no-validate",
    default=None,
    help="Validate generated QA pairs (default: from config)",
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
    patterns_file,
    instances,
    validate,
):
    """Generate QA pairs from path patterns.

    This command instantiates paths from real data, converts them to
    natural language questions, and extracts answers (target table row IDs).

    \b
    Examples:
        # Generate QA pairs with settings from config.yml
        talk2metadata qa generate

        # Use custom patterns file
        talk2metadata qa generate --patterns-file custom_patterns.json

        # Save to custom location
        talk2metadata qa generate --output custom_qa_pairs.json

        # Set number of instances per pattern
        talk2metadata qa generate --instances 10

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

    # Load data
    loader = CLIDataLoader(config)
    schema, tables, config, run_id = loader.load_schema_and_tables(
        schema_file=schema_file, run_id=run_id
    )

    # Load patterns
    handler = QAHandler(config)
    try:
        patterns, patterns_path = handler.load_patterns(
            schema=schema,
            patterns_file=patterns_file,
            run_id=run_id,
        )
        out.success(f"Loaded {len(patterns)} patterns from {patterns_path}")
    except FileNotFoundError as e:
        out.error(f"Patterns file not found: {e}")
        out.info("Run 'talk2metadata qa path-generate' first.")
        raise click.Abort()

    # Generate QA pairs
    out.section("🔍 Generating QA pairs...")
    out.stats(
        {
            "Patterns": len(patterns),
            "Instances per pattern": instances
            or config.get("qa_generation.instances_per_pattern", 5),
            "Validation": (
                "enabled"
                if (
                    validate
                    if validate is not None
                    else config.get("qa_generation.validate", True)
                )
                else "disabled"
            ),
        }
    )

    qa_pairs, output_path = handler.generate_qa_pairs(
        schema=schema,
        tables=tables,
        patterns=patterns,
        instances_per_pattern=instances,
        validate=validate,
        output_file=output,
        run_id=run_id,
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

        if stats["difficulties"]:
            out.section("   Difficulty distribution:")
            out.stats(stats["difficulties"], indent="     ")

    if output_path:
        out.success(f"Saved QA pairs to {output_path}")

    # Show sample QA pairs
    if qa_pairs:
        out.section("📝 Sample QA pairs:")
        for i, qa in enumerate(qa_pairs[:3], 1):
            click.echo(f"\n   {i}. Question: {qa.question[:100]}...")
            click.echo(f"      Answer: {len(qa.answer_row_ids)} row ID(s)")
            click.echo(f"      Difficulty: {qa.difficulty}")
            if qa.is_valid is False:
                out.warning(f"      Invalid: {', '.join(qa.validation_errors)}")

    out.section("✅ QA generation complete!")
