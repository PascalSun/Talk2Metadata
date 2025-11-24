"""QA generation commands for creating evaluation datasets."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd

from talk2metadata.core.qa_generation import QAGenerator
from talk2metadata.core.qa_generation.pattern_review import (
    review_patterns_interactive,
)
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import find_schema_file, get_metadata_dir, get_qa_dir

logger = get_logger(__name__)


@click.group(name="qa")
def qa_group():
    """QA generation commands for creating evaluation datasets."""
    pass


def _load_schema_and_tables():
    """Load schema and tables from config.

    Returns:
        Tuple of (schema, tables, config, run_id)
    """
    config = get_config()
    run_id = config.get("run_id")

    # Load schema
    metadata_dir = get_metadata_dir(run_id, config)
    schema_path = find_schema_file(metadata_dir)

    if not schema_path.exists():
        click.echo(f"❌ Schema file not found: {schema_path}", err=True)
        click.echo("   Run 'talk2metadata ingest' first to generate schema.", err=True)
        raise click.Abort()

    try:
        schema = SchemaMetadata.load(schema_path)
        click.echo(f"✓ Loaded schema from {schema_path}")
        click.echo(f"   Target table: {schema.target_table}")
        click.echo(f"   Tables: {len(schema.tables)}")
    except Exception as e:
        click.echo(f"❌ Failed to load schema: {e}", err=True)
        raise click.Abort()

    # Load tables
    data_dir = Path(config.get("data.raw_dir", "./data/raw"))
    if not data_dir.exists():
        possible_dirs = [
            Path("./data/wamex"),
            Path("./data/raw"),
            schema_path.parent.parent / "wamex",
        ]
        for pd in possible_dirs:
            if pd.exists() and any(pd.glob("*.csv")):
                data_dir = pd
                break
        else:
            click.echo(f"❌ Data directory not found: {data_dir}", err=True)
            click.echo("   Please set data.raw_dir in config.yml", err=True)
            raise click.Abort()

    click.echo(f"✓ Using data directory: {data_dir}")

    click.echo("\n📥 Loading tables...")
    try:
        loader = CSVLoader(str(data_dir))
        tables_dict = loader.load_tables()

        tables = {}
        for table_name in schema.tables.keys():
            if table_name in tables_dict:
                tables[table_name] = tables_dict[table_name]
                click.echo(f"   ✓ {table_name}: {len(tables[table_name])} rows")
            else:
                click.echo(f"   ⚠ {table_name}: not found in data directory")

        if not tables:
            click.echo("❌ No tables found matching schema", err=True)
            raise click.Abort()

        return schema, tables, config, run_id

    except Exception as e:
        click.echo(f"❌ Failed to load tables: {e}", err=True)
        raise click.Abort()


@qa_group.command(name="path-generate")
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    help="Output file path (default: qa/kg_paths.json in run directory)",
)
@click.pass_context
def qa_path_generate_cmd(ctx, output_file):
    """Generate path patterns for QA generation.

    This command uses LLM to analyze your database schema and generate
    meaningful path patterns that represent query patterns users might ask.

    \b
    Examples:
        # Generate patterns with settings from config.yml
        talk2metadata qa path-generate

        # Save to custom location
        talk2metadata qa path-generate --output custom_patterns.json
    """
    config = get_config()
    run_id = config.get("run_id")
    qa_config = config.get("qa_generation", {})
    num_patterns = qa_config.get("num_patterns", 15)
    agent_config = config.get("agent", {})
    provider = agent_config.get("provider")
    model = agent_config.get("model")

    # Load schema and tables
    schema, tables, config, run_id = _load_schema_and_tables()

    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        qa_dir = get_qa_dir(run_id, config)
        qa_dir.mkdir(parents=True, exist_ok=True)
        output_path = qa_dir / "kg_paths.json"

    # Generate patterns
    click.echo(f"\n🔍 Generating {num_patterns} path patterns...")
    try:
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

        click.echo(f"\n✓ Generated and saved {len(patterns)} patterns to {output_path}")
        click.echo("\n📝 Next steps:")
        click.echo(f"   1. Review patterns: talk2metadata qa review")
        click.echo(f"   2. Generate QA pairs: talk2metadata qa generate")

    except Exception as e:
        click.echo(f"❌ Failed to generate patterns: {e}", err=True)
        logger.exception("Pattern generation failed")
        raise click.Abort()


@qa_group.command(name="review")
@click.option(
    "--patterns-file",
    "-p",
    "patterns_file",
    type=click.Path(exists=True),
    help="Path to patterns file (default: qa/kg_paths.json in run directory)",
)
@click.pass_context
def qa_review_cmd(ctx, patterns_file):
    """Review and edit path patterns in web browser.

    Opens a web-based interface for reviewing and editing path patterns.
    Changes are saved directly to the patterns file.

    \b
    Examples:
        # Review patterns from default location
        talk2metadata qa review

        # Review patterns from custom file
        talk2metadata qa review --patterns-file custom_patterns.json
    """
    config = get_config()
    run_id = config.get("run_id")

    # Determine patterns file path
    if patterns_file:
        patterns_path = Path(patterns_file)
    else:
        qa_dir = get_qa_dir(run_id, config)
        patterns_path = qa_dir / "kg_paths.json"

    if not patterns_path.exists():
        click.echo(f"❌ Patterns file not found: {patterns_path}", err=True)
        click.echo("   Run 'talk2metadata qa path-generate' first.", err=True)
        raise click.Abort()

    # Load patterns
    try:
        # Load schema (needed for QAGenerator initialization)
        metadata_dir = get_metadata_dir(run_id, config)
        schema_path = find_schema_file(metadata_dir)
        schema = SchemaMetadata.load(schema_path)
        
        # Create minimal generator (tables not needed for loading patterns)
        generator = QAGenerator(
            schema=schema,
            tables={},  # Not needed for review
        )
        patterns = generator.load_patterns(patterns_path)
        click.echo(f"✓ Loaded {len(patterns)} patterns from {patterns_path}")
    except Exception as e:
        click.echo(f"❌ Failed to load patterns: {e}", err=True)
        raise click.Abort()

    # Review patterns
    click.echo(f"\n📝 Opening pattern review interface...")
    try:
        reviewed_patterns = review_patterns_interactive(patterns, patterns_path)
        click.echo(f"\n✓ Review complete, {len(reviewed_patterns)} patterns saved")
    except Exception as e:
        click.echo(f"❌ Review failed: {e}", err=True)
        logger.exception("Pattern review failed")
        raise click.Abort()


@qa_group.command(name="generate")
@click.option(
    "--patterns-file",
    "-p",
    "patterns_file",
    type=click.Path(exists=True),
    help="Path to patterns file (default: qa/kg_paths.json in run directory)",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    type=click.Path(),
    help="Output file path for QA pairs (default: qa/qa_pairs.json in run directory)",
)
@click.pass_context
def qa_generate_cmd(ctx, patterns_file, output_file):
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
    """
    config = get_config()
    run_id = config.get("run_id")
    qa_config = config.get("qa_generation", {})
    instances_per_pattern = qa_config.get("instances_per_pattern", 5)
    validate = qa_config.get("validate", True)
    filter_valid = qa_config.get("filter_valid", True)
    auto_save = qa_config.get("auto_save", True)
    agent_config = config.get("agent", {})
    provider = agent_config.get("provider")
    model = agent_config.get("model")

    # Load schema and tables
    schema, tables, config, run_id = _load_schema_and_tables()

    # Determine patterns file path
    if patterns_file:
        patterns_path = Path(patterns_file)
    else:
        qa_dir = get_qa_dir(run_id, config)
        patterns_path = qa_dir / "kg_paths.json"

    if not patterns_path.exists():
        click.echo(f"❌ Patterns file not found: {patterns_path}", err=True)
        click.echo("   Run 'talk2metadata qa path-generate' first.", err=True)
        raise click.Abort()

    # Determine output path
    if output_file:
        qa_output_path = Path(output_file)
    elif auto_save:
        qa_dir = get_qa_dir(run_id, config)
        qa_dir.mkdir(parents=True, exist_ok=True)
        qa_output_path = qa_dir / "qa_pairs.json"
    else:
        qa_output_path = None

    # Load patterns
    click.echo(f"\n📥 Loading patterns from {patterns_path}...")
    try:
        generator = QAGenerator(
            schema=schema,
            tables=tables,
            provider=provider,
            model=model,
        )
        patterns = generator.load_patterns(patterns_path)
        click.echo(f"✓ Loaded {len(patterns)} patterns")
    except Exception as e:
        click.echo(f"❌ Failed to load patterns: {e}", err=True)
        raise click.Abort()

    # Generate QA pairs
    click.echo(f"\n🔍 Generating QA pairs...")
    click.echo(f"   Patterns: {len(patterns)}")
    click.echo(f"   Instances per pattern: {instances_per_pattern}")
    click.echo(f"   Validation: {'enabled' if validate else 'disabled'}")

    try:
        qa_pairs = generator.generate_qa_from_patterns(
            patterns=patterns,
            instances_per_pattern=instances_per_pattern,
            validate=validate,
            filter_valid=filter_valid,
        )

        click.echo(f"\n✓ Generated {len(qa_pairs)} QA pairs")

        # Show statistics
        if qa_pairs:
            valid_count = sum(1 for qa in qa_pairs if qa.is_valid)
            difficulty_stats = {}
            for qa in qa_pairs:
                diff = qa.difficulty or "unknown"
                difficulty_stats[diff] = difficulty_stats.get(diff, 0) + 1

            click.echo("\n📊 Statistics:")
            click.echo(f"   Valid: {valid_count}/{len(qa_pairs)}")
            click.echo("   Difficulty distribution:")
            for diff, count in sorted(difficulty_stats.items()):
                click.echo(f"     - {diff}: {count}")

    except Exception as e:
        click.echo(f"❌ Failed to generate QA pairs: {e}", err=True)
        logger.exception("QA generation failed")
        raise click.Abort()

    # Save QA pairs
    click.echo(f"\n💾 Saving QA pairs...")
    try:
        saved_path = generator.save(
            qa_pairs,
            output_path=qa_output_path,
            auto_save=auto_save,
            run_id=run_id,
        )
        click.echo(f"✓ Saved {len(qa_pairs)} QA pairs to {saved_path}")
    except Exception as e:
        click.echo(f"❌ Failed to save QA pairs: {e}", err=True)
        raise click.Abort()

    # Show sample QA pairs
    if qa_pairs:
        click.echo("\n📝 Sample QA pairs:")
        for i, qa in enumerate(qa_pairs[:3], 1):
            click.echo(f"\n   {i}. Question: {qa.question}")
            click.echo(f"      Answer: {len(qa.answer_row_ids)} row ID(s)")
            click.echo(f"      Difficulty: {qa.difficulty}")
            if qa.is_valid is False:
                click.echo(f"      ⚠ Invalid: {', '.join(qa.validation_errors)}")

    click.echo("\n✅ QA generation complete!")

