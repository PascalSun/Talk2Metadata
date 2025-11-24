"""Schema management commands - ingest, validate, visualize."""

from __future__ import annotations

from pathlib import Path

import click

from talk2metadata.cli.decorators import handle_errors, with_run_id, with_schema_file
from talk2metadata.cli.handlers import IngestHandler, SchemaHandler
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.group(name="schema")
def schema_group():
    """Schema management commands for data ingestion and validation.

    This command group provides tools for ingesting data, validating schema,
    and visualizing database relationships.
    """
    pass


@schema_group.command(name="ingest")
@click.argument(
    "source_type",
    required=False,
    type=click.Choice(["csv", "database", "db"], case_sensitive=False),
)
@click.argument("source_path", required=False)
@click.option(
    "--target",
    "-t",
    "target_table",
    required=False,
    help="Target table name (can also be set in config.yml)",
)
@click.option(
    "--schema",
    "-s",
    "schema_file",
    type=click.Path(exists=True),
    help="Optional schema JSON file with FK definitions",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    help="Output directory for metadata (default: data/metadata)",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate HTML visualization of schema after detection",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate schema and show errors/warnings before saving",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip schema validation (use with caution)",
)
@with_run_id
@handle_errors
@click.pass_context
def ingest_cmd(
    ctx,
    source_type,
    source_path,
    target_table,
    schema_file,
    output_dir,
    visualize,
    validate,
    skip_validation,
    run_id,
):
    """Ingest data from CSV files or database.

    SOURCE_TYPE: csv, database, or db (optional if set in config.yml)

    SOURCE_PATH: Path to CSV directory or database connection string (optional if set in config.yml)

    \b
    Examples:
        # Ingest from CSV files
        talk2metadata schema ingest csv ./data/csv --target orders

        # Ingest from PostgreSQL
        talk2metadata schema ingest database "postgresql://localhost/mydb" --target orders

        # Ingest from SQLite
        talk2metadata schema ingest database "sqlite:///mydb.db" --target orders

        # Ingest with provided schema
        talk2metadata schema ingest csv ./data/csv --target orders --schema schema.json

        # Ingest with specific run ID
        talk2metadata schema ingest csv ./data/csv --target orders --run-id my_run

        # Ingest using config.yml settings
        talk2metadata schema ingest
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Read from config if not provided via CLI
    if source_type is None:
        source_type = config.get("ingest.data_type")
    if source_path is None:
        source_path = config.get("ingest.source_path")
    if target_table is None:
        target_table = config.get("ingest.target_table")

    # Validate required parameters
    if source_type is None:
        out.error(
            "source_type is required. Provide it as an argument or set 'ingest.data_type' in config.yml"
        )
        raise click.Abort()
    if source_path is None:
        out.error(
            "source_path is required. Provide it as an argument or set 'ingest.source_path' in config.yml"
        )
        raise click.Abort()
    if target_table is None:
        out.error(
            "target_table is required. Provide it with --target option or set 'ingest.target_table' in config.yml"
        )
        raise click.Abort()

    out.section(f"🔧 Ingesting from {source_type}: {source_path}")
    out.stats({"Target table": target_table})

    # Initialize handler
    handler = IngestHandler(config)

    # 1. Create connector and load tables
    out.progress_start("Loading tables...")
    try:
        connector = handler.create_connector(source_type, source_path, target_table)
        tables = connector.load_tables()

        out.success(f"Loaded {len(tables)} tables:")
        for name, df in tables.items():
            out.table_summary(name, len(df), len(df.columns))
    except Exception as e:
        out.error(f"Failed to load tables: {e}")
        raise click.Abort()

    # 2. Detect schema and FKs
    out.progress_start("Detecting schema and foreign keys...")
    try:
        provided_schema = handler.load_provided_schema(schema_file)
        if schema_file:
            out.info(f"Using provided schema from {schema_file}")

        metadata = handler.detect_schema(tables, target_table, provided_schema)

        out.success("Schema detection complete:")
        out.stats(
            {
                "Tables": len(metadata.tables),
                "Foreign keys": len(metadata.foreign_keys),
            }
        )

        if metadata.foreign_keys:
            out.section("   Foreign key relationships:")
            for fk in metadata.foreign_keys:
                coverage_icon = "✓" if fk.coverage >= 0.9 else "⚠"
                click.echo(
                    f"     {coverage_icon} {fk.child_table}.{fk.child_column} → "
                    f"{fk.parent_table}.{fk.parent_column} "
                    f"(coverage: {fk.coverage:.1%})"
                )
    except Exception as e:
        out.error(f"Schema detection failed: {e}")
        raise click.Abort()

    # 3. Validate schema (if requested or by default)
    if validate or not skip_validation:
        out.section("🔍 Validating schema...")
        validation_result = handler.validate_schema_metadata(metadata)

        if validation_result["errors"]:
            out.section("❌ Schema validation found errors:")
            out.list_items(validation_result["errors"])
            out.warning("Schema has errors. Please review and fix before indexing.")
            out.info(
                "Use 'talk2metadata schema validate --edit' to modify the schema file."
            )

            if not click.confirm("\n   Continue anyway?", default=False):
                raise click.Abort()

        if validation_result["warnings"]:
            out.section("⚠️  Schema validation warnings:")
            out.list_items(validation_result["warnings"])

        if not validation_result["errors"] and not validation_result["warnings"]:
            out.success("Schema validation passed!")

    # 4. Save metadata
    out.section("💾 Saving metadata...")
    try:
        metadata_path = handler.save_metadata(metadata, output_dir, run_id)
        out.success(f"Metadata saved to {metadata_path}")
    except Exception as e:
        out.error(f"Failed to save metadata: {e}")
        raise click.Abort()

    # 5. Generate visualization if requested
    if visualize:
        out.section("🎨 Generating schema visualization...")
        try:
            viz_path = handler.generate_visualization(metadata, output_dir, run_id)
            out.success(f"Visualization saved to {viz_path}")
            click.echo(f"   Open in browser: file://{viz_path.absolute()}")
        except Exception as e:
            out.warning(f"Failed to generate visualization: {e}")
            # Don't abort, visualization is optional

    # 6. Save tables for indexing
    out.section("💾 Saving processed tables...")
    try:
        tables_path = handler.save_tables(tables, run_id)
        out.success(f"Tables saved to {tables_path}")
    except Exception as e:
        out.error(f"Failed to save tables: {e}")
        raise click.Abort()

    # Success
    out.section("✅ Ingestion complete!")
    next_steps = ["Review schema: talk2metadata schema validate"]
    if not visualize:
        next_steps.append("Visualize schema: talk2metadata schema visualize")
    next_steps.append("Build index: talk2metadata search index")
    out.next_steps("Next steps:", next_steps)


@schema_group.command(name="validate")
@with_schema_file
@with_run_id
@click.option(
    "--edit",
    "-e",
    is_flag=True,
    help="Open schema in editor for modification",
)
@click.option(
    "--export",
    type=click.Path(),
    help="Export schema in review-friendly format to specified file",
)
@handle_errors
@click.pass_context
def validate_cmd(ctx, schema_file, run_id, edit, export):
    """Validate schema metadata and check for errors.

    This command validates your database schema metadata, checking for
    issues with tables, columns, foreign keys, and relationships.

    \b
    Examples:
        # Validate schema
        talk2metadata schema validate

        # Edit schema interactively
        talk2metadata schema validate --edit

        # Export schema for review
        talk2metadata schema validate --export schema_review.json

        # Use specific run ID
        talk2metadata schema validate --run-id wamex_run
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Load schema
    loader = CLIDataLoader(config)
    schema = loader.load_schema(schema_file=schema_file, run_id=run_id)

    # Get schema path for operations that need it
    from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

    metadata_dir = get_metadata_dir(run_id or config.get("run_id"), config)
    target_table = config.get("ingest.target_table")
    schema_path = (
        Path(schema_file)
        if schema_file
        else find_schema_file(metadata_dir, target_table=target_table)
    )

    # Display schema summary
    handler = SchemaHandler(config)
    summary = handler.get_schema_summary(schema)

    out.section("📊 Schema Summary:")
    out.stats(
        {
            "Target Table": summary["target_table"],
            "Tables": summary["num_tables"],
            "Foreign Keys": summary["num_foreign_keys"],
        }
    )

    if schema.tables:
        out.section("   Tables:")
        for name, meta in schema.tables.items():
            is_target = " (target)" if name == schema.target_table else ""
            click.echo(
                f"     - {name}{is_target}: {meta.row_count} rows, "
                f"{len(meta.columns)} columns, PK={meta.primary_key or 'None'}"
            )

    if schema.foreign_keys:
        out.section("   Foreign Keys:")
        for fk in schema.foreign_keys:
            coverage_icon = "✓" if fk.coverage >= 0.9 else "⚠"
            click.echo(
                f"     {coverage_icon} {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} "
                f"(coverage: {fk.coverage:.1%})"
            )

    # Validate schema
    out.section("🔍 Validating schema...")
    validation_result = handler.validate(schema)

    if validation_result["errors"]:
        out.section("❌ Errors found:")
        out.list_items(validation_result["errors"])

    if validation_result["warnings"]:
        out.section("⚠️  Warnings:")
        out.list_items(validation_result["warnings"])

    if not validation_result["errors"] and not validation_result["warnings"]:
        out.success("Schema validation passed with no errors or warnings!")

    # Edit schema if requested
    if edit:
        out.section(f"✏️  Opening schema for editing: {schema_path}")
        try:
            schema = handler.edit_schema(schema_path)

            # After editing, reload and validate
            out.section("🔄 Reloading schema after edit...")
            validation_result = handler.validate(schema)

            if validation_result["errors"]:
                out.section("❌ Schema has errors after editing:")
                out.list_items(validation_result["errors"])
            else:
                out.success("Schema is valid!")

        except Exception as e:
            out.error(f"Failed to open editor: {e}")
            out.info(f"Please edit the schema file manually: {schema_path.absolute()}")

    # Export schema for review if requested
    if export:
        export_path = Path(export)
        out.section(f"📄 Exporting schema for review to {export_path}...")
        try:
            handler.export_for_review(
                schema=schema,
                output_file=str(export_path),
                include_validation=True,
            )
            out.success("Schema exported successfully")
            out.next_steps(
                "Review the file and use it with:",
                [f"talk2metadata schema ingest <source> --schema {export_path}"],
            )
        except Exception as e:
            out.error(f"Failed to export schema: {e}")
            raise click.Abort()


@schema_group.command(name="visualize")
@with_schema_file
@with_run_id
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for visualization HTML (default: schema_visualization.html)",
)
@handle_errors
@click.pass_context
def visualize_cmd(ctx, schema_file, run_id, output):
    """Generate HTML visualization of schema relationships.

    This command creates an interactive HTML visualization showing tables,
    columns, foreign keys, and their relationships in your database schema.

    \b
    Examples:
        # Generate visualization
        talk2metadata schema visualize

        # Generate with custom output path
        talk2metadata schema visualize --output my_schema.html

        # Use specific run ID
        talk2metadata schema visualize --run-id wamex_run
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Load schema
    loader = CLIDataLoader(config)
    schema = loader.load_schema(schema_file=schema_file, run_id=run_id)

    # Get schema path
    from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

    metadata_dir = get_metadata_dir(run_id or config.get("run_id"), config)
    target_table = config.get("ingest.target_table")
    schema_path = (
        Path(schema_file)
        if schema_file
        else find_schema_file(metadata_dir, target_table=target_table)
    )

    # Generate visualization
    out.section("🎨 Generating visualization...")
    handler = SchemaHandler(config)
    try:
        viz_path = handler.generate_visualization(
            schema=schema,
            output_file=output,
            schema_path=schema_path,
        )
        out.success(f"Visualization saved to {viz_path}")
        click.echo(f"   Open in browser: file://{viz_path.absolute()}")
    except Exception as e:
        out.error(f"Failed to generate visualization: {e}")
        raise click.Abort()
