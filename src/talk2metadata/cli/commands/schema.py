"""Schema review and validation commands - refactored with modular structure."""

from __future__ import annotations

from pathlib import Path

import click

from talk2metadata.cli.decorators import handle_errors, with_run_id, with_schema_file
from talk2metadata.cli.handlers import SchemaHandler
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.command(name="schema")
@with_schema_file
@with_run_id
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate HTML visualization",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for visualization HTML (default: schema_visualization.html)",
)
@click.option(
    "--validate",
    is_flag=True,
    help="Validate schema and show errors/warnings",
)
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
def schema_cmd(ctx, schema_file, run_id, visualize, output, validate, edit, export):
    """Review and validate schema metadata.

    This command provides tools for reviewing, validating, and visualizing
    your database schema metadata, including tables, columns, and foreign keys.

    \b
    Examples:
        # View schema summary
        talk2metadata schema

        # Validate schema
        talk2metadata schema --validate

        # Generate visualization
        talk2metadata schema --visualize

        # Generate visualization with custom output
        talk2metadata schema --visualize --output my_schema.html

        # Edit schema interactively
        talk2metadata schema --edit

        # Export schema for review
        talk2metadata schema --export schema_review.json

        # Use specific run ID
        talk2metadata schema --run-id wamex_run
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
    schema_path = Path(schema_file) if schema_file else find_schema_file(metadata_dir)

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

    # Validate if requested
    if validate:
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

    # Generate visualization if requested
    if visualize:
        out.section("🎨 Generating visualization...")
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
                include_validation=validate,
            )
            out.success("Schema exported successfully")
            out.next_steps(
                "Review the file and use it with:",
                [f"talk2metadata ingest <source> --schema {export_path}"],
            )
        except Exception as e:
            out.error(f"Failed to export schema: {e}")
            raise click.Abort()
