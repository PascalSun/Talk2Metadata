"""Schema review and validation command."""

from __future__ import annotations

import json
from pathlib import Path

import click

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.core.schema_viz import (
    export_schema_for_review,
    generate_html_visualization,
    validate_schema,
)
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.command(name="schema")
@click.option(
    "--schema-file",
    "-s",
    "schema_file",
    type=click.Path(exists=True),
    help="Path to schema JSON file (default: data/metadata/schema.json)",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate HTML visualization",
)
@click.option(
    "--output",
    "-o",
    "output_file",
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
@click.pass_context
def schema_cmd(ctx, schema_file, visualize, output_file, validate, edit, export):
    """Review and validate schema metadata.

    \b
    Examples:
        # View schema summary
        talk2metadata schema

        # Validate schema
        talk2metadata schema --validate

        # Generate visualization
        talk2metadata schema --visualize

        # Edit schema
        talk2metadata schema --edit
    """
    config = get_config()

    # Determine schema file path
    if schema_file:
        schema_path = Path(schema_file)
    else:
        metadata_dir = Path(config.get("data.metadata_dir", "./data/metadata"))
        schema_path = metadata_dir / "schema.json"

    if not schema_path.exists():
        click.echo(f"❌ Schema file not found: {schema_path}", err=True)
        click.echo("   Run 'talk2metadata ingest' first to generate schema.", err=True)
        raise click.Abort()

    # Load schema
    try:
        schema = SchemaMetadata.load(schema_path)
        click.echo(f"✓ Loaded schema from {schema_path}")
    except Exception as e:
        click.echo(f"❌ Failed to load schema: {e}", err=True)
        raise click.Abort()

    # Display schema summary
    click.echo("\n📊 Schema Summary:")
    click.echo(f"   Target Table: {schema.target_table}")
    click.echo(f"   Tables: {len(schema.tables)}")
    click.echo(f"   Foreign Keys: {len(schema.foreign_keys)}")

    if schema.tables:
        click.echo("\n   Tables:")
        for name, meta in schema.tables.items():
            is_target = " (target)" if name == schema.target_table else ""
            click.echo(
                f"     - {name}{is_target}: {meta.row_count} rows, "
                f"{len(meta.columns)} columns, PK={meta.primary_key or 'None'}"
            )

    if schema.foreign_keys:
        click.echo("\n   Foreign Keys:")
        for fk in schema.foreign_keys:
            coverage_icon = "✓" if fk.coverage >= 0.9 else "⚠"
            click.echo(
                f"     {coverage_icon} {fk.child_table}.{fk.child_column} → "
                f"{fk.parent_table}.{fk.parent_column} "
                f"(coverage: {fk.coverage:.1%})"
            )

    # Validate if requested
    if validate:
        click.echo("\n🔍 Validating schema...")
        validation_result = validate_schema(schema)

        if validation_result["errors"]:
            click.echo("\n❌ Errors found:")
            for error in validation_result["errors"]:
                click.echo(f"   - {error}", err=True)

        if validation_result["warnings"]:
            click.echo("\n⚠️  Warnings:")
            for warning in validation_result["warnings"]:
                click.echo(f"   - {warning}")

        if not validation_result["errors"] and not validation_result["warnings"]:
            click.echo("\n✓ Schema validation passed with no errors or warnings!")

    # Generate visualization if requested
    if visualize:
        if output_file:
            viz_path = Path(output_file)
        else:
            viz_path = schema_path.parent / "schema_visualization.html"

        click.echo(f"\n🎨 Generating visualization...")
        try:
            generate_html_visualization(schema, viz_path)
            click.echo(f"✓ Visualization saved to {viz_path}")
            click.echo(f"   Open in browser: file://{viz_path.absolute()}")
        except Exception as e:
            click.echo(f"❌ Failed to generate visualization: {e}", err=True)
            raise click.Abort()

    # Edit schema if requested
    if edit:
        click.echo(f"\n✏️  Opening schema for editing: {schema_path}")
        try:
            import subprocess
            import sys

            # Try to open in default editor
            editor = config.get("editor", None)
            if editor:
                subprocess.run([editor, str(schema_path)])
            else:
                # Try common editors
                for editor_cmd in ["code", "vim", "nano", "vi"]:
                    try:
                        subprocess.run([editor_cmd, str(schema_path)], check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                else:
                    click.echo(
                        "   Please edit the schema file manually: "
                        f"{schema_path.absolute()}",
                        err=True
                    )

            # After editing, reload and validate
            click.echo("\n🔄 Reloading schema after edit...")
            try:
                schema = SchemaMetadata.load(schema_path)
                validation_result = validate_schema(schema)

                if validation_result["errors"]:
                    click.echo("\n❌ Schema has errors after editing:")
                    for error in validation_result["errors"]:
                        click.echo(f"   - {error}", err=True)
                else:
                    click.echo("✓ Schema is valid!")
            except Exception as e:
                click.echo(f"❌ Failed to reload schema: {e}", err=True)

        except Exception as e:
            click.echo(f"❌ Failed to open editor: {e}", err=True)

    # Export schema for review if requested
    if export:
        export_path = Path(export)
        click.echo(f"\n📄 Exporting schema for review to {export_path}...")
        try:
            export_schema_for_review(schema, export_path, include_validation=validate)
            click.echo(f"✓ Schema exported successfully")
            click.echo(f"   Review the file and use it with:")
            click.echo(f"   talk2metadata ingest <source> --schema {export_path}")
        except Exception as e:
            click.echo(f"❌ Failed to export schema: {e}", err=True)
            raise click.Abort()

