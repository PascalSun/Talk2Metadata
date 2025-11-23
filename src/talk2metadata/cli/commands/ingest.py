"""Ingest command for loading data."""

from __future__ import annotations

import re
from pathlib import Path

import click

from talk2metadata.connectors import ConnectorFactory
from talk2metadata.core.schema import SchemaDetector
from talk2metadata.core.schema_viz import generate_html_visualization, validate_schema
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_metadata_dir, get_processed_dir

logger = get_logger(__name__)


@click.command(name="ingest")
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
):
    """Ingest data from CSV files or database.

    SOURCE_TYPE: csv, database, or db (optional if set in config.yml)

    SOURCE_PATH: Path to CSV directory or database connection string (optional if set in config.yml)

    \b
    Examples:
        # Ingest from CSV files (with CLI arguments)
        talk2metadata ingest csv ./data/csv --target orders

        # Ingest from PostgreSQL
        talk2metadata ingest database "postgresql://localhost/mydb" --target orders

        # Ingest from SQLite
        talk2metadata ingest database "sqlite:///mydb.db" --target orders

        # Ingest with provided schema
        talk2metadata ingest csv ./data/csv --target orders --schema schema.json

        # Ingest using config.yml settings (no arguments needed)
        talk2metadata ingest
    """
    config = get_config()

    # Read from config if not provided via CLI
    if source_type is None:
        source_type = config.get("ingest.data_type")
    if source_path is None:
        source_path = config.get("ingest.source_path")
    if target_table is None:
        target_table = config.get("ingest.target_table")

    # Validate required parameters
    if source_type is None:
        click.echo(
            "❌ Error: source_type is required. Provide it as an argument or set 'ingest.data_type' in config.yml",
            err=True,
        )
        raise click.Abort()
    if source_path is None:
        click.echo(
            "❌ Error: source_path is required. Provide it as an argument or set 'ingest.source_path' in config.yml",
            err=True,
        )
        raise click.Abort()
    if target_table is None:
        click.echo(
            "❌ Error: target_table is required. Provide it with --target option or set 'ingest.target_table' in config.yml",
            err=True,
        )
        raise click.Abort()

    click.echo(f"🔧 Ingesting from {source_type}: {source_path}")
    click.echo(f"   Target table: {target_table}")

    # 1. Create connector
    try:
        if source_type in ["csv"]:
            connector = ConnectorFactory.create_connector(
                "csv",
                data_dir=source_path,
                target_table=target_table,
            )
        else:  # database/db
            connector = ConnectorFactory.create_connector(
                "database",
                connection_string=source_path,
                target_table=target_table,
            )
    except Exception as e:
        click.echo(f"❌ Failed to create connector: {e}", err=True)
        raise click.Abort()

    # 2. Load tables
    click.echo("📥 Loading tables...")
    try:
        tables = connector.load_tables()
        click.echo(f"✓ Loaded {len(tables)} tables:")
        for name, df in tables.items():
            click.echo(f"   - {name}: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        click.echo(f"❌ Failed to load tables: {e}", err=True)
        raise click.Abort()

    # 3. Load provided schema if available
    provided_schema = None
    if schema_file:
        import json

        click.echo(f"📄 Loading schema from {schema_file}")
        with open(schema_file, "r") as f:
            provided_schema = json.load(f)

    # 4. Detect schema and FKs
    click.echo("🔍 Detecting schema and foreign keys...")
    try:
        detector = SchemaDetector()
        metadata = detector.detect(
            tables,
            target_table=target_table,
            provided_schema=provided_schema,
        )

        click.echo("✓ Schema detection complete:")
        click.echo(f"   - Tables: {len(metadata.tables)}")
        click.echo(f"   - Foreign keys: {len(metadata.foreign_keys)}")

        if metadata.foreign_keys:
            click.echo("   Foreign key relationships:")
            for fk in metadata.foreign_keys:
                click.echo(
                    f"     - {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column} "
                    f"(coverage: {fk.coverage:.1%})"
                )
    except Exception as e:
        click.echo(f"❌ Schema detection failed: {e}", err=True)
        raise click.Abort()

    # 4.5. Validate schema (if requested or by default)
    if validate or not skip_validation:
        click.echo("\n🔍 Validating schema...")
        validation_result = validate_schema(metadata)

        if validation_result["errors"]:
            click.echo("\n❌ Schema validation found errors:")
            for error in validation_result["errors"]:
                click.echo(f"   - {error}", err=True)
            click.echo(
                "\n⚠️  Schema has errors. Please review and fix before indexing.",
                err=True,
            )
            click.echo(
                "   Use 'talk2metadata schema --edit' to modify the schema file.",
                err=True,
            )
            if not click.confirm("\n   Continue anyway?", default=False):
                raise click.Abort()

        if validation_result["warnings"]:
            click.echo("\n⚠️  Schema validation warnings:")
            for warning in validation_result["warnings"]:
                click.echo(f"   - {warning}")

        if not validation_result["errors"] and not validation_result["warnings"]:
            click.echo("✓ Schema validation passed!")

    # 5. Save metadata
    run_id = config.get("run_id")

    if output_dir:
        metadata_dir = Path(output_dir)
    else:
        metadata_dir = get_metadata_dir(run_id, config)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    # Generate filename with target table name
    target_table_safe = re.sub(r"[^\w\-_.]", "_", metadata.target_table)
    metadata_path = metadata_dir / f"schema_{target_table_safe}.json"

    click.echo(f"\n💾 Saving metadata to {metadata_path}")
    try:
        metadata.save(metadata_path)
        click.echo("✓ Metadata saved successfully")
    except Exception as e:
        click.echo(f"❌ Failed to save metadata: {e}", err=True)
        raise click.Abort()

    # 5.5. Generate visualization if requested
    if visualize:
        viz_path = metadata_dir / f"schema_visualization_{target_table_safe}.html"
        click.echo("\n🎨 Generating schema visualization...")
        try:
            generate_html_visualization(metadata, viz_path)
            click.echo(f"✓ Visualization saved to {viz_path}")
            click.echo(f"   Open in browser: file://{viz_path.absolute()}")
        except Exception as e:
            click.echo(f"⚠️  Failed to generate visualization: {e}")
            # Don't abort, visualization is optional

    # 6. Save raw tables (for indexing)
    processed_dir = get_processed_dir(run_id, config)
    processed_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"💾 Saving processed tables to {processed_dir}")
    try:
        import pickle

        tables_path = processed_dir / "tables.pkl"
        with open(tables_path, "wb") as f:
            pickle.dump(tables, f)
        click.echo(f"✓ Tables saved to {tables_path}")
    except Exception as e:
        click.echo(f"❌ Failed to save tables: {e}", err=True)
        raise click.Abort()

    click.echo("\n✅ Ingestion complete!")
    click.echo("\nNext steps:")
    click.echo("   1. Review schema: talk2metadata schema --validate")
    if not visualize:
        click.echo("   2. Visualize schema: talk2metadata schema --visualize")
    click.echo("   3. Build index: talk2metadata index")
