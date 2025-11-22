"""Ingest command for loading data."""

from __future__ import annotations

from pathlib import Path

import click

from talk2metadata.connectors import ConnectorFactory
from talk2metadata.core.schema import SchemaDetector
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.command(name="ingest")
@click.argument(
    "source_type",
    type=click.Choice(["csv", "database", "db"], case_sensitive=False),
)
@click.argument("source_path")
@click.option(
    "--target",
    "-t",
    "target_table",
    required=True,
    help="Target table name",
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
@click.pass_context
def ingest_cmd(ctx, source_type, source_path, target_table, schema_file, output_dir):
    """Ingest data from CSV files or database.

    SOURCE_TYPE: csv, database, or db

    SOURCE_PATH: Path to CSV directory or database connection string

    \b
    Examples:
        # Ingest from CSV files
        talk2metadata ingest csv ./data/csv --target orders

        # Ingest from PostgreSQL
        talk2metadata ingest database "postgresql://localhost/mydb" --target orders

        # Ingest from SQLite
        talk2metadata ingest database "sqlite:///mydb.db" --target orders

        # Ingest with provided schema
        talk2metadata ingest csv ./data/csv --target orders --schema schema.json
    """
    config = get_config()

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

        click.echo(f"✓ Schema detection complete:")
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

    # 5. Save metadata
    if output_dir:
        metadata_dir = Path(output_dir)
    else:
        metadata_dir = Path(config.get("data.metadata_dir", "./data/metadata"))

    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "schema.json"

    click.echo(f"💾 Saving metadata to {metadata_path}")
    try:
        metadata.save(metadata_path)
        click.echo(f"✓ Metadata saved successfully")
    except Exception as e:
        click.echo(f"❌ Failed to save metadata: {e}", err=True)
        raise click.Abort()

    # 6. Save raw tables (for indexing)
    processed_dir = Path(config.get("data.processed_dir", "./data/processed"))
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
    click.echo(f"\nNext step: Run 'talk2metadata index' to build search index")
