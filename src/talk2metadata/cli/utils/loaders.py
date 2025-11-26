"""Common data loading utilities for CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import pandas as pd

from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import Config, get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

logger = get_logger(__name__)


class CLIDataLoader:
    """Centralized data loading for CLI commands.

    This class provides consistent data loading patterns across all CLI commands,
    handling common cases like loading schemas, tables, and related metadata.

    Example:
        >>> loader = CLIDataLoader()
        >>> schema = loader.load_schema()
        >>> tables = loader.load_tables(schema)
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize loader with config.

        Args:
            config: Optional Config instance. If None, uses global config.
        """
        self.config = config or get_config()

    def load_schema(
        self,
        schema_file: Optional[str] = None,
        run_id: Optional[str] = None,
        target_table: Optional[str] = None,
        echo: bool = True,
    ) -> SchemaMetadata:
        """Load schema from file or config.

        Args:
            schema_file: Optional explicit path to schema file
            run_id: Optional run ID (overrides config)
            target_table: Optional target table name (reads from config if not provided)
            echo: Whether to echo progress to console

        Returns:
            SchemaMetadata instance

        Raises:
            click.Abort: If schema cannot be loaded
        """
        run_id = run_id or self.config.get("run_id")

        # Read target_table from config if not provided
        if target_table is None:
            target_table = self.config.get("ingest.target_table")

        # Determine schema file path
        if schema_file:
            schema_path = Path(schema_file)
        else:
            metadata_dir = get_metadata_dir(run_id, self.config)
            try:
                schema_path = find_schema_file(metadata_dir, target_table=target_table)
            except FileNotFoundError:
                if echo:
                    click.echo(f"❌ Schema file not found in {metadata_dir}", err=True)
                    if target_table:
                        click.echo(
                            f"   Expected: schema_{target_table}.json or schema.json",
                            err=True,
                        )
                    click.echo(
                        "   Run 'talk2metadata schema ingest' first to generate schema.",
                        err=True,
                    )
                raise click.Abort()

        if not schema_path.exists():
            if echo:
                click.echo(f"❌ Schema file not found: {schema_path}", err=True)
            raise click.Abort()

        # Load schema
        try:
            schema = SchemaMetadata.load(schema_path)

            # Validate that loaded schema matches expected target_table
            if target_table and schema.target_table != target_table:
                # If schema_file was explicitly provided but doesn't match, try to find correct one
                if schema_file:
                    metadata_dir = get_metadata_dir(run_id, self.config)
                    try:
                        correct_schema_path = find_schema_file(
                            metadata_dir, target_table=target_table
                        )
                        if (
                            correct_schema_path.exists()
                            and correct_schema_path != schema_path
                        ):
                            if echo:
                                click.echo(
                                    "⚠️  Schema mismatch detected!",
                                    err=True,
                                )
                                click.echo(
                                    f"   Provided: {schema_path} (target: {schema.target_table})",
                                    err=True,
                                )
                                click.echo(
                                    f"   Expected: {correct_schema_path} (target: {target_table})",
                                    err=True,
                                )
                                click.echo(
                                    f"   Auto-correcting to use: {correct_schema_path}",
                                    err=True,
                                )
                            logger.warning(
                                f"Schema target table mismatch: expected {target_table}, "
                                f"got {schema.target_table} from {schema_path}. "
                                f"Switching to {correct_schema_path}"
                            )
                            schema_path = correct_schema_path
                            schema = SchemaMetadata.load(schema_path)
                    except FileNotFoundError:
                        # Correct schema not found, proceed with warning
                        if echo:
                            click.echo(
                                "⚠️  Warning: Schema target table mismatch!",
                                err=True,
                            )
                            click.echo(
                                f"   Expected: {target_table}, but loaded: {schema.target_table}",
                                err=True,
                            )
                            click.echo(
                                f"   Schema file: {schema_path}",
                                err=True,
                            )
                        logger.warning(
                            f"Schema target table mismatch: expected {target_table}, "
                            f"got {schema.target_table} from {schema_path}"
                        )
                else:
                    # Schema was auto-selected but doesn't match - this shouldn't happen
                    if echo:
                        click.echo(
                            "⚠️  Warning: Schema target table mismatch!",
                            err=True,
                        )
                        click.echo(
                            f"   Expected: {target_table}, but loaded: {schema.target_table}",
                            err=True,
                        )
                    logger.warning(
                        f"Schema target table mismatch: expected {target_table}, "
                        f"got {schema.target_table} from {schema_path}"
                    )

            if echo:
                click.echo(f"✓ Loaded schema from {schema_path}")
                click.echo(f"   Target table: {schema.target_table}")
                click.echo(f"   Tables: {len(schema.tables)}")
            return schema
        except Exception as e:
            if echo:
                click.echo(f"❌ Failed to load schema: {e}", err=True)
            logger.exception("Failed to load schema")
            raise click.Abort()

    def load_tables(
        self,
        schema: SchemaMetadata,
        data_dir: Optional[Path] = None,
        echo: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Load tables matching schema.

        Args:
            schema: Schema metadata defining which tables to load
            data_dir: Optional data directory path
            echo: Whether to echo progress to console

        Returns:
            Dict mapping table_name -> DataFrame

        Raises:
            click.Abort: If tables cannot be loaded
        """
        # Determine data directory
        if data_dir is None:
            # Priority 1: Use ingest.source_path if data_type is csv (most explicit)
            ingest_data_type = self.config.get("ingest.data_type")
            ingest_source_path = self.config.get("ingest.source_path")
            if ingest_data_type == "csv" and ingest_source_path:
                data_dir = Path(ingest_source_path)
            else:
                # Priority 2: Try data.raw_dir (for backward compatibility)
                raw_dir = self.config.get("data.raw_dir")
                if raw_dir:
                    data_dir = Path(raw_dir)
                else:
                    # Priority 3: Default fallback
                    data_dir = Path("./data/raw")

        if not data_dir.exists():
            # Try common fallback locations
            possible_dirs = [
                Path("./data/wamex"),
                Path("./data/raw"),
                Path("./data/processed"),
            ]
            # Also try ingest.source_path if it's a CSV directory
            ingest_data_type = self.config.get("ingest.data_type")
            ingest_source_path = self.config.get("ingest.source_path")
            if ingest_data_type == "csv" and ingest_source_path:
                possible_dirs.insert(0, Path(ingest_source_path))

            for pd_path in possible_dirs:
                if pd_path.exists() and any(pd_path.glob("*.csv")):
                    data_dir = pd_path
                    break
            else:
                if echo:
                    click.echo(f"❌ Data directory not found: {data_dir}", err=True)
                    click.echo(
                        "   Please set 'data.raw_dir' or 'ingest.source_path' (when data_type is csv) in config.yml",
                        err=True,
                    )
                raise click.Abort()

        if echo:
            click.echo(f"✓ Using data directory: {data_dir}")
            click.echo("\n📥 Loading tables...")

        try:
            loader = CSVLoader(str(data_dir))
            tables_dict = loader.load_tables()

            tables = {}
            missing_tables = []
            for table_name in schema.tables.keys():
                if table_name in tables_dict:
                    tables[table_name] = tables_dict[table_name]
                    if echo:
                        click.echo(f"   ✓ {table_name}: {len(tables[table_name])} rows")
                else:
                    missing_tables.append(table_name)
                    if echo:
                        click.echo(f"   ⚠ {table_name}: not found in data directory")

            if not tables:
                if echo:
                    click.echo("❌ No tables found matching schema", err=True)
                    if missing_tables:
                        click.echo(
                            f"   Missing tables: {', '.join(missing_tables)}", err=True
                        )
                        click.echo(f"   Data directory used: {data_dir}", err=True)
                        # Show expected directory from config
                        expected_dir = self.config.get("ingest.source_path")
                        if expected_dir:
                            click.echo(
                                f"   Expected from config: {expected_dir}", err=True
                            )
                raise click.Abort()

            # Warn if some tables are missing but we have at least some
            if missing_tables and echo:
                click.echo(
                    f"\n⚠️  Warning: {len(missing_tables)} table(s) from schema not found in {data_dir}:"
                )
                for table_name in missing_tables:
                    click.echo(f"   - {table_name}")

            return tables

        except Exception as e:
            if echo:
                click.echo(f"❌ Failed to load tables: {e}", err=True)
            logger.exception("Failed to load tables")
            raise click.Abort()

    def load_schema_and_tables(
        self,
        schema_file: Optional[str] = None,
        data_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        target_table: Optional[str] = None,
        echo: bool = True,
    ) -> Tuple[SchemaMetadata, Dict[str, pd.DataFrame], Config, str]:
        """Load both schema and tables in one call.

        Convenience method that combines load_schema and load_tables.

        Args:
            schema_file: Optional path to schema file
            data_dir: Optional data directory path
            run_id: Optional run ID
            target_table: Optional target table name (reads from config if not provided)
            echo: Whether to echo progress to console

        Returns:
            Tuple of (schema, tables, config, run_id)

        Raises:
            click.Abort: If loading fails
        """
        schema = self.load_schema(
            schema_file=schema_file, run_id=run_id, target_table=target_table, echo=echo
        )
        tables = self.load_tables(schema=schema, data_dir=data_dir, echo=echo)
        run_id = run_id or self.config.get("run_id")
        return schema, tables, self.config, run_id
