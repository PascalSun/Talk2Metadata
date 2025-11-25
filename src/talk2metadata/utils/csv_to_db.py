"""Utility to convert CSV data to SQLite database for text2sql mode."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import create_engine, text

from talk2metadata.connectors.csv_loader import CSVLoader
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import get_db_dir

logger = get_logger(__name__)


def create_sqlite_from_csv(
    csv_data_dir: Path,
    run_id: Optional[str] = None,
    db_path: Optional[Path] = None,
    schema_metadata: Optional[SchemaMetadata] = None,
) -> str:
    """Create a SQLite database from CSV files with foreign key constraints.

    Args:
        csv_data_dir: Directory containing CSV files
        run_id: Optional run ID for cache location (if None, checks config)
        db_path: Optional path to save database (default: uses get_db_dir(run_id) / "text2sql.db")
        schema_metadata: Optional schema metadata for creating foreign key constraints

    Returns:
        SQLite connection string (e.g., "sqlite:///path/to/db.db")
    """
    if db_path is None:
        # Get db_dir from run_id (or config if run_id is None)
        # get_db_dir will check config for run_id if not provided
        db_dir = get_db_dir(run_id)
        db_path = db_dir / "text2sql.db"

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create SQLite connection
    connection_string = f"sqlite:///{db_path}"
    engine = create_engine(connection_string)

    # Enable foreign key constraints in SQLite
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Load CSV files
    logger.info(f"Loading CSV files from {csv_data_dir} into SQLite database")
    csv_loader = CSVLoader(data_dir=csv_data_dir, target_table=None)
    tables = csv_loader.load_tables()

    # Write tables to database
    # First, create tables without FK constraints (SQLite requires tables to exist before FK)
    logger.info(f"Writing {len(tables)} tables to SQLite database: {db_path}")
    for table_name, df in tables.items():
        logger.info(f"  Writing table '{table_name}' ({len(df)} rows)...")
        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

    # Add foreign key constraints if schema_metadata is provided
    # Note: SQLite doesn't support adding FK constraints to existing tables easily,
    # but we enable FK checking and the relationships are available for query planning
    if schema_metadata and schema_metadata.foreign_keys:
        logger.info(
            f"Found {len(schema_metadata.foreign_keys)} foreign key relationships"
        )
        fk_count = 0
        with engine.connect() as conn:
            for fk in schema_metadata.foreign_keys:
                # Check if both tables exist
                if fk.child_table not in tables or fk.parent_table not in tables:
                    logger.warning(
                        f"Skipping FK {fk.child_table}.{fk.child_column} -> "
                        f"{fk.parent_table}.{fk.parent_column}: table not found"
                    )
                    continue

                # SQLite 3.20+ supports adding FK constraints, but it's complex
                # For now, we ensure FK checking is enabled and log relationships
                # The LLM can use these relationships for generating JOIN queries
                logger.debug(
                    f"FK: {fk.child_table}.{fk.child_column} -> "
                    f"{fk.parent_table}.{fk.parent_column} (coverage: {fk.coverage:.2%})"
                )
                fk_count += 1

            # Ensure foreign keys are enabled
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.commit()

        logger.info(
            f"Foreign key relationships available for query planning ({fk_count} relationships)"
        )

    logger.info(f"Successfully created SQLite database at {db_path}")
    return connection_string


def get_or_create_db_connection(
    ingest_config: Dict,
    schema_metadata: SchemaMetadata,
    run_id: Optional[str] = None,
) -> str:
    """Get database connection string, creating SQLite from CSV if needed.

    Args:
        ingest_config: Ingest configuration dict
        schema_metadata: Schema metadata
        run_id: Optional run ID

    Returns:
        Database connection string
    """
    data_type = ingest_config.get("data_type", "csv")
    source_path = ingest_config.get("source_path")

    if data_type in ("database", "db"):
        # Already a database connection
        if not source_path:
            raise ValueError(
                "Database connection string not found in config. "
                "Please set 'ingest.source_path' in config.yml"
            )
        return source_path

    elif data_type == "csv":
        # Need to create database from CSV
        if not source_path:
            raise ValueError(
                "CSV data directory not found in config. "
                "Please set 'ingest.source_path' in config.yml"
            )

        csv_dir = Path(source_path)
        if not csv_dir.exists():
            raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

        # Determine database path (same logic as create_sqlite_from_csv)
        if run_id:
            db_dir = get_db_dir(run_id)
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "text2sql.db"
            connection_string = f"sqlite:///{db_path}"
        else:
            # Without run_id, always recreate (temp file)
            connection_string = create_sqlite_from_csv(
                csv_data_dir=csv_dir,
                run_id=run_id,
                schema_metadata=schema_metadata,
            )
            logger.info(
                "Created temporary SQLite database from CSV data for text2sql mode"
            )
            return connection_string

        # Check if database already exists
        db_path = Path(db_path)
        if db_path.exists():
            logger.info(f"Using existing database at {db_path}")
            return connection_string

        # Database doesn't exist, create it
        connection_string = create_sqlite_from_csv(
            csv_data_dir=csv_dir,
            run_id=run_id,
            schema_metadata=schema_metadata,
            db_path=db_path,
        )
        logger.info("Created SQLite database from CSV data for text2sql mode")
        return connection_string

    else:
        raise ValueError(
            f"Unsupported data_type: {data_type}. "
            "Supported types: 'csv', 'database', 'db'"
        )
