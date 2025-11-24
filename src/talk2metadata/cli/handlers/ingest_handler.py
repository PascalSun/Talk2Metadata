"""Business logic for data ingestion commands."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from talk2metadata.connectors import ConnectorFactory
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.core.schema.schema import SchemaDetector
from talk2metadata.core.schema.schema_viz import (
    generate_html_visualization,
    validate_schema,
)
from talk2metadata.utils.config import Config
from talk2metadata.utils.paths import get_metadata_dir, get_processed_dir


class IngestHandler:
    """Handler for data ingestion operations.

    Encapsulates business logic for data ingestion, schema detection,
    and metadata management.
    """

    def __init__(self, config: Config):
        """Initialize handler.

        Args:
            config: Configuration instance
        """
        self.config = config

    def create_connector(
        self,
        source_type: str,
        source_path: str,
        target_table: str,
    ):
        """Create data connector.

        Args:
            source_type: Type of data source ('csv', 'database', 'db')
            source_path: Path to data or connection string
            target_table: Target table name

        Returns:
            Connector instance

        Raises:
            Exception: If connector creation fails
        """
        if source_type in ["csv"]:
            return ConnectorFactory.create_connector(
                "csv",
                data_dir=source_path,
                target_table=target_table,
            )
        else:  # database/db
            return ConnectorFactory.create_connector(
                "database",
                connection_string=source_path,
                target_table=target_table,
            )

    def load_provided_schema(self, schema_file: str) -> Optional[Dict]:
        """Load schema from file if provided.

        Args:
            schema_file: Path to schema JSON file

        Returns:
            Schema dictionary or None
        """
        if not schema_file:
            return None

        with open(schema_file, "r") as f:
            return json.load(f)

    def detect_schema(
        self,
        tables: Dict[str, pd.DataFrame],
        target_table: str,
        provided_schema: Optional[Dict] = None,
    ) -> SchemaMetadata:
        """Detect schema and foreign keys.

        Args:
            tables: Dictionary of DataFrames
            target_table: Target table name
            provided_schema: Optional provided schema

        Returns:
            SchemaMetadata instance
        """
        detector = SchemaDetector()
        return detector.detect(
            tables,
            target_table=target_table,
            provided_schema=provided_schema,
        )

    def validate_schema_metadata(self, metadata: SchemaMetadata) -> Dict[str, list]:
        """Validate schema metadata.

        Args:
            metadata: Schema metadata to validate

        Returns:
            Dictionary with 'errors' and 'warnings' keys
        """
        return validate_schema(metadata)

    def save_metadata(
        self,
        metadata: SchemaMetadata,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Save schema metadata.

        Args:
            metadata: Schema metadata to save
            output_dir: Optional output directory
            run_id: Optional run ID

        Returns:
            Path to saved metadata file
        """
        if output_dir:
            metadata_dir = Path(output_dir)
        else:
            metadata_dir = get_metadata_dir(
                run_id or self.config.get("run_id"), self.config
            )

        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with target table name
        target_table_safe = re.sub(r"[^\w\-_.]", "_", metadata.target_table)
        metadata_path = metadata_dir / f"schema_{target_table_safe}.json"

        metadata.save(metadata_path)
        return metadata_path

    def generate_visualization(
        self,
        metadata: SchemaMetadata,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Path:
        """Generate HTML visualization of schema.

        Args:
            metadata: Schema metadata
            output_dir: Optional output directory
            run_id: Optional run ID

        Returns:
            Path to visualization file
        """
        if output_dir:
            metadata_dir = Path(output_dir)
        else:
            metadata_dir = get_metadata_dir(
                run_id or self.config.get("run_id"), self.config
            )

        target_table_safe = re.sub(r"[^\w\-_.]", "_", metadata.target_table)
        viz_path = metadata_dir / f"schema_visualization_{target_table_safe}.html"

        generate_html_visualization(metadata, viz_path)
        return viz_path

    def save_tables(
        self,
        tables: Dict[str, pd.DataFrame],
        run_id: Optional[str] = None,
    ) -> Path:
        """Save tables for later indexing.

        Args:
            tables: Dictionary of DataFrames
            run_id: Optional run ID

        Returns:
            Path to saved tables file
        """
        processed_dir = get_processed_dir(
            run_id or self.config.get("run_id"), self.config
        )
        processed_dir.mkdir(parents=True, exist_ok=True)

        tables_path = processed_dir / "tables.pkl"
        with open(tables_path, "wb") as f:
            pickle.dump(tables, f)

        return tables_path

    def ingest(
        self,
        source_type: str,
        source_path: str,
        target_table: str,
        schema_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Tuple[SchemaMetadata, Dict[str, pd.DataFrame], Path, Path]:
        """Complete ingestion workflow.

        Args:
            source_type: Type of data source
            source_path: Path to data
            target_table: Target table name
            schema_file: Optional schema file
            output_dir: Optional output directory
            run_id: Optional run ID

        Returns:
            Tuple of (metadata, tables, metadata_path, tables_path)
        """
        # Create connector and load tables
        connector = self.create_connector(source_type, source_path, target_table)
        tables = connector.load_tables()

        # Load provided schema if available
        provided_schema = self.load_provided_schema(schema_file)

        # Detect schema
        metadata = self.detect_schema(tables, target_table, provided_schema)

        # Save metadata and tables
        metadata_path = self.save_metadata(metadata, output_dir, run_id)
        tables_path = self.save_tables(tables, run_id)

        return metadata, tables, metadata_path, tables_path
