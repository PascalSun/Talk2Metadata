"""Indexing module for generating embeddings and FAISS index."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


class Indexer:
    """Indexer for creating searchable embeddings from tables."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """Initialize indexer.

        Args:
            model_name: Sentence-transformers model name
            device: Device ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
        """
        # Use provided parameters or defaults (mode-specific config should be passed via kwargs)
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.batch_size = batch_size or 32
        self.normalize = normalize if normalize is not None else True

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, embedding dimension: {self.embedding_dim}")

    def build_index(
        self,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
    ) -> Dict[str, Tuple[faiss.IndexFlatL2, List[Dict]]]:
        """Build FAISS index for all tables using record embedding mode.

        This method builds multi-table indices where each table gets its own index
        with record-level embeddings. This enables cross-table search via RecordVoter.

        Args:
            tables: Dict of table_name -> DataFrame
            schema_metadata: Schema metadata with FK relationships

        Returns:
            Dict mapping table_name -> (FAISS index, list of record metadata)

        Example:
            >>> indexer = Indexer()
            >>> table_indices = indexer.build_index(tables, schema_metadata)
            >>> customers_index, customers_records = table_indices["customers"]
        """
        return self.build_multi_table_index(tables, schema_metadata)

    def build_multi_table_index(
        self,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
    ) -> Dict[str, Tuple[faiss.IndexFlatL2, List[Dict]]]:
        """Build FAISS index for all tables (record embedding mode).

        Each table gets its own index with record-level embeddings. This enables
        cross-table search where records from any table can vote for target table rows
        via foreign key relationships (used by RecordVoter).

        **Record Embedding Strategy:**
        - Each row in each table is embedded independently
        - No denormalization or FK joins during indexing
        - Simple text format: "# Record from {table}\ncolumn1: value1\n..."
        - Each table maintains its own FAISS index

        Args:
            tables: Dict of table_name -> DataFrame
            schema_metadata: Schema metadata with FK relationships

        Returns:
            Dict mapping table_name -> (FAISS index, list of record metadata)

        Example:
            >>> indexer = Indexer()
            >>> table_indices = indexer.build_multi_table_index(tables, schema_metadata)
            >>> customers_index, customers_records = table_indices["customers"]
        """
        logger.info("Building multi-table index (record embedding mode)")

        table_indices = {}

        for table_name, df in tables.items():
            logger.info(f"Indexing table: {table_name} ({len(df)} rows)")

            # Create texts for each row in this table
            texts, records = self._create_table_texts(
                df, table_name, tables, schema_metadata
            )

            logger.info(f"Generated {len(texts)} texts for {table_name}")

            # Generate embeddings
            embeddings = self._encode_texts(texts)

            # Build FAISS index
            index = self._build_faiss_index(embeddings)

            table_indices[table_name] = (index, records)
            logger.info(f"Index built for {table_name}: {index.ntotal} vectors")

        logger.info(
            f"Multi-table index built successfully: {len(table_indices)} tables"
        )
        return table_indices

    def _create_table_texts(
        self,
        df: pd.DataFrame,
        table_name: str,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
    ) -> Tuple[List[str], List[Dict]]:
        """Create text representations for each row in a table.

        Args:
            df: DataFrame for the table
            table_name: Name of the table
            tables: All tables
            schema_metadata: Schema metadata

        Returns:
            Tuple of (texts, record_metadata)
        """
        texts = []
        records = []

        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Creating texts for {table_name}"
        ):
            text = self._row_to_simple_text(row, table_name)
            texts.append(text)

            records.append(
                {
                    "row_id": idx,
                    "table": table_name,
                    "data": row.to_dict(),
                }
            )

        return texts, records

    def _row_to_simple_text(self, row: pd.Series, table_name: str) -> str:
        """Convert a row to simple text representation (without FK joins).

        Args:
            row: Row from table
            table_name: Table name

        Returns:
            Text representation
        """
        parts = [f"# Record from {table_name}"]

        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}: {val}")

        return "\n".join(parts)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (N x D)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embeddings.astype("float32")

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings (N x D)

        Returns:
            FAISS IndexFlatL2
        """
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index

    def save_multi_table_index(
        self,
        table_indices: Dict[str, Tuple[faiss.IndexFlatL2, List[Dict]]],
        base_dir: str | Path,
    ) -> None:
        """Save multi-table FAISS indices.

        Args:
            table_indices: Dict mapping table_name -> (index, records)
            base_dir: Base directory to save indices (each table gets its own subdirectory)
        """
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        for table_name, (index, records) in table_indices.items():
            table_dir = base_dir / table_name
            table_dir.mkdir(parents=True, exist_ok=True)

            index_path = table_dir / "index.faiss"
            records_path = table_dir / "records.pkl"

            faiss.write_index(index, str(index_path))
            with open(records_path, "wb") as f:
                pickle.dump(records, f)

            logger.info(
                f"Saved index for {table_name}: {index.ntotal} vectors to {table_dir}"
            )

    @staticmethod
    def load_multi_table_index(
        base_dir: str | Path,
    ) -> Dict[str, Tuple[faiss.IndexFlatL2, List[Dict]]]:
        """Load multi-table FAISS indices.

        Args:
            base_dir: Base directory containing table subdirectories

        Returns:
            Dict mapping table_name -> (index, records)
        """
        base_dir = Path(base_dir)
        table_indices = {}

        for table_dir in base_dir.iterdir():
            if not table_dir.is_dir():
                continue

            table_name = table_dir.name
            index_path = table_dir / "index.faiss"
            records_path = table_dir / "records.pkl"

            if not index_path.exists() or not records_path.exists():
                logger.warning(
                    f"Skipping {table_name}: index or records file not found"
                )
                continue

            index = faiss.read_index(str(index_path))
            with open(records_path, "rb") as f:
                records = pickle.load(f)

            table_indices[table_name] = (index, records)
            logger.info(
                f"Loaded index for {table_name}: {index.ntotal} vectors from {table_dir}"
            )

        return table_indices
