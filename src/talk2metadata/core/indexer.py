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

from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
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
        config = get_config()
        self.model_name = model_name or config.get(
            "embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device or config.get("embedding.device")
        self.batch_size = batch_size or config.get("embedding.batch_size", 32)
        self.normalize = (
            normalize
            if normalize is not None
            else config.get("embedding.normalize", True)
        )

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded, embedding dimension: {self.embedding_dim}")

    def build_index(
        self,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
        return_texts: bool = False,
    ) -> (
        Tuple[faiss.IndexFlatL2, List[Dict]]
        | Tuple[faiss.IndexFlatL2, List[Dict], List[str]]
    ):
        """Build FAISS index from tables.

        Args:
            tables: Dict of table_name -> DataFrame
            schema_metadata: Schema metadata with FK relationships
            return_texts: Whether to return the denormalized texts

        Returns:
            Tuple of (FAISS index, list of record metadata) or
            Tuple of (FAISS index, list of record metadata, texts) if return_texts=True

        Example:
            >>> indexer = Indexer()
            >>> index, records = indexer.build_index(tables, schema_metadata)
            >>> index, records, texts = indexer.build_index(tables, schema_metadata, return_texts=True)
        """
        logger.info("Building index from tables")

        # 1. Generate denormalized text for target table
        target_table = schema_metadata.target_table
        if target_table not in tables:
            raise ValueError(f"Target table {target_table} not found")

        target_df = tables[target_table]
        logger.info(f"Target table: {target_table} ({len(target_df)} rows)")

        texts, records = self._create_denormalized_texts(
            target_df, target_table, tables, schema_metadata
        )

        logger.info(f"Generated {len(texts)} searchable texts")

        # 2. Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self._encode_texts(texts)

        # 3. Build FAISS index
        logger.info("Building FAISS index...")
        index = self._build_faiss_index(embeddings)

        logger.info(f"Index built successfully: {index.ntotal} vectors")

        if return_texts:
            return index, records, texts
        return index, records

    def _create_denormalized_texts(
        self,
        target_df: pd.DataFrame,
        target_table: str,
        tables: Dict[str, pd.DataFrame],
        schema_metadata: SchemaMetadata,
    ) -> Tuple[List[str], List[Dict]]:
        """Create denormalized text representations for each row.

        Args:
            target_df: Target table DataFrame
            target_table: Target table name
            tables: All tables
            schema_metadata: Schema metadata

        Returns:
            Tuple of (texts, record_metadata)
        """
        texts = []
        records = []

        # Get foreign keys for target table
        fks = schema_metadata.get_foreign_keys_for_table(
            target_table, direction="outgoing"
        )

        logger.info(f"Found {len(fks)} foreign keys to join")

        for idx, row in tqdm(
            target_df.iterrows(), total=len(target_df), desc="Creating texts"
        ):
            text = self._row_to_text(row, target_table, tables, fks, schema_metadata)
            texts.append(text)

            # Store metadata for retrieval
            records.append(
                {
                    "row_id": idx,
                    "table": target_table,
                    "data": row.to_dict(),
                }
            )

        return texts, records

    def _row_to_text(
        self,
        row: pd.Series,
        table_name: str,
        tables: Dict[str, pd.DataFrame],
        fks: List,
        schema_metadata: SchemaMetadata,
    ) -> str:
        """Convert a row to denormalized searchable text.

        Args:
            row: Row from target table
            table_name: Table name
            tables: All tables
            fks: Foreign keys for this table
            schema_metadata: Schema metadata

        Returns:
            Denormalized text representation
        """
        parts = [f"# Record from {table_name}"]

        # Add target table columns
        for col, val in row.items():
            if pd.notna(val):
                parts.append(f"{col}: {val}")

        # Add related table data via foreign keys
        for fk in fks:
            if fk.child_column not in row:
                continue

            fk_value = row[fk.child_column]
            if pd.isna(fk_value):
                continue

            parent_table = fk.parent_table
            parent_pk = fk.parent_column

            if parent_table not in tables:
                continue

            parent_df = tables[parent_table]

            # Find matching parent row
            matching_rows = parent_df[parent_df[parent_pk] == fk_value]

            if len(matching_rows) > 0:
                parent_row = matching_rows.iloc[0]
                parts.append(f"\n## Related {parent_table}")

                for col, val in parent_row.items():
                    if pd.notna(val) and col != parent_pk:  # Skip PK duplication
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

    def save_index(
        self,
        index: faiss.IndexFlatL2,
        records: List[Dict],
        index_path: str | Path,
        records_path: str | Path,
    ) -> None:
        """Save FAISS index and record metadata.

        Args:
            index: FAISS index
            records: List of record metadata
            index_path: Path to save index
            records_path: Path to save records
        """
        index_path = Path(index_path)
        records_path = Path(records_path)

        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save records
        with open(records_path, "wb") as f:
            pickle.dump(records, f)
        logger.info(f"Saved {len(records)} records to {records_path}")

    @staticmethod
    def load_index(
        index_path: str | Path,
        records_path: str | Path,
    ) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
        """Load FAISS index and record metadata.

        Args:
            index_path: Path to FAISS index
            records_path: Path to records file

        Returns:
            Tuple of (FAISS index, records)
        """
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index from {index_path} ({index.ntotal} vectors)")

        # Load records
        with open(records_path, "rb") as f:
            records = pickle.load(f)
        logger.info(f"Loaded {len(records)} records from {records_path}")

        return index, records
