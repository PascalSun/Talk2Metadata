"""Retrieval module for searching indexed records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import TimingContext, timed

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Search result for a single record."""

    row_id: int | str
    table: str
    data: Dict
    score: float  # Similarity score (lower is better for L2 distance)
    rank: int  # Result rank (1-indexed)

    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, table={self.table}, row_id={self.row_id}, score={self.score:.4f})"


class Retriever:
    """Retriever for searching indexed records."""

    def __init__(
        self,
        index: faiss.IndexFlatL2,
        records: List[Dict],
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """Initialize retriever.

        Args:
            index: FAISS index
            records: List of record metadata
            model_name: Sentence-transformers model name (must match indexer)
            device: Device for encoding queries
            normalize: Whether to normalize query embeddings
        """
        self.index = index
        self.records = records

        config = get_config()
        self.model_name = model_name or config.get(
            "embedding.model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = device or config.get("embedding.device")
        self.normalize = (
            normalize
            if normalize is not None
            else config.get("embedding.normalize", True)
        )

        logger.info(f"Loading embedding model for queries: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    @classmethod
    @timed("retriever.load", log_level="info")
    def from_paths(
        cls,
        index_path: str | Path,
        records_path: str | Path,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Retriever:
        """Create retriever from saved index and records.

        Args:
            index_path: Path to FAISS index
            records_path: Path to records file
            model_name: Embedding model name
            device: Device for queries

        Returns:
            Retriever instance

        Example:
            >>> retriever = Retriever.from_paths(
            ...     "data/indexes/index.faiss",
            ...     "data/indexes/records.pkl"
            ... )
        """
        from talk2metadata.core.index.indexer import Indexer

        with TimingContext("index_load"):
            index, records = Indexer.load_index(index_path, records_path)

        return cls(
            index=index,
            records=records,
            model_name=model_name,
            device=device,
        )

    @classmethod
    def from_config(cls, config: Optional[Dict] = None) -> Retriever:
        """Create retriever from configuration.

        Args:
            config: Configuration dict (uses global config if None)

        Returns:
            Retriever instance
        """
        if config is None:
            config = get_config()

        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))
        index_path = index_dir / "index.faiss"
        records_path = index_dir / "records.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                "Please run 'talk2metadata index' first."
            )

        return cls.from_paths(index_path, records_path)

    @timed("retriever.search")
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Search for relevant records.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of SearchResult objects, ranked by relevance

        Example:
            >>> results = retriever.search("customers in healthcare industry", top_k=5)
            >>> for result in results:
            ...     print(f"Rank {result.rank}: {result.data}")
        """
        logger.debug(f"Searching for: '{query}' (top_k={top_k})")

        # 1. Encode query
        with TimingContext("query_encoding"):
            query_embedding = self._encode_query(query)

        # 2. Search FAISS index
        with TimingContext("faiss_search"):
            distances, indices = self.index.search(query_embedding, top_k)

        # 3. Format results
        with TimingContext("result_formatting"):
            results = []
            for rank, (distance, idx) in enumerate(
                zip(distances[0], indices[0]), start=1
            ):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    break

                record = self.records[idx]
                results.append(
                    SearchResult(
                        row_id=record["row_id"],
                        table=record["table"],
                        data=record["data"],
                        score=float(distance),
                        rank=rank,
                    )
                )

        logger.debug(f"Found {len(results)} results")
        return results

    @timed("retriever.search_batch")
    def search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List[List[SearchResult]]:
        """Search multiple queries in batch.

        Args:
            queries: List of queries
            top_k: Number of results per query

        Returns:
            List of result lists (one per query)
        """
        logger.info(f"Batch searching {len(queries)} queries")

        # Encode all queries
        with TimingContext("batch_query_encoding"):
            query_embeddings = self._encode_queries(queries)

        # Search
        with TimingContext("batch_faiss_search"):
            distances, indices = self.index.search(query_embeddings, top_k)

        # Format results
        with TimingContext("batch_result_formatting"):
            all_results = []
            for query_distances, query_indices in zip(distances, indices):
                results = []
                for rank, (distance, idx) in enumerate(
                    zip(query_distances, query_indices), start=1
                ):
                    if idx == -1:
                        break

                    record = self.records[idx]
                    results.append(
                        SearchResult(
                            row_id=record["row_id"],
                            table=record["table"],
                            data=record["data"],
                            score=float(distance),
                            rank=rank,
                        )
                    )
                all_results.append(results)

        return all_results

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to embedding.

        Args:
            query: Query string

        Returns:
            Query embedding (1 x D)
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embedding.astype("float32")

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode multiple queries to embeddings.

        Args:
            queries: List of query strings

        Returns:
            Query embeddings (N x D)
        """
        embeddings = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )

        return embeddings.astype("float32")

    def get_record_by_id(self, row_id: int | str) -> Optional[Dict]:
        """Get a record by its row ID.

        Args:
            row_id: Row ID to retrieve

        Returns:
            Record dict or None if not found
        """
        for record in self.records:
            if record["row_id"] == row_id:
                return record["data"]
        return None

    def get_stats(self) -> Dict:
        """Get retriever statistics.

        Returns:
            Dict with statistics
        """
        return {
            "total_records": len(self.records),
            "index_size": self.index.ntotal,
            "embedding_dimension": self.index.d,
            "model": self.model_name,
        }

    def __repr__(self) -> str:
        return f"Retriever(records={len(self.records)}, model={self.model_name})"
