"""Record Voter Retriever for cross-table search with voting mechanism.

This module implements a record-level embedding strategy with voting-based result aggregation.
See class docstrings for detailed explanation of the embedding and voting mechanisms.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from talk2metadata.core.modes.record_embedding.search_result import SearchResult
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.timing import TimingContext, timed

logger = get_logger(__name__)


@dataclass
class RecordVoteSearchResult(SearchResult):
    """Search result with voting metadata.

    Attributes:
        match_count: Number of votes (matches) this target row received
        matched_tables: List of table names that voted for this row
    """

    match_count: int  # Number of votes this target row received
    matched_tables: List[str]  # Tables that voted for this row

    def __repr__(self) -> str:
        return (
            f"RecordVoteSearchResult(rank={self.rank}, table={self.table}, "
            f"row_id={self.row_id}, score={self.score:.4f}, "
            f"votes={self.match_count}, voters={self.matched_tables})"
        )


class RecordVoter:
    """Record Voter Retriever for cross-table search with voting mechanism.

    This retriever uses a "record embedding" strategy combined with a "voting" mechanism
    to aggregate search results across multiple tables.

    ## Record Embedding Strategy

    **How records are embedded:**

    1. **Per-table indexing**: Each table in the database gets its own FAISS index
    2. **Row-level embedding**: Each row in each table is converted to a text representation
       and embedded independently. The text format is:
       ```
       # Record from {table_name}
       column1: value1
       column2: value2
       ...
       ```
    3. **No denormalization**: Records are NOT denormalized with foreign key joins
       with foreign key joins. Each record stands alone in its embedding space.
    4. **Independent search**: Each table can be searched independently, allowing
       queries to match records in any table.

    **Example:**
    - Table "customers": Each customer row gets its own embedding
    - Table "orders": Each order row gets its own embedding
    - Table "products": Each product row gets its own embedding

    ## Voting Mechanism

    **How results are aggregated via voting:**

    1. **Multi-table search**: For a given query, search ALL tables independently
       and retrieve top-k results from each table (e.g., top 5 from each).

    2. **FK linking**: For each matched record, use foreign key relationships to
       find which target table rows it links to:
       - If a customer record matches, find all orders linked to that customer
       - If a product record matches, find all orders containing that product

    3. **Voting**: Each matched record "votes" for the target table rows it links to:
       - Customer C1 matches → votes for Order O1, O2, O3 (all orders by C1)
       - Product P1 matches → votes for Order O1, O5, O7 (all orders with P1)
       - Order O1 matches → votes for itself

    4. **Vote counting**: Count how many votes each target table row receives:
       - Order O1: 3 votes (from Customer C1, Product P1, and itself)
       - Order O2: 1 vote (from Customer C1)
       - Order O3: 1 vote (from Customer C1)

    5. **Ranking**: Rank target table rows by:
       - Primary: Number of votes (descending)
       - Secondary: Best similarity score (ascending, lower is better)

    6. **Result**: Return top-k target table rows with highest vote counts.

    **Example voting scenario:**

    Query: "healthcare customers"
    - Matches Customer C1 (healthcare industry) → votes for Orders O1, O2
    - Matches Customer C2 (healthcare industry) → votes for Orders O3, O4
    - Matches Order O1 (contains healthcare-related products) → votes for itself

    Final results:
    - Order O1: 2 votes (from C1 and itself) → Rank 1
    - Order O2: 1 vote (from C1) → Rank 2
    - Order O3: 1 vote (from C2) → Rank 3
    - Order O4: 1 vote (from C2) → Rank 4

    This voting mechanism ensures that target table rows linked to multiple
    relevant records across different tables rank higher, providing better
    cross-table search results.
    """

    def __init__(
        self,
        table_indices: Dict[str, Tuple[faiss.IndexFlatL2, List[Dict]]],
        schema_metadata: SchemaMetadata,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize: bool = True,
        per_table_top_k: int = 5,
    ):
        """Initialize Record Voter retriever.

        Args:
            table_indices: Dict mapping table_name -> (FAISS index, records)
                Each table has its own index built with record embeddings
            schema_metadata: Schema metadata with FK relationships
            model_name: Sentence-transformers model name (must match indexer)
            device: Device for encoding queries
            normalize: Whether to normalize query embeddings
            per_table_top_k: Number of results to retrieve per table before voting
                (each result can vote for multiple target rows via FK links)
        """
        self.table_indices = table_indices
        self.schema_metadata = schema_metadata
        self.target_table = schema_metadata.target_table
        self.per_table_top_k = per_table_top_k

        # Use provided parameters or defaults (mode-specific config should be passed via kwargs)
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.device = device
        self.normalize = normalize if normalize is not None else True

        logger.info(f"Loading embedding model for queries: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Build FK lookup maps for efficient reverse lookup
        self._build_fk_lookup_maps()

    def _build_fk_lookup_maps(self) -> None:
        """Build maps for efficient FK reverse lookup.

        Creates maps to quickly find target table rows from other table rows.
        """
        # Map: (source_table, source_row_id) -> List[target_table_row_id]
        self.fk_to_target: Dict[Tuple[str, int | str], List[int | str]] = defaultdict(
            list
        )

        # Process all foreign keys to build lookup maps
        for fk in self.schema_metadata.foreign_keys:
            # Case 1: Other table -> Target table (child -> parent)
            # If we find a row in child table, we can link to parent (target) via FK
            if (
                fk.child_table != self.target_table
                and fk.parent_table == self.target_table
            ):
                child_table = fk.child_table
                if child_table not in self.table_indices:
                    continue

                _, child_records = self.table_indices[child_table]
                target_pk = fk.parent_column

                # Get target table records for lookup
                _, target_records = self.table_indices[self.target_table]
                target_pk_map = {
                    r["data"].get(target_pk): r["row_id"]
                    for r in target_records
                    if r["data"].get(target_pk) is not None
                }

                for child_record in child_records:
                    child_row_id = child_record["row_id"]
                    fk_value = child_record["data"].get(fk.child_column)

                    if fk_value is not None and fk_value in target_pk_map:
                        target_row_id = target_pk_map[fk_value]
                        key = (child_table, child_row_id)
                        self.fk_to_target[key].append(target_row_id)

            # Case 2: Target table -> Other table (parent -> child)
            # If we find a row in other table, we can link back to target via reverse FK
            elif (
                fk.parent_table != self.target_table
                and fk.child_table == self.target_table
            ):
                parent_table = fk.parent_table
                if parent_table not in self.table_indices:
                    continue

                _, parent_records = self.table_indices[parent_table]
                parent_pk = fk.parent_column

                # Get target table records
                _, target_records = self.table_indices[self.target_table]
                target_fk_map = defaultdict(list)
                for target_record in target_records:
                    fk_value = target_record["data"].get(fk.child_column)
                    if fk_value is not None:
                        target_fk_map[fk_value].append(target_record["row_id"])

                for parent_record in parent_records:
                    parent_row_id = parent_record["row_id"]
                    parent_pk_value = parent_record["data"].get(parent_pk)

                    if parent_pk_value is not None and parent_pk_value in target_fk_map:
                        key = (parent_table, parent_row_id)
                        self.fk_to_target[key].extend(target_fk_map[parent_pk_value])

        logger.info(
            f"Built FK lookup maps: {len(self.fk_to_target)} source->target mappings"
        )

    @classmethod
    @timed("record_voter.load", log_level="info")
    def from_paths(
        cls,
        base_dir: str | Path,
        schema_metadata_path: str | Path,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        per_table_top_k: int = 5,
    ) -> RecordVoter:
        """Create retriever from saved multi-table indices.

        Args:
            base_dir: Base directory containing table subdirectories
            schema_metadata_path: Path to schema metadata JSON
            model_name: Embedding model name
            device: Device for queries
            per_table_top_k: Number of results per table before voting

        Returns:
            RecordVoter instance
        """
        from talk2metadata.core.modes.record_embedding.indexer import Indexer
        from talk2metadata.core.schema.schema import SchemaMetadata

        with TimingContext("multi_table_index_load"):
            table_indices = Indexer.load_multi_table_index(base_dir)

        with TimingContext("schema_metadata_load"):
            schema_metadata = SchemaMetadata.load(schema_metadata_path)

        return cls(
            table_indices=table_indices,
            schema_metadata=schema_metadata,
            model_name=model_name,
            device=device,
            per_table_top_k=per_table_top_k,
        )

    @timed("record_voter.search")
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RecordVoteSearchResult]:
        """Search across all tables and aggregate results via voting mechanism.

        This method implements the voting mechanism:
        1. Search all tables independently
        2. Link matched records to target table rows via FK relationships
        3. Count votes for each target row
        4. Return top-k rows ranked by vote count

        Args:
            query: Natural language query
            top_k: Number of target table results to return

        Returns:
            List of RecordVoteSearchResult objects, ranked by vote count

        Example:
            >>> voter = RecordVoter.from_paths(...)
            >>> results = voter.search("customers in healthcare", top_k=5)
            >>> for result in results:
            ...     print(f"Rank {result.rank}: {result.data} ({result.match_count} votes)")
        """
        logger.debug(
            f"Record voter search for: '{query}' (top_k={top_k}, per_table_top_k={self.per_table_top_k})"
        )

        # 1. Encode query
        with TimingContext("query_encoding"):
            query_embedding = self._encode_query(query)

        # 2. Search each table
        with TimingContext("multi_table_search"):
            all_matches = self._search_all_tables(query_embedding)

        # 3. Aggregate results via voting mechanism
        with TimingContext("voting"):
            aggregated = self._vote_and_aggregate(all_matches, top_k)

        logger.debug(f"Record voter search returned {len(aggregated)} results")
        return aggregated

    def _search_all_tables(
        self, query_embedding: np.ndarray
    ) -> Dict[str, List[Tuple[int, float]]]:
        """Search all tables and return matches.

        Args:
            query_embedding: Query embedding vector

        Returns:
            Dict mapping table_name -> List of (record_index, distance) tuples
        """
        all_matches = {}

        for table_name, (index, records) in self.table_indices.items():
            distances, indices = index.search(query_embedding, self.per_table_top_k)

            matches = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    break
                matches.append((idx, float(distance)))

            all_matches[table_name] = matches
            logger.debug(f"Found {len(matches)} matches in {table_name}")

        return all_matches

    def _vote_and_aggregate(
        self,
        all_matches: Dict[str, List[Tuple[int, float]]],
        top_k: int,
    ) -> List[RecordVoteSearchResult]:
        """Aggregate search results via voting mechanism.

        Each matched record votes for target table rows it links to via FK relationships.
        Results are ranked by vote count (number of votes received).

        Args:
            all_matches: Dict mapping table_name -> List of (record_index, distance)
            top_k: Number of results to return

        Returns:
            List of RecordVoteSearchResult objects ranked by vote count
        """
        # Count votes: how many times each target table row is voted for
        target_vote_counts: Counter[int | str] = Counter()
        target_scores: Dict[int | str, List[float]] = defaultdict(list)
        target_voters: Dict[int | str, List[str]] = defaultdict(list)

        # Process matches from each table
        for table_name, matches in all_matches.items():
            _, records = self.table_indices[table_name]

            for record_idx, distance in matches:
                record = records[record_idx]
                source_row_id = record["row_id"]

                # Find target table rows linked via FK (this record votes for these rows)
                linked_target_rows = self._find_linked_target_rows(
                    table_name, source_row_id, record["data"]
                )

                # Each linked target row receives a vote
                for target_row_id in linked_target_rows:
                    target_vote_counts[target_row_id] += 1  # Vote count
                    target_scores[target_row_id].append(distance)  # Track scores
                    if table_name not in target_voters[target_row_id]:
                        target_voters[target_row_id].append(table_name)  # Track voters

        # Get target table records
        target_index, target_records = self.table_indices[self.target_table]
        target_records_dict = {r["row_id"]: r for r in target_records}

        # Sort by vote count (descending), then by best score (ascending)
        # Rows with more votes rank higher
        sorted_targets = sorted(
            target_vote_counts.items(),
            key=lambda x: (-x[1], min(target_scores.get(x[0], [float("inf")]))),
        )[:top_k]

        # Build results
        results = []
        for rank, (target_row_id, vote_count) in enumerate(sorted_targets, start=1):
            if target_row_id not in target_records_dict:
                continue

            target_record = target_records_dict[target_row_id]
            # Use best (lowest) score from all votes
            best_score = min(target_scores[target_row_id])

            results.append(
                RecordVoteSearchResult(
                    row_id=target_row_id,
                    table=self.target_table,
                    data=target_record["data"],
                    score=best_score,
                    rank=rank,
                    match_count=vote_count,  # Number of votes received
                    matched_tables=target_voters[target_row_id],  # Which tables voted
                )
            )

        return results

    def _find_linked_target_rows(
        self, source_table: str, source_row_id: int | str, source_data: Dict
    ) -> List[int | str]:
        """Find target table rows linked via foreign keys.

        Args:
            source_table: Source table name
            source_row_id: Source row ID
            source_data: Source row data

        Returns:
            List of target table row IDs
        """
        linked_rows = []

        # If source table is target table, return itself
        if source_table == self.target_table:
            return [source_row_id]

        # Check direct FK link via lookup map (fast path)
        key = (source_table, source_row_id)
        if key in self.fk_to_target:
            linked_rows.extend(self.fk_to_target[key])
            return list(set(linked_rows))  # Return early if found in cache

        # Fallback: check FKs dynamically (shouldn't happen if maps are built correctly)
        for fk in self.schema_metadata.foreign_keys:
            if fk.child_table == source_table and fk.parent_table == self.target_table:
                # Source table (child) -> Target table (parent)
                fk_value = source_data.get(fk.child_column)
                if fk_value is not None:
                    _, target_records = self.table_indices[self.target_table]
                    for target_record in target_records:
                        if target_record["data"].get(fk.parent_column) == fk_value:
                            linked_rows.append(target_record["row_id"])

            elif (
                fk.parent_table == source_table and fk.child_table == self.target_table
            ):
                # Source table (parent) -> Target table (child)
                # Find target rows where FK column matches source PK
                source_table_meta = self.schema_metadata.tables.get(source_table)
                if source_table_meta and source_table_meta.primary_key:
                    source_pk_value = source_data.get(source_table_meta.primary_key)
                    if source_pk_value is not None:
                        _, target_records = self.table_indices[self.target_table]
                        for target_record in target_records:
                            if (
                                target_record["data"].get(fk.child_column)
                                == source_pk_value
                            ):
                                linked_rows.append(target_record["row_id"])

        return list(set(linked_rows))  # Remove duplicates

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

    def get_stats(self) -> Dict:
        """Get retriever statistics.

        Returns:
            Dict with statistics
        """
        total_records = sum(len(records) for _, records in self.table_indices.values())
        return {
            "total_tables": len(self.table_indices),
            "total_records": total_records,
            "target_table": self.target_table,
            "per_table_top_k": self.per_table_top_k,
            "model": self.model_name,
            "fk_mappings": len(self.fk_to_target),
        }

    def __repr__(self) -> str:
        return (
            f"RecordVoter(tables={len(self.table_indices)}, "
            f"target={self.target_table}, model={self.model_name})"
        )
