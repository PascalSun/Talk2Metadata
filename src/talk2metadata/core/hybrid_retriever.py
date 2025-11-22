"""Hybrid retrieval combining BM25 and semantic search."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from talk2metadata.core.retriever import Retriever, SearchResult
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HybridSearchResult(SearchResult):
    """Hybrid search result with component scores."""

    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    fusion_method: str = "rrf"  # rrf, weighted_sum, etc.


class BM25Index:
    """BM25 index for keyword-based search."""

    def __init__(self, texts: List[str], tokenizer=None):
        """Initialize BM25 index.

        Args:
            texts: List of text documents
            tokenizer: Optional tokenizer function (defaults to simple split)
        """
        self.texts = texts
        self.tokenizer = tokenizer or self._default_tokenizer

        # Tokenize documents
        logger.info(f"Tokenizing {len(texts)} documents for BM25")
        tokenized_docs = [self.tokenizer(text) for text in texts]

        # Build BM25 index
        logger.info("Building BM25 index")
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"BM25 index built with {len(tokenized_docs)} documents")

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        """Default tokenizer: lowercase and split on whitespace.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.lower().split()

    def search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """Search using BM25.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            Tuple of (indices, scores)
        """
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]

        return top_indices.tolist(), top_scores.tolist()

    def save(self, path: str | Path) -> None:
        """Save BM25 index to file.

        Args:
            path: Path to save the index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "texts": self.texts,
                },
                f,
            )
        logger.info(f"Saved BM25 index to {path}")

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        """Load BM25 index from file.

        Args:
            path: Path to the saved index

        Returns:
            BM25Index instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        index = cls.__new__(cls)
        index.bm25 = data["bm25"]
        index.texts = data["texts"]
        index.tokenizer = cls._default_tokenizer

        logger.info(f"Loaded BM25 index from {path}")
        return index


class HybridRetriever:
    """Hybrid retriever combining BM25 and semantic search."""

    def __init__(
        self,
        semantic_retriever: Retriever,
        bm25_index: BM25Index,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
        rrf_k: int = 60,
    ):
        """Initialize hybrid retriever.

        Args:
            semantic_retriever: Semantic retriever (FAISS)
            bm25_index: BM25 index
            alpha: Weight for combining scores (0=BM25 only, 1=semantic only)
                   Only used for weighted_sum fusion
            fusion_method: Method to combine results ('rrf' or 'weighted_sum')
            rrf_k: K parameter for Reciprocal Rank Fusion (default 60)
        """
        self.semantic_retriever = semantic_retriever
        self.bm25_index = bm25_index
        self.alpha = alpha
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

        logger.info(
            f"Initialized HybridRetriever with fusion={fusion_method}, "
            f"alpha={alpha}, rrf_k={rrf_k}"
        )

    @classmethod
    def from_paths(
        cls,
        index_path: str | Path,
        records_path: str | Path,
        bm25_path: str | Path,
        model_name: Optional[str] = None,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
    ) -> HybridRetriever:
        """Create hybrid retriever from saved files.

        Args:
            index_path: Path to FAISS index
            records_path: Path to records file
            bm25_path: Path to BM25 index
            model_name: Embedding model name
            alpha: Combination weight
            fusion_method: Fusion method

        Returns:
            HybridRetriever instance
        """
        # Load semantic retriever
        semantic_retriever = Retriever.from_paths(
            index_path, records_path, model_name=model_name
        )

        # Load BM25 index
        bm25_index = BM25Index.load(bm25_path)

        return cls(
            semantic_retriever=semantic_retriever,
            bm25_index=bm25_index,
            alpha=alpha,
            fusion_method=fusion_method,
        )

    @classmethod
    def from_config(cls, config: Optional[Dict] = None) -> HybridRetriever:
        """Create hybrid retriever from configuration.

        Args:
            config: Configuration dict

        Returns:
            HybridRetriever instance
        """
        if config is None:
            config = get_config()

        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))
        index_path = index_dir / "index.faiss"
        records_path = index_dir / "records.pkl"
        bm25_path = index_dir / "bm25.pkl"

        if not bm25_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {bm25_path}. "
                "Please run 'talk2metadata index --hybrid' first."
            )

        alpha = config.get("retrieval.hybrid.alpha", 0.5)
        fusion_method = config.get("retrieval.hybrid.fusion_method", "rrf")

        return cls.from_paths(
            index_path,
            records_path,
            bm25_path,
            alpha=alpha,
            fusion_method=fusion_method,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[HybridSearchResult]:
        """Hybrid search combining BM25 and semantic search.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of HybridSearchResult objects
        """
        logger.debug(f"Hybrid search for: '{query}' (top_k={top_k})")

        # 1. Get semantic search results
        semantic_results = self.semantic_retriever.search(query, top_k=top_k * 2)

        # 2. Get BM25 results
        bm25_indices, bm25_scores = self.bm25_index.search(query, top_k=top_k * 2)

        # 3. Fuse results
        if self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(
                semantic_results, bm25_indices, bm25_scores, top_k
            )
        elif self.fusion_method == "weighted_sum":
            fused_results = self._weighted_sum_fusion(
                semantic_results, bm25_indices, bm25_scores, top_k
            )
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        logger.debug(f"Hybrid search returned {len(fused_results)} results")
        return fused_results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: List[SearchResult],
        bm25_indices: List[int],
        bm25_scores: List[float],
        top_k: int,
    ) -> List[HybridSearchResult]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank_i)) for each ranking

        Args:
            semantic_results: Results from semantic search
            bm25_indices: Indices from BM25 search
            bm25_scores: Scores from BM25 search
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        rrf_scores = {}

        # Add semantic search scores
        for result in semantic_results:
            idx = result.row_id
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + result.rank)

        # Add BM25 scores
        for rank, (idx, score) in enumerate(zip(bm25_indices, bm25_scores), start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Create result objects
        results = []
        for rank, (idx, rrf_score) in enumerate(sorted_indices, start=1):
            record = self.semantic_retriever.records[idx]

            # Find original scores
            semantic_score = None
            for r in semantic_results:
                if r.row_id == idx:
                    semantic_score = r.score
                    break

            bm25_score = None
            if idx in bm25_indices:
                bm25_idx = bm25_indices.index(idx)
                bm25_score = bm25_scores[bm25_idx]

            results.append(
                HybridSearchResult(
                    row_id=record["row_id"],
                    table=record["table"],
                    data=record["data"],
                    score=float(rrf_score),
                    rank=rank,
                    bm25_score=bm25_score,
                    semantic_score=semantic_score,
                    fusion_method="rrf",
                )
            )

        return results

    def _weighted_sum_fusion(
        self,
        semantic_results: List[SearchResult],
        bm25_indices: List[int],
        bm25_scores: List[float],
        top_k: int,
    ) -> List[HybridSearchResult]:
        """Combine results using weighted sum.

        Score = alpha * semantic_score + (1 - alpha) * bm25_score

        Args:
            semantic_results: Results from semantic search
            bm25_indices: Indices from BM25 search
            bm25_scores: Scores from BM25 search
            top_k: Number of results to return

        Returns:
            Fused and ranked results
        """
        # Normalize scores to [0, 1]
        def normalize_scores(scores):
            if not scores or max(scores) == 0:
                return scores
            return [s / max(scores) for s in scores]

        # Get semantic scores normalized
        semantic_scores_dict = {}
        semantic_score_list = [r.score for r in semantic_results]
        normalized_semantic = normalize_scores(semantic_score_list)

        for result, norm_score in zip(semantic_results, normalized_semantic):
            # For L2 distance, lower is better, so invert
            semantic_scores_dict[result.row_id] = 1 - norm_score

        # Get BM25 scores normalized
        bm25_scores_dict = {}
        normalized_bm25 = normalize_scores(bm25_scores)
        for idx, norm_score in zip(bm25_indices, normalized_bm25):
            bm25_scores_dict[idx] = norm_score

        # Combine scores
        combined_scores = {}
        all_indices = set(semantic_scores_dict.keys()) | set(bm25_scores_dict.keys())

        for idx in all_indices:
            sem_score = semantic_scores_dict.get(idx, 0)
            bm_score = bm25_scores_dict.get(idx, 0)
            combined_scores[idx] = self.alpha * sem_score + (1 - self.alpha) * bm_score

        # Sort and take top-k
        sorted_indices = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Create result objects
        results = []
        for rank, (idx, combined_score) in enumerate(sorted_indices, start=1):
            record = self.semantic_retriever.records[idx]

            results.append(
                HybridSearchResult(
                    row_id=record["row_id"],
                    table=record["table"],
                    data=record["data"],
                    score=float(combined_score),
                    rank=rank,
                    bm25_score=bm25_scores_dict.get(idx),
                    semantic_score=1
                    - semantic_scores_dict.get(idx, 1),  # Invert back
                    fusion_method="weighted_sum",
                )
            )

        return results

    def get_stats(self) -> Dict:
        """Get hybrid retriever statistics.

        Returns:
            Dict with statistics
        """
        semantic_stats = self.semantic_retriever.get_stats()
        return {
            **semantic_stats,
            "bm25_enabled": True,
            "fusion_method": self.fusion_method,
            "alpha": self.alpha,
            "rrf_k": self.rrf_k,
        }
