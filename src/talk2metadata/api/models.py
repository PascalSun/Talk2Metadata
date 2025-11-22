"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., description="Natural language search query", min_length=1)
    top_k: int = Field(5, description="Number of results to return", ge=1, le=100)
    hybrid: bool = Field(False, description="Use hybrid search (BM25 + semantic)")
    alpha: float = Field(0.5, description="Weight for semantic vs BM25 (0=BM25 only, 1=semantic only)", ge=0.0, le=1.0)
    fusion_method: str = Field("rrf", description="Fusion method: 'rrf' or 'weighted_sum'")

    model_config = {"json_schema_extra": {"examples": [{"query": "healthcare customers with high revenue", "top_k": 5, "hybrid": True, "alpha": 0.5, "fusion_method": "rrf"}]}}


class SearchResultItem(BaseModel):
    """Single search result."""

    rank: int = Field(..., description="Result rank (1-indexed)")
    table: str = Field(..., description="Table name")
    row_id: int | str = Field(..., description="Row ID")
    score: float = Field(..., description="Combined similarity score")
    data: Dict[str, Any] = Field(..., description="Record data")
    bm25_score: Optional[float] = Field(None, description="BM25 component score (hybrid search only)")
    semantic_score: Optional[float] = Field(None, description="Semantic component score (hybrid search only)")
    fusion_method: Optional[str] = Field(None, description="Fusion method used (hybrid search only)")


class SearchResponse(BaseModel):
    """Search response model."""

    query: str = Field(..., description="Original query")
    top_k: int = Field(..., description="Requested number of results")
    search_mode: str = Field("semantic", description="Search mode: 'semantic' or 'hybrid'")
    results: List[SearchResultItem] = Field(..., description="Search results")
    total_records: int = Field(..., description="Total records in index")
    fusion_method: Optional[str] = Field(None, description="Fusion method used (hybrid only)")
    alpha: Optional[float] = Field(None, description="Alpha value used (hybrid only)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "healthcare customers",
                    "top_k": 5,
                    "results": [
                        {
                            "rank": 1,
                            "table": "orders",
                            "row_id": 1001,
                            "score": 0.234,
                            "data": {
                                "id": 1001,
                                "customer_id": 1,
                                "amount": 50000,
                            },
                        }
                    ],
                    "total_records": 20,
                }
            ]
        }
    }


class TableInfo(BaseModel):
    """Table metadata."""

    name: str
    columns: Dict[str, str]
    primary_key: Optional[str]
    row_count: int


class ForeignKeyInfo(BaseModel):
    """Foreign key relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    coverage: float


class SchemaResponse(BaseModel):
    """Schema information response."""

    target_table: str
    tables: Dict[str, TableInfo]
    foreign_keys: List[ForeignKeyInfo]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "target_table": "orders",
                    "tables": {
                        "orders": {
                            "name": "orders",
                            "columns": {"id": "int64", "customer_id": "int64"},
                            "primary_key": "id",
                            "row_count": 20,
                        }
                    },
                    "foreign_keys": [
                        {
                            "child_table": "orders",
                            "child_column": "customer_id",
                            "parent_table": "customers",
                            "parent_column": "id",
                            "coverage": 1.0,
                        }
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    index_loaded: bool = Field(..., description="Whether search index is loaded")
    total_records: Optional[int] = Field(None, description="Total records in index")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
