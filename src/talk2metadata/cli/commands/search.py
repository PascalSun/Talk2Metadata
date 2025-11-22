"""Search command for querying records."""

from __future__ import annotations

import json
from pathlib import Path

import click

from talk2metadata.core.retriever import Retriever
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.command(name="search")
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--index",
    "-i",
    "index_dir",
    type=click.Path(exists=True),
    help="Path to index directory (default: data/indexes)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
)
@click.option(
    "--show-score",
    is_flag=True,
    help="Show similarity scores",
)
@click.option(
    "--hybrid",
    is_flag=True,
    default=False,
    help="Use hybrid search (BM25 + semantic) for better results",
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    help="Weight for semantic vs BM25 (0=BM25 only, 1=semantic only)",
)
@click.option(
    "--fusion",
    type=click.Choice(["rrf", "weighted_sum"], case_sensitive=False),
    default="rrf",
    help="Fusion method for combining results",
)
@click.pass_context
def search_cmd(ctx, query, top_k, index_dir, output_format, show_score, hybrid, alpha, fusion):
    """Search for relevant records using natural language.

    QUERY: Natural language search query

    \b
    Examples:
        # Simple search
        talk2metadata search "customers in healthcare industry"

        # Get top 10 results
        talk2metadata search "high value orders" --top-k 10

        # JSON output
        talk2metadata search "recent orders" --format json

        # Show similarity scores
        talk2metadata search "VIP customers" --show-score
    """
    config = get_config()

    # 1. Load index and create retriever
    if not index_dir:
        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))
    else:
        index_dir = Path(index_dir)

    index_path = index_dir / "index.faiss"
    records_path = index_dir / "records.pkl"

    if not index_path.exists():
        click.echo(
            f"❌ Index not found at {index_path}\n"
            "   Please run 'talk2metadata index' first.",
            err=True,
        )
        raise click.Abort()

    if output_format == "text":
        mode_str = "hybrid (BM25 + semantic)" if hybrid else "semantic"
        click.echo(f"🔍 Searching: \"{query}\" [{mode_str}]")
        click.echo(f"   Top-K: {top_k}\n")

    try:
        if hybrid:
            # Use hybrid retriever
            from talk2metadata.core.hybrid_retriever import HybridRetriever

            bm25_path = index_dir / "bm25.pkl"
            if not bm25_path.exists():
                click.echo(
                    f"❌ BM25 index not found at {bm25_path}\n"
                    "   Please run 'talk2metadata index --hybrid' first.",
                    err=True,
                )
                raise click.Abort()

            retriever = HybridRetriever.from_paths(
                index_path, records_path, bm25_path, alpha=alpha, fusion_method=fusion
            )
        else:
            # Use semantic retriever only
            retriever = Retriever.from_paths(index_path, records_path)
    except Exception as e:
        click.echo(f"❌ Failed to load index: {e}", err=True)
        raise click.Abort()

    # 2. Search
    try:
        results = retriever.search(query, top_k=top_k)
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)
        raise click.Abort()

    # 3. Display results
    if output_format == "json":
        # JSON output
        result_dicts = []
        for r in results:
            result_dict = {
                "rank": r.rank,
                "table": r.table,
                "row_id": r.row_id,
                "score": r.score,
                "data": r.data,
            }
            # Add hybrid-specific fields if available
            if hybrid and hasattr(r, "bm25_score"):
                result_dict["bm25_score"] = r.bm25_score
                result_dict["semantic_score"] = r.semantic_score
                result_dict["fusion_method"] = r.fusion_method
            result_dicts.append(result_dict)

        output = {
            "query": query,
            "top_k": top_k,
            "search_mode": "hybrid" if hybrid else "semantic",
            "results": result_dicts,
        }
        if hybrid:
            output["fusion_method"] = fusion
            output["alpha"] = alpha

        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        if not results:
            click.echo("❌ No results found")
            return

        click.echo(f"Found {len(results)} results:\n")

        for result in results:
            click.echo(f"{'='*80}")
            click.echo(f"Rank #{result.rank}")
            if show_score:
                click.echo(f"Combined Score: {result.score:.4f}")
                if hybrid and hasattr(result, "bm25_score"):
                    if result.bm25_score is not None:
                        click.echo(f"  BM25 Score: {result.bm25_score:.4f}")
                    if result.semantic_score is not None:
                        click.echo(f"  Semantic Score: {result.semantic_score:.4f}")
            click.echo(f"Table: {result.table}")
            click.echo(f"Row ID: {result.row_id}")
            click.echo(f"\nData:")

            for key, value in result.data.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                click.echo(f"  {key}: {value_str}")

            click.echo()

        click.echo(f"{'='*80}")
        click.echo(f"\n✅ Retrieved {len(results)} records")
