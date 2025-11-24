"""Search command for querying records."""

from __future__ import annotations

import json
from pathlib import Path

import click

# Ensure modes are registered
import talk2metadata.core.modes  # noqa: F401
from talk2metadata.core.modes import (
    RecordVoter,
    get_active_mode,
    get_mode_retriever_config,
    get_registry,
)
from talk2metadata.core.modes.comparison import ModeComparator
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import (
    find_schema_file,
    get_indexes_dir,
    get_metadata_dir,
)

logger = get_logger(__name__)


def _handle_comparison_mode(
    query, top_k, index_dir, output_format, show_score, run_id, config, registry
):
    """Handle comparison mode - run search on all modes and compare."""
    click.echo(f'🔍 Comparison Mode: "{query}"')
    click.echo(f"   Top-K: {top_k}\n")

    # Get modes to compare
    comparison_config = config.get("modes.compare", {})
    modes_to_compare = comparison_config.get("modes", [])
    if not modes_to_compare:
        modes_to_compare = registry.get_all_enabled()

    if not modes_to_compare:
        click.echo("❌ No enabled modes found for comparison", err=True)
        raise click.Abort()

    click.echo(
        f"📊 Comparing {len(modes_to_compare)} mode(s): {', '.join(modes_to_compare)}\n"
    )

    # Initialize retrievers for each mode
    base_index_dir = index_dir if index_dir else get_indexes_dir(run_id, config)
    base_index_dir = Path(base_index_dir)

    retrievers = {}
    for mode_name in modes_to_compare:
        mode_info = registry.get(mode_name)
        if not mode_info or not mode_info.enabled:
            continue

        mode_index_dir = base_index_dir / mode_name
        schema_path = mode_index_dir / "schema_metadata.json"
        if not schema_path.exists():
            metadata_dir = get_metadata_dir(run_id, config)
            schema_path = find_schema_file(metadata_dir)
            if not schema_path or not Path(schema_path).exists():
                click.echo(f"⚠️  Skipping {mode_name}: index not found")
                continue

        try:
            if mode_name == "record_embedding":
                retrievers[mode_name] = RecordVoter.from_paths(
                    mode_index_dir, schema_path
                )
            else:
                click.echo(f"⚠️  Skipping {mode_name}: retriever not implemented")
        except Exception as e:
            click.echo(f"⚠️  Failed to load {mode_name}: {e}")
            continue

    if not retrievers:
        click.echo("❌ No retrievers loaded for comparison", err=True)
        raise click.Abort()

    # Run comparison
    comparator = ModeComparator(modes=list(retrievers.keys()))
    comparison_result = comparator.compare(query, top_k=top_k, retrievers=retrievers)

    # Display comparison results
    if output_format == "json":
        output = {
            "query": query,
            "top_k": top_k,
            "mode": "comparison",
            "modes_compared": list(retrievers.keys()),
            "comparison": {
                "common_results_count": len(comparison_result.common_results),
                "overlap_stats": comparison_result.overlap_stats,
                "mode_results": {
                    mode: [
                        {
                            "rank": r.rank,
                            "table": r.table,
                            "row_id": r.row_id,
                            "score": r.score,
                        }
                        for r in results
                    ]
                    for mode, results in comparison_result.mode_results.items()
                },
            },
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        click.echo("=" * 80)
        click.echo("COMPARISON RESULTS")
        click.echo("=" * 80)
        click.echo(
            f"\nCommon Results (appear in all modes): {len(comparison_result.common_results)}"
        )
        if comparison_result.common_results:
            for r in comparison_result.common_results[:top_k]:
                click.echo(f"  - {r.table}.{r.row_id} (score: {r.score:.4f})")

        click.echo("\nOverlap Statistics:")
        for mode_name, overlap in comparison_result.overlap_stats.items():
            click.echo(f"  - {mode_name}: {overlap}% overlap")

        click.echo("\nMode-specific Results:")
        for mode_name, results in comparison_result.mode_results.items():
            click.echo(f"\n  {mode_name} ({len(results)} results):")
            unique = comparison_result.unique_results.get(mode_name, [])
            click.echo(f"    - Unique: {len(unique)}")
            for r in results[:5]:
                is_unique = any(
                    ur.row_id == r.row_id and ur.table == r.table for ur in unique
                )
                marker = "★" if is_unique else " "
                click.echo(
                    f"    {marker} Rank {r.rank}: {r.table}.{r.row_id} (score: {r.score:.4f})"
                )


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
    "--per-table-top-k",
    type=int,
    default=5,
    help="Number of results per table before voting (record_embedding mode only)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Search mode (default: from config or 'record_embedding')",
)
@click.option(
    "--compare",
    is_flag=True,
    default=False,
    help="Compare results from all enabled modes",
)
@click.pass_context
def search_cmd(
    ctx,
    query,
    top_k,
    index_dir,
    output_format,
    show_score,
    per_table_top_k,
    mode,
    compare,
):
    """Search for relevant records using natural language.

    QUERY: Natural language search query

    \b
    Examples:
        # Simple search with active mode
        talk2metadata search "customers in healthcare industry"

        # Search with specific mode
        talk2metadata search "high value orders" --mode record_embedding

        # Compare all modes
        talk2metadata search "recent orders" --compare

        # JSON output
        talk2metadata search "recent orders" --format json
    """
    config = get_config()
    run_id = config.get("run_id")
    registry = get_registry()

    # 1. Determine if comparing modes
    if compare:
        _handle_comparison_mode(
            query, top_k, index_dir, output_format, show_score, run_id, config, registry
        )
        return

    # 2. Determine mode to use
    if mode:
        active_mode = mode
    else:
        active_mode = get_active_mode() or "record_embedding"

    mode_info = registry.get(active_mode)
    if not mode_info:
        click.echo(f"❌ Mode '{active_mode}' not found", err=True)
        click.echo(f"   Available modes: {', '.join(registry.get_all_enabled())}")
        raise click.Abort()

    # 3. Load index and create retriever
    if not index_dir:
        base_index_dir = get_indexes_dir(run_id, config)
        index_dir = base_index_dir / active_mode
    else:
        index_dir = Path(index_dir)
        # If base dir provided, use mode subdirectory
        if (index_dir / active_mode).exists():
            index_dir = index_dir / active_mode

    if output_format == "text":
        click.echo(f'🔍 Searching: "{query}" [Mode: {active_mode}]')
        click.echo(f"   Top-K: {top_k}\n")

    try:
        # Load retriever for this mode
        schema_path = index_dir / "schema_metadata.json"
        if not schema_path.exists():
            metadata_dir = get_metadata_dir(run_id, config)
            schema_path = find_schema_file(metadata_dir)
            if not schema_path or not Path(schema_path).exists():
                click.echo(
                    f"❌ Schema metadata not found for mode '{active_mode}'\n"
                    f"   Please run 'talk2metadata index --mode {active_mode}' first.",
                    err=True,
                )
                raise click.Abort()

        # Initialize retriever based on mode (use mode-specific config)
        mode_retriever_config = get_mode_retriever_config(active_mode)

        if active_mode == "record_embedding":
            retriever = RecordVoter.from_paths(
                index_dir,
                schema_path,
                per_table_top_k=per_table_top_k
                or mode_retriever_config.get("per_table_top_k", 5),
            )
        else:
            # Future modes would be initialized here
            click.echo(
                f"❌ Retriever for mode '{active_mode}' not implemented", err=True
            )
            raise click.Abort()

    except Exception as e:
        click.echo(f"❌ Failed to load index: {e}", err=True)
        raise click.Abort()

    # 4. Search
    try:
        results = retriever.search(query, top_k=top_k)
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)
        raise click.Abort()

    # 5. Display results
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
            # Add RecordVoter fields
            if hasattr(r, "match_count"):
                result_dict["match_count"] = r.match_count  # Vote count
                result_dict["matched_tables"] = r.matched_tables  # Voter tables
            result_dicts.append(result_dict)

        output = {
            "query": query,
            "top_k": top_k,
            "search_mode": active_mode,
            "per_table_top_k": (
                per_table_top_k if active_mode == "record_embedding" else None
            ),
            "results": result_dicts,
        }

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
                click.echo(f"Score: {result.score:.4f}")
            if hasattr(result, "match_count"):
                click.echo(f"Votes: {result.match_count}")
                click.echo(f"Voter Tables: {', '.join(result.matched_tables)}")
            click.echo(f"Table: {result.table}")
            click.echo(f"Row ID: {result.row_id}")
            click.echo("\nData:")

            for key, value in result.data.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                click.echo(f"  {key}: {value_str}")

            click.echo()

        click.echo(f"{'='*80}")
        click.echo(f"\n✅ Retrieved {len(results)} records")
