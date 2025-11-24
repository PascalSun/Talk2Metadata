"""Search management commands - index and retrieve."""

from __future__ import annotations

import json
from pathlib import Path

import click

# Ensure modes are registered
import talk2metadata.core.modes  # noqa: F401
from talk2metadata.cli.decorators import handle_errors, with_run_id
from talk2metadata.cli.handlers import IndexHandler, SearchHandler
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.core.modes import get_active_mode
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.group(name="search")
def search_group():
    """Search management commands for indexing and retrieval.

    This command group provides tools for building search indexes and
    retrieving records using natural language queries.
    """
    pass


@search_group.command(name="index")
@click.option(
    "--metadata",
    "-m",
    "metadata_path",
    type=click.Path(exists=True),
    help="Path to schema metadata JSON (default: data/metadata/schema.json)",
)
@click.option(
    "--tables",
    "-t",
    "tables_path",
    type=click.Path(exists=True),
    help="Path to tables pickle file (default: data/processed/tables.pkl)",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    help="Output directory for index (default: data/indexes)",
)
@click.option(
    "--model",
    "model_name",
    help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for embedding generation (default: from config)",
)
@click.option(
    "--mode",
    type=str,
    default=None,
    help="Indexing mode (default: from config or 'record_embedding')",
)
@click.option(
    "--all-modes",
    is_flag=True,
    default=False,
    help="Build index for all enabled modes",
)
@with_run_id
@handle_errors
@click.pass_context
def index_cmd(
    ctx,
    metadata_path,
    tables_path,
    output_dir,
    model_name,
    batch_size,
    mode,
    all_modes,
    run_id,
):
    """Build search index from ingested data.

    This command builds indexes using the specified mode (or active mode from config).
    Use --all-modes to build indexes for all enabled modes.

    \b
    Examples:
        # Build index with active mode (from config)
        talk2metadata search index

        # Build index with specific mode
        talk2metadata search index --mode record_embedding

        # Build indexes for all enabled modes
        talk2metadata search index --all-modes

        # Specify custom paths
        talk2metadata search index --metadata schema.json --tables tables.pkl

        # Use specific run ID
        talk2metadata search index --run-id my_run
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Read output_dir from config if not provided via CLI
    if output_dir is None:
        output_dir_str = config.get("data.indexes_dir")
        output_dir = Path(output_dir_str) if output_dir_str else None

    # Initialize handler
    handler = IndexHandler(config)

    # 1. Load schema metadata (reads from config if metadata_path is None)
    loader = CLIDataLoader(config)
    out.section("📄 Loading schema metadata...")
    schema_metadata = loader.load_schema(schema_file=metadata_path, run_id=run_id)

    # 2. Load tables (reads from config if tables_path is None)
    out.section("📥 Loading tables...")
    try:
        tables = handler.load_tables_from_pickle(
            tables_path=Path(tables_path) if tables_path else None,
            run_id=run_id,
        )
        out.success(f"Loaded {len(tables)} tables")
    except FileNotFoundError as e:
        out.error(f"Tables not found: {e}")
        out.info("Please run 'talk2metadata schema ingest' first.")
        raise click.Abort()

    # 3. Determine modes to build (reads from config if mode is None)
    try:
        modes_to_build = handler.determine_modes_to_build(
            mode=mode, all_modes=all_modes
        )
        if len(modes_to_build) > 1:
            out.section(
                f"📋 Building indexes for {len(modes_to_build)} mode(s): {', '.join(modes_to_build)}"
            )
    except ValueError as e:
        out.error(str(e))
        raise click.Abort()

    # 4. Build indexes for each mode
    from talk2metadata.utils.paths import find_schema_file, get_metadata_dir

    metadata_dir = get_metadata_dir(run_id, config)
    target_table = config.get("ingest.target_table")
    schema_path = (
        Path(metadata_path)
        if metadata_path
        else find_schema_file(metadata_dir, target_table=target_table)
    )

    for mode_name in modes_to_build:
        out.section(f"\n{'='*60}")
        out.section(f"🔨 Building index for mode: {mode_name}")
        mode_info = handler.registry.get(mode_name)
        if mode_info:
            out.info(f"Description: {mode_info.description}")
        out.section(f"{'='*60}")

        # Build index
        # Note: build_index_for_mode already reads from config if model_name/batch_size are None
        try:
            table_indices, indexer = handler.build_index_for_mode(
                mode_name=mode_name,
                tables=tables,
                schema_metadata=schema_metadata,
                model_name=model_name,  # None = read from config
                batch_size=batch_size,  # None = read from config
            )

            out.success(f"Index built successfully for {mode_name}:")
            stats = handler.get_index_stats(table_indices)
            for table_name, table_stats in stats.items():
                out.stats(
                    {
                        f"{table_name}": f"{table_stats['vectors']} vectors, {table_stats['records']} records"
                    },
                    indent="   ",
                )
        except Exception as e:
            out.error(f"Index building failed for {mode_name}: {e}")
            logger.exception(f"Failed to build index for {mode_name}")
            continue

        # Save index
        out.section(f"💾 Saving index for {mode_name}...")
        try:
            mode_index_dir = handler.save_index_for_mode(
                mode_name=mode_name,
                table_indices=table_indices,
                indexer=indexer,
                schema_metadata_path=schema_path,
                output_dir=output_dir,  # Already Path or None from config
                run_id=run_id,
            )
            out.success(f"Index saved to {mode_index_dir}")
        except Exception as e:
            out.error(f"Failed to save index for {mode_name}: {e}")

    # Success
    out.section(f"{'='*60}")
    out.section("✅ Indexing complete!")

    if len(modes_to_build) == 1:
        out.next_steps(
            "Next step:",
            [
                f"Run 'talk2metadata search retrieve \"your query\"' to search using {modes_to_build[0]}"
            ],
        )
    else:
        out.next_steps(
            "Next step:",
            [
                "Run 'talk2metadata search retrieve \"your query\" --compare' to compare all modes"
            ],
        )


def _display_comparison_results(
    comparison_result, query, top_k, output_format, retrievers
):
    """Display comparison mode results."""
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
        out.section("=" * 80)
        out.section("COMPARISON RESULTS")
        out.section("=" * 80)

        click.echo(
            f"\nCommon Results (appear in all modes): {len(comparison_result.common_results)}"
        )
        if comparison_result.common_results:
            for r in comparison_result.common_results[:top_k]:
                click.echo(f"  - {r.table}.{r.row_id} (score: {r.score:.4f})")

        out.section("\nOverlap Statistics:")
        for mode_name, overlap in comparison_result.overlap_stats.items():
            click.echo(f"  - {mode_name}: {overlap}% overlap")

        out.section("\nMode-specific Results:")
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


def _display_search_results(
    results, query, top_k, mode_name, output_format, show_score, per_table_top_k=None
):
    """Display single mode search results."""
    if output_format == "json":
        result_dicts = []
        for r in results:
            result_dict = {
                "rank": r.rank,
                "table": r.table,
                "row_id": r.row_id,
                "score": r.score,
                "data": r.data,
            }
            # Add RecordVoter fields if available
            if hasattr(r, "match_count"):
                result_dict["match_count"] = r.match_count
                result_dict["matched_tables"] = r.matched_tables
            result_dicts.append(result_dict)

        output = {
            "query": query,
            "top_k": top_k,
            "search_mode": mode_name,
            "per_table_top_k": (
                per_table_top_k if mode_name == "record_embedding" else None
            ),
            "results": result_dicts,
        }
        click.echo(json.dumps(output, indent=2))
    else:
        # Text output
        if not results:
            out.error("No results found")
            return

        out.success(f"Found {len(results)} results:\n")

        for result in results:
            out.section("=" * 80)
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
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                click.echo(f"  {key}: {value_str}")

            click.echo()

        out.section("=" * 80)
        out.success(f"Retrieved {len(results)} records")


@search_group.command(name="retrieve")
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=None,
    help="Number of results to return (default: from config or 5)",
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
@with_run_id
@handle_errors
@click.pass_context
def retrieve_cmd(
    ctx,
    query,
    top_k,
    index_dir,
    output_format,
    show_score,
    per_table_top_k,
    mode,
    compare,
    run_id,
):
    """Retrieve relevant records using natural language query.

    QUERY: Natural language search query

    \b
    Examples:
        # Simple search with active mode
        talk2metadata search retrieve "customers in healthcare industry"

        # Search with specific mode
        talk2metadata search retrieve "high value orders" --mode record_embedding

        # Compare all modes
        talk2metadata search retrieve "recent orders" --compare

        # JSON output
        talk2metadata search retrieve "recent orders" --format json

        # Use specific run ID
        talk2metadata search retrieve "my query" --run-id my_run
    """
    config = get_config()

    # Update config with CLI options
    if run_id:
        config.set("run_id", run_id)

    # Initialize handler
    handler = SearchHandler(config)

    # Handle comparison mode
    if compare:
        if output_format == "text":
            out.section(f'🔍 Comparison Mode: "{query}"')
            out.stats({"Top-K": top_k})

        try:
            # Load retrievers and run comparison
            retrievers = handler.load_retrievers_for_comparison(
                index_dir=Path(index_dir) if index_dir else None,
                run_id=run_id,
            )

            if not retrievers:
                out.error("No retrievers loaded for comparison")
                raise click.Abort()

            if output_format == "text":
                out.section(
                    f"📊 Comparing {len(retrievers)} mode(s): {', '.join(retrievers.keys())}\n"
                )

            comparison_result = handler.compare_modes(
                query=query,
                top_k=top_k,
                index_dir=Path(index_dir) if index_dir else None,
                run_id=run_id,
            )

            # Display results
            _display_comparison_results(
                comparison_result, query, top_k, output_format, retrievers
            )

        except Exception as e:
            out.error(f"Comparison failed: {e}")
            logger.exception("Comparison search failed")
            raise click.Abort()

        return

    # Single mode search
    if output_format == "text":
        mode_name = mode or get_active_mode() or "record_embedding"
        out.section(f'🔍 Searching: "{query}" [Mode: {mode_name}]')
        out.stats({"Top-K": top_k})

    try:
        results = handler.search(
            query=query,
            top_k=top_k,
            mode_name=mode,
            index_dir=Path(index_dir) if index_dir else None,
            run_id=run_id,
            per_table_top_k=per_table_top_k,
        )

        # Display results
        mode_name = mode or get_active_mode() or "record_embedding"
        _display_search_results(
            results, query, top_k, mode_name, output_format, show_score, per_table_top_k
        )

    except (FileNotFoundError, NotImplementedError) as e:
        out.error(str(e))
        out.info("Please run 'talk2metadata search index' first.")
        raise click.Abort()
    except Exception as e:
        out.error(f"Search failed: {e}")
        logger.exception("Search failed")
        raise click.Abort()
