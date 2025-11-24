"""Index command - refactored with modular structure."""

from __future__ import annotations

from pathlib import Path

import click

# Ensure modes are registered
import talk2metadata.core.modes  # noqa: F401
from talk2metadata.cli.decorators import handle_errors, with_run_id
from talk2metadata.cli.handlers import IndexHandler
from talk2metadata.cli.output import OutputFormatter
from talk2metadata.cli.utils import CLIDataLoader
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)
out = OutputFormatter()


@click.command(name="index")
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
        talk2metadata index

        # Build index with specific mode
        talk2metadata index --mode record_embedding

        # Build indexes for all enabled modes
        talk2metadata index --all-modes

        # Specify custom paths
        talk2metadata index --metadata schema.json --tables tables.pkl

        # Use specific run ID
        talk2metadata index --run-id my_run
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
        out.info("Please run 'talk2metadata ingest' first.")
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
    schema_path = (
        Path(metadata_path) if metadata_path else find_schema_file(metadata_dir)
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
                f"Run 'talk2metadata search \"your query\"' to search using {modes_to_build[0]}"
            ],
        )
    else:
        out.next_steps(
            "Next step:",
            [
                "Run 'talk2metadata search \"your query\" --compare' to compare all modes"
            ],
        )
