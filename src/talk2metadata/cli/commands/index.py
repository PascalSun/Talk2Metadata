"""Index command for building search index."""

from __future__ import annotations

import pickle
from pathlib import Path

import click

# Ensure modes are registered
import talk2metadata.core.modes  # noqa: F401
from talk2metadata.core.modes import (  # noqa: F401
    Indexer,
    get_active_mode,
    get_mode_indexer_config,
    get_registry,
)
from talk2metadata.core.schema.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger
from talk2metadata.utils.paths import (
    find_schema_file,
    get_indexes_dir,
    get_metadata_dir,
    get_processed_dir,
)

logger = get_logger(__name__)


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
    default=32,
    help="Batch size for embedding generation",
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
@click.pass_context
def index_cmd(
    ctx, metadata_path, tables_path, output_dir, model_name, batch_size, mode, all_modes
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
    """
    config = get_config()
    run_id = config.get("run_id")

    # 1. Load schema metadata
    if not metadata_path:
        metadata_dir = get_metadata_dir(run_id, config)
        metadata_path = find_schema_file(metadata_dir)

    click.echo(f"📄 Loading schema metadata from {metadata_path}")

    if not Path(metadata_path).exists():
        click.echo(
            f"❌ Metadata not found at {metadata_path}\n"
            "   Please run 'talk2metadata ingest' first.",
            err=True,
        )
        raise click.Abort()

    try:
        schema_metadata = SchemaMetadata.load(metadata_path)
        click.echo("✓ Loaded schema:")
        click.echo(f"   - Target table: {schema_metadata.target_table}")
        click.echo(f"   - Tables: {len(schema_metadata.tables)}")
        click.echo(f"   - Foreign keys: {len(schema_metadata.foreign_keys)}")
    except Exception as e:
        click.echo(f"❌ Failed to load metadata: {e}", err=True)
        raise click.Abort()

    # 2. Load tables
    if not tables_path:
        processed_dir = get_processed_dir(run_id, config)
        tables_path = processed_dir / "tables.pkl"

    click.echo(f"📥 Loading tables from {tables_path}")

    if not Path(tables_path).exists():
        click.echo(
            f"❌ Tables not found at {tables_path}\n"
            "   Please run 'talk2metadata ingest' first.",
            err=True,
        )
        raise click.Abort()

    try:
        with open(tables_path, "rb") as f:
            tables = pickle.load(f)
        click.echo(f"✓ Loaded {len(tables)} tables")
    except Exception as e:
        click.echo(f"❌ Failed to load tables: {e}", err=True)
        raise click.Abort()

    # 3. Determine modes to build
    registry = get_registry()
    if all_modes:
        modes_to_build = registry.get_all_enabled()
        if not modes_to_build:
            click.echo("❌ No enabled modes found", err=True)
            raise click.Abort()
        click.echo(
            f"\n📋 Building indexes for {len(modes_to_build)} mode(s): {', '.join(modes_to_build)}"
        )
    else:
        # Use specified mode or active mode from config
        if mode:
            active_mode = mode
        else:
            active_mode = get_active_mode() or "record_embedding"

        if not registry.get(active_mode):
            click.echo(f"❌ Mode '{active_mode}' not found", err=True)
            click.echo(f"   Available modes: {', '.join(registry.get_all_enabled())}")
            raise click.Abort()

        modes_to_build = [active_mode]

    # 4. Build indexes for each mode
    base_index_dir = Path(output_dir) if output_dir else get_indexes_dir(run_id, config)

    for mode_name in modes_to_build:
        mode_info = registry.get(mode_name)
        if not mode_info or not mode_info.enabled:
            click.echo(f"⚠️  Skipping disabled mode: {mode_name}")
            continue

        click.echo(f"\n{'='*60}")
        click.echo(f"🔨 Building index for mode: {mode_name}")
        click.echo(f"   Description: {mode_info.description}")
        click.echo(f"{'='*60}")

        # Initialize indexer for this mode (use mode-specific config)
        try:
            mode_indexer_config = get_mode_indexer_config(mode_name)
            # CLI args override config
            indexer = Indexer(
                model_name=model_name or mode_indexer_config.get("model_name"),
                device=mode_indexer_config.get("device"),
                batch_size=batch_size or mode_indexer_config.get("batch_size", 32),
                normalize=mode_indexer_config.get("normalize", True),
            )
        except Exception as e:
            click.echo(
                f"❌ Failed to initialize indexer for {mode_name}: {e}", err=True
            )
            continue

        # Build index
        try:
            table_indices = indexer.build_index(tables, schema_metadata)
            click.echo(f"✓ Index built successfully for {mode_name}:")
            for table_name, (idx, records) in table_indices.items():
                click.echo(
                    f"   - {table_name}: {idx.ntotal} vectors, {len(records)} records"
                )
        except Exception as e:
            click.echo(f"❌ Index building failed for {mode_name}: {e}", err=True)
            continue

        # Save index (mode-specific directory)
        mode_index_dir = base_index_dir / mode_name
        mode_index_dir.mkdir(parents=True, exist_ok=True)

        click.echo(f"\n💾 Saving index to {mode_index_dir}")
        try:
            indexer.save_multi_table_index(table_indices, mode_index_dir)
            click.echo(f"✓ Index saved for {mode_name}")

            # Save schema metadata
            metadata_path_obj = Path(metadata_path)
            schema_copy_path = mode_index_dir / "schema_metadata.json"
            import shutil

            shutil.copy(metadata_path_obj, schema_copy_path)
        except Exception as e:
            click.echo(f"❌ Failed to save index for {mode_name}: {e}", err=True)

    click.echo(f"\n{'='*60}")
    click.echo("✅ Indexing complete!")
    if len(modes_to_build) == 1:
        click.echo(
            f"\nNext step: Run 'talk2metadata search \"your query\"' to search using {modes_to_build[0]}"
        )
    else:
        click.echo(
            "\nNext step: Run 'talk2metadata search \"your query\" --compare' to compare all modes"
        )
