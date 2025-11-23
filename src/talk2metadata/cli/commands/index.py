"""Index command for building search index."""

from __future__ import annotations

import pickle
from pathlib import Path

import click

from talk2metadata.core.indexer import Indexer
from talk2metadata.core.schema import SchemaMetadata
from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

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
    "--hybrid",
    is_flag=True,
    default=False,
    help="Build hybrid index (FAISS + BM25) for better search quality",
)
@click.pass_context
def index_cmd(
    ctx, metadata_path, tables_path, output_dir, model_name, batch_size, hybrid
):
    """Build search index from ingested data.

    This command:
    1. Loads tables and schema metadata
    2. Generates denormalized text for target table rows
    3. Creates embeddings using sentence-transformers
    4. Builds FAISS index for fast similarity search

    \b
    Examples:
        # Build index with defaults
        talk2metadata index

        # Specify custom paths
        talk2metadata index --metadata schema.json --tables tables.pkl

        # Use different embedding model
        talk2metadata index --model sentence-transformers/all-mpnet-base-v2
    """
    config = get_config()

    # 1. Load schema metadata
    if not metadata_path:
        metadata_path = (
            Path(config.get("data.metadata_dir", "./data/metadata")) / "schema.json"
        )

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
        tables_path = (
            Path(config.get("data.processed_dir", "./data/processed")) / "tables.pkl"
        )

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

    # 3. Initialize indexer
    click.echo("\n🤖 Initializing indexer...")
    if model_name:
        click.echo(f"   Model: {model_name}")

    try:
        indexer = Indexer(
            model_name=model_name,
            batch_size=batch_size,
        )
    except Exception as e:
        click.echo(f"❌ Failed to initialize indexer: {e}", err=True)
        raise click.Abort()

    # 4. Build index
    if hybrid:
        click.echo("\n🔨 Building hybrid search index (FAISS + BM25)...")
    else:
        click.echo("\n🔨 Building search index...")
    click.echo("   This may take a while...")

    try:
        if hybrid:
            # Build both FAISS and BM25 indexes
            index, records, texts = indexer.build_index(
                tables, schema_metadata, return_texts=True
            )
            click.echo("✓ FAISS index built successfully:")
            click.echo(f"   - Vectors: {index.ntotal}")
            click.echo(f"   - Dimension: {index.d}")
            click.echo(f"   - Records: {len(records)}")

            # Build BM25 index
            click.echo("\n🔨 Building BM25 index...")
            from talk2metadata.core.hybrid_retriever import BM25Index

            bm25_index = BM25Index(texts)
            click.echo("✓ BM25 index built successfully")
        else:
            index, records = indexer.build_index(tables, schema_metadata)
            click.echo("✓ Index built successfully:")
            click.echo(f"   - Vectors: {index.ntotal}")
            click.echo(f"   - Dimension: {index.d}")
            click.echo(f"   - Records: {len(records)}")
            bm25_index = None
    except Exception as e:
        click.echo(f"❌ Index building failed: {e}", err=True)
        raise click.Abort()

    # 5. Save index
    if output_dir:
        index_dir = Path(output_dir)
    else:
        index_dir = Path(config.get("data.indexes_dir", "./data/indexes"))

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "index.faiss"
    records_path = index_dir / "records.pkl"

    click.echo(f"\n💾 Saving index to {index_dir}")

    try:
        indexer.save_index(index, records, index_path, records_path)
        click.echo("✓ Index saved:")
        click.echo(f"   - {index_path}")
        click.echo(f"   - {records_path}")

        if hybrid and bm25_index:
            bm25_path = index_dir / "bm25.pkl"
            bm25_index.save(bm25_path)
            click.echo(f"   - {bm25_path}")
    except Exception as e:
        click.echo(f"❌ Failed to save index: {e}", err=True)
        raise click.Abort()

    click.echo("\n✅ Indexing complete!")
    if hybrid:
        click.echo(
            "\nNext step: Run 'talk2metadata search \"your query\" --hybrid' to use hybrid search"
        )
    else:
        click.echo(
            "\nNext step: Run 'talk2metadata search \"your query\"' to search records"
        )
