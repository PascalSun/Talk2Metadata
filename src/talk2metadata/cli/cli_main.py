"""CLI entry point for Talk2Metadata."""

from __future__ import annotations

import click

from talk2metadata import __version__
from talk2metadata.utils.config import load_config
from talk2metadata.utils.logging import setup_logging

# Import commands
from talk2metadata.cli.commands import index, ingest, schema, search


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to config.yml file",
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging level",
)
@click.pass_context
def cli(ctx, config, log_level):
    """Talk2Metadata - Question-driven multi-table record retrieval.

    \b
    Examples:
        # Ingest CSV files
        talk2metadata ingest csv ./data/csv --target orders

        # Build search index
        talk2metadata index

        # Search for records
        talk2metadata search "customers in healthcare industry"
    """
    ctx.ensure_object(dict)

    # Setup logging
    setup_logging(level=log_level)

    # Load config if provided
    if config:
        ctx.obj["config"] = load_config(config)


# Register commands
cli.add_command(ingest.ingest_cmd)
cli.add_command(index.index_cmd)
cli.add_command(schema.schema_cmd)
cli.add_command(search.search_cmd)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
