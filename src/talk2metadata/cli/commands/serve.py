"""Serve command for starting API server."""

from __future__ import annotations

import click

from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.command(name="serve")
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to bind to (default: 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload (development mode)",
)
@click.pass_context
def serve_cmd(ctx, host, port, reload):
    """Start FastAPI server for Talk2Metadata.

    This command starts a REST API server that exposes search functionality.

    NOTE: Requires 'full' installation: pip install talk2metadata[full]

    \b
    Examples:
        # Start server
        talk2metadata serve

        # Custom host and port
        talk2metadata serve --host localhost --port 8080

        # Development mode with auto-reload
        talk2metadata serve --reload
    """
    click.echo(f"🚀 Starting Talk2Metadata API server...")
    click.echo(f"   Host: {host}")
    click.echo(f"   Port: {port}")

    try:
        import uvicorn
        from talk2metadata.api.server import create_app
    except ImportError:
        click.echo(
            "❌ FastAPI/Uvicorn not installed.\n"
            "   Please install with: pip install talk2metadata[full]",
            err=True,
        )
        raise click.Abort()

    try:
        # Create app
        app = create_app()

        # Start server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except Exception as e:
        click.echo(f"❌ Server failed: {e}", err=True)
        raise click.Abort()
