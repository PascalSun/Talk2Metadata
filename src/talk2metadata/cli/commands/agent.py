"""Agent-related commands for managing LLM providers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import click

from talk2metadata.utils.config import get_config
from talk2metadata.utils.logging import get_logger

logger = get_logger(__name__)


@click.group(name="agent")
def agent_group():
    """Manage LLM agent providers and servers."""
    pass


@agent_group.command(name="vllm-server")
@click.option(
    "--model",
    "-m",
    help="Model name/identifier (overrides config.yml)",
)
@click.option(
    "--host",
    help="Host to bind the server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    help="Port to bind the server to (default: 8000)",
)
@click.option(
    "--tensor-parallel-size",
    type=int,
    help="Number of tensor parallel replicas",
)
@click.option(
    "--gpu-memory-utilization",
    type=float,
    help="Fraction of GPU memory to use (0.0 to 1.0)",
)
@click.option(
    "--max-model-len",
    type=int,
    help="Maximum sequence length",
)
@click.option(
    "--dtype",
    type=click.Choice(["auto", "float16", "bfloat16", "float32"], case_sensitive=False),
    help="Data type for model weights",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    help="Trust remote code when loading model",
)
@click.option(
    "--download-dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    help="Directory to download and load model",
)
@click.option(
    "--api-key",
    help="API key for authentication (optional)",
)
@click.option(
    "--served-model-name",
    help="Model name to serve (defaults to model name)",
)
def vllm_server_cmd(
    model: str | None,
    host: str | None,
    port: int | None,
    tensor_parallel_size: int | None,
    gpu_memory_utilization: float | None,
    max_model_len: int | None,
    dtype: str | None,
    trust_remote_code: bool,
    download_dir: str | None,
    api_key: str | None,
    served_model_name: str | None,
):
    """Start a vLLM OpenAI-compatible API server.

    This command starts a vLLM server that provides an OpenAI-compatible API
    for local high-performance LLM inference.

    Model and port are read from config.yml (agent.vllm.model and agent.vllm.base_url)
    if not specified via command-line options.

    \b
    Examples:
        # Start server using model from config.yml
        talk2metadata agent vllm-server

        # Override model from command line
        talk2metadata agent vllm-server --model meta-llama/Llama-2-7b-chat-hf

        # Start server on custom port
        talk2metadata agent vllm-server --port 8080

        # Start server with GPU memory limit
        talk2metadata agent vllm-server --gpu-memory-utilization 0.9

        # Start server with tensor parallelism
        talk2metadata agent vllm-server --tensor-parallel-size 2
    """
    try:
        # Check if vllm is installed
        import vllm  # noqa: F401
    except ImportError:
        click.echo(
            "❌ vLLM is not installed.\n" "   Install it with: pip install vllm",
            err=True,
        )
        sys.exit(1)

    # Load config
    config = get_config()
    agent_config = config.get("agent", {})
    vllm_config = agent_config.get("vllm", {})

    # Resolve model: command-line > config.vllm.model > config.model
    if model is None:
        model = vllm_config.get("model") or agent_config.get("model")
        if model is None:
            click.echo(
                "❌ Model not specified and not found in config.yml.\n"
                "   Please specify --model or set agent.vllm.model in config.yml",
                err=True,
            )
            sys.exit(1)

    # Resolve host and port from base_url if provided
    if host is None:
        host = "0.0.0.0"  # Default host

    if port is None:
        # Try to extract port from base_url
        base_url = vllm_config.get("base_url", "")
        if base_url:
            try:
                parsed = urlparse(base_url)
                if parsed.port:
                    port = parsed.port
                elif parsed.scheme == "http":
                    port = 80
                elif parsed.scheme == "https":
                    port = 443
                else:
                    port = 8000
            except Exception:
                port = 8000
        else:
            port = 8000

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]

    # Add optional arguments
    if tensor_parallel_size is not None:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])
    if max_model_len is not None:
        cmd.extend(["--max-model-len", str(max_model_len)])
    if dtype:
        cmd.extend(["--dtype", dtype])
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if download_dir:
        download_path = Path(download_dir)
        download_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--download-dir", str(download_path)])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])

    click.echo("🚀 Starting vLLM server...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Endpoint: http://{host}:{port}/v1")
    click.echo(f"\n   Command: {' '.join(cmd)}\n")

    try:
        # Run the server (this will block)
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n\n⚠️  Server stopped by user")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        click.echo(f"\n❌ Server failed to start: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Unexpected error: {e}", err=True)
        sys.exit(1)
