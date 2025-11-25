# MCP Server Quick Start

## Installation

```bash
# Install with MCP support
./setup.sh --mcp
# or
pip install -e ".[mcp]"
```

## Prepare Data

```bash
# Ingest CSV files
talk2metadata schema ingest csv ./data/raw --target orders

# Prepare modes (builds indexes or loads databases)
talk2metadata search prepare
```

## Start Server

```bash
# Copy and edit configuration
cp config.mcp.example.yml config.mcp.yml

# Start server
talk2metadata-mcp sse
```

Server runs at `http://localhost:8010`.

## Configuration

Edit `config.mcp.yml`:

```yaml
server:
  host: 0.0.0.0
  port: 8010

oauth:
  discovery_url: http://localhost:8000/o/.well-known/openid-configuration
  client_id: talk2metadata-mcp-client
  client_secret: your-secret-here
```

## Available Tools

1. **search** - Natural language search across all tables
2. **list_tables** - List all available tables
3. **get_schema** - Get complete schema with foreign keys
4. **get_table_info** - Get details about a specific table

## Testing

```bash
# Health check
curl http://localhost:8010/health

# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8010
```
