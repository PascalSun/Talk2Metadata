# MCP Server Integration

Expose Talk2Metadata functionality to AI agents via Model Context Protocol.

## Overview

The Talk2Metadata MCP server provides:
- **Semantic Search**: Natural language queries across multi-table data
- **Hybrid Search**: BM25 + semantic for optimal results
- **Schema Exploration**: Discover tables, columns, and foreign keys
- **OAuth 2.0 Security**: Secure access with OIDC authentication

## Quick Start

```bash
# Install with MCP support
./setup.sh --mcp

# Prepare data
talk2metadata schema ingest csv ./data/raw --target orders
talk2metadata search prepare

# Start server
talk2metadata-mcp sse --host 0.0.0.0 --port 8010
```

## Configuration

Create `config.mcp.yml`:

```yaml
server:
  host: 0.0.0.0
  port: 8010
  base_url: http://localhost:8010

oauth:
  discovery_url: http://localhost:8000/o/.well-known/openid-configuration
  client_id: your-client-id
  client_secret: your-secret
```

## Available Tools

### search

Search across all tables using natural language:

```json
{
  "query": "customers in healthcare industry",
  "top_k": 10,
  "hybrid": true
}
```

### list_tables

Get all available tables:

```json
{}
```

### get_schema

Get complete schema with foreign key relationships:

```json
{}
```

### get_table_info

Get detailed information about a specific table:

```json
{
  "table_name": "orders"
}
```

## Client Integration

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "talk2metadata": {
      "url": "http://localhost:8010/mcp",
      "authorization": "Bearer YOUR_TOKEN_HERE"
    }
  }
}
```
