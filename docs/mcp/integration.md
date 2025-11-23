# MCP Server Integration

Expose Talk2Metadata functionality to AI agents (Claude Desktop, ChatGPT, Cursor) via Model Context Protocol.

## Overview

The Talk2Metadata MCP server provides:
- **Semantic Search**: Natural language queries across multi-table data
- **Hybrid Search**: BM25 + semantic for optimal results
- **Schema Exploration**: Discover tables, columns, and foreign keys
- **OAuth 2.0 Security**: Secure access with OIDC authentication

## Quick Start

### Installation

```bash
pip install -e ".[mcp,full]"
```

### Prepare Data

```bash
# Ingest your data
talk2metadata ingest csv ./data/csv --target orders

# Build search index (with hybrid support)
talk2metadata index --hybrid
```

### Run Server

```bash
# Start MCP server
talk2metadata-mcp sse --host 0.0.0.0 --port 8010

# Custom configuration
talk2metadata-mcp sse --config config.mcp.yml --port 8080
```

Server available at: `http://localhost:8010/mcp`

---

## Configuration

Create `config.mcp.yml`:

```yaml
server:
  host: 0.0.0.0
  port: 8010
  base_url: http://localhost:8010
  data_dir: ./data/processed    # Optional
  index_dir: ./data/indexes     # Optional

oauth:
  discovery_url: http://localhost:8000/o/.well-known/openid-configuration
  public_base_url: http://localhost:8000/o
  client_id: talk2metadata-mcp-client
  client_secret: your-secret-here
  use_introspection: true
  verify_ssl: false
  timeout: 5.0
```

Or use environment variables:

```bash
export MCP_PORT=8010
export MCP_DATA_DIR=./data/processed
export OIDC_CLIENT_ID=your-client-id
export OIDC_CLIENT_SECRET=your-secret
```

---

## Authentication

The server supports OAuth 2.0 / OIDC authentication:

### Token Validation Methods

1. **Token Introspection** (default) - RFC 7662 for opaque tokens
2. **JWT Validation** - For self-contained JWT tokens using JWKS

### Setup OAuth Provider

```yaml
oauth:
  discovery_url: http://your-oauth-server/o/.well-known/openid-configuration
  client_id: your-client-id
  client_secret: your-client-secret
  use_introspection: true  # or false for JWT
```

### For Development (No Auth)

To disable authentication for local development, modify `server.py` middleware configuration.

---

## Available Tools

### 1. search

Search across all tables using natural language:

```json
{
  "query": "customers in healthcare industry",
  "top_k": 10,
  "hybrid": true
}
```

### 2. list_tables

Get all available tables:

```json
{}
```

### 3. get_schema

Get complete schema with foreign key relationships:

```json
{}
```

### 4. get_table_info

Get detailed information about a specific table:

```json
{
  "table_name": "orders"
}
```

---

## Resources

Access data via URI:

- `resource://talk2metadata/info` - Server information
- `resource://talk2metadata/status` - Health status
- `resource://talk2metadata/schema` - Complete schema metadata
- `resource://talk2metadata/table/{name}` - Specific table details

---

## Prompts

- **help** - Usage guide and best practices
- **search_guide** - Effective search strategies

---

## Testing

### Health Check

```bash
curl http://localhost:8010/health
```

### Get Metadata

```bash
curl http://localhost:8010/metadata
```

### With MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8010
```

### With Bearer Token

```bash
curl -H "Authorization: Bearer $TOKEN" \
     -X POST http://localhost:8010/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

---

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

### Python Client

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    mcp_servers=[{
        "url": "http://localhost:8010/mcp",
        "authorization": f"Bearer {access_token}"
    }],
    messages=[{
        "role": "user",
        "content": "Search for customers in the healthcare industry"
    }]
)
```

---

## Troubleshooting

### Index not found

```bash
talk2metadata index --hybrid
```

### Schema not found

```bash
talk2metadata ingest csv ./data/csv --target orders
```

### OAuth errors

1. Check OAuth server is running
2. Verify client credentials in `config.mcp.yml`
3. Test discovery: `curl http://localhost:8000/o/.well-known/openid-configuration`
4. Check logs with `--log-level debug`

### Port in use

```bash
talk2metadata-mcp sse --port 8011
```

---

## Advanced

### Hybrid Search

For best results, enable hybrid search (BM25 + semantic):

```json
{
  "query": "high-value enterprise customers",
  "hybrid": true,
  "top_k": 20
}
```

### Custom Index Location

```bash
export MCP_INDEX_DIR=/path/to/your/indexes
talk2metadata-mcp sse
```

### Production Deployment

1. Enable SSL verification: `verify_ssl: true`
2. Use strong client credentials
3. Configure rate limiting
4. Set up monitoring and logging
5. Use reverse proxy (nginx) with HTTPS

---

## See Also

- [Quick Start Guide](quickstart.md)
- [Implementation Details](implementation.md)
- [Main Documentation](../index.md)
