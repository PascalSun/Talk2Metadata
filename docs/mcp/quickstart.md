# Quick Start: Talk2Metadata MCP Server

## 🚀 Get Started in 3 Steps

### 1. Install with MCP Support

```bash
# From the project directory
pip install -e ".[mcp,full]"
```

### 2. Prepare Your Data

```bash
# Ingest CSV files
talk2metadata ingest csv ./data/csv --target orders

# Build search index (with hybrid support)
talk2metadata index --hybrid
```

### 3. Start the MCP Server

```bash
# Copy and edit configuration
cp config.mcp.example.yml config.mcp.yml

# Start server
talk2metadata-mcp sse
```

## 🔧 Configuration

Edit `config.mcp.yml`:

```yaml
server:
  host: 0.0.0.0
  port: 8010
  base_url: http://localhost:8010

oauth:
  discovery_url: http://localhost:8000/o/.well-known/openid-configuration
  client_id: talk2metadata-mcp-client
  client_secret: your-secret-here
```

Or use environment variables:

```bash
export MCP_PORT=8010
export OIDC_CLIENT_ID=your-client-id
export OIDC_CLIENT_SECRET=your-secret
```

## 🧪 Test the Server

```bash
# Health check
curl http://localhost:8010/health

# Get metadata
curl http://localhost:8010/metadata

# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8010
```

## 📚 Available Tools

1. **search** - Natural language search across all tables
   ```json
   {"query": "customers in healthcare", "top_k": 10, "hybrid": true}
   ```

2. **list_tables** - List all available tables
   ```json
   {}
   ```

3. **get_schema** - Get complete schema with foreign keys
   ```json
   {}
   ```

4. **get_table_info** - Get details about a specific table
   ```json
   {"table_name": "orders"}
   ```

## 🔐 OAuth Setup (Optional)

For production, set up OAuth 2.0:

1. Configure Django OAuth Toolkit or similar
2. Register your MCP server as a client
3. Update `config.mcp.yml` with credentials
4. Connect with Bearer token authentication

For development, you can disable OAuth by modifying the middleware configuration.

## 📖 Documentation

- [MCP Server Guide](integration.md) - Complete user documentation
- [Implementation Details](implementation.md) - Technical details
- Configuration example in root: `config.mcp.example.yml`

## 🆘 Troubleshooting

**"Index not found"**
```bash
talk2metadata index --hybrid
```

**"Schema not found"**
```bash
talk2metadata ingest csv ./data/csv --target orders
```

**Port already in use**
```bash
talk2metadata-mcp sse --port 8011
```

## 💡 Tips

- Use `hybrid=true` for best search results (combines keyword + semantic)
- Check `get_schema` to understand table relationships
- Use `search_guide` prompt for search best practices
- Monitor logs with `--log-level debug`

## 🎯 Next Steps

1. Read the [full documentation](integration.md)
2. Explore the tools with MCP Inspector
3. Integrate with Claude or ChatGPT
4. Configure OAuth for production use

Happy searching! 🔍
