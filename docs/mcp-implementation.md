# MCP Server Implementation

Technical documentation for the Talk2Metadata MCP server implementation.

## Architecture

### Directory Structure

```
talk2metadata/mcp/
├── __init__.py              # Package exports
├── server.py                # Main MCP server with OAuth
├── config.py                # Configuration management
├── cli.py                   # CLI interface
├── auth/                    # Authentication
│   ├── oidc_client.py       # OIDC resource server
│   └── oauth_proxy.py       # OAuth proxy endpoints
├── tools/                   # MCP tools
│   ├── __init__.py
│   ├── search.py            # Semantic/hybrid search
│   ├── list_tables.py       # List tables
│   ├── get_schema.py        # Schema metadata
│   └── get_table_info.py    # Table details
├── resources/               # MCP resources
│   └── __init__.py          # Resource handlers
├── prompts/                 # MCP prompts
│   ├── help.py              # General help
│   └── search_guide.py      # Search best practices
├── common/                  # Shared utilities
│   ├── retriever.py         # Global retriever instance
│   └── schema_index.py      # Schema access
└── static/                  # Static assets
```

## Components

### 1. Server (`server.py`)

**Main MCP server** with OAuth 2.0 integration.

**Key Features:**
- StreamableHTTP transport (MCP protocol 2025-03-26)
- JWT authentication middleware
- CORS support
- OAuth endpoint proxying
- Health and metadata endpoints

**Endpoints:**
- `GET /mcp` - SSE stream for server messages
- `POST /mcp` - JSON-RPC requests
- `DELETE /mcp` - Session termination
- `GET /health` - Health check
- `GET /metadata` - Server discovery
- `/.well-known/*` - OAuth discovery endpoints
- `/oauth/*` - OAuth flow endpoints

### 2. Configuration (`config.py`)

**Three-tier configuration system:**

1. CLI arguments (highest priority)
2. Environment variables
3. YAML config file
4. Defaults (lowest priority)

**Classes:**
- `ServerConfig` - Server settings (host, port, paths)
- `OAuthConfig` - OAuth/OIDC settings
- `MCPConfig` - Main configuration container

### 3. Authentication (`auth/`)

#### OIDC Client (`oidc_client.py`)

**OIDCResourceServer** class for token validation:

- Discovers OIDC endpoints
- Validates JWT tokens using JWKS
- Validates opaque tokens via introspection (RFC 7662)
- Handles token expiration
- Provides user info endpoint

**Methods:**
- `discover_endpoints()` - OIDC discovery
- `verify_token()` - Token validation (auto-selects method)
- `verify_token_jwt()` - JWT validation
- `verify_token_introspection()` - Token introspection
- `get_userinfo()` - Fetch user information

#### OAuth Proxy (`oauth_proxy.py`)

Proxies OAuth endpoints from the authentication server:

- Authorization server metadata
- Protected resource metadata
- OpenID configuration
- Client registration
- OAuth callbacks
- Login pages
- Static assets

### 4. Tools (`tools/`)

MCP tools implement the callable functions exposed to clients.

#### Tool Registration Pattern

```python
# Each tool module defines:
TOOL_SPEC = {
    "name": "tool_name",
    "description": "What it does",
    "inputSchema": {...}
}

async def handle_tool_name(args: dict) -> list[TextContent]:
    # Implementation
    ...
```

**Available Tools:**

1. **search** - Semantic/hybrid search
   - Uses global retriever instance
   - Supports top-k and hybrid mode
   - Returns ranked results with scores

2. **list_tables** - List all tables
   - Uses global schema instance
   - Shows basic table information
   - Identifies target table

3. **get_schema** - Complete schema
   - Foreign key relationships
   - Sample values
   - Data types and statistics

4. **get_table_info** - Table details
   - Column information
   - Related tables
   - Foreign keys (incoming/outgoing)

### 5. Resources (`resources/`)

URI-based data access following MCP resource pattern.

**Static Resources:**
- `resource://talk2metadata/info`
- `resource://talk2metadata/status`

**Dynamic Resources:**
- `resource://talk2metadata/schema`
- `resource://talk2metadata/table/{name}`

### 6. Prompts (`prompts/`)

Guidance and help for users.

**Prompts:**
- `help` - Comprehensive usage guide
- `search_guide` - Search best practices

### 7. Common Utilities (`common/`)

Shared instances and utilities.

#### Retriever Manager (`retriever.py`)

Global retriever singleton:
```python
retriever = get_retriever(use_hybrid=True)
results = retriever.search(query, top_k=10)
```

#### Schema Access (`schema_index.py`)

Global schema metadata:
```python
schema = get_schema()
tables = schema.tables
fks = schema.foreign_keys
```

## Protocol

### Transport: StreamableHTTP

MCP protocol version: **2025-03-26**

**Request Flow:**
1. Client connects via POST `/mcp`
2. Server validates OAuth token (JWT middleware)
3. Request routed to MCP transport
4. Response via SSE or JSON

**Message Format:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"query": "...", "top_k": 10}
  },
  "id": 1
}
```

### Authentication Flow

**OAuth 2.0 / OIDC:**

1. **Discovery** - Client fetches `.well-known/openid-configuration`
2. **Authorization** - User authorizes via OAuth provider
3. **Token** - Client receives access token
4. **Request** - Client includes `Authorization: Bearer {token}`
5. **Validation** - Server validates token (JWT or introspection)
6. **Response** - Protected resource returned

**Middleware Chain:**
```
Request → CORS → JWTAuth → MCP Handler → Response
```

## Configuration Management

### Priority Order

1. **CLI arguments**
   ```bash
   talk2metadata-mcp sse --port 8010 --host 0.0.0.0
   ```

2. **Environment variables**
   ```bash
   export MCP_PORT=8010
   export OIDC_CLIENT_ID=your-client-id
   ```

3. **YAML config file**
   ```yaml
   server:
     port: 8010
   ```

4. **Defaults**
   ```python
   ServerConfig(port=8010)
   ```

### Configuration Loading

```python
# Load with priority
config = MCPConfig.load(config_path)

# From file only
config = MCPConfig.from_file("config.mcp.yml")

# From env only
config = MCPConfig.from_env()
```

## Error Handling

### Tool Errors

Tools return error responses in JSON:

```json
{
  "error": "Index not found",
  "message": "...",
  "hint": "Run 'talk2metadata index' first"
}
```

### Resource Errors

Resources return error objects:

```json
{
  "error": "Table not found",
  "error_code": "TABLE_NOT_FOUND",
  "table_name": "...",
  "available_tables": [...]
}
```

### Authentication Errors

HTTP 401 with RFC 6750 format:

```
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer realm="Talk2Metadata MCP"

{
  "error": "unauthorized",
  "error_description": "Invalid or expired token"
}
```

## Integration with Talk2Metadata Core

### Dependencies

```python
# Core retrieval
from talk2metadata.core.retriever import Retriever
from talk2metadata.core.hybrid_retriever import HybridRetriever

# Schema
from talk2metadata.core.schema import SchemaMetadata

# Configuration
from talk2metadata.utils.config import get_config

# Logging
from talk2metadata.utils.logging import get_logger
```

### Singleton Pattern

Global instances for efficiency:

```python
# Retriever (lazy loaded)
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever.from_paths(...)
    return _retriever

# Schema (lazy loaded)
_schema = None
def get_schema():
    global _schema
    if _schema is None:
        _schema = SchemaMetadata.load(...)
    return _schema
```

## Testing

### Manual Testing

```bash
# Health check
curl http://localhost:8010/health

# List tools (requires auth)
curl -H "Authorization: Bearer $TOKEN" \
     -X POST http://localhost:8010/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Call search tool
curl -H "Authorization: Bearer $TOKEN" \
     -X POST http://localhost:8010/mcp \
     -H "Content-Type: application/json" \
     -d '{
       "jsonrpc":"2.0",
       "method":"tools/call",
       "params":{
         "name":"search",
         "arguments":{"query":"healthcare","top_k":5}
       },
       "id":2
     }'
```

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector http://localhost:8010
```

## Performance Considerations

### Caching

- Retriever instance cached globally
- Schema metadata cached globally
- JWKS keys cached by PyJWKClient

### Lazy Loading

- Retriever loaded on first search
- Schema loaded on first access
- OIDC endpoints discovered on first token validation

### Connection Pooling

- httpx async client for OAuth requests
- Starlette ASGI for concurrent requests

## Security

### Token Validation

**JWT Method:**
- Validates signature using JWKS
- Checks expiration (exp claim)
- Verifies issuer (iss claim)
- Optional audience check

**Introspection Method:**
- POST to introspection endpoint
- Checks active=true
- Validates client credentials
- RFC 7662 compliant

### Protected Endpoints

Only `/mcp` requires authentication. Public endpoints:
- `/health`
- `/metadata`
- `/.well-known/*`

### SSL/TLS

Production checklist:
- Enable SSL verification: `verify_ssl: true`
- Use HTTPS for base_url
- Strong client secrets
- Secure token storage

## Deployment

### Development

```bash
pip install -e ".[mcp,full]"
talk2metadata-mcp sse --log-level debug
```

### Production

```bash
# Install from PyPI
pip install talk2metadata[mcp,full]

# Production config
cp config.mcp.example.yml config.mcp.yml
# Edit with production settings

# Run with uvicorn
talk2metadata-mcp sse --config config.mcp.yml
```

### Docker (Future)

Create `Dockerfile.mcp`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install .[mcp,full]
CMD ["talk2metadata-mcp", "sse"]
```

## Future Enhancements

### Short Term
- Unit tests for MCP components
- Docker deployment
- Rate limiting
- Permission controls

### Medium Term
- Real-time index updates
- Query caching
- Usage analytics
- WebSocket transport option

### Long Term
- Multi-tenancy
- Result streaming
- Advanced filters
- Plugin system

## See Also

- [User Guide](mcp-integration.md)
- [Quick Start](mcp-quickstart.md)
- [Main Documentation](../README.md)
