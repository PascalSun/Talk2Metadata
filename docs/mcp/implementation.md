# MCP Server Implementation

Technical documentation for the Talk2Metadata MCP server.

## Architecture

```
talk2metadata/mcp/
├── server.py                # Main MCP server with OAuth
├── config.py                # Configuration management
├── cli.py                   # CLI interface
├── auth/                    # Authentication
│   ├── oidc_client.py       # OIDC resource server
│   └── oauth_proxy.py       # OAuth proxy endpoints
├── tools/                   # MCP tools
│   ├── search.py            # Semantic/hybrid search
│   ├── list_tables.py       # List tables
│   ├── get_schema.py        # Schema metadata
│   └── get_table_info.py    # Table details
├── resources/               # MCP resources
└── prompts/                 # MCP prompts
```

## Components

### Server

Main MCP server with OAuth 2.0 integration:

- StreamableHTTP transport (MCP protocol 2025-03-26)
- JWT authentication middleware
- CORS support
- Health and metadata endpoints

### Tools

MCP tools implement callable functions:

- **search**: Semantic/hybrid search
- **list_tables**: List all tables
- **get_schema**: Complete schema with FKs
- **get_table_info**: Table details

### Authentication

OAuth 2.0 / OIDC support:

- Token introspection (RFC 7662)
- JWT validation using JWKS
- User info endpoint

## Configuration

Three-tier configuration:

1. CLI arguments (highest priority)
2. Environment variables
3. YAML config file
4. Defaults (lowest priority)
