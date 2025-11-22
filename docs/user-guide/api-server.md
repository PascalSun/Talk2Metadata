# API Server

Talk2Metadata includes a FastAPI-based REST API server for programmatic access.

## Starting the Server

### Basic Usage

```bash
uv run talk2metadata serve
```

Server will start on `http://0.0.0.0:8000`

### Custom Host and Port

```bash
# Bind to localhost only
uv run talk2metadata serve --host localhost --port 8080

# Development mode with auto-reload
uv run talk2metadata serve --reload
```

### Production Deployment

For production, use a proper ASGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn talk2metadata.api.server:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Endpoints

### Root

```http
GET /
```

Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Talk2Metadata API",
  "version": "0.1.0",
  "description": "Question-driven multi-table record retrieval",
  "endpoints": {
    "health": "/health",
    "search": "/api/v1/search",
    "schema": "/api/v1/schema",
    "docs": "/docs"
  }
}
```

### Health Check

```http
GET /health
```

Check service health and index status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "index_loaded": true,
  "total_records": 20
}
```

### Search

```http
POST /api/v1/search
```

Search for relevant records using natural language.

**Request Body:**
```json
{
  "query": "healthcare customers with high revenue",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "healthcare customers with high revenue",
  "top_k": 5,
  "total_records": 20,
  "results": [
    {
      "rank": 1,
      "table": "orders",
      "row_id": 1001,
      "score": 0.234,
      "data": {
        "id": 1001,
        "customer_id": 1,
        "amount": 50000,
        "status": "completed"
      }
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "healthcare customers",
    "top_k": 5
  }'
```

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/search",
    json={
        "query": "healthcare customers",
        "top_k": 5
    }
)

results = response.json()
for result in results["results"]:
    print(f"Rank {result['rank']}: {result['data']}")
```

### Get Schema

```http
GET /api/v1/schema
```

Get complete schema information including tables and foreign keys.

**Response:**
```json
{
  "target_table": "orders",
  "tables": {
    "orders": {
      "name": "orders",
      "columns": {
        "id": "int64",
        "customer_id": "int64",
        "amount": "float64"
      },
      "primary_key": "id",
      "row_count": 20
    }
  },
  "foreign_keys": [
    {
      "child_table": "orders",
      "child_column": "customer_id",
      "parent_table": "customers",
      "parent_column": "id",
      "coverage": 1.0
    }
  ]
}
```

### List Tables

```http
GET /api/v1/schema/tables
```

Get a simplified list of tables.

**Response:**
```json
{
  "target_table": "orders",
  "tables": [
    {
      "name": "customers",
      "row_count": 10,
      "column_count": 6
    },
    {
      "name": "orders",
      "row_count": 20,
      "column_count": 8
    }
  ]
}
```

### Search Status

```http
GET /api/v1/search/status
```

Get detailed search service status.

**Response:**
```json
{
  "index_loaded": true,
  "total_records": 20,
  "index_size": 20,
  "embedding_dimension": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

## Interactive Documentation

The API includes auto-generated interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide:
- Complete API reference
- Request/response schemas
- Try-it-out functionality
- Example requests

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `404`: Resource not found (e.g., index not built)
- `422`: Validation error (invalid request)
- `500`: Server error

**Error Response:**
```json
{
  "error": "not_found",
  "message": "Index not found",
  "detail": "Please run 'talk2metadata index' first"
}
```

## CORS

CORS is enabled by default for all origins in development. For production, configure allowed origins in `api/server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify your domain
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Authentication

The current version does not include authentication. For production use, consider adding:

- API key authentication
- OAuth 2.0
- JWT tokens

Example with API key middleware:

```python
from fastapi import Header, HTTPException

API_KEY = "your-secret-key"

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
```

## Rate Limiting

Consider adding rate limiting for production:

```bash
pip install slowapi

# In server.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/v1/search")
@limiter.limit("5/minute")
async def search(request: Request, ...):
    ...
```

## Monitoring

For production deployment, add monitoring:

```python
from prometheus_fastapi_instrumentator import Instrumentator

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
```

Metrics will be available at `/metrics`.

## Configuration

For production deployments, customize your `config.yml`:

```yaml
# data directories
data:
  indexes_dir: "./data/indexes"
  metadata_dir: "./data/metadata"

# embedding settings
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: null  # auto-detect
  batch_size: 32

# retrieval settings
retrieval:
  top_k: 5
  hybrid:
    alpha: 0.5  # 0=BM25 only, 1=semantic only
    fusion_method: "rrf"  # or "weighted_sum"
```

See `config.example.yml` for all available options.

## Next Steps

- [REST API Reference](../api-reference/rest-api.md) - Complete API documentation
- [Installation Guide](../getting-started/installation.md) - Setup and dependencies
