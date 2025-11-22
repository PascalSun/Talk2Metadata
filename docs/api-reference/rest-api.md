# REST API Reference

Complete reference for Talk2Metadata REST API endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. See [API Server](../user-guide/api-server.md#authentication) for adding authentication in production.

## Endpoints

### Search Records

Search for relevant records using natural language queries.

**Endpoint:** `POST /api/v1/search`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | Natural language search query (min length: 1) |
| top_k | integer | No | Number of results to return (default: 5, range: 1-100) |

**Response:** `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| query | string | Original query |
| top_k | integer | Requested number of results |
| total_records | integer | Total records in index |
| results | array | Array of search result objects |

**Result Object:**

| Field | Type | Description |
|-------|------|-------------|
| rank | integer | Result rank (1-indexed) |
| table | string | Table name |
| row_id | int/string | Row ID from original table |
| score | float | Similarity score (L2 distance, lower is better) |
| data | object | Complete record data |

**Example Request:**

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "healthcare customers with high revenue",
    "top_k": 3
  }'
```

**Example Response:**

```json
{
  "query": "healthcare customers with high revenue",
  "top_k": 3,
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
        "product_id": 101,
        "amount": 50000,
        "quantity": 1,
        "order_date": "2023-02-01",
        "status": "completed",
        "sales_rep": "John Smith"
      }
    }
  ]
}
```

**Error Responses:**

- `422 Unprocessable Entity`: Invalid request (e.g., empty query, top_k out of range)
- `500 Internal Server Error`: Index not loaded or search failed

---

### Get Schema Information

Retrieve complete schema metadata including tables and foreign keys.

**Endpoint:** `GET /api/v1/schema`

**Response:** `200 OK`

| Field | Type | Description |
|-------|------|-------------|
| target_table | string | Name of the target table |
| tables | object | Dictionary of table metadata |
| foreign_keys | array | Array of foreign key relationships |

**Table Metadata:**

| Field | Type | Description |
|-------|------|-------------|
| name | string | Table name |
| columns | object | Column name to data type mapping |
| primary_key | string/null | Primary key column name |
| row_count | integer | Number of rows in table |

**Foreign Key Object:**

| Field | Type | Description |
|-------|------|-------------|
| child_table | string | Table with foreign key |
| child_column | string | Foreign key column |
| parent_table | string | Referenced table |
| parent_column | string | Referenced column (usually PK) |
| coverage | float | FK coverage ratio (0.0-1.0) |

**Example Request:**

```bash
curl "http://localhost:8000/api/v1/schema"
```

**Example Response:**

```json
{
  "target_table": "orders",
  "tables": {
    "customers": {
      "name": "customers",
      "columns": {
        "id": "int64",
        "name": "object",
        "industry": "object",
        "region": "object",
        "annual_revenue": "int64",
        "created_date": "object"
      },
      "primary_key": "id",
      "row_count": 10
    },
    "orders": {
      "name": "orders",
      "columns": {
        "id": "int64",
        "customer_id": "int64",
        "product_id": "int64",
        "amount": "int64",
        "quantity": "int64",
        "order_date": "object",
        "status": "object",
        "sales_rep": "object"
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
    },
    {
      "child_table": "orders",
      "child_column": "product_id",
      "parent_table": "products",
      "parent_column": "id",
      "coverage": 1.0
    }
  ]
}
```

**Error Responses:**

- `404 Not Found`: Schema metadata not found (need to run ingestion)
- `500 Internal Server Error`: Failed to load schema

---

### List Tables

Get a simplified list of tables with basic information.

**Endpoint:** `GET /api/v1/schema/tables`

**Response:** `200 OK`

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
    },
    {
      "name": "products",
      "row_count": 10,
      "column_count": 5
    }
  ]
}
```

---

### Search Service Status

Get detailed information about the search service and loaded index.

**Endpoint:** `GET /api/v1/search/status`

**Response:** `200 OK`

```json
{
  "index_loaded": true,
  "total_records": 20,
  "index_size": 20,
  "embedding_dimension": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| index_loaded | boolean | Whether search index is loaded |
| total_records | integer | Total records in index |
| index_size | integer | Number of vectors in FAISS index |
| embedding_dimension | integer | Dimension of embeddings |
| model | string | Embedding model name |

---

### Health Check

Check service health and basic status.

**Endpoint:** `GET /health`

**Response:** `200 OK`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "index_loaded": true,
  "total_records": 20
}
```

**Status Values:**

- `healthy`: Service running normally with index loaded
- `degraded`: Service running but index not loaded

---

### Root Endpoint

Get API information and available endpoints.

**Endpoint:** `GET /`

**Response:** `200 OK`

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

---

## Error Responses

All endpoints may return these error responses:

### 422 Unprocessable Entity

Request validation failed.

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

Server error occurred.

```json
{
  "error": "internal_server_error",
  "message": "An unexpected error occurred",
  "detail": "Index not loaded"
}
```

## Rate Limiting

Currently no rate limiting. See [API Server Guide](../user-guide/api-server.md#rate-limiting) for adding rate limiting in production.

## Client Libraries

### Python

```python
import requests

class Talk2MetadataClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def search(self, query: str, top_k: int = 5):
        response = requests.post(
            f"{self.base_url}/api/v1/search",
            json={"query": query, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()

    def get_schema(self):
        response = requests.get(f"{self.base_url}/api/v1/schema")
        response.raise_for_status()
        return response.json()

# Usage
client = Talk2MetadataClient()
results = client.search("healthcare customers", top_k=5)
```

### JavaScript/TypeScript

```typescript
class Talk2MetadataClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async search(query: string, topK: number = 5) {
    const response = await fetch(`${this.baseUrl}/api/v1/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getSchema() {
    const response = await fetch(`${this.baseUrl}/api/v1/schema`);
    if (!response.ok) {
      throw new Error(`Get schema failed: ${response.statusText}`);
    }
    return response.json();
  }
}

// Usage
const client = new Talk2MetadataClient();
const results = await client.search("healthcare customers", 5);
```

## Next Steps

- [API Server Guide](../user-guide/api-server.md)
- [Quick Start Tutorial](../getting-started/quickstart.md)
