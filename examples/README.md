# Talk2Metadata Examples

This directory contains example scripts demonstrating various usage patterns of Talk2Metadata.

## Prerequisites

Before running the examples, ensure you have:

1. Installed Talk2Metadata with all dependencies:
   ```bash
   uv sync --all-extras
   ```

2. Prepared sample data (already included in `../data/raw/`):
   - `customers.csv`
   - `products.csv`
   - `orders.csv`

## Examples Overview

### 1. Complete Workflow (`complete_workflow.py`)

Demonstrates the full end-to-end workflow:

```bash
cd examples
uv run python complete_workflow.py
```

This script:
- Loads data from CSV files
- Detects schema and foreign keys
- Builds search index
- Performs multiple searches

**What you'll learn:**
- How to use connectors to load data
- How schema detection works
- How to build and save indexes
- How to perform searches programmatically

### 2. Python API Usage (`python_api_example.py`)

Shows various Python API usage patterns:

```bash
# First, ingest and index data
uv run talk2metadata ingest csv ../data/raw --target orders
uv run talk2metadata index

# Then run examples
cd examples
uv run python python_api_example.py
```

**What you'll learn:**
- Basic search operations
- Batch searching
- Accessing schema metadata
- Filtering and processing results
- Getting retriever statistics

### 3. REST API Client (`api_client_example.py`)

Demonstrates interaction with the REST API:

```bash
# Terminal 1: Start server
uv run talk2metadata serve

# Terminal 2: Run client examples
cd examples
uv run python api_client_example.py
```

**What you'll learn:**
- How to create a simple API client
- All available API endpoints
- Error handling
- Processing API responses

## Step-by-Step Tutorial

### Step 1: Ingest Sample Data

```bash
uv run talk2metadata ingest csv data/raw --target orders
```

This will:
- Load 3 CSV files
- Detect 2 foreign key relationships:
  - `orders.customer_id` → `customers.id`
  - `orders.product_id` → `products.id`
- Save metadata to `data/metadata/schema.json`

### Step 2: Build Search Index

```bash
uv run talk2metadata index
```

This will:
- Generate denormalized text for each order (with customer and product info)
- Create embeddings using sentence-transformers
- Build FAISS index
- Save to `data/indexes/`

### Step 3: Try Searches

```bash
# Search from CLI
uv run talk2metadata search "healthcare customers with high revenue"

# Get JSON output
uv run talk2metadata search "technology companies" --format json

# Get more results
uv run talk2metadata search "completed orders" --top-k 10
```

### Step 4: Use Python API

Create a Python script:

```python
from talk2metadata import Retriever

# Load retriever
retriever = Retriever.from_paths(
    "data/indexes/index.faiss",
    "data/indexes/records.pkl"
)

# Search
results = retriever.search("healthcare customers", top_k=5)

# Process results
for result in results:
    print(f"Rank {result.rank}: {result.data}")
```

### Step 5: Start API Server

```bash
# Start server
uv run talk2metadata serve

# In another terminal, test it
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "healthcare customers", "top_k": 5}'

# Or visit interactive docs
open http://localhost:8000/docs
```

## Sample Queries to Try

Here are some interesting queries to try with the sample data:

### By Industry
- "healthcare customers with high revenue"
- "orders from technology companies"
- "finance industry customers"

### By Product
- "machine learning and AI products"
- "consulting services orders"
- "enterprise software purchases"

### By Status
- "pending orders"
- "completed orders"
- "in progress transactions"

### By Region
- "orders from US-West region"
- "European customers"
- "US-East technology companies"

### By Amount
- "high value orders"
- "large transactions"
- "orders above 50000"

### Complex Queries
- "recent healthcare orders for consulting services"
- "pending technology orders in US-West"
- "completed high-value orders from finance companies"

## Custom Data

To use your own data:

1. Prepare CSV files in a directory
2. Ensure tables have ID columns
3. Use consistent naming for foreign keys (e.g., `customer_id` referencing `customers.id`)
4. Run ingestion:
   ```bash
   uv run talk2metadata ingest csv /path/to/your/data --target your_target_table
   ```

## Troubleshooting

### Index Not Found
```
Error: Index not found at data/indexes/index.faiss
```
**Solution:** Run `uv run talk2metadata index`

### Metadata Not Found
```
Error: Schema metadata not found at data/metadata/schema.json
```
**Solution:** Run `uv run talk2metadata ingest csv data/raw --target orders`

### Model Download Issues
If the embedding model fails to download:
```bash
# Pre-download the model
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### API Server Connection Error
```
Error: Could not connect to API server
```
**Solution:** Start the server with `uv run talk2metadata serve`

## Next Steps

- Read the [User Guide](../docs/user-guide/)
- Explore [API Reference](../docs/api-reference/)
- Check [Architecture docs](../docs/architecture/)
- Try with your own data!
