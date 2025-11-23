# Quick Start

This guide will get you up and running with Talk2Metadata in 5 minutes.

## Step 1: Install Talk2Metadata

```bash
uv sync
```

## Step 2: Prepare Sample Data

Use the provided sample data:

```bash
ls data/raw/
# customers.csv  orders.csv  products.csv
```

The sample data includes:
- **customers**: 10 customers from various industries
- **products**: 10 products (software and services)
- **orders**: 20 orders linking customers to products

## Step 3: Ingest Data

Load the CSV files and detect schema:

```bash
uv run talk2metadata ingest csv data/raw --target orders
```

This command:
- Loads all CSV files from `data/raw/`
- Detects foreign key relationships automatically
- Marks `orders` as the target table (the table we'll search)
- Saves metadata to `data/metadata/schema.json`

Expected output:
```
🔧 Ingesting from csv: data/raw
   Target table: orders
📥 Loading tables...
✓ Loaded 3 tables:
   - customers: 10 rows, 6 columns
   - orders: 20 rows, 8 columns
   - products: 10 rows, 5 columns
🔍 Detecting schema and foreign keys...
✓ Schema detection complete:
   - Tables: 3
   - Foreign keys: 2
   Foreign key relationships:
     - orders.customer_id -> customers.id (coverage: 100.0%)
     - orders.product_id -> products.id (coverage: 100.0%)
💾 Saving metadata to data/metadata/schema.json
✓ Metadata saved successfully
```

## Step 4: Build Search Index

Generate embeddings and create FAISS index:

```bash
uv run talk2metadata index
```

This command:
- Loads the target table (`orders`)
- Joins related tables using detected FKs
- Creates denormalized text for each row
- Generates embeddings using sentence-transformers
- Builds FAISS index for fast search

Expected output:
```
📄 Loading schema metadata from data/metadata/schema.json
✓ Loaded schema:
   - Target table: orders
   - Tables: 3
   - Foreign keys: 2
📥 Loading tables from data/processed/tables.pkl
✓ Loaded 3 tables

🤖 Initializing indexer...
   Model: sentence-transformers/all-MiniLM-L6-v2

🔨 Building search index...
   This may take a while...
Creating texts: 100%|████████| 20/20 [00:00<00:00]
Batches: 100%|████████| 1/1 [00:02<00:00]
✓ Index built successfully:
   - Vectors: 20
   - Dimension: 384
   - Records: 20

💾 Saving index to data/indexes
✓ Index saved
```

## Step 5: Search for Records

Now you can search using natural language:

```bash
# Find healthcare customers
uv run talk2metadata search "healthcare customers with high revenue"

# Find recent technology orders
uv run talk2metadata search "orders from technology companies"

# Find specific products
uv run talk2metadata search "machine learning and AI products"

# Show top 10 results
uv run talk2metadata search "completed orders" --top-k 10

# JSON output
uv run talk2metadata search "pending orders" --format json
```

Example output:
```
🔍 Searching: "healthcare customers with high revenue"
   Top-K: 5

Found 5 results:

================================================================================
Rank #1
Table: orders
Row ID: 1

Data:
  id: 1001
  customer_id: 1
  product_id: 101
  amount: 50000
  quantity: 1
  order_date: 2023-02-01
  status: completed
  sales_rep: John Smith

================================================================================
...
```

## Using Python API

You can also use Talk2Metadata programmatically:

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
    print(f"Rank {result.rank}: {result.data['customer_id']}")
    print(f"  Amount: ${result.data['amount']:,}")
    print(f"  Score: {result.score:.4f}")
```

## Next Steps

- [MCP Quick Start Guide](../mcp/quickstart.md) - Running the MCP server for AI integration
- [MCP Integration Guide](../mcp/integration.md) - Complete MCP documentation
- Check `examples/` directory for more Python API usage examples
