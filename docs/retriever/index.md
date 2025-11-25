# Retriever

## Overview

The Retriever module indexes database records and retrieves them based on natural language queries. It bridges the gap between user questions and actual database records through semantic search.

## Key Features

- **Semantic Indexing**: Convert database records into searchable embeddings
- **Hybrid Retrieval**: Combine semantic search (vector similarity) with keyword search (BM25)
- **Schema-Aware**: Leverage foreign key relationships for denormalized indexing
- **Efficient Search**: FAISS-based vector indexing for fast similarity search

## Architecture

```
Indexing Phase:
1. Schema Detection → Identify target table and foreign keys
2. Denormalization → Join related tables, create rich text representations
3. Embedding & Indexing → Generate embeddings, build FAISS index

Query Phase:
1. Query Understanding → Parse natural language, generate query embedding
2. Retrieval → Semantic search (vector similarity) or hybrid (BM25 + semantic)
3. Ranking → Return top-k results
```

## Usage

### Indexing

```python
from talk2metadata.core.indexer import Indexer
from talk2metadata.core.schema import SchemaDetector

# Detect schema
schema = SchemaDetector(tables, target_table="orders").detect()

# Create indexer
indexer = Indexer(
    schema=schema,
    embedding_model="all-MiniLM-L6-v2"
)

# Index all records
indexer.index()
indexer.save("orders_index")
```

### Retrieval

```python
from talk2metadata.core.retriever import Retriever

# Load index
retriever = Retriever.load("orders_index")

# Search
results = retriever.search(
    query="Find orders from Healthcare customers",
    top_k=10
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Record: {result.record}")
```

## Retrieval Strategies

### Semantic Search (Default)

Uses embedding similarity:

```python
results = retriever.search(
    query="healthcare customers",
    top_k=10
)
```

### Hybrid Search

Combines semantic and keyword search:

```python
results = retriever.search(
    query="healthcare customers",
    strategy="hybrid",
    top_k=10
)
```

## Configuration

```yaml
retriever:
  strategy: "hybrid"  # "semantic", "bm25", or "hybrid"
  top_k: 5
  min_score: 0.5
```
