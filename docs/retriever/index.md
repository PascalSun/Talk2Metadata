# Retriever

## Overview

The Retriever module is responsible for indexing database records and retrieving them based on natural language queries. It bridges the gap between user questions and actual database records through semantic search and structured query understanding.

## Key Features

- **Semantic Indexing**: Convert database records into searchable embeddings
- **Hybrid Retrieval**: Combine semantic search (vector similarity) with keyword search (BM25)
- **Schema-Aware**: Leverage foreign key relationships for denormalized indexing
- **Efficient Search**: FAISS-based vector indexing for fast similarity search
- **Configurable**: Support multiple embedding models and retrieval strategies

## Architecture

```
Retrieval Pipeline:

1. Indexing Phase
   ├─ Schema Detection
   │  └─ Identify target table and foreign keys
   │
   ├─ Denormalization
   │  ├─ Join related tables
   │  ├─ Create rich text representations
   │  └─ Include related entity information
   │
   ├─ Text Generation
   │  ├─ Format records as readable text
   │  ├─ Include column names and values
   │  └─ Add related table context
   │
   └─ Embedding & Indexing
      ├─ Generate embeddings (Sentence-Transformers)
      ├─ Build vector index (FAISS)
      └─ Build keyword index (BM25) [optional]

2. Query Phase
   ├─ Query Understanding
   │  ├─ Parse natural language question
   │  └─ Generate query embedding
   │
   ├─ Retrieval
   │  ├─ Semantic search (vector similarity)
   │  ├─ Keyword search (BM25) [optional]
   │  └─ Hybrid fusion (RRF) [optional]
   │
   └─ Ranking & Filtering
      ├─ Re-rank by relevance score
      ├─ Apply confidence thresholds
      └─ Return top-k results
```

## Indexing Strategy

### Denormalization

Talk2Metadata creates denormalized text representations by joining related tables:

```
Original Records (Normalized):

orders (id: 1001)
├─ customer_id: 1
├─ product_id: 101
├─ amount: 50000
└─ status: completed

customers (id: 1)
├─ name: Acme Healthcare
├─ industry: Healthcare
└─ region: US-West

products (id: 101)
├─ name: Enterprise Analytics
├─ category: Software
└─ price: 50000

Denormalized Text:

# Record from orders
id: 1001
customer_id: 1
product_id: 101
amount: 50000
status: completed

## Related customers
name: Acme Healthcare
industry: Healthcare
region: US-West

## Related products
name: Enterprise Analytics Platform
category: Software
price: 50000
```

### Why Denormalization?

1. **Richer Context**: Single record contains information from multiple tables
2. **Better Embeddings**: More semantic information for vector representations
3. **Fewer Lookups**: Related data already included in search results
4. **Natural Language Friendly**: Text format matches how users ask questions

### Indexing Process

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

# Save index
indexer.save("orders_index")
```

## Retrieval Strategies

### 1. Semantic Search (Vector Similarity)

**Default strategy** - Uses embedding similarity:

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

**How it works**:
1. Embed the query using same model as indexing
2. Compute cosine similarity with all indexed records
3. Return top-k most similar records

**Best for**:
- Semantic understanding ("healthcare companies" matches "medical organizations")
- Paraphrased queries
- Conceptual similarity

### 2. Keyword Search (BM25)

**Lexical matching** - Uses term frequency:

```python
retriever = Retriever.load("orders_index", enable_bm25=True)

results = retriever.search(
    query="Healthcare Software",
    strategy="bm25",
    top_k=10
)
```

**How it works**:
1. Tokenize query and documents
2. Compute BM25 scores based on term frequency
3. Return top-k by BM25 score

**Best for**:
- Exact term matching ("SKU-12345")
- Rare/specific keywords
- Domain-specific terminology

### 3. Hybrid Search (Semantic + Keyword)

**Combines both approaches** using Reciprocal Rank Fusion:

```python
results = retriever.search(
    query="Find Healthcare customers buying Software",
    strategy="hybrid",
    semantic_weight=0.7,
    keyword_weight=0.3,
    top_k=10
)
```

**How it works**:
1. Run both semantic and keyword search
2. Normalize scores from each method
3. Combine using weighted fusion
4. Re-rank by fused score

**Best for**:
- General-purpose retrieval
- Balancing precision and recall
- Handling diverse query types

## Configuration

### Indexer Configuration

```yaml
indexer:
  # Target table
  target_table: "orders"

  # Embedding model
  embedding:
    model: "all-MiniLM-L6-v2"  # or "all-mpnet-base-v2", "e5-large", etc.
    device: "cpu"  # or "cuda"
    batch_size: 32

  # Denormalization settings
  denormalization:
    max_depth: 2  # Maximum JOIN depth
    include_related: true
    max_related_records: 10  # Limit related records per FK

  # Text formatting
  text_format:
    include_column_names: true
    include_table_names: true
    separator: "\n"

  # Performance
  cache_embeddings: true
  parallel_processing: true
  num_workers: 4
```

### Retriever Configuration

```yaml
retriever:
  # Search strategy
  strategy: "hybrid"  # "semantic", "bm25", or "hybrid"

  # Semantic search
  semantic:
    similarity_metric: "cosine"  # or "euclidean", "dot_product"
    index_type: "HNSW"  # FAISS index type

  # Keyword search
  bm25:
    k1: 1.5  # Term frequency saturation
    b: 0.75  # Length normalization

  # Hybrid fusion
  hybrid:
    semantic_weight: 0.7
    keyword_weight: 0.3
    fusion_method: "rrf"  # Reciprocal Rank Fusion

  # Result filtering
  min_score: 0.5  # Minimum similarity score
  max_results: 100
  deduplicate: true
```

## Embedding Models

### Supported Models

Talk2Metadata supports any Sentence-Transformers model:

**Recommended Models**:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ | Default, fast |
| `all-mpnet-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐⭐ | Better quality |
| `e5-large-v2` | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | State-of-the-art |
| `bge-large-en-v1.5` | 1024 | ⚡ | ⭐⭐⭐⭐⭐ | Multilingual |

### Custom Models

```python
from sentence_transformers import SentenceTransformer

# Load custom model
custom_model = SentenceTransformer("your-model-name")

# Use in indexer
indexer = Indexer(
    schema=schema,
    embedding_model=custom_model
)
```

### Fine-Tuning

For domain-specific data, consider fine-tuning:

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Prepare training data
train_examples = [
    InputExample(
        texts=["Find Healthcare orders", "## Related customers\nindustry: Healthcare"],
        label=1.0
    ),
    InputExample(
        texts=["Find Software products", "## Related customers\nindustry: Healthcare"],
        label=0.3
    ),
]

# Fine-tune
model = SentenceTransformer("all-MiniLM-L6-v2")
train_dataloader = DataLoader(train_examples, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3
)
```

## Performance Optimization

### Indexing Performance

**Large Datasets**:
```python
# Batch processing
indexer = Indexer(
    schema=schema,
    batch_size=1000,  # Process 1000 records at a time
    num_workers=8,    # Parallel processing
    cache_embeddings=True
)

# Incremental indexing
indexer.index_incremental(new_records)  # Only index new data
```

**Memory Management**:
```python
# For very large datasets, use disk-based index
indexer = Indexer(
    schema=schema,
    index_type="IVF",  # Inverted File Index (faster for large scale)
    use_disk_index=True,
    disk_path="/path/to/index"
)
```

### Search Performance

**Fast Search**:
```python
# Pre-load index
retriever = Retriever.load("orders_index", preload=True)

# Use approximate search for speed
results = retriever.search(
    query="Healthcare orders",
    approximate=True,  # Faster but slightly less accurate
    top_k=10
)
```

**Batch Search**:
```python
# Search multiple queries at once
queries = [
    "Healthcare customers",
    "Software products",
    "High-value orders"
]

batch_results = retriever.search_batch(queries, top_k=10)
```

## Evaluation

### Metrics

Evaluate retrieval quality using QA pairs:

```python
from talk2metadata.evaluation import RetrievalEvaluator

# Load QA dataset
qa_pairs = load_qa_dataset("qa_dataset.jsonl")

# Evaluate
evaluator = RetrievalEvaluator(retriever)
metrics = evaluator.evaluate(qa_pairs)

print(f"Recall@10: {metrics['recall@10']:.3f}")
print(f"MRR: {metrics['mrr']:.3f}")
print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
```

**Key Metrics**:

- **Recall@k**: Percentage of ground truth records in top-k results
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first correct result
- **NDCG@k**: Normalized Discounted Cumulative Gain (ranking quality)
- **Precision@k**: Percentage of top-k results that are correct

### A/B Testing

Compare different retrieval strategies:

```python
# Strategy A: Semantic only
retriever_a = Retriever.load("index", strategy="semantic")

# Strategy B: Hybrid
retriever_b = Retriever.load("index", strategy="hybrid")

# Evaluate both
metrics_a = evaluator.evaluate(qa_pairs, retriever_a)
metrics_b = evaluator.evaluate(qa_pairs, retriever_b)

# Compare
print(f"Semantic Recall@10: {metrics_a['recall@10']:.3f}")
print(f"Hybrid Recall@10: {metrics_b['recall@10']:.3f}")
```

## Advanced Features

### Re-Ranking

Apply secondary ranking after initial retrieval:

```python
from talk2metadata.retriever import CrossEncoderReranker

# Initial retrieval
results = retriever.search(query, top_k=100)

# Re-rank top 100 using cross-encoder
reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-MiniLM-L6-v2")
reranked = reranker.rerank(query, results, top_k=10)
```

### Query Expansion

Expand queries for better recall:

```python
from talk2metadata.retriever import QueryExpander

expander = QueryExpander(method="synonym")
expanded_query = expander.expand("Healthcare orders")
# Output: "Healthcare medical orders transactions"

results = retriever.search(expanded_query, top_k=10)
```

### Filter-Based Retrieval

Combine semantic search with structured filters:

```python
results = retriever.search(
    query="High-value orders",
    filters={
        "amount": {"$gt": 10000},
        "status": {"$eq": "completed"}
    },
    top_k=10
)
```

### Multi-Hop Retrieval

Retrieve across multiple tables:

```python
# First hop: Find customers
customers = retriever.search(
    query="Healthcare companies",
    target_table="customers",
    top_k=10
)

# Second hop: Find their orders
orders = retriever.search_related(
    source_records=customers,
    target_table="orders",
    relationship="customer_id"
)
```

## Best Practices

### 1. Choose the Right Strategy

- **Semantic search**: Best for conceptual queries, paraphrasing
- **Keyword search**: Best for exact matches, IDs, codes
- **Hybrid**: Best for general-purpose, production use

### 2. Tune Hyperparameters

```python
# Grid search over parameters
from talk2metadata.tuning import HyperparameterTuner

tuner = HyperparameterTuner(retriever, qa_pairs)
best_params = tuner.search(
    param_grid={
        "semantic_weight": [0.5, 0.6, 0.7, 0.8],
        "min_score": [0.3, 0.4, 0.5],
    },
    metric="recall@10"
)
```

### 3. Monitor Performance

```python
# Track retrieval metrics over time
from talk2metadata.monitoring import MetricsTracker

tracker = MetricsTracker()
tracker.log_search(query, results, ground_truth)

# Analyze
stats = tracker.get_statistics(time_window="7d")
print(f"Average Recall@10: {stats['avg_recall@10']:.3f}")
```

### 4. Handle Updates

```python
# Incremental updates (for new records)
indexer.add_records(new_records)

# Full re-index (for schema changes)
indexer.reindex()
```

## Troubleshooting

### Common Issues

**Issue**: Low recall (missing relevant results)
- **Solutions**:
  - Use hybrid search instead of semantic-only
  - Increase `top_k`
  - Check if denormalization includes necessary columns
  - Try a better embedding model

**Issue**: Slow search performance
- **Solutions**:
  - Use approximate search (`approximate=True`)
  - Reduce embedding dimensions
  - Use faster index type (IVF instead of Flat)
  - Enable GPU acceleration

**Issue**: Poor quality results
- **Solutions**:
  - Fine-tune embedding model on your domain
  - Improve denormalization (include more context)
  - Add re-ranking step
  - Filter out low-confidence results

**Issue**: Index is too large
- **Solutions**:
  - Use smaller embedding model
  - Enable compression (`use_compression=True`)
  - Sample records (for very large tables)

## Integration with QA

The Retriever is designed to work with QA-generated datasets:

```python
# Generate QA pairs
qa_pairs = qa_generator.generate_batch(distribution)

# Index records
indexer.index()

# Evaluate retrieval
metrics = evaluator.evaluate(qa_pairs)

# Iterate and improve
if metrics['recall@10'] < 0.8:
    # Try different strategy or model
    retriever.update_strategy("hybrid")
    metrics = evaluator.evaluate(qa_pairs)
```

## Future Enhancements

- [ ] Multi-modal retrieval (text + images)
- [ ] Learned sparse retrieval (SPLADE)
- [ ] Neural re-rankers (ColBERT)
- [ ] Query understanding (entity extraction, intent classification)
- [ ] Federated search (across multiple databases)
- [ ] Real-time index updates
- [ ] Explainability (why this record was retrieved?)

## Related Documentation

- **[Schema Detection](../schema/index.md)** - Understand how schemas are detected for indexing
- **[QA Generation](../qa/index.md)** - Generate evaluation datasets
- **[Getting Started](../getting-started/quickstart.md)** - Quick setup guide

## API Reference

See `src/talk2metadata/core/indexer.py` and `src/talk2metadata/core/retriever.py` for detailed API documentation.
