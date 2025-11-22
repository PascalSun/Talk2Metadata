"""Python API usage examples for Talk2Metadata."""

from pathlib import Path

from talk2metadata import Retriever, SchemaMetadata

# Example 1: Load and search with Retriever
def example_basic_search():
    """Basic search example."""
    print("Example 1: Basic Search")
    print("=" * 60)

    # Load retriever
    retriever = Retriever.from_paths(
        index_path="data/indexes/index.faiss",
        records_path="data/indexes/records.pkl",
    )

    # Perform search
    results = retriever.search("healthcare customers", top_k=5)

    # Process results
    for result in results:
        print(f"\nRank {result.rank}:")
        print(f"  Table: {result.table}")
        print(f"  Row ID: {result.row_id}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Data: {result.data}")


# Example 2: Batch search
def example_batch_search():
    """Batch search example."""
    print("\n\nExample 2: Batch Search")
    print("=" * 60)

    retriever = Retriever.from_paths(
        "data/indexes/index.faiss",
        "data/indexes/records.pkl",
    )

    queries = [
        "healthcare industry",
        "technology companies",
        "large orders",
    ]

    results_list = retriever.search_batch(queries, top_k=3)

    for query, results in zip(queries, results_list):
        print(f"\nQuery: '{query}'")
        print(f"Top result: {results[0].data if results else 'No results'}")


# Example 3: Access schema metadata
def example_schema_access():
    """Schema metadata access example."""
    print("\n\nExample 3: Schema Access")
    print("=" * 60)

    # Load schema
    metadata = SchemaMetadata.load("data/metadata/schema.json")

    print(f"Target table: {metadata.target_table}")
    print(f"Number of tables: {len(metadata.tables)}")

    # Get related tables
    related = metadata.get_related_tables("orders")
    print(f"Tables related to 'orders': {related}")

    # Get foreign keys
    fks = metadata.get_foreign_keys_for_table("orders", direction="outgoing")
    print(f"\nForeign keys from 'orders':")
    for fk in fks:
        print(f"  {fk.child_column} -> {fk.parent_table}.{fk.parent_column}")


# Example 4: Filter and process results
def example_filter_results():
    """Filter and process results example."""
    print("\n\nExample 4: Filter and Process Results")
    print("=" * 60)

    retriever = Retriever.from_paths(
        "data/indexes/index.faiss",
        "data/indexes/records.pkl",
    )

    results = retriever.search("orders", top_k=20)

    # Filter by status
    completed_orders = [
        r for r in results
        if r.data.get("status") == "completed"
    ]
    print(f"Completed orders: {len(completed_orders)}")

    # Filter by amount
    large_orders = [
        r for r in results
        if r.data.get("amount", 0) > 50000
    ]
    print(f"Large orders (>$50k): {len(large_orders)}")

    # Calculate total
    total_amount = sum(r.data.get("amount", 0) for r in results)
    print(f"Total amount: ${total_amount:,}")


# Example 5: Get retriever statistics
def example_stats():
    """Retriever statistics example."""
    print("\n\nExample 5: Retriever Statistics")
    print("=" * 60)

    retriever = Retriever.from_paths(
        "data/indexes/index.faiss",
        "data/indexes/records.pkl",
    )

    stats = retriever.get_stats()

    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get a specific record
    record = retriever.get_record_by_id(1001)
    if record:
        print(f"\nRecord 1001: {record}")


# Example 6: Custom embedding model
def example_custom_model():
    """Custom embedding model example."""
    print("\n\nExample 6: Custom Embedding Model")
    print("=" * 60)

    # Note: This assumes you've built an index with a custom model
    # See indexer example for building with custom models

    retriever = Retriever.from_paths(
        "data/indexes/index.faiss",
        "data/indexes/records.pkl",
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    results = retriever.search("healthcare", top_k=3)
    print(f"Found {len(results)} results using custom model")


def main():
    """Run all examples."""
    try:
        example_basic_search()
        example_batch_search()
        example_schema_access()
        example_filter_results()
        example_stats()

        # Uncomment if you have a custom model index
        # example_custom_model()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you've run:")
        print("1. talk2metadata ingest csv data/raw --target orders")
        print("2. talk2metadata index")


if __name__ == "__main__":
    main()
