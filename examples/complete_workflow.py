"""Complete workflow example for Talk2Metadata.

This script demonstrates the full workflow:
1. Load data from connectors
2. Detect schema and foreign keys
3. Build search index
4. Perform searches
"""

from pathlib import Path

from talk2metadata.connectors import CSVLoader
from talk2metadata.core.indexer import Indexer
from talk2metadata.core.retriever import Retriever
from talk2metadata.core.schema import SchemaDetector

# Configuration
DATA_DIR = Path("../data/raw")
METADATA_DIR = Path("../data/metadata")
INDEX_DIR = Path("../data/indexes")
TARGET_TABLE = "orders"


def main():
    """Run complete workflow."""
    print("=" * 80)
    print("Talk2Metadata - Complete Workflow Example")
    print("=" * 80)

    # Step 1: Load data
    print("\n[Step 1] Loading data from CSV files...")
    loader = CSVLoader(
        data_dir=DATA_DIR,
        target_table=TARGET_TABLE,
    )
    tables = loader.load_tables()
    print(f"✓ Loaded {len(tables)} tables:")
    for name, df in tables.items():
        print(f"  - {name}: {len(df)} rows, {len(df.columns)} columns")

    # Step 2: Detect schema
    print("\n[Step 2] Detecting schema and foreign keys...")
    detector = SchemaDetector()
    metadata = detector.detect(tables, target_table=TARGET_TABLE)

    print(f"✓ Schema detection complete:")
    print(f"  - Target table: {metadata.target_table}")
    print(f"  - Tables: {len(metadata.tables)}")
    print(f"  - Foreign keys: {len(metadata.foreign_keys)}")

    if metadata.foreign_keys:
        print("\n  Detected foreign key relationships:")
        for fk in metadata.foreign_keys:
            print(f"    • {fk.child_table}.{fk.child_column} → "
                  f"{fk.parent_table}.{fk.parent_column} "
                  f"(coverage: {fk.coverage:.1%})")

    # Save metadata
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = METADATA_DIR / "schema.json"
    metadata.save(metadata_path)
    print(f"\n✓ Saved metadata to {metadata_path}")

    # Step 3: Build index
    print("\n[Step 3] Building search index...")
    indexer = Indexer()
    index, records = indexer.build_index(tables, metadata)

    print(f"✓ Index built:")
    print(f"  - Vectors: {index.ntotal}")
    print(f"  - Dimension: {index.d}")
    print(f"  - Records: {len(records)}")

    # Save index
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / "index.faiss"
    records_path = INDEX_DIR / "records.pkl"
    indexer.save_index(index, records, index_path, records_path)
    print(f"\n✓ Saved index to {INDEX_DIR}")

    # Step 4: Perform searches
    print("\n[Step 4] Performing searches...")
    retriever = Retriever(index, records)

    queries = [
        "healthcare customers with high revenue",
        "orders from technology companies",
        "machine learning and AI products",
        "pending orders",
        "completed sales in US-West region",
    ]

    for query in queries:
        print(f"\n{'─' * 80}")
        print(f"Query: \"{query}\"")
        print(f"{'─' * 80}")

        results = retriever.search(query, top_k=3)

        for result in results:
            print(f"\nRank #{result.rank} (score: {result.score:.4f})")
            print(f"Table: {result.table}, Row ID: {result.row_id}")

            # Show key fields
            data = result.data
            if 'customer_id' in data:
                print(f"  Customer ID: {data['customer_id']}")
            if 'amount' in data:
                print(f"  Amount: ${data['amount']:,}")
            if 'status' in data:
                print(f"  Status: {data['status']}")
            if 'order_date' in data:
                print(f"  Order Date: {data['order_date']}")

    print("\n" + "=" * 80)
    print("Workflow complete!")
    print("=" * 80)

    # Print statistics
    stats = retriever.get_stats()
    print(f"\nIndex Statistics:")
    print(f"  Total records: {stats['total_records']}")
    print(f"  Index size: {stats['index_size']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Model: {stats['model']}")


if __name__ == "__main__":
    main()
