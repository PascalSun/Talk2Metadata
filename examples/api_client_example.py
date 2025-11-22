"""API client example for Talk2Metadata REST API.

This script demonstrates how to interact with the Talk2Metadata API server.

Prerequisites:
  1. Start the server: uv run talk2metadata serve
  2. Ensure index is built: uv run talk2metadata index
"""

import json
from typing import Dict, List

import requests


class Talk2MetadataClient:
    """Simple client for Talk2Metadata API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> Dict:
        """Check API health.

        Returns:
            Health status dict
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def search(self, query: str, top_k: int = 5) -> Dict:
        """Search for records.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            Search response dict with results
        """
        response = requests.post(
            f"{self.base_url}/api/v1/search",
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        return response.json()

    def get_schema(self) -> Dict:
        """Get schema information.

        Returns:
            Schema metadata dict
        """
        response = requests.get(f"{self.base_url}/api/v1/schema")
        response.raise_for_status()
        return response.json()

    def list_tables(self) -> Dict:
        """List all tables.

        Returns:
            Tables list dict
        """
        response = requests.get(f"{self.base_url}/api/v1/schema/tables")
        response.raise_for_status()
        return response.json()

    def search_status(self) -> Dict:
        """Get search service status.

        Returns:
            Search status dict
        """
        response = requests.get(f"{self.base_url}/api/v1/search/status")
        response.raise_for_status()
        return response.json()


def main():
    """Run API client examples."""
    print("=" * 80)
    print("Talk2Metadata API Client Examples")
    print("=" * 80)

    # Initialize client
    client = Talk2MetadataClient()

    # Example 1: Health check
    print("\n[Example 1] Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(json.dumps(health, indent=2))

        if not health.get("index_loaded"):
            print("\n⚠️  Warning: Index not loaded!")
            print("Please run: uv run talk2metadata index")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("Please start the server: uv run talk2metadata serve")
        return

    # Example 2: Search status
    print("\n[Example 2] Search Status")
    print("-" * 60)
    status = client.search_status()
    print(json.dumps(status, indent=2))

    # Example 3: List tables
    print("\n[Example 3] List Tables")
    print("-" * 60)
    tables = client.list_tables()
    print(f"Target table: {tables['target_table']}")
    print(f"All tables ({len(tables['tables'])}):")
    for table in tables["tables"]:
        print(f"  • {table['name']}: {table['row_count']} rows, "
              f"{table['column_count']} columns")

    # Example 4: Get full schema
    print("\n[Example 4] Get Schema")
    print("-" * 60)
    schema = client.get_schema()
    print(f"Target table: {schema['target_table']}")
    print(f"Foreign keys ({len(schema['foreign_keys'])}):")
    for fk in schema["foreign_keys"]:
        print(f"  • {fk['child_table']}.{fk['child_column']} → "
              f"{fk['parent_table']}.{fk['parent_column']} "
              f"(coverage: {fk['coverage']:.1%})")

    # Example 5: Simple search
    print("\n[Example 5] Simple Search")
    print("-" * 60)
    query = "healthcare customers with high revenue"
    print(f"Query: '{query}'")

    results = client.search(query, top_k=3)
    print(f"\nFound {len(results['results'])} results:")

    for result in results["results"]:
        print(f"\n  Rank #{result['rank']} (score: {result['score']:.4f})")
        print(f"  Table: {result['table']}, Row ID: {result['row_id']}")
        data = result["data"]
        if "customer_id" in data:
            print(f"  Customer ID: {data['customer_id']}")
        if "amount" in data:
            print(f"  Amount: ${data['amount']:,}")
        if "status" in data:
            print(f"  Status: {data['status']}")

    # Example 6: Multiple queries
    print("\n[Example 6] Multiple Queries")
    print("-" * 60)
    queries = [
        "technology companies",
        "pending orders",
        "machine learning products",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = client.search(query, top_k=2)

        if results["results"]:
            top_result = results["results"][0]
            print(f"  Top result: Row {top_result['row_id']} "
                  f"(score: {top_result['score']:.4f})")
        else:
            print("  No results found")

    # Example 7: Filter results by status
    print("\n[Example 7] Filter by Status")
    print("-" * 60)
    results = client.search("orders", top_k=20)

    status_counts = {}
    for result in results["results"]:
        status = result["data"].get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print("Order status distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Example 8: Calculate aggregates
    print("\n[Example 8] Calculate Aggregates")
    print("-" * 60)
    results = client.search("all orders", top_k=50)

    amounts = [r["data"].get("amount", 0) for r in results["results"]]
    if amounts:
        print(f"Total amount: ${sum(amounts):,}")
        print(f"Average amount: ${sum(amounts) / len(amounts):,.2f}")
        print(f"Max amount: ${max(amounts):,}")
        print(f"Min amount: ${min(amounts):,}")

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure server is running: uv run talk2metadata serve")
        print("2. Ensure index is built: uv run talk2metadata index")
        print("3. Check server logs for errors")
