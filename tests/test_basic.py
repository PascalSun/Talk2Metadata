"""Basic tests for Talk2Metadata."""

import pandas as pd
import pytest

from talk2metadata.core.schema import SchemaDetector


def test_schema_detector():
    """Test schema detection."""
    # Create sample tables
    customers = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "industry": ["Healthcare", "Tech", "Finance"],
        }
    )

    orders = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "customer_id": [1, 2, 1],
            "amount": [1000, 2000, 1500],
        }
    )

    tables = {"customers": customers, "orders": orders}

    # Detect schema
    detector = SchemaDetector()
    metadata = detector.detect(tables, target_table="orders")

    # Assertions
    assert len(metadata.tables) == 2
    assert metadata.target_table == "orders"
    assert len(metadata.foreign_keys) >= 1

    # Check FK detection
    fk = metadata.foreign_keys[0]
    assert fk.child_table == "orders"
    assert fk.child_column == "customer_id"
    assert fk.parent_table == "customers"
    assert fk.parent_column == "id"
    assert fk.coverage == 1.0  # 100% coverage


def test_version():
    """Test version is set."""
    from talk2metadata import __version__

    assert __version__ == "0.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
