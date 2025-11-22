# Sample Data

This directory contains sample CSV files for testing Talk2Metadata.

## Schema

```
customers (id, name, industry, region, annual_revenue, created_date)
    ↑
    |
orders (id, customer_id, product_id, amount, quantity, order_date, status, sales_rep)
    |
    ↓
products (id, name, category, price, description)
```

## Foreign Keys

- `orders.customer_id` → `customers.id`
- `orders.product_id` → `products.id`

## Target Table

The target table is **orders** - it contains the main records we want to search.

## Sample Queries

Try these natural language queries:

- "healthcare customers with high revenue"
- "orders from technology companies"
- "recent orders in US-West region"
- "completed orders for enterprise software"
- "customers who bought consulting services"
- "pending orders for machine learning products"

## Quick Start

```bash
# 1. Ingest the data
talk2metadata ingest csv data/raw --target orders

# 2. Build index
talk2metadata index

# 3. Search
talk2metadata search "healthcare customers with high revenue"
```
