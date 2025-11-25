# Quick Start

This guide will get you up and running with Talk2Metadata in 5 minutes.

## Step 1: Install Talk2Metadata

```bash
# Clone repository
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Run setup script
./setup.sh

# Activate environment
source .venv/bin/activate
```

The setup script will:
- Check Python version (3.11+)
- Install uv package manager if needed
- Create virtual environment
- Install Talk2Metadata and dependencies
- Create project directories
- Set up configuration files

## Step 2: Ingest Data

```bash
talk2metadata schema ingest csv data/raw --target orders
```

This command:
- Loads all CSV files from `data/raw/`
- Detects foreign key relationships automatically
- Marks `orders` as the target table
- Saves metadata to `data/metadata/schema.json`

## Step 3: Generate QA Pairs (Optional)

```bash
talk2metadata qa generate
```

This generates evaluation questions based on difficulty classification for testing retrieval strategies.

## Step 4: Build Search Index

```bash
talk2metadata search prepare
```

This command:
- Loads the target table (`orders`)
- Joins related tables using detected FKs
- Creates denormalized text for each row
- Generates embeddings using sentence-transformers
- Builds FAISS index for fast search
- Includes BM25 index for hybrid search

## Step 5: Evaluate Strategies (Optional)

```bash
talk2metadata search evaluate
```

This evaluates different retrieval strategies and finds the best solution.

## Step 6: Search for Records

```bash
# Find healthcare customers
talk2metadata search "orders from healthcare customers buying software"
```


