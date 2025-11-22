#!/bin/bash
# Test workflow script for Talk2Metadata

set -e

echo "============================================"
echo "Talk2Metadata - Test Workflow"
echo "============================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check uv is installed
if ! command -v uv &> /dev/null; then
    error "uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

info "Step 1: Cleaning previous data..."
rm -rf data/metadata/schema.json data/processed/tables.pkl data/indexes/*.faiss data/indexes/*.pkl
info "✓ Cleaned"

info "Step 2: Ingesting sample data..."
uv run talk2metadata ingest csv data/raw --target orders
if [ $? -ne 0 ]; then
    error "Ingestion failed"
fi
info "✓ Ingestion complete"

info "Step 3: Building search index..."
uv run talk2metadata index
if [ $? -ne 0 ]; then
    error "Indexing failed"
fi
info "✓ Index built"

info "Step 4: Testing searches..."

# Test query 1
echo ""
echo "Query 1: 'healthcare customers'"
uv run talk2metadata search "healthcare customers" --top-k 3
echo ""

# Test query 2
echo "Query 2: 'technology companies'"
uv run talk2metadata search "technology companies" --top-k 3
echo ""

# Test JSON output
echo "Query 3: 'pending orders' (JSON)"
uv run talk2metadata search "pending orders" --format json | head -20
echo ""

info "✓ Searches complete"

echo ""
echo "============================================"
echo "All tests passed! ✓"
echo "============================================"
echo ""
echo "Next steps:"
echo "  • Start API server: uv run talk2metadata serve"
echo "  • Run Python examples: cd examples && uv run python complete_workflow.py"
echo "  • View docs: cd docs && mkdocs serve"
