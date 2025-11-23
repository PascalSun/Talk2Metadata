#!/bin/bash
set -e

echo "================================"
echo "Running Format & Tests"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section header
print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    echo ""
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
    echo ""
}

# Target directories
TARGET_DIRS="src tests"

# ================================
# FORMATTING & LINTING
# ================================

print_header "STEP 1: Code Formatting & Linting"

# Run isort
echo -e "${YELLOW}>> Running isort (import sorting)...${NC}"
if uv run isort $TARGET_DIRS; then
    print_success "isort completed successfully"
else
    print_error "isort found issues"
    exit 1
fi

# Run black
echo -e "${YELLOW}>> Running black (code formatting)...${NC}"
if uv run black $TARGET_DIRS; then
    print_success "black completed successfully"
else
    print_error "black found issues"
    exit 1
fi

# Run flake8
echo -e "${YELLOW}>> Running flake8 (linting)...${NC}"
if uv run flake8 $TARGET_DIRS; then
    print_success "flake8 completed successfully"
else
    print_error "flake8 found issues"
    exit 1
fi

# ================================
# TESTS
# ================================

print_header "STEP 2: Running Tests"

if uv run pytest; then
    print_success "All tests passed"
else
    print_error "Tests failed"
    exit 1
fi

# ================================
# INTEGRATION TESTS
# ================================

print_header "STEP 3: Integration Tests"

# Check if data directory exists
if [ -d "data/raw" ]; then
    echo -e "${YELLOW}>> Cleaning previous data...${NC}"
    rm -rf data/metadata/schema.json data/processed/tables.pkl data/indexes/*.faiss data/indexes/*.pkl
    print_success "Cleaned previous data"

    echo -e "${YELLOW}>> Ingesting sample data...${NC}"
    if uv run talk2metadata ingest csv data/raw --target orders; then
        print_success "Ingestion complete"
    else
        print_error "Ingestion failed"
        exit 1
    fi

    echo -e "${YELLOW}>> Building search index...${NC}"
    if uv run talk2metadata index; then
        print_success "Index built"
    else
        print_error "Indexing failed"
        exit 1
    fi

    echo -e "${YELLOW}>> Testing searches...${NC}"
    echo ""
    echo "Query 1: 'healthcare customers'"
    uv run talk2metadata search "healthcare customers" --top-k 3
    echo ""

    echo "Query 2: 'technology companies'"
    uv run talk2metadata search "technology companies" --top-k 3
    echo ""

    echo "Query 3: 'pending orders' (JSON)"
    uv run talk2metadata search "pending orders" --format json | head -20
    echo ""

    print_success "Integration tests passed"
else
    echo -e "${YELLOW}⚠ Skipping integration tests (data/raw directory not found)${NC}"
    echo ""
fi

# ================================
# SUCCESS
# ================================

echo ""
echo "================================"
echo -e "${GREEN}✓ All checks passed!${NC}"
echo "================================"
echo ""
echo "Summary:"
echo "  ✓ Code formatting (isort, black)"
echo "  ✓ Linting (flake8)"
echo "  ✓ Unit tests (pytest)"
if [ -d "data/raw" ]; then
    echo "  ✓ Integration tests (ingest, index, search)"
else
    echo "  ⚠ Integration tests skipped"
fi
