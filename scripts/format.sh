#!/bin/bash
set -e

echo "================================"
echo "Running Code Formatters & Linters"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Target directories
TARGET_DIRS="src tests"

# Function to print section header
print_header() {
    echo -e "${YELLOW}>> $1${NC}"
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

# Run isort
print_header "Running isort (import sorting)..."
if uv run isort $TARGET_DIRS; then
    print_success "isort completed successfully"
else
    print_error "isort found issues"
    exit 1
fi

# Run black
print_header "Running black (code formatting)..."
if uv run black $TARGET_DIRS; then
    print_success "black completed successfully"
else
    print_error "black found issues"
    exit 1
fi

# Run flake8
print_header "Running flake8 (linting)..."
if uv run flake8 $TARGET_DIRS; then
    print_success "flake8 completed successfully"
else
    print_error "flake8 found issues"
    exit 1
fi

echo "================================"
echo -e "${GREEN}All checks passed!${NC}"
echo "================================"
