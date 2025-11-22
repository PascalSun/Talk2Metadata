#!/bin/bash
# Setup script for Unix-based systems (macOS, Linux, WSL)
# Usage: ./setup.sh [--mcp] [--full]
# Options:
#   --mcp   Install with MCP server support
#   --full  Install all features (API server, MCP, hybrid search)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo_step() {
    echo -e "${BLUE}==>${NC} $1"
}

# Parse arguments
INSTALL_MCP=""
INSTALL_FULL=""

for arg in "$@"; do
    case $arg in
        --mcp)
            INSTALL_MCP="true"
            ;;
        --full)
            INSTALL_FULL="true"
            ;;
        --help|-h)
            echo "Usage: ./setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mcp     Install with MCP server support"
            echo "  --full    Install all features (API, MCP, hybrid search)"
            echo "  --help    Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./setup.sh              # Basic installation"
            echo "  ./setup.sh --mcp        # With MCP server"
            echo "  ./setup.sh --full       # Everything"
            exit 0
            ;;
        *)
            echo_error "Unknown argument: $arg"
            echo_info "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo_step "Starting Talk2Metadata development environment setup..."
echo ""

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo_warn "uv is not installed. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            echo_error "Failed to install uv. Please install manually:"
            echo_info "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        echo_info "uv installed successfully!"
    else
        echo_info "✓ uv is already installed"
    fi
}

# Check Python version
check_python_version() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            echo_info "✓ Python $PYTHON_VERSION detected (3.11+ supported)"
            return 0
        else
            echo_error "Python 3.11+ required, but found $PYTHON_VERSION"
            echo_info "Please install Python 3.11 or higher"
            return 1
        fi
    else
        echo_error "Python 3 not found. Please install Python 3.11 or higher."
        return 1
    fi
}

# Create necessary directories
create_directories() {
    echo_info "Creating project directories..."
    mkdir -p data/{raw,processed,indexes}
    mkdir -p logs
    mkdir -p examples
    echo_info "✓ Directories created"
}

# Copy configuration files
setup_config() {
    echo_info "Setting up configuration files..."

    # Main config
    if [ ! -f "config.yml" ]; then
        if [ -f "config.example.yml" ]; then
            cp config.example.yml config.yml
            echo_info "✓ Created config.yml from example"
        fi
    else
        echo_warn "config.yml already exists, skipping..."
    fi

    # MCP config
    if [ -f "config.mcp.example.yml" ] && [ ! -f "config.mcp.yml" ]; then
        cp config.mcp.example.yml config.mcp.yml
        echo_info "✓ Created config.mcp.yml from example"
        echo_warn "  Remember to update OAuth credentials in config.mcp.yml"
    fi
}

# Determine installation extras
determine_extras() {
    local extras="dev"

    if [ "$INSTALL_FULL" = "true" ]; then
        extras="full,mcp,agent,dev"
        echo_info "Installing with all features" >&2
    elif [ "$INSTALL_MCP" = "true" ]; then
        extras="mcp,full,dev"
        echo_info "Installing with MCP server support" >&2
    else
        # Ask user if they want MCP
        read -p "Install with MCP server support? (Y/n): " -n 1 -r </dev/tty
        echo >&2
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            extras="mcp,full,dev"
            echo_info "Installing with MCP server support" >&2
        else
            extras="full,dev"
            echo_info "Installing basic version" >&2
        fi
    fi

    echo "$extras"
}

# Main setup
setup_environment() {
    echo_step "Step 1: Checking Python version..."
    if ! check_python_version; then
        exit 1
    fi
    echo ""

    echo_step "Step 2: Checking uv package manager..."
    check_uv
    echo ""

    echo_step "Step 3: Creating virtual environment..."
    if [ ! -d ".venv" ]; then
        echo_info "Creating virtual environment with uv..."
        uv venv
    else
        echo_warn "Virtual environment already exists, skipping..."
    fi
    echo ""

    echo_step "Step 4: Installing Talk2Metadata..."
    source .venv/bin/activate

    EXTRAS=$(determine_extras)
    echo_info "Installing with extras: ${EXTRAS}"
    uv pip install -e ".[${EXTRAS}]"
    echo ""

    echo_step "Step 5: Creating project structure..."
    create_directories
    echo ""

    echo_step "Step 6: Setting up configuration..."
    setup_config
    echo ""
}

# Run setup
setup_environment

# Verify installation
echo_step "Step 7: Verifying installation..."
echo ""

if python -c "import talk2metadata" 2>/dev/null; then
    VERSION=$(python -c "import talk2metadata; print(talk2metadata.__version__)")
    echo_info "✓ Talk2Metadata v$VERSION successfully installed!"
else
    echo_warn "Package import test failed. Please check the installation."
fi

# Check if MCP is available
if python -c "import talk2metadata.mcp" 2>/dev/null; then
    echo_info "✓ MCP server support available"
fi

echo ""
echo_step "=========================================="
echo_step "Setup complete! 🎉"
echo_step "=========================================="
echo ""
echo_info "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo_info "Quick start commands:"
echo "  talk2metadata --help              # Show all commands"
echo "  talk2metadata ingest csv <path>   # Ingest CSV files"
echo "  talk2metadata index --hybrid      # Build search index"
echo "  talk2metadata search \"query\"      # Search records"
echo ""

# Show MCP instructions if installed
if python -c "import talk2metadata.mcp" 2>/dev/null; then
    echo_info "MCP Server commands:"
    echo "  talk2metadata-mcp sse             # Start MCP server"
    echo "  talk2metadata-mcp sse --port 8010 # Custom port"
    echo ""
    echo_info "Next steps for MCP:"
    echo "  1. Edit config.mcp.yml with your OAuth credentials"
    echo "  2. Ingest and index your data (see commands above)"
    echo "  3. Start the MCP server"
    echo "  4. See docs/mcp-quickstart.md for details"
    echo ""
fi

echo_info "Documentation:"
echo "  README.md                    # Main documentation"
echo "  docs/mcp-quickstart.md       # MCP quick start"
echo "  docs/mcp-integration.md      # MCP integration guide"
echo ""
echo_info "Run tests:"
echo "  pytest tests/"
echo ""
echo_step "=========================================="
