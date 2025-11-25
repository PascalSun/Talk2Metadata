# Installation

## Requirements

- Python 3.11 or 3.12
- 4GB RAM minimum

## Quick Setup (Recommended)

**Unix/macOS/Linux/WSL:**

```bash
# Clone repository
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Run setup script
./setup.sh          # Basic installation
./setup.sh --mcp    # With MCP server support

# Activate environment
source .venv/bin/activate
```

**Windows:**

```bash
# Clone repository
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Run setup script
setup.bat          # Basic installation
setup.bat --mcp    # With MCP server support

# Activate environment
.venv\Scripts\activate.bat
```

The setup script automatically:
- ✅ Checks Python version (3.11+)
- ✅ Installs uv package manager
- ✅ Creates virtual environment
- ✅ Installs Talk2Metadata with dependencies
- ✅ Creates project directories (data, logs, examples)
- ✅ Copies configuration templates

## Manual Installation

If you prefer manual installation:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/PascalSun/Talk2Metadata.git
cd Talk2Metadata

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat

# Install dependencies
uv sync
```

## Verify Installation

```bash
# Check CLI is available
talk2metadata --version

# Or with uv
uv run talk2metadata --version
```

## Next Steps

- [Quick Start Tutorial](quickstart.md)
