@echo off
REM Setup script for Windows
REM Usage: setup.bat [--mcp]
REM Options:
REM   --mcp   Install with MCP server support

setlocal enabledelayedexpansion

SET INSTALL_MCP=

REM Parse arguments
:parse_args
IF "%~1"=="" GOTO args_done
IF /I "%~1"=="--mcp" (
    SET INSTALL_MCP=true
    SHIFT
    GOTO parse_args
)
IF /I "%~1"=="--help" GOTO show_help
IF /I "%~1"=="-h" GOTO show_help
echo [ERROR] Unknown argument: %~1
echo [INFO] Use --help for usage information
exit /b 1

:show_help
echo Usage: setup.bat [OPTIONS]
echo.
echo Options:
echo   --mcp     Install with MCP server support
echo   --help    Show this help message
echo.
echo Examples:
echo   setup.bat              # Basic installation
echo   setup.bat --mcp        # With MCP server
exit /b 0

:args_done

echo.
echo ==^> Starting Talk2Metadata development environment setup...
echo.

REM Check if uv is installed
where uv >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [WARN] uv is not installed. Installing uv...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>nul
    IF !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install uv. Please install manually:
        echo [INFO]   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        exit /b 1
    )
    echo [INFO] uv installed successfully!
) ELSE (
    echo [INFO] uv is already installed
)

REM Check if Python is installed
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Please install Python 3.11 or higher.
    echo [INFO] Visit: https://www.python.org/downloads/
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Python %PYTHON_VERSION% detected
echo.

REM Step 1: Create virtual environment
echo ==^> Step 1: Creating virtual environment...
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment with uv...
    uv venv
) ELSE (
    echo [WARN] Virtual environment already exists, skipping...
)
echo.

REM Step 2: Determine installation extras
echo ==^> Step 2: Determining installation options...
SET "EXTRAS=dev"

IF "!INSTALL_MCP!"=="true" (
    SET "EXTRAS=mcp,dev"
    echo [INFO] Installing with MCP server support
) ELSE (
    REM Ask user
    set /p "MCP_CHOICE=Install with MCP server support? (Y/n): "
    IF /I "!MCP_CHOICE!"=="" SET "MCP_CHOICE=Y"
    IF /I "!MCP_CHOICE!"=="Y" (
        SET "EXTRAS=mcp,dev"
        echo [INFO] Installing with MCP server support
    ) ELSE (
        SET "EXTRAS=dev"
        echo [INFO] Installing basic version
    )
)
echo.

REM Step 3: Activate and install
echo ==^> Step 3: Installing Talk2Metadata...
call .venv\Scripts\activate.bat
echo [INFO] Installing with extras: !EXTRAS!
uv pip install -e ".[!EXTRAS!]"
echo.

REM Step 4: Create directories
echo ==^> Step 4: Creating project structure...
IF NOT EXIST "data\raw" mkdir data\raw
IF NOT EXIST "data\processed" mkdir data\processed
IF NOT EXIST "data\indexes" mkdir data\indexes
IF NOT EXIST "logs" mkdir logs
IF NOT EXIST "examples" mkdir examples
echo [INFO] Directories created
echo.

REM Step 5: Setup configuration
echo ==^> Step 5: Setting up configuration files...

IF NOT EXIST "config.yml" (
    IF EXIST "config.example.yml" (
        copy config.example.yml config.yml >nul
        echo [INFO] Created config.yml from example
    )
) ELSE (
    echo [WARN] config.yml already exists, skipping...
)

IF EXIST "config.mcp.example.yml" (
    IF NOT EXIST "config.mcp.yml" (
        copy config.mcp.example.yml config.mcp.yml >nul
        echo [INFO] Created config.mcp.yml from example
        echo [WARN]   Remember to update OAuth credentials in config.mcp.yml
    )
)
echo.

REM Step 6: Verify installation
echo ==^> Step 6: Verifying installation...
echo.

python -c "import talk2metadata" 2>nul
IF %ERRORLEVEL% EQU 0 (
    for /f "delims=" %%i in ('python -c "import talk2metadata; print(talk2metadata.__version__)"') do set VERSION=%%i
    echo [INFO] Talk2Metadata v!VERSION! successfully installed!
) ELSE (
    echo [WARN] Package import test failed. Please check the installation.
)

REM Check if MCP is available
python -c "import talk2metadata.mcp" 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo [INFO] MCP server support available
    SET HAS_MCP=true
)

echo.
echo ==========================================
echo Setup complete! 🎉
echo ==========================================
echo.
echo [INFO] To activate the environment:
echo   .venv\Scripts\activate.bat
echo.
echo [INFO] Quick start commands:
echo   talk2metadata --help              # Show all commands
echo   talk2metadata ingest csv ^<path^>   # Ingest CSV files
echo   talk2metadata index --hybrid      # Build search index
echo   talk2metadata search "query"      # Search records
echo.

IF "!HAS_MCP!"=="true" (
    echo [INFO] MCP Server commands:
    echo   talk2metadata-mcp sse             # Start MCP server
    echo   talk2metadata-mcp sse --port 8010 # Custom port
    echo.
    echo [INFO] Next steps for MCP:
    echo   1. Edit config.mcp.yml with your OAuth credentials
    echo   2. Ingest and index your data (see commands above^)
    echo   3. Start the MCP server
    echo   4. See docs\mcp-quickstart.md for details
    echo.
)

echo [INFO] Documentation:
echo   README.md                    # Main documentation
echo   docs\mcp-quickstart.md       # MCP quick start
echo   docs\mcp-integration.md      # MCP integration guide
echo.
echo [INFO] Run tests:
echo   pytest tests\
echo.
echo ==========================================

endlocal
