# AI IDE Build Script for Windows
# Sets up VSCodium development environment and builds extensions

param(
    [switch]$SkipVSCodium,
    [switch]$SkipExtension,
    [switch]$SkipBackend
)

Write-Host "🚀 Building AI IDE..." -ForegroundColor Green

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if we're in the right directory
if (-not (Test-Path "vscodium")) {
    Write-Error "VSCodium directory not found. Please run this script from the ai-ide directory."
    exit 1
}

# Step 1: Build VSCodium (if needed)
if (-not $SkipVSCodium) {
    Write-Status "Checking VSCodium build status..."
    if (-not (Test-Path "vscodium/out")) {
        Write-Status "VSCodium needs to be built. Please follow VSCodium build instructions manually."
        Write-Warning "Building VSCodium from source requires specific setup. Skipping for now."
    } else {
        Write-Status "VSCodium already built, continuing..."
    }
}

# Step 2: Build AI Assistant Extension
if (-not $SkipExtension) {
    Write-Status "Building AI Assistant extension..."
    Set-Location "extensions/ai-assistant"

    # Install dependencies
    if (-not (Test-Path "node_modules")) {
        Write-Status "Installing extension dependencies..."
        npm install
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install npm dependencies"
            Set-Location "../.."
            exit 1
        }
    }

    # Compile TypeScript
    Write-Status "Compiling TypeScript..."
    npm run compile
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to compile TypeScript"
        Set-Location "../.."
        exit 1
    }

    # Package extension (optional)
    if (Get-Command vsce -ErrorAction SilentlyContinue) {
        Write-Status "Packaging extension..."
        vsce package
    } else {
        Write-Warning "vsce not found, skipping extension packaging"
    }

    Set-Location "../.."
}

# Step 3: Set up Python backend
if (-not $SkipBackend) {
    Write-Status "Setting up Python backend..."
    Set-Location "backend"

    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Status "Creating Python virtual environment..."
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create Python virtual environment"
            Set-Location ".."
            exit 1
        }
    }

    # Activate virtual environment and install dependencies
    Write-Status "Installing Python dependencies..."
    & "venv/Scripts/Activate.ps1"
    
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    } else {
        Write-Warning "requirements.txt not found, skipping Python dependencies"
    }

    Set-Location ".."
}

# Step 4: Create launch configuration
Write-Status "Creating launch configuration..."
if (-not (Test-Path ".vscode")) {
    New-Item -ItemType Directory -Path ".vscode" | Out-Null
}

$launchConfig = @'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch AI IDE Extension",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}/extensions/ai-assistant"
            ],
            "outFiles": [
                "${workspaceFolder}/extensions/ai-assistant/out/**/*.js"
            ],
            "preLaunchTask": "npm: compile - extensions/ai-assistant"
        }
    ]
}
'@

$launchConfig | Out-File -FilePath ".vscode/launch.json" -Encoding UTF8

# Step 5: Create tasks configuration
$tasksConfig = @'
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "npm",
            "script": "compile",
            "path": "extensions/ai-assistant/",
            "group": "build",
            "presentation": {
                "panel": "shared"
            },
            "problemMatcher": "$tsc"
        },
        {
            "type": "npm",
            "script": "watch",
            "path": "extensions/ai-assistant/",
            "group": "build",
            "presentation": {
                "panel": "shared"
            },
            "problemMatcher": "$tsc-watch"
        }
    ]
}
'@

$tasksConfig | Out-File -FilePath ".vscode/tasks.json" -Encoding UTF8

Write-Status "✅ AI IDE build completed successfully!" 
Write-Status ""
Write-Status "Next steps:"
Write-Status "1. Open this project in VS Code"
Write-Status "2. Press F5 to launch the extension development host"
Write-Status "3. Test the AI Assistant extension"
Write-Status ""
Write-Status "Extension location: extensions/ai-assistant"
Write-Status "Backend service: backend/main.py"