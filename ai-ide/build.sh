#!/bin/bash

# AI IDE Build Script
# Sets up VSCodium development environment and builds extensions

set -e

echo "🚀 Building AI IDE..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -d "vscodium" ]; then
    print_error "VSCodium directory not found. Please run this script from the ai-ide directory."
    exit 1
fi

# Step 1: Build VSCodium (if needed)
print_status "Checking VSCodium build status..."
if [ ! -d "vscodium/out" ]; then
    print_status "Building VSCodium from source..."
    cd vscodium
    
    # Check if we have the VS Code source
    if [ ! -d "vscode" ]; then
        print_status "Preparing VS Code source..."
        ./prepare_vscode.sh
    fi
    
    # Build VSCodium
    print_status "Building VSCodium..."
    ./build.sh
    
    cd ..
else
    print_status "VSCodium already built, skipping..."
fi

# Step 2: Build AI Assistant Extension
print_status "Building AI Assistant extension..."
cd extensions/ai-assistant

# Install dependencies
if [ ! -d "node_modules" ]; then
    print_status "Installing extension dependencies..."
    npm install
fi

# Compile TypeScript
print_status "Compiling TypeScript..."
npm run compile

# Package extension (optional)
if command -v vsce &> /dev/null; then
    print_status "Packaging extension..."
    vsce package
else
    print_warning "vsce not found, skipping extension packaging"
fi

cd ../..

# Step 3: Set up Python backend
print_status "Setting up Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt 2>/dev/null || print_warning "requirements.txt not found, skipping Python dependencies"

cd ..

# Step 4: Create launch configuration
print_status "Creating launch configuration..."
mkdir -p .vscode

cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Launch AI IDE",
            "type": "extensionHost",
            "request": "launch",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}/extensions/ai-assistant"
            ],
            "outFiles": [
                "${workspaceFolder}/extensions/ai-assistant/out/**/*.js"
            ],
            "preLaunchTask": "npm: compile - extensions/ai-assistant"
        },
        {
            "name": "Launch VSCodium with Extension",
            "type": "node",
            "request": "launch",
            "program": "${workspaceFolder}/vscodium/out/main.js",
            "args": [
                "--extensionDevelopmentPath=${workspaceFolder}/extensions/ai-assistant"
            ],
            "console": "integratedTerminal"
        }
    ]
}
EOF

# Step 5: Create tasks configuration
cat > .vscode/tasks.json << 'EOF'
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
EOF

print_status "✅ AI IDE build completed successfully!"
print_status ""
print_status "Next steps:"
print_status "1. Open this project in VS Code"
print_status "2. Press F5 to launch the extension development host"
print_status "3. Test the AI Assistant extension"
print_status ""
print_status "Extension location: extensions/ai-assistant"
print_status "Backend service: backend/main.py"