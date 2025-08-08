#!/bin/bash

# AI IDE Extension Packaging Script
# Creates VSCodium marketplace package with agent configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTENSION_DIR="$PROJECT_ROOT/extensions/ai-assistant"
DIST_DIR="$PROJECT_ROOT/dist"
VERSION=${1:-$(node -p "require('$EXTENSION_DIR/package.json').version")}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[PACKAGE]${NC} $1"
}

print_header "Packaging AI IDE Extension v$VERSION"

# Create dist directory
mkdir -p "$DIST_DIR"

# Change to extension directory
cd "$EXTENSION_DIR"

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v npm &> /dev/null; then
    print_error "npm is not installed"
    exit 1
fi

if ! command -v vsce &> /dev/null; then
    print_warning "vsce not found, installing..."
    npm install -g @vscode/vsce
fi

# Install dependencies
print_status "Installing dependencies..."
npm ci

# Run tests
print_status "Running tests..."
npm run test:all

# Compile TypeScript
print_status "Compiling TypeScript..."
npm run compile

# Update version if provided
if [ "$1" ]; then
    print_status "Updating version to $VERSION..."
    npm version "$VERSION" --no-git-tag-version
fi

# Create agent configuration bundle
print_status "Creating agent configuration bundle..."
mkdir -p "./agent-configs"

# Copy backend configurations
cp -r "$PROJECT_ROOT/backend/config" "./agent-configs/"
cp "$PROJECT_ROOT/backend/requirements.txt" "./agent-configs/"

# Create MCP server configurations
cat > "./agent-configs/mcp-servers.json" << 'EOF'
{
  "mcpServers": {
    "ai-ide-websearch": {
      "command": "python",
      "args": ["-m", "web_search_agent"],
      "env": {
        "AI_IDE_ENV": "production"
      },
      "disabled": false,
      "autoApprove": ["search", "summarize"]
    },
    "ai-ide-rag": {
      "command": "python",
      "args": ["-m", "rag_system"],
      "env": {
        "AI_IDE_ENV": "production"
      },
      "disabled": false,
      "autoApprove": ["query", "index"]
    },
    "ai-ide-reasoning": {
      "command": "python",
      "args": ["-m", "chain_of_thought_engine"],
      "env": {
        "AI_IDE_ENV": "production"
      },
      "disabled": false,
      "autoApprove": ["reason", "analyze"]
    }
  }
}
EOF

# Create installation script
cat > "./agent-configs/install.sh" << 'EOF'
#!/bin/bash
# AI IDE Agent Installation Script

set -e

echo "Installing AI IDE backend dependencies..."

# Create virtual environment
python3 -m venv ai-ide-env
source ai-ide-env/bin/activate

# Install requirements
pip install -r requirements.txt

# Set up database
python -c "
from database.connection import DatabaseManager
from database.migrations import run_migrations

db = DatabaseManager()
db.create_database()
run_migrations()
print('Database setup complete')
"

echo "AI IDE agents installed successfully!"
echo "Activate environment with: source ai-ide-env/bin/activate"
EOF

chmod +x "./agent-configs/install.sh"

# Create Windows installation script
cat > "./agent-configs/install.ps1" << 'EOF'
# AI IDE Agent Installation Script for Windows

Write-Host "Installing AI IDE backend dependencies..." -ForegroundColor Green

# Create virtual environment
python -m venv ai-ide-env
& "ai-ide-env\Scripts\Activate.ps1"

# Install requirements
pip install -r requirements.txt

# Set up database
python -c @"
from database.connection import DatabaseManager
from database.migrations import run_migrations

db = DatabaseManager()
db.create_database()
run_migrations()
print('Database setup complete')
"@

Write-Host "AI IDE agents installed successfully!" -ForegroundColor Green
Write-Host "Activate environment with: ai-ide-env\Scripts\Activate.ps1" -ForegroundColor Yellow
EOF

# Update package.json with agent configurations
print_status "Updating package.json with agent configurations..."
node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));

// Add agent configuration contributions
pkg.contributes.configuration = {
  title: 'AI IDE',
  properties: {
    'ai-ide.backend.url': {
      type: 'string',
      default: 'http://localhost:8000',
      description: 'AI IDE backend service URL'
    },
    'ai-ide.agents.enabled': {
      type: 'boolean',
      default: true,
      description: 'Enable AI agents'
    },
    'ai-ide.reasoning.mode': {
      type: 'string',
      enum: ['fast', 'deep', 'chain-of-thought'],
      default: 'fast',
      description: 'AI reasoning mode'
    },
    'ai-ide.search.semantic.enabled': {
      type: 'boolean',
      default: true,
      description: 'Enable semantic search'
    },
    'ai-ide.web.search.enabled': {
      type: 'boolean',
      default: true,
      description: 'Enable web search integration'
    },
    'ai-ide.rag.enabled': {
      type: 'boolean',
      default: true,
      description: 'Enable RAG (Retrieval-Augmented Generation)'
    },
    'ai-ide.mcp.autoDiscover': {
      type: 'boolean',
      default: true,
      description: 'Automatically discover MCP servers'
    }
  }
};

// Add activation events for agent features
pkg.activationEvents.push(
  'onCommand:ai-assistant.startAgents',
  'onCommand:ai-assistant.configureAgents',
  'workspaceContains:**/*.py',
  'workspaceContains:**/*.js',
  'workspaceContains:**/*.ts'
);

// Add agent management commands
pkg.contributes.commands.push(
  {
    command: 'ai-assistant.startAgents',
    title: 'Start AI Agents',
    category: 'AI Assistant',
    icon: '\$(play)'
  },
  {
    command: 'ai-assistant.stopAgents',
    title: 'Stop AI Agents',
    category: 'AI Assistant',
    icon: '\$(stop)'
  },
  {
    command: 'ai-assistant.configureAgents',
    title: 'Configure AI Agents',
    category: 'AI Assistant',
    icon: '\$(settings-gear)'
  },
  {
    command: 'ai-assistant.installAgents',
    title: 'Install AI Agents',
    category: 'AI Assistant',
    icon: '\$(cloud-download)'
  }
);

fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
console.log('Updated package.json with agent configurations');
"

# Package extension
print_status "Packaging extension..."
vsce package --out "$DIST_DIR/ai-ide-extension-v$VERSION.vsix"

# Create marketplace metadata
print_status "Creating marketplace metadata..."
cat > "$DIST_DIR/marketplace-info.json" << EOF
{
  "name": "AI IDE Extension",
  "version": "$VERSION",
  "description": "Advanced AI-powered coding assistant with multi-agent system, semantic search, and self-improving capabilities",
  "publisher": "ai-ide",
  "repository": "https://github.com/ai-ide/ai-ide",
  "homepage": "https://ai-ide.dev",
  "bugs": "https://github.com/ai-ide/ai-ide/issues",
  "license": "MIT",
  "keywords": [
    "ai",
    "assistant",
    "coding",
    "semantic-search",
    "multi-agent",
    "reasoning",
    "rag",
    "web-search",
    "mcp"
  ],
  "categories": [
    "Machine Learning",
    "Other",
    "Snippets",
    "Programming Languages"
  ],
  "engines": {
    "vscode": "^1.85.0"
  },
  "features": [
    "Multi-agent AI system",
    "Semantic code search",
    "Chain-of-thought reasoning",
    "Web search integration",
    "RAG (Retrieval-Augmented Generation)",
    "MCP server integration",
    "Self-improving AI models",
    "Performance benchmarking"
  ],
  "requirements": {
    "python": ">=3.11",
    "node": ">=18",
    "memory": "4GB",
    "storage": "2GB"
  }
}
EOF

# Create installation guide
print_status "Creating installation guide..."
cat > "$DIST_DIR/INSTALLATION.md" << 'EOF'
# AI IDE Extension Installation Guide

## Prerequisites

- VSCode or VSCodium 1.85.0 or higher
- Python 3.11 or higher
- Node.js 18 or higher
- 4GB RAM minimum
- 2GB free storage

## Installation Steps

### 1. Install Extension

#### From VSIX file:
```bash
code --install-extension ai-ide-extension-v*.vsix
```

#### From VSCode Marketplace:
1. Open VSCode
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "AI IDE"
4. Click Install

### 2. Install AI Agents

After installing the extension:

1. Open Command Palette (Ctrl+Shift+P)
2. Run "AI Assistant: Install AI Agents"
3. Follow the installation prompts

Or manually:

#### Linux/macOS:
```bash
cd ~/.vscode/extensions/ai-ide.ai-assistant-*/agent-configs
chmod +x install.sh
./install.sh
```

#### Windows:
```powershell
cd $env:USERPROFILE\.vscode\extensions\ai-ide.ai-assistant-*\agent-configs
.\install.ps1
```

### 3. Configure Backend Services

1. Start the AI IDE backend:
```bash
source ai-ide-env/bin/activate  # Linux/macOS
# or
ai-ide-env\Scripts\Activate.ps1  # Windows

python main.py
```

2. Configure extension settings:
   - Open VSCode Settings
   - Search for "AI IDE"
   - Set backend URL (default: http://localhost:8000)

### 4. Verify Installation

1. Open Command Palette (Ctrl+Shift+P)
2. Run "AI Assistant: Open AI Chat"
3. Test with a simple query

## Features

- **Multi-Agent System**: Specialized agents for different tasks
- **Semantic Search**: Find code by meaning, not just keywords
- **Chain-of-Thought Reasoning**: Deep AI reasoning for complex problems
- **Web Search Integration**: Real-time information retrieval
- **RAG System**: Context-aware code assistance
- **MCP Integration**: Extensible tool ecosystem

## Troubleshooting

### Common Issues

1. **Backend connection failed**
   - Ensure Python backend is running
   - Check backend URL in settings
   - Verify firewall settings

2. **Agents not responding**
   - Restart VSCode
   - Check Python environment activation
   - Review extension logs

3. **Performance issues**
   - Increase memory allocation
   - Disable unused features
   - Check system resources

### Getting Help

- Documentation: https://ai-ide.dev/docs
- Issues: https://github.com/ai-ide/ai-ide/issues
- Discussions: https://github.com/ai-ide/ai-ide/discussions

## Configuration

### Basic Settings

```json
{
  "ai-ide.backend.url": "http://localhost:8000",
  "ai-ide.agents.enabled": true,
  "ai-ide.reasoning.mode": "fast",
  "ai-ide.search.semantic.enabled": true,
  "ai-ide.web.search.enabled": true,
  "ai-ide.rag.enabled": true,
  "ai-ide.mcp.autoDiscover": true
}
```

### Advanced Configuration

See the full configuration guide at: https://ai-ide.dev/docs/configuration
EOF

# Clean up temporary files
rm -rf "./agent-configs"

print_header "✅ Extension packaging completed successfully!"
print_status ""
print_status "Package created: $DIST_DIR/ai-ide-extension-v$VERSION.vsix"
print_status "Marketplace info: $DIST_DIR/marketplace-info.json"
print_status "Installation guide: $DIST_DIR/INSTALLATION.md"
print_status ""
print_status "Next steps:"
print_status "1. Test the extension locally"
print_status "2. Upload to VSCode Marketplace"
print_status "3. Create GitHub release"