# AI IDE Installation Guide

## Overview

AI IDE is an advanced AI-powered development environment built on VSCodium that integrates multiple cutting-edge AI technologies including multi-agent systems, semantic search, chain-of-thought reasoning, web search integration, and self-improving AI models.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Memory**: 8GB RAM
- **Storage**: 5GB free space
- **CPU**: Intel i5 or AMD Ryzen 5 equivalent
- **Network**: Stable internet connection for AI model downloads

### Recommended Requirements
- **Memory**: 16GB+ RAM
- **Storage**: 10GB+ free space (SSD recommended)
- **CPU**: Intel i7 or AMD Ryzen 7 equivalent
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for local AI models)

### Software Prerequisites
- **Python**: 3.11 or higher
- **Node.js**: 18.0 or higher
- **Git**: Latest version
- **VSCode/VSCodium**: 1.85.0 or higher

## Installation Methods

### Method 1: Quick Install (Recommended)

#### Windows
```powershell
# Download and run the installer
Invoke-WebRequest -Uri "https://github.com/ai-ide/ai-ide/releases/latest/download/ai-ide-installer.exe" -OutFile "ai-ide-installer.exe"
.\ai-ide-installer.exe
```

#### macOS
```bash
# Download and install via Homebrew
brew tap ai-ide/tap
brew install ai-ide
```

#### Linux
```bash
# Download and install via package manager
curl -fsSL https://install.ai-ide.dev | bash
```

### Method 2: Manual Installation

#### Step 1: Install Prerequisites

**Python 3.11+**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS (via Homebrew)
brew install python@3.11

# Windows (via Python.org or Microsoft Store)
# Download from https://python.org/downloads/
```

**Node.js 18+**
```bash
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS (via Homebrew)
brew install node@18

# Windows (via Node.js website)
# Download from https://nodejs.org/
```

#### Step 2: Clone Repository
```bash
git clone https://github.com/ai-ide/ai-ide.git
cd ai-ide
```

#### Step 3: Install Backend Dependencies
```bash
cd backend
python3.11 -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 4: Install Extension Dependencies
```bash
cd ../extensions/ai-assistant
npm install
npm run compile
```

#### Step 5: Set Up Database
```bash
cd ../../backend

# Install PostgreSQL with pgvector extension
# Ubuntu/Debian:
sudo apt install postgresql postgresql-contrib
sudo -u postgres psql -c "CREATE EXTENSION vector;"

# macOS (via Homebrew):
brew install postgresql pgvector
brew services start postgresql

# Create database
python -c "
from database.connection import DatabaseManager
from database.migrations import run_migrations

db = DatabaseManager()
db.create_database()
run_migrations()
print('Database setup complete')
"
```

#### Step 6: Configure Services
```bash
# Copy configuration templates
cp config/development.yaml.template config/development.yaml

# Edit configuration file
nano config/development.yaml
```

#### Step 7: Start Services
```bash
# Start backend services
python main.py &

# Start additional services
python web_search_agent.py &
python rag_system.py &
python mcp_server_framework.py &
```

#### Step 8: Install VSCode Extension
```bash
# Package extension
cd ../extensions/ai-assistant
npm run package

# Install extension
code --install-extension ai-ide-extension-*.vsix
```

### Method 3: Docker Installation

#### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

#### Quick Start
```bash
# Clone repository
git clone https://github.com/ai-ide/ai-ide.git
cd ai-ide

# Start all services
docker-compose up -d

# Install VSCode extension
code --install-extension dist/ai-ide-extension-*.vsix
```

#### Production Deployment
```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale ai-ide-backend=3
```

## Configuration

### Basic Configuration

Create or edit `~/.ai-ide/config.yaml`:

```yaml
# AI IDE Configuration
backend:
  url: "http://localhost:8000"
  timeout: 30

agents:
  enabled: true
  max_concurrent: 5
  
reasoning:
  mode: "fast"  # Options: fast, deep, chain-of-thought
  max_steps: 10
  
search:
  semantic:
    enabled: true
    model: "sentence-transformers/all-MiniLM-L6-v2"
  web:
    enabled: true
    engines: ["google", "bing", "duckduckgo"]
    
rag:
  enabled: true
  chunk_size: 512
  max_context: 8192
  
mcp:
  auto_discover: true
  servers:
    - name: "websearch"
      enabled: true
    - name: "rag"
      enabled: true
```

### Advanced Configuration

#### LM Studio Integration
```yaml
ai_models:
  lm_studio:
    url: "http://localhost:1234"
    model: "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_tokens: 4096
    temperature: 0.1
    timeout: 60
```

#### Database Configuration
```yaml
database:
  url: "postgresql://user:password@localhost:5432/ai_ide"
  pool_size: 20
  max_overflow: 30
  echo: false
```

#### Security Configuration
```yaml
security:
  jwt_secret: "your-secret-key"
  cors_origins:
    - "http://localhost:3000"
    - "https://your-domain.com"
  rate_limiting:
    requests_per_minute: 100
    burst_size: 20
```

## Verification

### Health Checks
```bash
# Check backend health
curl http://localhost:8000/health

# Check all services
curl http://localhost:8000/health/detailed
```

### Test Installation
1. Open VSCode/VSCodium
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
3. Type "AI Assistant: Open AI Chat"
4. Test with a simple query: "Hello, can you help me write a Python function?"

### Performance Test
```bash
# Run benchmark suite
cd backend
python ../scripts/run-benchmarks.py --mini-only
```

## Troubleshooting

### Common Issues

#### 1. Backend Connection Failed
**Symptoms**: Extension shows "Backend unavailable" error

**Solutions**:
```bash
# Check if backend is running
ps aux | grep python | grep main.py

# Check backend logs
tail -f backend/logs/ai-ide.log

# Restart backend
cd backend
source venv/bin/activate
python main.py
```

#### 2. Python Dependencies Error
**Symptoms**: Import errors or missing modules

**Solutions**:
```bash
# Reinstall dependencies
cd backend
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.11+
```

#### 3. Database Connection Error
**Symptoms**: Database-related errors in logs

**Solutions**:
```bash
# Check PostgreSQL status
sudo systemctl status postgresql  # Linux
brew services list | grep postgresql  # macOS

# Reset database
cd backend
python -c "
from database.connection import DatabaseManager
db = DatabaseManager()
db.reset_database()
"
```

#### 4. Extension Not Loading
**Symptoms**: AI Assistant panel not visible

**Solutions**:
1. Check VSCode extension is installed: `code --list-extensions | grep ai-ide`
2. Reload VSCode window: `Ctrl+Shift+P` → "Developer: Reload Window"
3. Check extension logs: `Ctrl+Shift+P` → "Developer: Show Logs" → "Extension Host"

#### 5. Performance Issues
**Symptoms**: Slow responses or high memory usage

**Solutions**:
```bash
# Check system resources
htop  # Linux/macOS
# Task Manager on Windows

# Optimize configuration
# Edit ~/.ai-ide/config.yaml:
reasoning:
  mode: "fast"
agents:
  max_concurrent: 2
```

### Getting Help

- **Documentation**: https://ai-ide.dev/docs
- **GitHub Issues**: https://github.com/ai-ide/ai-ide/issues
- **Discussions**: https://github.com/ai-ide/ai-ide/discussions
- **Discord**: https://discord.gg/ai-ide
- **Email Support**: support@ai-ide.dev

### Diagnostic Information

When reporting issues, please include:

```bash
# Generate diagnostic report
cd backend
python -c "
import sys
import platform
import pkg_resources

print('=== AI IDE Diagnostic Report ===')
print(f'Python Version: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
print('Installed Packages:')
for pkg in pkg_resources.working_set:
    print(f'  {pkg.key}=={pkg.version}')
"
```

## Next Steps

After successful installation:

1. **Read the User Guide**: [USER_GUIDE.md](USER_GUIDE.md)
2. **Configure Agents**: [AGENT_CONFIGURATION.md](AGENT_CONFIGURATION.md)
3. **Set Up MCP Servers**: [MCP_INTEGRATION.md](MCP_INTEGRATION.md)
4. **Explore Features**: [FEATURES.md](FEATURES.md)
5. **Join Community**: [COMMUNITY.md](COMMUNITY.md)

## Updates

AI IDE automatically checks for updates. To manually update:

```bash
# Update via package manager
brew upgrade ai-ide  # macOS
# or
sudo apt update && sudo apt upgrade ai-ide  # Ubuntu

# Update from source
git pull origin main
cd backend && pip install -r requirements.txt
cd ../extensions/ai-assistant && npm install && npm run compile
```