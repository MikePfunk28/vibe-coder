# AI IDE Setup Guide

## Building AI IDE on VSCode OSS Foundation

This guide walks you through building AI IDE using VSCode OSS as the base, just like Cursor does.

## 🎯 What We're Building

AI IDE is built on the complete VSCode OSS codebase with comprehensive AI enhancements:

- **Complete VSCode functionality** - All features, menus, extensions, etc.
- **Multi-Agent AI System** - Specialized AI agents for different tasks
- **Web Search Integration** - Playwright-powered web research
- **Semantic Code Search** - Intelligent code discovery
- **RAG System** - Enhanced code assistance with knowledge retrieval
- **Self-Improving AI** - Darwin-Gödel model that learns and improves

## 📋 Prerequisites

### System Requirements

- **Node.js** 18.0.0 or later
- **Python** 3.11 or later
- **Yarn** 1.22.0 or later (required for VSCode OSS)
- **Git** for repository management

### Platform-Specific Requirements

#### Windows
- **Visual Studio Build Tools** 2019 or later
- **Windows SDK** 10.0.17763.0 or later
- **Python for Windows** with pip

#### macOS
- **Xcode Command Line Tools**
- **macOS** 10.15 or later
- **Homebrew** (recommended for dependencies)

#### Linux
- **build-essential** package
- **Python 3.11-dev** package
- Required libraries: `libnss3-dev`, `libatk-bridge2.0-dev`, `libdrm2`, `libxcomposite1`, `libxdamage1`, `libxrandr2`, `libgbm1`, `libxss1`, `libasound2`

## 🚀 Step-by-Step Setup

### Step 1: Clone and Install Dependencies

```bash
# Clone the AI IDE repository
git clone https://github.com/your-username/ai-ide.git
cd ai-ide

# Install all dependencies (Node.js, Python, extensions)
npm run install:all
```

### Step 2: Setup VSCode OSS Foundation

```bash
# This downloads VSCode OSS and configures it for AI IDE
npm run setup
```

**What this does:**
- Downloads VSCode OSS v1.85.0 from Microsoft's repository
- Copies it to `ai-ide-build/` directory
- Updates `product.json` with AI IDE branding
- Configures extension marketplace access
- Adds AI-specific configuration options
- Integrates our AI backend and extensions

### Step 3: Build AI IDE

```bash
# Build the complete AI IDE
npm run build
```

**What this does:**
- Installs VSCode OSS dependencies with Yarn
- Compiles TypeScript code
- Builds all AI extensions
- Integrates AI backend
- Creates the complete AI IDE application

### Step 4: Test the Build

```bash
# Run AI IDE in development mode
npm run dev
```

This starts AI IDE with:
- Full VSCode functionality
- AI chat panel
- Semantic search
- Web search integration
- Multi-agent system
- All AI features enabled

### Step 5: Create Executables

```bash
# Create executables for current platform
npm run package

# Create executables for all platforms
npm run package:all
```

**Output:**
- **Windows**: `.exe` installer and portable version
- **macOS**: `.dmg` installer and `.app` bundle
- **Linux**: `.AppImage`, `.deb`, and `.rpm` packages

## 📁 Project Structure After Setup

```
ai-ide/
├── code-oss/                  # Downloaded VSCode OSS source
├── ai-ide-build/              # Our AI IDE build (based on VSCode OSS)
│   ├── src/                   # VSCode source with AI enhancements
│   ├── extensions/            # Built-in AI extensions
│   ├── ai-backend/           # Integrated Python AI backend
│   ├── product.json          # AI IDE branding and configuration
│   └── package.json          # Build configuration
├── backend/                   # AI backend source code
├── extensions/               # AI extension source code
├── dist/                     # Built executables
├── setup-code-oss.js        # VSCode OSS setup script
├── build-ai-ide.js          # AI IDE build script
└── build-config-oss.json    # Build configuration
```

## 🔧 Configuration

### AI Features Configuration

Edit `build-config-oss.json` to customize AI features:

```json
{
  "aiFeatures": {
    "enableAI": true,
    "enableWebSearch": true,
    "enablePlaywright": true,
    "enableMultiAgent": true,
    "enableSemanticSearch": true,
    "enableRAG": true,
    "enableDarwinGodel": true,
    "enableReinforcementLearning": true,
    "defaultAIProvider": "lm-studio",
    "backendPort": 8000
  }
}
```

### Branding Configuration

Update branding in `build-config-oss.json`:

```json
{
  "branding": {
    "nameShort": "AI IDE",
    "nameLong": "AI IDE - Advanced AI-Powered Development Environment",
    "applicationName": "ai-ide",
    "dataFolderName": ".ai-ide"
  }
}
```

## 🤖 AI Backend Setup

The AI backend is automatically integrated during build, but you can run it separately:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install

# Run backend
python main.py --port 8000
```

## 🧩 Extension Development

AI IDE includes several built-in AI extensions:

- **ai-assistant** - Main AI chat and assistance
- **ai-chat** - Built-in chat interface
- **ai-semantic-search** - Semantic code search
- **ai-web-search** - Web search integration
- **ai-reasoning** - Advanced reasoning capabilities
- **ai-multi-agent** - Multi-agent coordination

### Adding Custom Extensions

1. Create extension in `extensions/` directory
2. Follow VSCode extension structure
3. Add to build configuration
4. Rebuild AI IDE

## 🔍 Troubleshooting

### Common Issues

#### VSCode OSS Download Fails
```bash
# Manually clone VSCode OSS
git clone --depth 1 --branch 1.85.0 https://github.com/microsoft/vscode.git code-oss
```

#### Python Backend Issues
```bash
# Check Python version
python --version  # Should be 3.11+

# Reinstall dependencies
cd backend
pip install --upgrade -r requirements.txt
```

#### Build Failures
```bash
# Clean and rebuild
npm run clean
npm run setup
npm run build
```

#### Yarn Issues
```bash
# Install Yarn globally
npm install -g yarn

# Clear Yarn cache
yarn cache clean
```

### Platform-Specific Issues

#### Windows
- Ensure Visual Studio Build Tools are installed
- Use PowerShell as Administrator for setup
- Check Windows Defender exclusions

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for Python: `brew install python@3.11`

#### Linux
- Install required libraries: `sudo apt-get install build-essential libnss3-dev libatk-bridge2.0-dev`
- Ensure Python 3.11-dev is installed

## 🚀 Distribution

### Creating Releases

```bash
# Create new version
npm run version:bump minor "Added new AI features"

# Build for all platforms
npm run build:all

# Package for distribution
npm run package:all

# Create release
npm run version:release 1.0.0 "Initial AI IDE release"
```

### Distribution Files

After building, you'll have:

- **Windows**: `AI-IDE-Setup-1.0.0.exe`, `AI-IDE-1.0.0-win.zip`
- **macOS**: `AI-IDE-1.0.0.dmg`, `AI-IDE-1.0.0-mac.zip`
- **Linux**: `AI-IDE-1.0.0.AppImage`, `ai-ide_1.0.0_amd64.deb`, `ai-ide-1.0.0.x86_64.rpm`

## 🎯 Comparison with Cursor

| Feature | AI IDE | Cursor |
|---------|--------|--------|
| **Base** | VSCode OSS (complete) | VSCode OSS (complete) |
| **AI Agents** | Multi-agent system | Single AI assistant |
| **Web Search** | Playwright integration | Limited |
| **Self-Improvement** | Darwin-Gödel model | Static |
| **Local Models** | LM Studio + Qwen Coder | Limited support |
| **Semantic Search** | Advanced similarity | Basic |
| **RAG System** | Comprehensive | Basic |
| **Open Source** | Fully open | Partially closed |

## 📚 Next Steps

1. **Customize AI Features** - Modify `build-config-oss.json`
2. **Add Extensions** - Create custom AI extensions
3. **Enhance Backend** - Add new AI capabilities
4. **Create Themes** - Design custom AI IDE themes
5. **Build Community** - Share and collaborate

---

**AI IDE** - The complete VSCode-based AI development environment that rivals Cursor and Windsurf!