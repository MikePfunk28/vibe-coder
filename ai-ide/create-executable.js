#!/usr/bin/env node

/**
 * AI IDE Executable Creator
 * Creates a portable executable for the AI IDE
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class ExecutableCreator {
    constructor() {
        this.projectRoot = __dirname;
        this.buildDir = path.join(this.projectRoot, 'ai-ide-build');
        this.distDir = path.join(this.projectRoot, 'dist');
    }

    async createExecutable() {
        console.log('📦 Creating AI IDE executable...');
        console.log('');

        try {
            // Step 1: Create distribution directory
            await this.createDistDirectory();
            
            // Step 2: Create portable launcher
            await this.createPortableLauncher();
            
            // Step 3: Create installation package
            await this.createInstallationPackage();
            
            // Step 4: Create documentation
            await this.createDocumentation();
            
            console.log('🎉 AI IDE executable created successfully!');
            this.displaySummary();
            
        } catch (error) {
            console.error('❌ Executable creation failed:', error.message);
            throw error;
        }
    }

    async createDistDirectory() {
        console.log('📁 Creating distribution directory...');
        
        if (fs.existsSync(this.distDir)) {
            fs.rmSync(this.distDir, { recursive: true, force: true });
        }
        fs.mkdirSync(this.distDir, { recursive: true });
        
        console.log('✅ Distribution directory created');
    }

    async createPortableLauncher() {
        console.log('🚀 Creating portable launcher...');
        
        // Create Windows batch launcher
        const windowsLauncher = `@echo off
title AI IDE - Advanced AI-Powered Development Environment
echo.
echo ╔══════════════════════════════════════════════════════════════════════════════╗
echo ║                           AI IDE - Starting Up                              ║
echo ║                                                                              ║
echo ║  🤖 Complete VSCode OSS foundation with Cursor-level AI features            ║
echo ║  ⚡ Universal AI provider support (OpenAI, Anthropic, Ollama, LM Studio)   ║
echo ║  🔍 Advanced semantic search and reasoning capabilities                      ║
echo ║  🌐 Web search integration and multi-agent systems                          ║
echo ║                                                                              ║
echo ║  Cursor-Style Features:                                                      ║
echo ║  • Ctrl+K: Inline code generation                                           ║
echo ║  • Ctrl+L: AI chat panel                                                    ║
echo ║  • Select code + Ctrl+K: Edit with AI                                       ║
echo ║                                                                              ║
echo ╚══════════════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is required but not found. Please install Node.js from https://nodejs.org
    echo.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is required but not found. Please install Python from https://python.org
    echo.
    pause
    exit /b 1
)

echo 🔧 Starting AI backend...
cd /d "%~dp0ai-ide-build\\ai-backend"
start /B python main.py --port 8000

echo 🚀 Launching AI IDE...
cd /d "%~dp0ai-ide-build"

REM Try different VSCode launch methods
if exist "scripts\\code.bat" (
    call scripts\\code.bat
) else if exist "scripts\\code.sh" (
    bash scripts\\code.sh
) else if exist "Code.exe" (
    start Code.exe
) else (
    echo 🔨 Building AI IDE first...
    call npm install
    call npm run compile
    if exist "scripts\\code.bat" (
        call scripts\\code.bat
    ) else (
        echo ❌ Could not find VSCode executable. Please run setup first.
        pause
    )
)

echo.
echo 🎯 AI IDE session ended. Thank you for using AI IDE!
pause
`;

        // Create Linux/Mac launcher
        const unixLauncher = `#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                           AI IDE - Starting Up                              ║"
echo "║                                                                              ║"
echo "║  🤖 Complete VSCode OSS foundation with Cursor-level AI features            ║"
echo "║  ⚡ Universal AI provider support (OpenAI, Anthropic, Ollama, LM Studio)   ║"
echo "║  🔍 Advanced semantic search and reasoning capabilities                      ║"
echo "║  🌐 Web search integration and multi-agent systems                          ║"
echo "║                                                                              ║"
echo "║  Cursor-Style Features:                                                      ║"
echo "║  • Ctrl+K: Inline code generation                                           ║"
echo "║  • Ctrl+L: AI chat panel                                                    ║"
echo "║  • Select code + Ctrl+K: Edit with AI                                       ║"
echo "║                                                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not found. Please install Node.js from https://nodejs.org"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is required but not found. Please install Python from https://python.org"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "🔧 Starting AI backend..."
cd "$SCRIPT_DIR/ai-ide-build/ai-backend"
python3 main.py --port 8000 &
BACKEND_PID=$!

echo "🚀 Launching AI IDE..."
cd "$SCRIPT_DIR/ai-ide-build"

# Try different VSCode launch methods
if [ -f "scripts/code.sh" ]; then
    bash scripts/code.sh
elif [ -f "scripts/code" ]; then
    ./scripts/code
elif [ -f "code" ]; then
    ./code
else
    echo "🔨 Building AI IDE first..."
    npm install
    npm run compile
    if [ -f "scripts/code.sh" ]; then
        bash scripts/code.sh
    else
        echo "❌ Could not find VSCode executable. Please run setup first."
        exit 1
    fi
fi

# Clean up backend process
kill $BACKEND_PID 2>/dev/null

echo ""
echo "🎯 AI IDE session ended. Thank you for using AI IDE!"
`;

        // Write launchers
        fs.writeFileSync(path.join(this.distDir, 'AI-IDE.bat'), windowsLauncher);
        fs.writeFileSync(path.join(this.distDir, 'AI-IDE.sh'), unixLauncher);
        
        // Make Unix launcher executable
        if (process.platform !== 'win32') {
            execSync(`chmod +x "${path.join(this.distDir, 'AI-IDE.sh')}"`);
        }
        
        console.log('✅ Portable launchers created');
    }

    async createInstallationPackage() {
        console.log('📦 Creating installation package...');
        
        // Copy the AI IDE build
        console.log('📁 Copying AI IDE build...');
        this.copyDirectory(this.buildDir, path.join(this.distDir, 'ai-ide-build'));
        
        // Create setup script
        const setupScript = `@echo off
title AI IDE Setup
echo.
echo 🚀 Setting up AI IDE...
echo.

REM Install Node.js dependencies
echo 📦 Installing dependencies...
cd ai-ide-build
call npm install

REM Install Python dependencies
echo 🐍 Installing Python dependencies...
cd ai-backend
pip install -r requirements.txt
cd ..

echo.
echo ✅ AI IDE setup completed!
echo.
echo To start AI IDE, run: AI-IDE.bat
echo.
pause
`;

        fs.writeFileSync(path.join(this.distDir, 'setup.bat'), setupScript);
        
        console.log('✅ Installation package created');
    }

    async createDocumentation() {
        console.log('📚 Creating documentation...');
        
        const readme = `# AI IDE - Advanced AI-Powered Development Environment

## Overview

AI IDE is a complete VSCode-based development environment enhanced with cutting-edge AI capabilities. Built on VSCode OSS foundation, it provides all standard IDE features plus Cursor-level AI assistance and advanced reasoning capabilities.

## Features

### 🤖 Complete VSCode Foundation
- **Full VSCode Compatibility**: All standard VSCode features (editing, debugging, git, extensions, themes)
- **Extension Support**: Compatible with all VSCode extensions from the marketplace
- **Familiar Interface**: Identical GUI and user experience to VSCode

### ⚡ Cursor-Level AI Features
- **Ctrl+K**: Inline code generation with natural language prompts
- **Ctrl+L**: AI chat panel for code discussions and assistance
- **AI Code Editing**: Select code + Ctrl+K to edit/refactor with AI
- **Smart Autocomplete**: AI-powered code completion with context awareness

### 🔍 Advanced AI Capabilities
- **Universal AI Provider Support**: OpenAI, Anthropic, OpenRouter, Ollama, LM Studio, llama.cpp
- **Semantic Code Search**: Find code by meaning, not just text
- **Multi-Agent System**: Specialized AI agents for different development tasks
- **Web Search Integration**: Real-time information retrieval and documentation lookup
- **RAG System**: Enhanced code assistance with retrieval-augmented generation
- **Self-Improving AI**: Darwin-Gödel model that learns and improves over time

### 🌐 Advanced Features
- **Chain-of-Thought Reasoning**: Step-by-step problem solving
- **Interleaved Context Management**: Efficient handling of large codebases
- **Reinforcement Learning**: Adapts to your coding preferences
- **MCP Integration**: Extensible tool and service integration

## Installation

### Prerequisites
- **Node.js** (v18 or higher): Download from [nodejs.org](https://nodejs.org)
- **Python** (v3.11 or higher): Download from [python.org](https://python.org)

### Quick Start
1. Run \`setup.bat\` (Windows) or \`setup.sh\` (Linux/Mac) to install dependencies
2. Run \`AI-IDE.bat\` (Windows) or \`AI-IDE.sh\` (Linux/Mac) to start AI IDE
3. Start coding with AI assistance!

## Usage

### Cursor-Style AI Features
- **Inline Generation**: Press \`Ctrl+K\` in any editor to generate code with natural language
- **AI Chat**: Press \`Ctrl+L\` to open the AI chat panel for discussions
- **Code Editing**: Select code and press \`Ctrl+K\` to edit/refactor with AI
- **Context Awareness**: AI understands your entire codebase for better suggestions

### AI Providers Setup
1. **OpenAI**: Set \`OPENAI_API_KEY\` environment variable
2. **Anthropic**: Set \`ANTHROPIC_API_KEY\` environment variable
3. **Local Models**: Install Ollama or LM Studio for local AI models
4. **Auto-Detection**: AI IDE automatically detects available providers

### Advanced Features
- **Semantic Search**: Use Ctrl+Shift+S for semantic code search
- **Web Search**: Integrated web search for documentation and examples
- **Multi-Agent**: Coordinate multiple AI agents for complex tasks
- **Reasoning**: Advanced problem-solving with chain-of-thought

## Architecture

AI IDE is built on a layered architecture:

1. **VSCode OSS Foundation**: Complete standard IDE functionality
2. **AI Extensions Layer**: Cursor-level AI features and enhancements
3. **Universal AI Provider**: Support for multiple AI models and services
4. **Advanced Intelligence**: Semantic search, reasoning, and self-improvement
5. **Integration Layer**: MCP servers, external tools, and web services

## Comparison

| Feature | AI IDE | Cursor | VSCode | Windsurf |
|---------|--------|--------|--------|----------|
| VSCode Foundation | ✅ Complete | ✅ Yes | ✅ Yes | ❌ No |
| Ctrl+K Generation | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| Ctrl+L Chat | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| Universal AI Support | ✅ Yes | ❌ Limited | ❌ No | ❌ Limited |
| Semantic Search | ✅ Yes | ❌ No | ❌ Basic | ❌ No |
| Multi-Agent System | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Web Search | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Self-Improvement | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Open Source | ✅ Yes | ❌ No | ✅ Yes | ❌ No |

## Contributing

AI IDE is built on open-source technologies:
- **VSCode OSS**: Microsoft's open-source editor
- **PocketFlow**: Advanced AI workflow management
- **LangChain**: AI orchestration and tool integration
- **Universal AI Providers**: Support for all major AI models

## License

AI IDE is released under the MIT License, same as VSCode OSS.

## Support

For issues, questions, or contributions:
- GitHub Issues: Report bugs and request features
- Documentation: Comprehensive guides and API reference
- Community: Join our developer community

---

**AI IDE** - The VSCode-based AI development environment that competes with Cursor and Windsurf while remaining fully open-source.
`;

        fs.writeFileSync(path.join(this.distDir, 'README.md'), readme);
        
        // Create quick start guide
        const quickStart = `# AI IDE Quick Start Guide

## 1. First Launch
- Run AI-IDE.bat (Windows) or AI-IDE.sh (Linux/Mac)
- Wait for the AI backend to start (about 10 seconds)
- VSCode will open with AI extensions loaded

## 2. Test Cursor-Level Features
- Open any code file or create a new one
- Press **Ctrl+K** to try inline code generation
- Press **Ctrl+L** to open the AI chat panel
- Select some code and press **Ctrl+K** to edit with AI

## 3. Configure AI Providers
- Set environment variables for your preferred AI service:
  - \`OPENAI_API_KEY=your_key_here\`
  - \`ANTHROPIC_API_KEY=your_key_here\`
- Or use local models with Ollama/LM Studio (auto-detected)

## 4. Explore Advanced Features
- **Semantic Search**: Ctrl+Shift+S
- **AI Reasoning**: Ctrl+Shift+R  
- **Web Search**: Integrated in chat panel
- **Multi-Agent**: Available in AI assistant panel

## 5. Troubleshooting
- Ensure Node.js and Python are installed
- Check that port 8000 is available for the AI backend
- Restart AI IDE if AI features don't respond

Enjoy coding with AI assistance! 🚀
`;

        fs.writeFileSync(path.join(this.distDir, 'QUICK-START.md'), quickStart);
        
        console.log('✅ Documentation created');
    }

    copyDirectory(src, dest, options = {}) {
        const { exclude = [] } = options;
        
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }

        const items = fs.readdirSync(src);
        
        for (const item of items) {
            if (exclude.includes(item)) continue;
            
            const srcPath = path.join(src, item);
            const destPath = path.join(dest, item);
            
            const stat = fs.statSync(srcPath);
            
            if (stat.isDirectory()) {
                this.copyDirectory(srcPath, destPath, options);
            } else {
                fs.copyFileSync(srcPath, destPath);
            }
        }
    }

    displaySummary() {
        console.log('');
        console.log('🎉 AI IDE Executable Package Created!');
        console.log('');
        console.log('📁 Distribution Contents:');
        console.log('   📄 AI-IDE.bat - Windows launcher');
        console.log('   📄 AI-IDE.sh - Linux/Mac launcher');
        console.log('   📄 setup.bat - Installation script');
        console.log('   📁 ai-ide-build/ - Complete AI IDE');
        console.log('   📚 README.md - Full documentation');
        console.log('   📚 QUICK-START.md - Quick start guide');
        console.log('');
        console.log('🚀 To use AI IDE:');
        console.log('   1. Run setup.bat to install dependencies');
        console.log('   2. Run AI-IDE.bat to start the IDE');
        console.log('   3. Press Ctrl+K for inline generation');
        console.log('   4. Press Ctrl+L for AI chat');
        console.log('');
        console.log('🎯 Features:');
        console.log('   ✅ Complete VSCode OSS foundation');
        console.log('   ✅ Cursor-level AI features (Ctrl+K, Ctrl+L)');
        console.log('   ✅ Universal AI provider support');
        console.log('   ✅ Advanced semantic search');
        console.log('   ✅ Multi-agent reasoning system');
        console.log('   ✅ Web search integration');
        console.log('   ✅ Self-improving AI capabilities');
        console.log('');
        console.log('🏆 Ready to compete with Cursor and Windsurf!');
        console.log(`📦 Package location: ${this.distDir}`);
    }
}

// CLI Interface
if (require.main === module) {
    const creator = new ExecutableCreator();
    creator.createExecutable().catch(console.error);
}

module.exports = ExecutableCreator;