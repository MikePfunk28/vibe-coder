# 🚀 Mike-AI-IDE Setup Guide

## What is Mike-AI-IDE?

**Mike-AI-IDE** is a complete AI-powered IDE built on VSCode OSS with **native AI features** integrated directly into the core. Unlike extensions, these AI features are built into the IDE itself, providing:

- **Native AI Code Generation** (Ctrl+K) - Like Cursor
- **Native AI Chat** (Ctrl+L) - Like Cursor  
- **Native AI Code Analysis** (Ctrl+Shift+E/R/F)
- **Multi-Model Support** - Ollama, LM Studio, OpenAI, Claude, etc.
- **Complete VSCode Compatibility** - All extensions, themes, settings work
- **Advanced AI Backend** - PocketFlow, LangChain, semantic search, etc.

## 🎯 Current Status

✅ **Complete VSCode OSS Foundation** - Full IDE with all features  
✅ **Native AI Features Integrated** - Built into the core, not extensions  
✅ **Advanced AI Backend** - Multi-model, semantic search, workflows  
✅ **Product Configuration** - Branded as Mike-AI-IDE  
✅ **Launcher Scripts** - Ready to run  

## 🚀 Quick Start

### 1. Launch Mike-AI-IDE

```powershell
# Run this command to start everything
./ai-ide/LAUNCH-MIKE-AI-IDE.ps1
```

This will:
- Start the AI backend (Python server)
- Start Ollama (if not running)
- Launch Mike-AI-IDE with native AI features

### 2. Verify AI Features Work

Once Mike-AI-IDE opens:

1. **Test AI Code Generation**: Press `Ctrl+K` and type "create a hello world function"
2. **Test AI Chat**: Press `Ctrl+L` to open the AI chat panel
3. **Test Code Analysis**: Select some code and press `Ctrl+Shift+E` to explain it

### 3. Check AI Status

Look at the bottom-right status bar for AI status:
- `🤖 AI: X Local` - Shows number of local models available
- Click it to see available models and providers

## 🔧 Manual Setup (if needed)

### Prerequisites

1. **Node.js** (already installed)
2. **Python** (already installed) 
3. **Ollama** (already installed with models)
4. **LM Studio** (optional, already available)

### Build Mike-AI-IDE (if needed)

```powershell
cd ai-ide/vscode-oss-complete

# Install dependencies
yarn install

# Build the IDE
yarn compile

# Start the IDE
node main.js
```

### Start AI Backend

```powershell
cd ai-ide/vscode-oss-complete/ai-backend
python main.py
```

### Start Ollama

```powershell
ollama serve
```

## 🎮 AI Features Usage

### Native AI Code Generation (Ctrl+K)
- Press `Ctrl+K` anywhere in a file
- Type what you want to generate
- AI will generate code inline

### Native AI Chat (Ctrl+L)  
- Press `Ctrl+L` to open AI chat
- Ask questions about your code
- Get explanations and suggestions

### Code Analysis
- `Ctrl+Shift+E` - Explain selected code
- `Ctrl+Shift+R` - Refactor selected code  
- `Ctrl+Shift+F` - Fix selected code

### Model Selection
- Click AI status in status bar
- Choose from available models
- Compare multiple models

## 🤖 Available AI Models

Your Ollama setup includes excellent coding models:
- **devstral** - Excellent for code generation
- **deepseek-r1:8b** - Great for reasoning
- **qwen2.5-coder** - Specialized for coding
- **gemma3:12b** - Good general purpose

## 🔧 Configuration

### AI Backend Configuration
Edit `ai-ide/vscode-oss-complete/ai-backend/main.py` to:
- Add API keys for cloud models
- Configure model preferences
- Adjust AI behavior

### IDE Configuration
Mike-AI-IDE uses standard VSCode settings:
- Settings are stored in `.mike-ai-ide` folder
- All VSCode extensions work
- Themes and customizations work

## 🚨 Troubleshooting

### AI Features Not Working
1. Check AI backend is running: `http://localhost:8000`
2. Check Ollama is running: `http://localhost:11434`
3. Look at status bar for AI status
4. Check Developer Console (F12) for errors

### Models Not Loading
1. Verify Ollama is running: `ollama list`
2. Check model availability in AI status
3. Try restarting the AI backend

### IDE Won't Start
1. Check Node.js version: `node --version`
2. Rebuild if needed: `yarn compile`
3. Check for port conflicts

## 🎯 Next Steps

1. **Test all AI features** to ensure they work
2. **Configure your preferred models** in the backend
3. **Add API keys** for cloud models if desired
4. **Customize the IDE** with your preferred extensions/themes

## 🆚 Mike-AI-IDE vs Others

| Feature | Mike-AI-IDE | Cursor | VSCode + Extensions |
|---------|-------------|--------|-------------------|
| AI Integration | Native | Native | Extension-based |
| VSCode Compatibility | 100% | ~90% | 100% |
| Local Models | ✅ | Limited | Limited |
| Multi-Model | ✅ | Limited | Varies |
| Customizable | ✅ | Limited | ✅ |
| Free | ✅ | Freemium | Varies |

**Mike-AI-IDE gives you Cursor-level AI features with complete VSCode compatibility and full control over your AI models.**

---

🚀 **Ready to experience the future of AI-powered development with Mike-AI-IDE!**