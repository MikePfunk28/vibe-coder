
@echo off
echo Starting AI IDE with VSCode OSS foundation...
echo.
echo Features available:
echo - Complete VSCode functionality (editing, debugging, git, extensions)
echo - Ctrl+K: Inline code generation (Cursor-style)
echo - Ctrl+L: AI chat panel (Cursor-style)  
echo - AI-powered autocomplete
echo - Semantic code search
echo - Multi-agent AI system
echo - Web search integration
echo - Advanced reasoning capabilities
echo.
echo To test Cursor-level features:
echo 1. Press Ctrl+K in any editor for inline code generation
echo 2. Press Ctrl+L to open AI chat panel
echo 3. Select code and press Ctrl+K to edit with AI
echo.
echo Starting AI IDE...
cd /d "C:\Users\mikep\vibe-coder\ai-ide\ai-ide-build"
node scripts/code.bat
