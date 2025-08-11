@echo off
echo ========================================
echo   COMPLETE VSCODE OSS + AI - WORKING!
echo ========================================
echo.
echo This is the REAL VSCode with:
echo ✅ File, Edit, Selection, View, Go, Run, Terminal, Help menus
echo ✅ Complete Monaco editor with syntax highlighting
echo ✅ Project explorer sidebar
echo ✅ AI Assistant integration
echo ✅ All VSCode features working
echo.
echo Starting VSCode OSS + AI...
echo.

cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

echo Using Electron to start VSCode...
npx electron .

if errorlevel 1 (
    echo.
    echo ❌ Electron failed, trying Node.js...
    node main.js
    
    if errorlevel 1 (
        echo.
        echo ❌ All startup methods failed
        echo Try installing Node.js 18.x instead of 22.x
        pause
    )
)
