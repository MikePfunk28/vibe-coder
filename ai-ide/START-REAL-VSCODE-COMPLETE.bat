@echo off
echo ========================================
echo   REAL VSCODE OSS - COMPLETE IDE
echo ========================================
echo.
echo Starting REAL VSCode with ALL features:
echo ✅ Complete File menu (New, Open, Save, Save As, Save All, Recent, Preferences, Exit)
echo ✅ Complete Edit menu (Undo, Redo, Cut, Copy, Paste, Find, Replace, etc.)
echo ✅ Complete View menu (Explorer, Search, Source Control, Debug, Extensions, etc.)
echo ✅ Complete debugging, terminal, git integration
echo ✅ Extension marketplace and all VSCode features
echo.

cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

REM Start REAL VSCode
call scripts\code.bat %*

if errorlevel 1 (
    echo.
    echo ❌ VSCode failed to start properly
    echo This means the build is incomplete
    pause
)
