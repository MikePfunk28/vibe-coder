@echo off
echo 🚀 Starting AI IDE (Complete VSCode OSS + AI Features)
echo.
echo ✨ Features Available:
echo   • Complete VSCode functionality (debugging, git, extensions, etc.)
echo   • Ctrl+K: AI inline code generation (like Cursor)
echo   • Ctrl+L: AI chat panel (like Cursor)  
echo   • Right-click → AI Explain Code
echo   • Full extension marketplace support
echo.
echo 🤖 AI Features Status: Demo Mode
echo    Connect to backend/main.py for full AI capabilities
echo.
echo Starting AI IDE...

"vscode-oss-working\Code.exe" --new-window --disable-telemetry %*
