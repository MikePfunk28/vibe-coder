@echo off
echo 🚀 AI IDE - VSCode OSS + AI Features
echo.
echo ✨ Features:
echo   • Complete VSCode functionality
echo   • Ctrl+K: AI code generation
echo   • Ctrl+L: AI chat
echo   • All debugging, git, extensions
echo.
echo Starting AI IDE...
cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"
if exist "scripts\code.bat" (
  call scripts\code.bat %*
) else if exist "Code.exe" (
  Code.exe %*
) else if exist "bin\code.cmd" (
  call bin\code.cmd %*
) else (
  echo ❌ Could not find VSCode startup script
  dir /b *.exe *.bat *.cmd
  pause
)
