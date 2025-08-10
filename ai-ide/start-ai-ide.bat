@echo off
echo 🚀 Starting AI IDE (Complete VSCode OSS + AI)
echo.
echo This is the REAL VSCode OSS with ALL features:
echo ✅ Complete VSCode functionality
echo ✅ All built-in extensions
echo ✅ Full debugging support
echo ✅ Integrated terminal
echo ✅ Git integration
echo ✅ Extension marketplace
echo ✅ AI features (Ctrl+K, Ctrl+L)
echo.
echo Starting AI IDE...
cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

REM Try different startup methods
if exist "scripts\code.bat" (
    call scripts\code.bat %*
) else if exist "scripts\code.js" (
    node scripts\code.js %*
) else if exist "out\main.js" (
    node out\main.js %*
) else (
    echo ❌ VSCode startup script not found
    echo Build may have failed. Check the output above.
    pause
)
