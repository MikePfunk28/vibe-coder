@echo off
echo ========================================
echo   SIMPLE VSCODE OSS + AI
echo ========================================
echo.
echo Starting VSCode with Electron...
echo.

cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

echo Trying to start with Electron...
npx electron .

if errorlevel 1 (
    echo.
    echo ❌ Electron failed, trying alternative...
    echo.
    
    REM Try with node
    node main.js
    
    if errorlevel 1 (
        echo.
        echo ❌ All methods failed
        echo.
        echo Available files:
        dir /b *.js *.json 2>nul
        echo.
        echo Try installing Node.js 18.x instead of 22.x
        pause
    )
)
