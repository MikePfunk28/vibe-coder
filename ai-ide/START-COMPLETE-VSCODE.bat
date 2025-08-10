@echo off
echo ========================================
echo   COMPLETE VSCODE OSS + AI FEATURES
echo ========================================
echo.
echo Starting VSCode OSS with Electron...
echo.

cd /d "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

REM Set environment variables
set NODE_ENV=development
set VSCODE_DEV=1

REM Try different startup methods
if exist "scripts\code.bat" (
    echo Using scripts\code.bat
    call scripts\code.bat %*
) else if exist "scripts\code.js" (
    echo Using scripts\code.js with Node
    node scripts\code.js %*
) else if exist "out\main.js" (
    echo Using out\main.js
    node out\main.js %*
) else (
    echo Using Electron directly
    npx electron . %*
)

if errorlevel 1 (
    echo.
    echo ❌ VSCode failed to start
    echo Trying alternative startup...
    echo.
    
    REM Try with yarn
    yarn electron
    
    if errorlevel 1 (
        echo.
        echo Available startup files:
        dir /b scripts\*.* out\*.* *.js 2>nul
        echo.
        echo Try running: npx electron .
        pause
    )
)
