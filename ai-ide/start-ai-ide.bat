@echo off
echo ========================================
echo        AI IDE - Starting...
echo ========================================
echo.

REM Start AI Backend in background
echo Starting AI Backend...
start /B "AI Backend" cmd /c "cd /d %~dp0backend && python main.py"

REM Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

REM Start VSCodium with AI IDE branding
echo Starting AI IDE (VSCodium + AI)...
start "" "C:\Users\%USERNAME%\AppData\Local\Programs\VSCodium\VSCodium.exe" --new-window .

echo.
echo AI IDE is starting...
echo - AI Backend running on localhost
echo - VSCodium with AI Assistant extension
echo.
echo Use Ctrl+K for inline AI generation
echo Use Ctrl+L for AI chat
echo Use Ctrl+Shift+S for semantic search
echo.
pause