@echo off
echo 🚀 AI IDE Build Environment Fixer
echo Creating a complete VSCode OSS + AI IDE like Cursor
echo.

REM Check if PowerShell is available
powershell -Command "Get-Host" >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ PowerShell is required but not found
    echo Please install PowerShell or run the .ps1 script directly
    pause
    exit /b 1
)

echo Running PowerShell build script...
powershell -ExecutionPolicy Bypass -File "%~dp0fix-build-environment.ps1" %*

if %errorlevel% neq 0 (
    echo.
    echo ❌ Build failed. Check the output above for errors.
    pause
    exit /b 1
)

echo.
echo 🎉 Build completed successfully!
echo Run launch-ai-ide.bat to start your AI IDE
pause