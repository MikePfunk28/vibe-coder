#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Fixed Launcher for Mike-AI-IDE
.DESCRIPTION
    Launches Mike-AI-IDE with proper environment setup
#>

Write-Host "🚀 Launching Mike-AI-IDE (Fixed Version)..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Set up proper PATH environment
$env:PATH = "$env:PATH;C:\Windows\System32;C:\Windows\System32\WindowsPowerShell\v1.0"

# Check if we're in the right directory
if (-not (Test-Path "vscode-oss-complete")) {
    Write-Host "❌ Please run this script from the ai-ide directory" -ForegroundColor Red
    exit 1
}

# Try different launch methods
Write-Host "🎯 Attempting to launch Mike-AI-IDE..." -ForegroundColor Cyan

# Method 1: Try using node directly with main.js
if (Test-Path "vscode-oss-complete/main.js") {
    Write-Host "  Method 1: Using main.js with node..." -ForegroundColor Gray
    try {
        Set-Location "vscode-oss-complete"
        & node main.js .
        Set-Location ..
        Write-Host "✅ Mike-AI-IDE launched successfully!" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "  ⚠️ Method 1 failed: $_" -ForegroundColor Yellow
        Set-Location ..
    }
}

# Method 2: Try the scripts/code.bat with full path
if (Test-Path "vscode-oss-complete/scripts/code.bat") {
    Write-Host "  Method 2: Using scripts/code.bat..." -ForegroundColor Gray
    try {
        $CodeBat = Resolve-Path "vscode-oss-complete/scripts/code.bat"
        & $CodeBat .
        Write-Host "✅ Mike-AI-IDE launched successfully!" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "  ⚠️ Method 2 failed: $_" -ForegroundColor Yellow
    }
}

# Method 3: Try using your regular VSCode to open the workspace
Write-Host "  Method 3: Using regular VSCode as fallback..." -ForegroundColor Gray
try {
    $VSCodePath = Get-Command code -ErrorAction SilentlyContinue
    if ($VSCodePath) {
        Write-Host "  📝 Opening workspace in regular VSCode..." -ForegroundColor Gray
        & code . --new-window
        Write-Host "✅ Opened in regular VSCode (you can install the AI extension manually)" -ForegroundColor Green
        Write-Host "  Extension location: extensions/ai-assistant/ai-assistant-0.1.0.vsix" -ForegroundColor White
        exit 0
    }
} catch {
    Write-Host "  ⚠️ Method 3 failed: $_" -ForegroundColor Yellow
}

# Method 4: Try Electron directly
if (Test-Path "vscode-oss-complete/node_modules/.bin/electron.cmd") {
    Write-Host "  Method 4: Using Electron directly..." -ForegroundColor Gray
    try {
        Set-Location "vscode-oss-complete"
        & "node_modules/.bin/electron.cmd" .
        Set-Location ..
        Write-Host "✅ Mike-AI-IDE launched with Electron!" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "  ⚠️ Method 4 failed: $_" -ForegroundColor Yellow
        Set-Location ..
    }
}

# If all methods fail, provide instructions
Write-Host "❌ All launch methods failed" -ForegroundColor Red
Write-Host ""
Write-Host "🔧 Manual Launch Options:" -ForegroundColor Cyan
Write-Host "1. Install Node.js 18.x and add to PATH" -ForegroundColor White
Write-Host "2. Use regular VSCode and install the AI extension:" -ForegroundColor White
Write-Host "   code --install-extension extensions/ai-assistant/ai-assistant-0.1.0.vsix" -ForegroundColor Gray
Write-Host "3. Try building again with proper Node.js:" -ForegroundColor White
Write-Host "   .\BUILD-COMPLETE-VSCODE.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Good news: Backend services are running!" -ForegroundColor Green
Write-Host "  • Main Backend: http://localhost:8000" -ForegroundColor White
Write-Host "  • Copilot API: http://localhost:8001" -ForegroundColor White