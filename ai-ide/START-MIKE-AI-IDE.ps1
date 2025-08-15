#!/usr/bin/env pwsh
<#
.SYNOPSIS
    One-Click Launcher for Mike-AI-IDE
.DESCRIPTION
    Starts all backend services and launches the AI IDE
#>

Write-Host "🚀 Starting Mike-AI-IDE..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if we're in the right directory
if (-not (Test-Path "backend/main.py")) {
    Write-Host "❌ Please run this script from the ai-ide directory" -ForegroundColor Red
    exit 1
}

# Start backend services in background
Write-Host "📡 Starting AI Backend Services..." -ForegroundColor Cyan

# Start main backend
Write-Host "  Starting main AI backend..." -ForegroundColor Gray
$BackendProcess = Start-Process -FilePath "python" -ArgumentList "backend/main.py", "--server" -PassThru -WindowStyle Hidden

# Wait a moment for backend to initialize
Start-Sleep -Seconds 3

# Start Copilot API
Write-Host "  Starting Copilot API server..." -ForegroundColor Gray
$CopilotProcess = Start-Process -FilePath "python" -ArgumentList "backend/copilot_api.py" -PassThru -WindowStyle Hidden

# Wait for services to start
Write-Host "  Waiting for services to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Test backend connectivity
try {
    Write-Host "🔍 Testing backend connectivity..." -ForegroundColor Cyan
    
    # Test main backend
    $BackendHealth = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET -TimeoutSec 5
    if ($BackendHealth) {
        Write-Host "  ✅ Main backend: Online" -ForegroundColor Green
    }
    
    # Test Copilot API
    $CopilotHealth = Invoke-RestMethod -Uri "http://localhost:8001/health" -Method GET -TimeoutSec 5
    if ($CopilotHealth) {
        Write-Host "  ✅ Copilot API: Online" -ForegroundColor Green
    }
    
} catch {
    Write-Host "  ⚠️ Backend services starting... (this is normal)" -ForegroundColor Yellow
}

# Launch Mike-AI-IDE
Write-Host "🎯 Launching Mike-AI-IDE..." -ForegroundColor Cyan

if (Test-Path "START-REAL-VSCODE-COMPLETE.bat") {
    & ".\START-REAL-VSCODE-COMPLETE.bat"
} elseif (Test-Path "vscode-oss-complete/scripts/code.bat") {
    & ".\vscode-oss-complete\scripts\code.bat" .
} else {
    Write-Host "❌ VSCode launcher not found. Please build VSCode first:" -ForegroundColor Red
    Write-Host "   .\BUILD-COMPLETE-VSCODE.ps1" -ForegroundColor Yellow
    
    # Clean up processes
    if ($BackendProcess -and !$BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
    }
    if ($CopilotProcess -and !$CopilotProcess.HasExited) {
        Stop-Process -Id $CopilotProcess.Id -Force -ErrorAction SilentlyContinue
    }
    exit 1
}

Write-Host ""
Write-Host "🎉 Mike-AI-IDE is starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 What's Available:" -ForegroundColor Cyan
Write-Host "  • Dual AI Chat (Ctrl+L)" -ForegroundColor White
Write-Host "  • GitHub Copilot (Ctrl+Shift+C)" -ForegroundColor White
Write-Host "  • Inline AI Completion (Ctrl+K)" -ForegroundColor White
Write-Host "  • Context Features (#File, #Folder, #Problems)" -ForegroundColor White
Write-Host "  • Multi-Model AI Support" -ForegroundColor White
Write-Host "  • Open VSX Extensions" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Backend Services:" -ForegroundColor Cyan
Write-Host "  • Main Backend: http://localhost:8000" -ForegroundColor White
Write-Host "  • Copilot API: http://localhost:8001" -ForegroundColor White
Write-Host ""
Write-Host "⚠️ To stop services later, close this window or press Ctrl+C" -ForegroundColor Yellow

# Keep script running to monitor processes
try {
    Write-Host "🔄 Monitoring backend services... (Press Ctrl+C to stop)" -ForegroundColor Gray
    
    while ($true) {
        Start-Sleep -Seconds 30
        
        # Check if processes are still running
        if ($BackendProcess.HasExited) {
            Write-Host "⚠️ Main backend stopped unexpectedly" -ForegroundColor Yellow
        }
        if ($CopilotProcess.HasExited) {
            Write-Host "⚠️ Copilot API stopped unexpectedly" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "🛑 Stopping backend services..." -ForegroundColor Yellow
    
    # Clean up processes
    if ($BackendProcess -and !$BackendProcess.HasExited) {
        Stop-Process -Id $BackendProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Main backend stopped" -ForegroundColor Green
    }
    if ($CopilotProcess -and !$CopilotProcess.HasExited) {
        Stop-Process -Id $CopilotProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Copilot API stopped" -ForegroundColor Green
    }
    
    Write-Host "👋 Mike-AI-IDE services stopped" -ForegroundColor Green
}