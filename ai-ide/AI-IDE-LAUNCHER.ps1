#!/usr/bin/env pwsh
<#
.SYNOPSIS
    AI IDE Complete Launcher
.DESCRIPTION
    Launches the complete AI IDE with all services
#>

Write-Host "🚀 AI IDE - Complete AI Development Environment" -ForegroundColor Cyan
Write-Host "Built on VSCodium with integrated AI capabilities" -ForegroundColor Green
Write-Host ""

# Check if LM Studio is running
Write-Host "🔍 Checking AI services..." -ForegroundColor Yellow
$lmStudioRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:1234/v1/models" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        $lmStudioRunning = $true
        Write-Host "✅ LM Studio is running" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  LM Studio not detected on localhost:1234" -ForegroundColor Yellow
    Write-Host "   You can start it manually for enhanced AI features" -ForegroundColor Gray
}

# Start AI Backend
Write-Host "🧠 Starting AI Backend..." -ForegroundColor Yellow
$backendPath = Join-Path $PSScriptRoot "backend"
$backendProcess = Start-Process -FilePath "python" -ArgumentList "main.py" -WorkingDirectory $backendPath -WindowStyle Hidden -PassThru

# Wait for backend to initialize
Start-Sleep -Seconds 3

# Check if backend started successfully
if ($backendProcess -and !$backendProcess.HasExited) {
    Write-Host "✅ AI Backend started successfully" -ForegroundColor Green
} else {
    Write-Host "⚠️  AI Backend may have issues - check logs" -ForegroundColor Yellow
}

# Start VSCodium with AI IDE configuration
Write-Host "🎨 Launching AI IDE (VSCodium + AI)..." -ForegroundColor Yellow

$vscodiumPath = "$env:LOCALAPPDATA\Programs\VSCodium\VSCodium.exe"
if (Test-Path $vscodiumPath) {
    # Launch VSCodium with the current workspace
    Start-Process -FilePath $vscodiumPath -ArgumentList @("--new-window", ".")
    
    Write-Host ""
    Write-Host "🎉 AI IDE is starting!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Available AI Features:" -ForegroundColor Cyan
    Write-Host "  • Ctrl+K - Inline AI code generation" -ForegroundColor White
    Write-Host "  • Ctrl+L - AI chat panel" -ForegroundColor White  
    Write-Host "  • Ctrl+Shift+S - Semantic code search" -ForegroundColor White
    Write-Host "  • Ctrl+Shift+R - AI reasoning" -ForegroundColor White
    Write-Host ""
    Write-Host "Backend Services:" -ForegroundColor Cyan
    Write-Host "  • PocketFlow reasoning engine" -ForegroundColor White
    Write-Host "  • Semantic search and indexing" -ForegroundColor White
    Write-Host "  • Multi-model AI support" -ForegroundColor White
    if ($lmStudioRunning) {
        Write-Host "  • LM Studio integration (active)" -ForegroundColor Green
    } else {
        Write-Host "  • LM Studio integration (start LM Studio for full features)" -ForegroundColor Gray
    }
    Write-Host ""
    Write-Host "🔧 Backend Process ID: $($backendProcess.Id)" -ForegroundColor Gray
    
} else {
    Write-Host "❌ VSCodium not found at expected location" -ForegroundColor Red
    Write-Host "   Expected: $vscodiumPath" -ForegroundColor Gray
    Write-Host "   Please install VSCodium or update the path" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")