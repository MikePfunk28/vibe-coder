#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Use Regular VSCode with AI Extensions
.DESCRIPTION
    Since the VSCode OSS build has PATH issues, use your regular VSCode with AI extensions
#>

Write-Host "🎯 Using Regular VSCode with AI Extensions" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check if regular VSCode is available
$VSCodePath = Get-Command code -ErrorAction SilentlyContinue
if (-not $VSCodePath) {
    Write-Host "❌ VSCode not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "📦 Installing AI Assistant extension..." -ForegroundColor Cyan

# Install the AI Assistant extension
if (Test-Path "extensions/ai-assistant/ai-assistant-0.1.0.vsix") {
    try {
        & code --install-extension "extensions/ai-assistant/ai-assistant-0.1.0.vsix" --force
        Write-Host "✅ AI Assistant extension installed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ Extension installation failed: $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "⚠️ AI Assistant extension not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Opening workspace in VSCode..." -ForegroundColor Cyan

# Open the current directory in a new VSCode window
& code . --new-window

Write-Host ""
Write-Host "🎉 VSCode opened with AI capabilities!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Available Features:" -ForegroundColor Cyan
Write-Host "  • AI Assistant Extension (if installed)" -ForegroundColor White
Write-Host "  • Backend Services Running:" -ForegroundColor White
Write-Host "    - Main Backend: http://localhost:8000" -ForegroundColor Gray
Write-Host "    - Copilot API: http://localhost:8001" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 To use AI features:" -ForegroundColor Cyan
Write-Host "  1. Check if AI Assistant appears in the activity bar" -ForegroundColor White
Write-Host "  2. If not, manually install: extensions/ai-assistant/ai-assistant-0.1.0.vsix" -ForegroundColor White
Write-Host "  3. Use Ctrl+L for AI chat" -ForegroundColor White
Write-Host "  4. Use Ctrl+K for inline AI completion" -ForegroundColor White
Write-Host ""
Write-Host "✨ Your regular VSCode now has AI superpowers!" -ForegroundColor Green