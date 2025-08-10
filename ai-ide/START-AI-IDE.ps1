#!/usr/bin/env pwsh
Write-Host "🚀 Starting AI IDE (Complete VSCode OSS + AI)" -ForegroundColor Cyan
Write-Host ""
Write-Host "This is the REAL VSCode OSS with ALL features:" -ForegroundColor Green
Write-Host "✅ Complete VSCode functionality"
Write-Host "✅ All built-in extensions"
Write-Host "✅ Full debugging support"
Write-Host "✅ Integrated terminal"
Write-Host "✅ Git integration"
Write-Host "✅ Extension marketplace"
Write-Host "✅ AI features (Ctrl+K, Ctrl+L)"
Write-Host ""

Set-Location "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

if (Test-Path "scripts\code.js") {
    node scripts\code.js @args
} elseif (Test-Path "out\main.js") {
    node out\main.js @args
} else {
    Write-Host "❌ VSCode startup script not found" -ForegroundColor Red
    Write-Host "Build may have failed. Check the build output." -ForegroundColor Red
}
