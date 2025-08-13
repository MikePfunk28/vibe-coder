# Launch Mike-AI-IDE - Native AI IDE with VSCode Foundation
Write-Host "🚀 Starting Mike-AI-IDE..." -ForegroundColor Green

# Set environment variables for AI features
$env:MIKE_AI_IDE = "true"
$env:AI_BACKEND_URL = "http://localhost:8000"
$env:OLLAMA_URL = "http://localhost:11434"
$env:LM_STUDIO_URL = "http://localhost:1234"

# Start AI backend if not running
$backendProcess = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*main.py*" }
if (-not $backendProcess) {
    Write-Host "🤖 Starting AI Backend..." -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "ai-ide/vscode-oss-complete/ai-backend/main.py" -WindowStyle Hidden
    Start-Sleep 3
}

# Check if Ollama is running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "🦙 Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 2
}

# Launch Mike-AI-IDE
Write-Host "🎯 Launching Mike-AI-IDE with native AI features..." -ForegroundColor Cyan
Write-Host "   • Ctrl+K: AI Code Generation" -ForegroundColor White
Write-Host "   • Ctrl+L: AI Chat" -ForegroundColor White
Write-Host "   • Ctrl+Shift+E: Explain Code" -ForegroundColor White
Write-Host "   • Ctrl+Shift+R: Refactor Code" -ForegroundColor White
Write-Host "   • Ctrl+Shift+F: Fix Code" -ForegroundColor White

# Launch the IDE
$idePath = "ai-ide/vscode-oss-complete/scripts/code.bat"
if (Test-Path $idePath) {
    & $idePath
} else {
    # Fallback to direct node execution
    Push-Location "ai-ide/vscode-oss-complete"
    node main.js
    Pop-Location
}

Write-Host "✅ Mike-AI-IDE launched successfully!" -ForegroundColor Green