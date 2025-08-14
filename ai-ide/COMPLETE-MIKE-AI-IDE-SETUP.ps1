# Complete Mike-AI-IDE Setup - Build, Test, and Launch
Write-Host "🚀 Complete Mike-AI-IDE Setup Starting..." -ForegroundColor Green
Write-Host "This will build, test, and launch Mike-AI-IDE with all features" -ForegroundColor White

$ErrorActionPreference = "Continue"

# Step 1: Build and Test
Write-Host ""
Write-Host "📋 Step 1: Building and Testing Mike-AI-IDE..." -ForegroundColor Cyan
try {
    & "./ai-ide/BUILD-AND-TEST-MIKE-AI-IDE.ps1"
    Write-Host "✅ Build and test completed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Build failed: $_" -ForegroundColor Red
    Write-Host "Please fix build issues before continuing" -ForegroundColor Yellow
    exit 1
}

# Step 2: Install Extensions
Write-Host ""
Write-Host "📋 Step 2: Installing All Extensions..." -ForegroundColor Cyan
try {
    & "./ai-ide/INSTALL-ALL-EXTENSIONS.ps1"
    Write-Host "✅ Extensions installed!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Some extensions may have failed to install" -ForegroundColor Yellow
}

# Step 3: Configure GitHub Copilot
Write-Host ""
Write-Host "📋 Step 3: GitHub Copilot Configuration..." -ForegroundColor Cyan
Write-Host "🔑 To enable GitHub Copilot:" -ForegroundColor Yellow
Write-Host "   1. Open Mike-AI-IDE" -ForegroundColor White
Write-Host "   2. Press Ctrl+Shift+P" -ForegroundColor White
Write-Host "   3. Type 'GitHub Copilot: Sign In'" -ForegroundColor White
Write-Host "   4. Follow authentication steps" -ForegroundColor White
Write-Host "   5. Our enhancer will automatically integrate!" -ForegroundColor White

# Step 4: Final Launch
Write-Host ""
Write-Host "📋 Step 4: Final Launch with All Features..." -ForegroundColor Cyan

# Ensure all services are running
Write-Host "🔧 Starting all services..." -ForegroundColor Yellow

# Start AI Backend
$backendProcess = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*main.py*" }
if (-not $backendProcess) {
    Write-Host "🤖 Starting AI Backend..." -ForegroundColor Cyan
    Start-Process -FilePath "python" -ArgumentList "ai-ide/vscode-oss-complete/ai-backend/main.py" -WindowStyle Hidden
    Start-Sleep 3
}

# Start Ollama
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "🦙 Starting Ollama..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 2
}

# Launch Mike-AI-IDE
Write-Host "🎯 Launching Mike-AI-IDE with all features..." -ForegroundColor Green

# Set environment variables
$env:MIKE_AI_IDE = "true"
$env:AI_BACKEND_URL = "http://localhost:8000"
$env:OLLAMA_URL = "http://localhost:11434"
$env:LM_STUDIO_URL = "http://localhost:1234"

# Launch
Push-Location "ai-ide/vscode-oss-complete"
Start-Process -FilePath "node" -ArgumentList "main.js"
Pop-Location

Write-Host ""
Write-Host "🎉 Mike-AI-IDE Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Your AI IDE Features:" -ForegroundColor Cyan
Write-Host "   🤖 Native AI (Built-in):" -ForegroundColor White
Write-Host "      • Ctrl+K: AI Code Generation" -ForegroundColor Gray
Write-Host "      • Ctrl+L: AI Chat" -ForegroundColor Gray
Write-Host "      • Ctrl+Shift+E: Explain Code" -ForegroundColor Gray
Write-Host "      • Ctrl+Shift+R: Refactor Code" -ForegroundColor Gray
Write-Host "      • Ctrl+Shift+F: Fix Code" -ForegroundColor Gray
Write-Host ""
Write-Host "   🚀 GitHub Copilot (Extension):" -ForegroundColor White
Write-Host "      • Tab: Copilot completions" -ForegroundColor Gray
Write-Host "      • Ctrl+I: Copilot inline chat" -ForegroundColor Gray
Write-Host "      • Copilot Chat panel" -ForegroundColor Gray
Write-Host ""
Write-Host "   ⚡ Enhanced Features:" -ForegroundColor White
Write-Host "      • Multi-model comparison" -ForegroundColor Gray
Write-Host "      • Local model support" -ForegroundColor Gray
Write-Host "      • Semantic search" -ForegroundColor Gray
Write-Host "      • Advanced workflows" -ForegroundColor Gray
Write-Host ""
Write-Host "   🔧 Available Models:" -ForegroundColor White
Write-Host "      • devstral (excellent for code)" -ForegroundColor Gray
Write-Host "      • deepseek-r1:8b (great reasoning)" -ForegroundColor Gray
Write-Host "      • qwen2.5-coder (coding specialist)" -ForegroundColor Gray
Write-Host "      • gemma3:12b (general purpose)" -ForegroundColor Gray
Write-Host "      • + GitHub Copilot (when authenticated)" -ForegroundColor Gray
Write-Host ""
Write-Host "🎮 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Authenticate GitHub Copilot (Ctrl+Shift+P → 'GitHub Copilot: Sign In')" -ForegroundColor White
Write-Host "   2. Test native AI features (Ctrl+K, Ctrl+L)" -ForegroundColor White
Write-Host "   3. Try multi-model completions" -ForegroundColor White
Write-Host "   4. Explore the AI status in the status bar" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Welcome to Mike-AI-IDE - The Ultimate AI Development Environment!" -ForegroundColor Green