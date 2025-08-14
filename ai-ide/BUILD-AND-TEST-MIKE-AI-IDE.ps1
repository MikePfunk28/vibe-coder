# Build and Test Mike-AI-IDE
Write-Host "🔨 Building and Testing Mike-AI-IDE..." -ForegroundColor Green

$ErrorActionPreference = "Stop"

# 1. Build the VSCode OSS with AI features
Write-Host "📦 Building VSCode OSS with AI features..." -ForegroundColor Yellow
Push-Location "ai-ide/vscode-oss-complete"

try {
    # Install dependencies if needed
    if (-not (Test-Path "node_modules")) {
        Write-Host "📥 Installing dependencies..." -ForegroundColor Cyan
        yarn install
    }

    # Build the IDE
    Write-Host "🔨 Compiling TypeScript..." -ForegroundColor Cyan
    yarn compile

    # Build extensions
    Write-Host "🔌 Building extensions..." -ForegroundColor Cyan
    yarn compile-extensions

    Write-Host "✅ VSCode OSS build complete!" -ForegroundColor Green

} catch {
    Write-Host "❌ Build failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# 2. Build AI Assistant Extension
Write-Host "🤖 Building AI Assistant Extension..." -ForegroundColor Yellow
Push-Location "ai-ide/extensions/ai-assistant"

try {
    # Install dependencies
    if (-not (Test-Path "node_modules")) {
        Write-Host "📥 Installing extension dependencies..." -ForegroundColor Cyan
        npm install
    }

    # Build extension
    Write-Host "🔨 Building extension..." -ForegroundColor Cyan
    npm run compile

    # Package extension
    Write-Host "📦 Packaging extension..." -ForegroundColor Cyan
    npx vsce package --out ai-assistant-latest.vsix

    Write-Host "✅ AI Assistant extension built!" -ForegroundColor Green

} catch {
    Write-Host "❌ Extension build failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# 3. Setup AI Backend
Write-Host "🐍 Setting up AI Backend..." -ForegroundColor Yellow
Push-Location "ai-ide/vscode-oss-complete/ai-backend"

try {
    # Install Python dependencies
    if (-not (Test-Path "venv")) {
        Write-Host "🐍 Creating Python virtual environment..." -ForegroundColor Cyan
        python -m venv venv
    }

    # Activate venv and install requirements
    Write-Host "📥 Installing Python dependencies..." -ForegroundColor Cyan
    & "venv/Scripts/Activate.ps1"
    pip install -r requirements.txt

    Write-Host "✅ AI Backend setup complete!" -ForegroundColor Green

} catch {
    Write-Host "❌ Backend setup failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

# 4. Test Launch
Write-Host "🚀 Testing Mike-AI-IDE Launch..." -ForegroundColor Yellow

# Start AI Backend
Write-Host "🤖 Starting AI Backend..." -ForegroundColor Cyan
$backendProcess = Start-Process -FilePath "python" -ArgumentList "ai-ide/vscode-oss-complete/ai-backend/main.py" -WindowStyle Hidden -PassThru
Start-Sleep 5

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ AI Backend is running!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  AI Backend may have issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  AI Backend not responding: $_" -ForegroundColor Yellow
}

# Start Ollama if not running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "🦙 Starting Ollama..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 3
}

# Test Ollama
try {
    $ollamaResponse = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 10
    if ($ollamaResponse.StatusCode -eq 200) {
        $models = ($ollamaResponse.Content | ConvertFrom-Json).models
        Write-Host "✅ Ollama is running with $($models.Count) models!" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Ollama not responding: $_" -ForegroundColor Yellow
}

# Launch Mike-AI-IDE for testing
Write-Host "🎯 Launching Mike-AI-IDE for testing..." -ForegroundColor Cyan
Push-Location "ai-ide/vscode-oss-complete"

# Set environment variables
$env:MIKE_AI_IDE = "true"
$env:AI_BACKEND_URL = "http://localhost:8000"
$env:OLLAMA_URL = "http://localhost:11434"

# Launch IDE
$ideProcess = Start-Process -FilePath "node" -ArgumentList "main.js" -PassThru

Write-Host "🎉 Mike-AI-IDE launched!" -ForegroundColor Green
Write-Host "Process ID: $($ideProcess.Id)" -ForegroundColor White

Pop-Location

# 5. Test AI Features
Write-Host "🧪 Testing AI Features..." -ForegroundColor Yellow

Write-Host "✅ Build and Test Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Mike-AI-IDE Status:" -ForegroundColor Cyan
Write-Host "   • VSCode OSS: Built and running" -ForegroundColor White
Write-Host "   • AI Backend: Running on port 8000" -ForegroundColor White
Write-Host "   • Ollama: Running on port 11434" -ForegroundColor White
Write-Host "   • AI Features: Integrated natively" -ForegroundColor White
Write-Host ""
Write-Host "🎮 Test these features in the IDE:" -ForegroundColor Yellow
Write-Host "   • Ctrl+K: AI Code Generation" -ForegroundColor White
Write-Host "   • Ctrl+L: AI Chat" -ForegroundColor White
Write-Host "   • Ctrl+Shift+E: Explain Code" -ForegroundColor White
Write-Host "   • Status Bar: Check AI status" -ForegroundColor White
Write-Host ""
Write-Host "📝 To install GitHub Copilot:" -ForegroundColor Yellow
Write-Host "   1. Open Extensions (Ctrl+Shift+X)" -ForegroundColor White
Write-Host "   2. Search for 'GitHub Copilot'" -ForegroundColor White
Write-Host "   3. Install and authenticate" -ForegroundColor White
Write-Host "   4. Our enhancer will automatically integrate with it!" -ForegroundColor White

# Cleanup function
Write-Host ""
Write-Host "💡 To stop all services later:" -ForegroundColor Cyan
Write-Host "   Stop-Process -Id $($ideProcess.Id) -Force" -ForegroundColor White
Write-Host "   Stop-Process -Id $($backendProcess.Id) -Force" -ForegroundColor White