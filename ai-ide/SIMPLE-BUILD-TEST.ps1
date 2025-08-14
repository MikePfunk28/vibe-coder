# Simple Build and Test for Mike-AI-IDE
Write-Host "🔨 Building Mike-AI-IDE..." -ForegroundColor Green

# Check if VSCode OSS directory exists
if (-not (Test-Path "ai-ide/vscode-oss-complete")) {
    Write-Host "❌ VSCode OSS directory not found" -ForegroundColor Red
    exit 1
}

# Build VSCode OSS
Write-Host "📦 Building VSCode OSS..." -ForegroundColor Yellow
Push-Location "ai-ide/vscode-oss-complete"

# Check if already built
if (Test-Path "out/main.js") {
    Write-Host "✅ VSCode OSS already built" -ForegroundColor Green
} else {
    Write-Host "🔨 Compiling..." -ForegroundColor Cyan
    if (Test-Path "node_modules") {
        yarn compile
    } else {
        Write-Host "📥 Installing dependencies first..." -ForegroundColor Cyan
        yarn install
        yarn compile
    }
}

Pop-Location

# Test launch
Write-Host "🚀 Testing launch..." -ForegroundColor Yellow
Push-Location "ai-ide/vscode-oss-complete"

# Set environment
$env:MIKE_AI_IDE = "true"

# Try to launch (will exit quickly)
Write-Host "🎯 Testing Mike-AI-IDE launch..." -ForegroundColor Cyan
node main.js --version

Pop-Location

Write-Host "✅ Build and test complete!" -ForegroundColor Green