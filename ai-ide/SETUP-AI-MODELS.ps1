# Setup AI Models for VSCodium AI IDE
Write-Host "=== AI Model Setup for VSCodium ===" -ForegroundColor Green

# 1. Start Ollama and install essential models
Write-Host "`n🤖 Setting up Ollama models..." -ForegroundColor Cyan

$ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaPath) {
    Write-Host "❌ Ollama not found. Install from https://ollama.ai" -ForegroundColor Red
} else {
    Write-Host "✅ Ollama found" -ForegroundColor Green
    
    # Start Ollama service
    Write-Host "🚀 Starting Ollama service..." -ForegroundColor Yellow
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep 3
    
    # Install essential coding models
    $models = @("qwen2.5-coder:7b", "llama3.2:3b", "deepseek-coder:6.7b")
    
    foreach ($model in $models) {
        Write-Host "Installing $model..." -ForegroundColor Yellow
        & ollama pull $model
    }
}

# 2. Install AI Extensions
Write-Host "`n🔌 Installing AI Extensions..." -ForegroundColor Cyan
$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

if (Test-Path $vscodium) {
    $extensions = @("saoudrizwan.claude-dev", "continue.continue", "tabnine.tabnine-vscode")
    
    foreach ($ext in $extensions) {
        Write-Host "Installing $ext..." -ForegroundColor Yellow
        & $vscodium --install-extension $ext --force
    }
}

Write-Host "`n✅ Setup complete! Restart VSCodium to see AI features." -ForegroundColor Green