#!/usr/bin/env powershell
# Quick start script for Vibe Coder workspace

Write-Host "🚀 Vibe Coder Workspace Quick Start" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    python --version
    Write-Host "✅ Python is installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    node --version
    Write-Host "✅ Node.js is installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Node.js not found. Install for VSCode extension development" -ForegroundColor Yellow
}

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Set up browser automation
if (Test-Path "browser-automation") {
    Write-Host "Setting up browser automation..." -ForegroundColor Yellow
    # Browser automation dependencies are in main requirements.txt
}

# Create .env template if it doesn't exist
if (!(Test-Path ".env")) {
    Write-Host "Creating .env template..." -ForegroundColor Yellow
    @'
# LLM Configuration
LMSTUDIO_BASE_URL=http://localhost:1234
OLLAMA_BASE_URL=http://localhost:11434

# Model Selection
DEFAULT_INSTRUCT_MODEL=phi-4-mini-instruct
DEFAULT_REASONING_MODEL=microsoft/phi-4-mini-reasoning
DEFAULT_REASONING_PLUS_MODEL=microsoft/phi-4-reasoning-plus

# Browser Automation
CHROME_PROFILE_PATH=C:/Users/[username]/AppData/Local/Google/Chrome/User Data/Default

# Logging
LOG_LEVEL=INFO
'@ | Out-File .env -Encoding utf8
    Write-Host "✅ Created .env template - please customize for your setup" -ForegroundColor Green
}

Write-Host ""
Write-Host "🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start LMStudio or Ollama:" -ForegroundColor White
Write-Host "   - LMStudio: Load phi-4-mini-instruct model" -ForegroundColor Gray
Write-Host "   - Ollama: ollama pull phi3" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Run the PocketFlow agent:" -ForegroundColor White
Write-Host "   python main.py --query 'Add error handling to my code'" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Try email automation:" -ForegroundColor White
Write-Host "   python main.py --query 'Send email to test@example.com'" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Interactive mode:" -ForegroundColor White
Write-Host "   python run_agent.py" -ForegroundColor Gray
