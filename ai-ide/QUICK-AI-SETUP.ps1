# Quick AI Setup - Single Run Script
Write-Host "=== Quick AI Setup for VSCodium ===" -ForegroundColor Green

$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

# Install the best AI extension available for VSCodium
Write-Host "Installing Claude Dev (Cline) - the best Cursor alternative..." -ForegroundColor Yellow
& $vscodium --install-extension saoudrizwan.claude-dev --force

Write-Host "✅ Done! Restart VSCodium and use Ctrl+Shift+P then type 'Cline' to access AI features" -ForegroundColor Green
Write-Host "Your Ollama models are ready to use!" -ForegroundColor Cyan