# Check VSCodium Extension Status
$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

Write-Host "=== VSCodium Extension Status Check ===" -ForegroundColor Green

# Check if VSCodium is installed
if (-not (Test-Path $vscodium)) {
    Write-Host "❌ VSCodium not found at $vscodium" -ForegroundColor Red
    exit 1
}

Write-Host "✅ VSCodium found at $vscodium" -ForegroundColor Green

# List currently installed extensions
Write-Host "`n📋 Currently installed extensions:" -ForegroundColor Cyan
try {
    $extensions = & $vscodium --list-extensions 2>$null
    if ($extensions) {
        foreach ($ext in $extensions) {
            Write-Host "  ✅ $ext" -ForegroundColor Green
        }
        Write-Host "`n📊 Total extensions installed: $($extensions.Count)" -ForegroundColor Yellow
    } else {
        Write-Host "  ⚠️  No extensions found or VSCodium not responding" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ❌ Error checking extensions: $_" -ForegroundColor Red
}

# Check if our AI Assistant extension is packaged
$extensionPath = "ai-ide/extensions/ai-assistant/ai-assistant-0.1.0.vsix"
if (Test-Path $extensionPath) {
    Write-Host "`n✅ Our AI Assistant extension is packaged at $extensionPath" -ForegroundColor Green
    
    # Try to install our extension
    Write-Host "📥 Installing our AI Assistant extension..." -ForegroundColor Yellow
    try {
        & $vscodium --install-extension $extensionPath --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ AI Assistant extension installed successfully!" -ForegroundColor Green
        } else {
            Write-Host "⚠️  AI Assistant extension installation may have issues" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Error installing AI Assistant extension: $_" -ForegroundColor Red
    }
} else {
    Write-Host "`n⚠️  AI Assistant extension not packaged yet" -ForegroundColor Yellow
    Write-Host "Run: cd ai-ide/extensions/ai-assistant && npx vsce package" -ForegroundColor Cyan
}

# Check marketplace configuration
Write-Host "`n🏪 Marketplace Configuration:" -ForegroundColor Cyan
Write-Host "  VSCodium uses Open VSX Registry (not Microsoft Marketplace)" -ForegroundColor White
Write-Host "  URL: https://open-vsx.org/" -ForegroundColor Gray

# List key AI extensions that ARE available on Open VSX
Write-Host "`n🤖 AI Extensions Available on Open VSX:" -ForegroundColor Cyan
$openVSXExtensions = @(
    @{Name="Claude Dev (Cline)"; ID="saoudrizwan.claude-dev"; Description="AI coding assistant similar to Cursor"},
    @{Name="Continue"; ID="continue.continue"; Description="Open-source AI code assistant"},
    @{Name="Tabnine"; ID="tabnine.tabnine-vscode"; Description="AI-powered code completions"},
    @{Name="Codeium"; ID="codeium.codeium"; Description="Free AI code completion"},
    @{Name="GitLens"; ID="eamodio.gitlens"; Description="Enhanced Git capabilities"},
    @{Name="Python"; ID="ms-python.python"; Description="Python language support"},
    @{Name="Prettier"; ID="esbenp.prettier-vscode"; Description="Code formatter"},
    @{Name="ESLint"; ID="dbaeumer.vscode-eslint"; Description="JavaScript linter"}
)

foreach ($ext in $openVSXExtensions) {
    Write-Host "  🔹 $($ext.Name) ($($ext.ID))" -ForegroundColor White
    Write-Host "    $($ext.Description)" -ForegroundColor Gray
}

Write-Host "`n❌ NOT Available on Open VSX:" -ForegroundColor Red
Write-Host "  🚫 GitHub Copilot (Microsoft exclusive)" -ForegroundColor Red
Write-Host "  🚫 Microsoft AI Toolkit (Microsoft exclusive)" -ForegroundColor Red
Write-Host "  🚫 Most Microsoft-branded extensions" -ForegroundColor Red

Write-Host "`n✅ Our Solution:" -ForegroundColor Green
Write-Host "  🤖 Custom AI Assistant Extension (provides Cursor-like features)" -ForegroundColor Green
Write-Host "  🤖 Claude Dev for additional AI coding assistance" -ForegroundColor Green
Write-Host "  🤖 Continue.dev for open-source AI features" -ForegroundColor Green
Write-Host "  🤖 Backend with multiple AI models (GPT, Claude, Qwen, local models)" -ForegroundColor Green

Write-Host "`n🎯 Next Steps:" -ForegroundColor Yellow
Write-Host "1. Package our AI Assistant extension if not done" -ForegroundColor White
Write-Host "2. Install Claude Dev manually from Open VSX" -ForegroundColor White
Write-Host "3. Test our AI features with the backend running" -ForegroundColor White
Write-Host "4. Configure AI models in the backend" -ForegroundColor White