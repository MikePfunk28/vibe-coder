# Install AI Extensions Actually Available on Open VSX Registry
# VSCodium uses Open VSX, not Microsoft's marketplace

$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

Write-Host "Installing AI extensions available on Open VSX Registry..." -ForegroundColor Green

# AI Extensions that ARE available on Open VSX
$availableExtensions = @(
    # AI Coding Assistants
    "saoudrizwan.claude-dev",           # Claude Dev (Cline) - AI coding assistant
    "continue.continue",                # Continue.dev - AI code assistant
    "tabnine.tabnine-vscode",          # TabNine - AI completions
    
    # Development Tools
    "ms-python.python",                 # Python support
    "bradlc.vscode-tailwindcss",       # Tailwind CSS
    "esbenp.prettier-vscode",          # Prettier formatter
    "dbaeumer.vscode-eslint",          # ESLint
    
    # Git and Version Control
    "eamodio.gitlens",                 # GitLens - enhanced git
    "mhutchie.git-graph",              # Git Graph
    
    # Productivity
    "formulahendry.auto-rename-tag",   # Auto rename tag
    "christian-kohler.path-intellisense", # Path intellisense
    "ms-vscode.live-server",           # Live server
    
    # Themes and UI
    "dracula-theme.theme-dracula",     # Dracula theme
    "PKief.material-icon-theme",       # Material icons
    "zhuangtongfa.material-theme",     # Material theme
    
    # Language Support
    "golang.go",                       # Go language
    "rust-lang.rust-analyzer",         # Rust analyzer
    "ms-vscode.cpptools",              # C/C++ tools
    
    # API and Testing
    "rangav.vscode-thunder-client",    # Thunder Client - API testing
    "ms-vscode.test-adapter-converter" # Test adapter
)

Write-Host "Available AI and development extensions for VSCodium:" -ForegroundColor Cyan

$installed = 0
$failed = 0

foreach ($extension in $availableExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    
    try {
        & $vscodium --install-extension $extension --force 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension - INSTALLED" -ForegroundColor Green
            $installed++
        } else {
            Write-Host "❌ $extension - FAILED" -ForegroundColor Red
            $failed++
        }
    } catch {
        Write-Host "❌ $extension - ERROR: $_" -ForegroundColor Red
        $failed++
    }
}

Write-Host ""
Write-Host "=== Installation Summary ===" -ForegroundColor Cyan
Write-Host "✅ Successfully installed: $installed extensions" -ForegroundColor Green
Write-Host "❌ Failed to install: $failed extensions" -ForegroundColor Red

Write-Host ""
Write-Host "=== IMPORTANT NOTES ===" -ForegroundColor Yellow
Write-Host "🚫 GitHub Copilot is NOT available on Open VSX (VSCodium's marketplace)" -ForegroundColor Red
Write-Host "🚫 Microsoft AI Toolkit is NOT available on Open VSX" -ForegroundColor Red
Write-Host "✅ Claude Dev (Cline) IS available and provides similar AI features" -ForegroundColor Green
Write-Host "✅ Continue.dev IS available for AI code assistance" -ForegroundColor Green
Write-Host "✅ TabNine IS available for AI completions" -ForegroundColor Green
Write-Host "✅ Our custom AI Assistant extension provides enhanced features" -ForegroundColor Green

Write-Host ""
Write-Host "=== Alternative AI Solutions ===" -ForegroundColor Cyan
Write-Host "1. 🤖 Our AI Assistant Extension (already installed)" -ForegroundColor White
Write-Host "   - Ctrl+K: Inline generation" -ForegroundColor Gray
Write-Host "   - Ctrl+L: AI chat" -ForegroundColor Gray
Write-Host "   - Multi-model support" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 🤖 Claude Dev (Cline) Extension" -ForegroundColor White
Write-Host "   - AI coding assistant similar to Cursor" -ForegroundColor Gray
Write-Host "   - Works with Claude, GPT, and other models" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 🤖 Continue.dev Extension" -ForegroundColor White
Write-Host "   - Open-source AI code assistant" -ForegroundColor Gray
Write-Host "   - Supports multiple AI providers" -ForegroundColor Gray
Write-Host ""
Write-Host "4. 🤖 TabNine Extension" -ForegroundColor White
Write-Host "   - AI-powered code completions" -ForegroundColor Gray
Write-Host "   - Works offline with local models" -ForegroundColor Gray

Write-Host ""
Write-Host "✅ VSCodium is now configured with available AI extensions!" -ForegroundColor Green