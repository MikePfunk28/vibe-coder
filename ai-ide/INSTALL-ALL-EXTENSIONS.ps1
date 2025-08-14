# Install All Extensions for Mike-AI-IDE
Write-Host "🔌 Installing All Extensions for Mike-AI-IDE..." -ForegroundColor Green

# Path to Mike-AI-IDE executable
$mikeAiIde = "ai-ide/vscode-oss-complete/scripts/code.bat"

# Check if Mike-AI-IDE is built
if (-not (Test-Path $mikeAiIde)) {
    Write-Host "❌ Mike-AI-IDE not found. Run BUILD-AND-TEST-MIKE-AI-IDE.ps1 first" -ForegroundColor Red
    exit 1
}

# AI Extensions (Priority 1)
Write-Host "🤖 Installing AI Extensions..." -ForegroundColor Cyan
$aiExtensions = @(
    "GitHub.copilot",                    # GitHub Copilot - AI code completion
    "GitHub.copilot-chat",               # GitHub Copilot Chat - AI chat
    "saoudrizwan.claude-dev",            # Claude Dev (Cline) - AI coding assistant
    "continue.continue",                 # Continue.dev - Open source AI assistant
    "tabnine.tabnine-vscode",           # TabNine - AI completions
    "ms-vscode.vscode-ai-toolkit"       # AI Toolkit (if available)
)

foreach ($extension in $aiExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Development Extensions (Priority 2)
Write-Host "🛠️  Installing Development Extensions..." -ForegroundColor Cyan
$devExtensions = @(
    "ms-python.python",                 # Python support
    "ms-vscode.cpptools",              # C/C++ tools
    "golang.go",                       # Go language support
    "rust-lang.rust-analyzer",         # Rust analyzer
    "ms-dotnettools.csharp",           # C# support
    "ms-vscode.vscode-typescript-next", # TypeScript support
    "bradlc.vscode-tailwindcss",       # Tailwind CSS
    "esbenp.prettier-vscode",          # Prettier formatter
    "dbaeumer.vscode-eslint",          # ESLint
    "ms-vscode.live-server"            # Live server
)

foreach ($extension in $devExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Git and Version Control (Priority 3)
Write-Host "📝 Installing Git Extensions..." -ForegroundColor Cyan
$gitExtensions = @(
    "eamodio.gitlens",                 # GitLens - enhanced git
    "mhutchie.git-graph",              # Git Graph
    "GitHub.vscode-pull-request-github" # GitHub Pull Requests
)

foreach ($extension in $gitExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Cloud and DevOps (Priority 4)
Write-Host "☁️  Installing Cloud Extensions..." -ForegroundColor Cyan
$cloudExtensions = @(
    "ms-vscode.azure-account",         # Azure Account
    "ms-azuretools.vscode-docker",     # Docker
    "ms-kubernetes-tools.vscode-kubernetes-tools", # Kubernetes
    "amazonwebservices.aws-toolkit-vscode" # AWS Toolkit
)

foreach ($extension in $cloudExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Productivity Extensions (Priority 5)
Write-Host "⚡ Installing Productivity Extensions..." -ForegroundColor Cyan
$productivityExtensions = @(
    "rangav.vscode-thunder-client",    # Thunder Client - API testing
    "formulahendry.auto-rename-tag",   # Auto rename tag
    "christian-kohler.path-intellisense", # Path intellisense
    "ms-vscode.test-adapter-converter", # Test adapter
    "gruntfuggly.todo-tree",          # TODO Tree
    "streetsidesoftware.code-spell-checker" # Code Spell Checker
)

foreach ($extension in $productivityExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Themes and UI (Priority 6)
Write-Host "🎨 Installing Themes..." -ForegroundColor Cyan
$themeExtensions = @(
    "dracula-theme.theme-dracula",     # Dracula theme
    "PKief.material-icon-theme",       # Material icons
    "zhuangtongfa.material-theme",     # Material theme
    "GitHub.github-vscode-theme"       # GitHub theme
)

foreach ($extension in $themeExtensions) {
    Write-Host "Installing $extension..." -ForegroundColor Yellow
    try {
        & $mikeAiIde --install-extension $extension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ $extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  $extension may not be available" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install $extension" -ForegroundColor Red
    }
}

# Install our custom AI Assistant extension
Write-Host "🤖 Installing Custom AI Assistant Extension..." -ForegroundColor Cyan
$customExtension = "ai-ide/extensions/ai-assistant/ai-assistant-latest.vsix"
if (Test-Path $customExtension) {
    try {
        & $mikeAiIde --install-extension $customExtension --force
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Custom AI Assistant extension installed" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Custom AI Assistant extension installation may have issues" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Failed to install custom AI Assistant extension" -ForegroundColor Red
    }
} else {
    Write-Host "⚠️  Custom AI Assistant extension not found. Build it first." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Extension installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 Mike-AI-IDE now has:" -ForegroundColor Cyan
Write-Host "   ✅ GitHub Copilot + Chat (if available)" -ForegroundColor White
Write-Host "   ✅ Native AI features (built-in)" -ForegroundColor White
Write-Host "   ✅ Claude Dev (Cline) for additional AI" -ForegroundColor White
Write-Host "   ✅ Continue.dev for open-source AI" -ForegroundColor White
Write-Host "   ✅ Full development toolchain" -ForegroundColor White
Write-Host "   ✅ Cloud and DevOps tools" -ForegroundColor White
Write-Host "   ✅ Productivity enhancements" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Restart Mike-AI-IDE to activate all extensions!" -ForegroundColor Yellow