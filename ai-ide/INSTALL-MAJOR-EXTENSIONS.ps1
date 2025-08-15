#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Install Major Extensions for Mike-AI-IDE
.DESCRIPTION
    Installs all the major helpful extensions for the AI IDE from Open VSX marketplace
#>

Write-Host "🚀 Installing Major Extensions for Mike-AI-IDE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Define the path to our VSCode OSS build
$VSCodePath = ".\vscode-oss-complete"
$CodeExecutable = "$VSCodePath\scripts\code.bat"

# Check if VSCode OSS is built
if (-not (Test-Path $CodeExecutable)) {
    Write-Host "❌ VSCode OSS not found. Please build it first." -ForegroundColor Red
    Write-Host "Run: .\BUILD-COMPLETE-VSCODE.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "📦 Installing AI and Development Extensions..." -ForegroundColor Cyan

# AI Extensions
$AIExtensions = @(
    "GitHub.copilot",                    # GitHub Copilot
    "GitHub.copilot-chat",               # GitHub Copilot Chat
    "continue.continue",                 # Continue (Claude Dev alternative)
    "ms-toolsai.jupyter",               # Jupyter Notebooks
    "ms-python.python",                 # Python
    "ms-python.vscode-pylance",         # Pylance
    "ms-vscode.vscode-typescript-next"  # TypeScript
)

# Development Tools
$DevExtensions = @(
    "ms-vscode.vscode-json",            # JSON
    "redhat.vscode-yaml",               # YAML
    "ms-vscode.vscode-eslint",          # ESLint
    "esbenp.prettier-vscode",           # Prettier
    "bradlc.vscode-tailwindcss",        # Tailwind CSS
    "ms-vscode.vscode-css-peek",        # CSS Peek
    "formulahendry.auto-rename-tag",    # Auto Rename Tag
    "christian-kohler.path-intellisense" # Path Intellisense
)

# Git and Version Control
$GitExtensions = @(
    "eamodio.gitlens",                  # GitLens
    "mhutchie.git-graph",               # Git Graph
    "donjayamanne.githistory",          # Git History
    "github.vscode-pull-request-github" # GitHub Pull Requests
)

# Cloud and DevOps
$CloudExtensions = @(
    "ms-vscode.vscode-docker",          # Docker
    "ms-kubernetes-tools.vscode-kubernetes-tools", # Kubernetes
    "amazonwebservices.aws-toolkit-vscode",        # AWS Toolkit
    "ms-azuretools.vscode-azureresourcegroups",    # Azure Tools
    "hashicorp.terraform"               # Terraform
)

# API and Testing
$APIExtensions = @(
    "rangav.vscode-thunder-client",     # Thunder Client
    "humao.rest-client",                # REST Client
    "ms-vscode.test-adapter-converter", # Test Explorer
    "hbenl.vscode-test-explorer"        # Test Explorer UI
)

# Productivity
$ProductivityExtensions = @(
    "ms-vscode.vscode-todo-highlight",  # TODO Highlight
    "aaron-bond.better-comments",       # Better Comments
    "streetsidesoftware.code-spell-checker", # Code Spell Checker
    "ms-vscode.vscode-markdown-preview-enhanced", # Markdown Preview
    "yzhang.markdown-all-in-one",       # Markdown All in One
    "ms-vscode.vscode-icons"            # VSCode Icons
)

# Function to install extensions
function Install-Extensions {
    param(
        [string[]]$Extensions,
        [string]$Category
    )
    
    Write-Host "📦 Installing $Category Extensions..." -ForegroundColor Yellow
    
    foreach ($extension in $Extensions) {
        Write-Host "  Installing: $extension" -ForegroundColor Gray
        try {
            & $CodeExecutable --install-extension $extension --force
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✅ $extension installed successfully" -ForegroundColor Green
            } else {
                Write-Host "  ⚠️ $extension installation failed" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "  ❌ Error installing $extension`: $_" -ForegroundColor Red
        }
    }
    Write-Host ""
}

# Install all extension categories
try {
    Install-Extensions -Extensions $AIExtensions -Category "AI & Machine Learning"
    Install-Extensions -Extensions $DevExtensions -Category "Development Tools"
    Install-Extensions -Extensions $GitExtensions -Category "Git & Version Control"
    Install-Extensions -Extensions $CloudExtensions -Category "Cloud & DevOps"
    Install-Extensions -Extensions $APIExtensions -Category "API & Testing"
    Install-Extensions -Extensions $ProductivityExtensions -Category "Productivity"
    
    Write-Host "🎉 Extension installation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📋 Summary:" -ForegroundColor Cyan
    Write-Host "  • AI Extensions: $($AIExtensions.Count)" -ForegroundColor White
    Write-Host "  • Development Tools: $($DevExtensions.Count)" -ForegroundColor White
    Write-Host "  • Git Extensions: $($GitExtensions.Count)" -ForegroundColor White
    Write-Host "  • Cloud Extensions: $($CloudExtensions.Count)" -ForegroundColor White
    Write-Host "  • API Extensions: $($APIExtensions.Count)" -ForegroundColor White
    Write-Host "  • Productivity Extensions: $($ProductivityExtensions.Count)" -ForegroundColor White
    Write-Host "  Total: $($AIExtensions.Count + $DevExtensions.Count + $GitExtensions.Count + $CloudExtensions.Count + $APIExtensions.Count + $ProductivityExtensions.Count) extensions" -ForegroundColor Green
    
    Write-Host ""
    Write-Host "🚀 Mike-AI-IDE is now ready with all major extensions!" -ForegroundColor Green
    Write-Host "Start the IDE with: .\START-REAL-VSCODE-COMPLETE.bat" -ForegroundColor Yellow
    
} catch {
    Write-Host "❌ Extension installation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ All major extensions installed successfully!" -ForegroundColor Green