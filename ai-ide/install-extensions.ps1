# Install Major Helpful Extensions for AI IDE
# This script installs all the essential extensions to make this a complete AI IDE

$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

Write-Host "Installing major helpful extensions for AI IDE..." -ForegroundColor Green

# AI and Coding Assistance Extensions
Write-Host "Installing AI and coding assistance extensions..." -ForegroundColor Yellow
& $vscodium --install-extension GitHub.copilot
& $vscodium --install-extension GitHub.copilot-chat
& $vscodium --install-extension saoudrizwan.claude-dev
& $vscodium --install-extension ms-vscode.vscode-ai-toolkit

# Development Tools
Write-Host "Installing development tools..." -ForegroundColor Yellow
& $vscodium --install-extension ms-vscode.vscode-json
& $vscodium --install-extension ms-vscode.vscode-typescript-next
& $vscodium --install-extension ms-python.python
& $vscodium --install-extension ms-python.pylint
& $vscodium --install-extension ms-python.black-formatter
& $vscodium --install-extension ms-toolsai.jupyter

# Git and Version Control
Write-Host "Installing git and version control extensions..." -ForegroundColor Yellow
& $vscodium --install-extension eamodio.gitlens
& $vscodium --install-extension GitHub.vscode-pull-request-github
& $vscodium --install-extension GitHub.vscode-github-actions

# Cloud and DevOps
Write-Host "Installing cloud and DevOps extensions..." -ForegroundColor Yellow
& $vscodium --install-extension amazonwebservices.aws-toolkit-vscode
& $vscodium --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
& $vscodium --install-extension ms-azuretools.vscode-docker

# API and Testing
Write-Host "Installing API and testing tools..." -ForegroundColor Yellow
& $vscodium --install-extension rangav.vscode-thunder-client
& $vscodium --install-extension ms-vscode.test-adapter-converter

# Code Quality and Formatting
Write-Host "Installing code quality and formatting extensions..." -ForegroundColor Yellow
& $vscodium --install-extension esbenp.prettier-vscode
& $vscodium --install-extension dbaeumer.vscode-eslint
& $vscodium --install-extension bradlc.vscode-tailwindcss

# Language Support
Write-Host "Installing language support extensions..." -ForegroundColor Yellow
& $vscodium --install-extension ms-vscode.cpptools
& $vscodium --install-extension golang.go
& $vscodium --install-extension rust-lang.rust-analyzer
& $vscodium --install-extension redhat.java
& $vscodium --install-extension ms-dotnettools.csharp

# Productivity Extensions
Write-Host "Installing productivity extensions..." -ForegroundColor Yellow
& $vscodium --install-extension vscodevim.vim
& $vscodium --install-extension ms-vscode.live-server
& $vscodium --install-extension formulahendry.auto-rename-tag
& $vscodium --install-extension christian-kohler.path-intellisense

# Themes and UI
Write-Host "Installing themes and UI extensions..." -ForegroundColor Yellow
& $vscodium --install-extension dracula-theme.theme-dracula
& $vscodium --install-extension PKief.material-icon-theme
& $vscodium --install-extension zhuangtongfa.material-theme

Write-Host "Extension installation complete!" -ForegroundColor Green
Write-Host "Restart VSCodium to ensure all extensions are loaded properly." -ForegroundColor Cyan