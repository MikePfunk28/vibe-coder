# Install Key Extensions for AI IDE using Open VSX Registry
$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

Write-Host "Installing key extensions for AI IDE..." -ForegroundColor Green

# Core AI and Development Extensions (using Open VSX compatible versions)
$extensions = @(
    "ms-python.python",
    "ms-python.pylint", 
    "ms-python.black-formatter",
    "eamodio.gitlens",
    "ms-azuretools.vscode-docker",
    "rangav.vscode-thunder-client",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "ms-vscode.vscode-json",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.cpptools",
    "golang.go",
    "rust-lang.rust-analyzer",
    "redhat.java",
    "ms-dotnettools.csharp",
    "formulahendry.auto-rename-tag",
    "christian-kohler.path-intellisense",
    "PKief.material-icon-theme",
    "dracula-theme.theme-dracula"
)

Write-Host "Installing $($extensions.Count) extensions..." -ForegroundColor Yellow

foreach ($ext in $extensions) {
    Write-Host "Installing $ext..." -ForegroundColor Cyan
    try {
        & $vscodium --install-extension $ext --force 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $ext installed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to install $ext" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Error installing $ext : $_" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 500
}

Write-Host "`nKey extension installation complete!" -ForegroundColor Green
Write-Host "Note: GitHub Copilot and Claude Dev may need to be installed manually from their respective sources." -ForegroundColor Yellow