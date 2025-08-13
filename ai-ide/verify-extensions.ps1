# Verify Installed Extensions for AI IDE
$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"

Write-Host "Checking installed extensions..." -ForegroundColor Green

# Get list of installed extensions
$extensions = & $vscodium --list-extensions 2>$null

if ($extensions) {
    Write-Host "Found $($extensions.Count) installed extensions:" -ForegroundColor Yellow
    
    # Key extensions we're looking for
    $keyExtensions = @(
        "GitHub.copilot",
        "GitHub.copilot-chat", 
        "saoudrizwan.claude-dev",
        "ms-python.python",
        "eamodio.gitlens",
        "ms-azuretools.vscode-docker",
        "rangav.vscode-thunder-client",
        "esbenp.prettier-vscode",
        "dbaeumer.vscode-eslint"
    )
    
    Write-Host "`nKey AI and Development Extensions:" -ForegroundColor Cyan
    foreach ($key in $keyExtensions) {
        if ($extensions -contains $key) {
            Write-Host "✓ $key - INSTALLED" -ForegroundColor Green
        } else {
            Write-Host "✗ $key - NOT FOUND" -ForegroundColor Red
        }
    }
    
    Write-Host "`nAll installed extensions:" -ForegroundColor Cyan
    $extensions | Sort-Object | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "No extensions found or VSCodium not responding properly." -ForegroundColor Red
}

Write-Host "`nExtension verification complete!" -ForegroundColor Green