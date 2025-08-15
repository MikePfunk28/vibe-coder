#!/usr/bin/env pwsh
# Fix PATH quote characters that are breaking VSCode

Write-Host "Fixing PATH quote characters..." -ForegroundColor Yellow

# Get current PATH and remove problematic quotes
$currentPath = $env:PATH
$cleanPath = $currentPath -replace '"', ''

# Set the cleaned PATH
$env:PATH = $cleanPath

Write-Host "PATH quotes removed" -ForegroundColor Green
Write-Host "Testing tools..."
Write-Host "Node.js: $(node --version)"
Write-Host "Yarn: $(yarn --version)"

Write-Host "PATH fixed. You can now run VSCode build commands."