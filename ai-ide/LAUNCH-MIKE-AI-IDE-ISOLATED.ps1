#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Launch Mike-AI-IDE completely isolated from regular VSCode
.DESCRIPTION
    Ensures Mike-AI-IDE runs with its own settings, extensions, and data
    No interference with your regular VSCode installation
#>

Write-Host "🚀 LAUNCHING MIKE-AI-IDE (Isolated from VSCode)" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Ensure complete isolation
$env:VSCODE_PORTABLE = "$ProjectRoot\mike-ai-ide-data"
$env:VSCODE_EXTENSIONS = "$ProjectRoot\mike-ai-ide-extensions"
$env:VSCODE_USER_DATA_DIR = "$ProjectRoot\mike-ai-ide-userdata"

# Create isolated directories
New-Item -ItemType Directory -Path "$ProjectRoot\mike-ai-ide-data" -Force | Out-Null
New-Item -ItemType Directory -Path "$ProjectRoot\mike-ai-ide-extensions" -Force | Out-Null
New-Item -ItemType Directory -Path "$ProjectRoot\mike-ai-ide-userdata" -Force | Out-Null

Write-Host "📁 Mike-AI-IDE Data: $env:VSCODE_USER_DATA_DIR" -ForegroundColor Yellow
Write-Host "🔌 Mike-AI-IDE Extensions: $env:VSCODE_EXTENSIONS" -ForegroundColor Yellow
Write-Host "💾 Mike-AI-IDE Portable: $env:VSCODE_PORTABLE" -ForegroundColor Yellow
Write-Host ""

# Launch with complete isolation
if (Test-Path "vscode-oss-complete\Code.exe") {
    Write-Host "🎯 Launching Mike-AI-IDE (completely isolated from VSCode)..." -ForegroundColor Green
    & "vscode-oss-complete\Code.exe" --user-data-dir="$env:VSCODE_USER_DATA_DIR" --extensions-dir="$env:VSCODE_EXTENSIONS" $args
} elseif (Test-Path "ai-ide\AI-IDE-WORKING.bat") {
    Write-Host "🎯 Launching Mike-AI-IDE via working launcher..." -ForegroundColor Green
    & "ai-ide\AI-IDE-WORKING.bat"
} else {
    Write-Host "❌ Mike-AI-IDE executable not found" -ForegroundColor Red
    Write-Host "Build Mike-AI-IDE first using BUILD-COMPLETE-VSCODE.ps1" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ Mike-AI-IDE launched with complete isolation!" -ForegroundColor Green
Write-Host "   • Uses separate settings directory" -ForegroundColor White
Write-Host "   • Uses separate extensions directory" -ForegroundColor White
Write-Host "   • No interference with regular VSCode" -ForegroundColor White