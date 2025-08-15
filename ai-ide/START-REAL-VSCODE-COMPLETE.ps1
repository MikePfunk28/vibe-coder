#!/usr/bin/env pwsh
Write-Host "========================================"
Write-Host "  REAL VSCODE OSS - COMPLETE IDE"
Write-Host "========================================"
Write-Host ""
Write-Host "Starting REAL VSCode with ALL features:"
Write-Host "✅ Complete File menu (New, Open, Save, Save As, Save All, Recent, Preferences, Exit)"
Write-Host "✅ Complete Edit menu (Undo, Redo, Cut, Copy, Paste, Find, Replace, etc.)"
Write-Host "✅ Complete View menu (Explorer, Search, Source Control, Debug, Extensions, etc.)"
Write-Host "✅ Complete debugging, terminal, git integration"
Write-Host "✅ Extension marketplace and all VSCode features"
Write-Host ""

# Change to the VSCode directory
Set-Location "C:\Users\mikep\vibe-coder\ai-ide\vscode-oss-complete"

# Ensure we have the right PATH with all necessary tools
$env:PATH = "C:\Program Files\nodejs\;C:\Users\mikep\AppData\Roaming\nvm\v22.17.0;C:\Users\mikep\AppData\Roaming\npm;C:\WINDOWS\System32;C:\WINDOWS;C:\WINDOWS\System32\Wbem;$env:PATH"

Write-Host "Updated PATH includes yarn: $(Test-Path (Get-Command yarn.cmd -ErrorAction SilentlyContinue))"

# Start REAL VSCode directly with PowerShell
try {
    # Set environment variables
    $env:NODE_ENV = "development"
    $env:VSCODE_DEV = "1"
    $env:VSCODE_CLI = "1"
    $env:ELECTRON_ENABLE_LOGGING = "1"
    $env:ELECTRON_ENABLE_STACK_DUMPING = "1"
    
    # Run the preLaunch steps manually with PowerShell
    Write-Host "Running preLaunch steps manually..."
    
    # Ensure node_modules exists
    if (-not (Test-Path "node_modules")) {
        Write-Host "Installing dependencies with yarn..."
        yarn install
        if ($LASTEXITCODE -ne 0) {
            throw "yarn install failed"
        }
    }
    
    # Get electron
    Write-Host "Getting electron..."
    yarn electron
    if ($LASTEXITCODE -ne 0) {
        throw "yarn electron failed"
    }
    
    # Ensure compiled
    if (-not (Test-Path "out")) {
        Write-Host "Compiling..."
        yarn compile
        if ($LASTEXITCODE -ne 0) {
            throw "yarn compile failed"
        }
    }
    
    Write-Host "Manual preLaunch completed successfully"
    
    # Get the executable name from product.json
    $productJson = Get-Content "product.json" | ConvertFrom-Json
    $nameShort = $productJson.nameShort + ".exe"
    $codePath = ".build\electron\$nameShort"
    
    Write-Host "Starting VSCode from: $codePath"
    
    # Launch VSCode
    if (Test-Path $codePath) {
        & $codePath "." $args
    } else {
        throw "VSCode executable not found at: $codePath"
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ VSCode failed to start properly"
        Write-Host "This means the build is incomplete"
        Read-Host "Press Enter to continue"
    }
} catch {
    Write-Host ""
    Write-Host "❌ Error starting VSCode: $_"
    Read-Host "Press Enter to continue"
}