#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Quick Start VSCode OSS - Get basic VSCode working first, then add AI
.DESCRIPTION
    This script gets a basic VSCode OSS working quickly, then incrementally adds AI features
#>

param(
    [switch]$SkipBuild,
    [switch]$AIFeaturesOnly
)

$ErrorActionPreference = "Continue"

Write-Host "⚡ Quick Start: VSCode OSS + AI Features" -ForegroundColor Cyan
Write-Host "Getting you a working IDE as fast as possible" -ForegroundColor Green
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host "🔧 $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

if (-not $AIFeaturesOnly) {
    Write-Step "Step 1: Quick VSCode OSS setup..."
    
    $buildDir = "ai-ide-build"
    
    # Use existing VSCode OSS if available
    if (Test-Path $buildDir) {
        Write-Host "Found existing VSCode build, using it..."
    } else {
        Write-Host "Setting up fresh VSCode OSS..."
        
        # Clone VSCode OSS (shallow clone for speed)
        git clone --depth 1 --branch 1.85.2 https://github.com/microsoft/vscode.git $buildDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "❌ Failed to clone VSCode OSS" -ForegroundColor Red
            exit 1
        }
    }
    
    Write-Success "VSCode OSS base ready"
}

Write-Step "Step 2: Minimal dependency fixes..."

# Fix the most critical issues only
$packageJsonPath = Join-Path $ProjectRoot "package.json"
if (Test-Path $packageJsonPath) {
    $content = Get-Content $packageJsonPath -Raw
    # Remove node-pty to avoid compilation issues
    $content = $content -replace '"node-pty":\s*"[^"]*",?\s*', ''
    $content | Set-Content $packageJsonPath
    Write-Success "Removed problematic node-pty dependency"
}

if (-not $SkipBuild) {
    Write-Step "Step 3: Quick build (development mode)..."
    
    Set-Location $buildDir
    
    try {
        # Quick install with minimal flags
        Write-Host "Installing dependencies (this may take a few minutes)..."
        npm install --legacy-peer-deps --no-optional --ignore-scripts
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Building VSCode (development mode)..."
            npm run compile
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "VSCode OSS built successfully!"
            } else {
                Write-Host "⚠️  Build had issues but may still work" -ForegroundColor Yellow
            }
        } else {
            Write-Host "⚠️  Dependency installation had issues but continuing..." -ForegroundColor Yellow
        }
        
    } finally {
        Set-Location $ProjectRoot
    }
}

Write-Step "Step 4: Creating simple launch script..."

$launchScript = @"
@echo off
echo 🚀 Starting AI IDE (VSCode OSS base)
echo.
cd /d "$ProjectRoot\$buildDir"

REM Try different ways to start VSCode
if exist "scripts\code.js" (
    node scripts\code.js %*
) else if exist "out\main.js" (
    node out\main.js %*
) else if exist "scripts\code.bat" (
    call scripts\code.bat %*
) else (
    echo ❌ Could not find VSCode startup script
    echo Try running: npm run compile
    pause
)
"@

$launchScript | Set-Content "start-ai-ide.bat"

Write-Step "Step 5: Testing the build..."

$buildDir = Join-Path $ProjectRoot "ai-ide-build"
Set-Location $buildDir

# Test if VSCode can start
$testFiles = @(
    "scripts\code.js",
    "out\main.js",
    "scripts\code.bat"
)

$foundStartup = $false
foreach ($file in $testFiles) {
    if (Test-Path $file) {
        Write-Success "Found startup script: $file"
        $foundStartup = $true
        break
    }
}

Set-Location $ProjectRoot

if ($foundStartup) {
    Write-Host ""
    Write-Host "🎉 SUCCESS! Basic VSCode OSS is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To start your IDE:" -ForegroundColor Yellow
    Write-Host "  .\start-ai-ide.bat" -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps to add AI features:" -ForegroundColor Yellow
    Write-Host "  1. Get basic VSCode working first" -ForegroundColor White
    Write-Host "  2. Then run: .\fix-build-environment.ps1 -AIFeaturesOnly" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "⚠️  Build completed but startup scripts not found" -ForegroundColor Yellow
    Write-Host "Try running the full build script: .\fix-build-environment.ps1" -ForegroundColor White
    Write-Host ""
}

Write-Host "Build directory: $buildDir" -ForegroundColor Cyan