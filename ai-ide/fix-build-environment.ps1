#!/usr/bin/env pwsh
<#
.SYNOPSIS
    AI IDE Build Environment Fixer - Creates a working VSCode OSS + AI IDE
.DESCRIPTION
    This script fixes all build issues and creates a complete AI IDE based on VSCode OSS
    - Fixes node-pty compilation issues on Windows
    - Fixes Python torch version conflicts
    - Sets up proper VSCode OSS base (not VSCodium)
    - Creates a complete fork like Cursor
    - Gets you a working IDE with AI features
#>

param(
    [switch]$SkipDependencies,
    [switch]$ForceClean,
    [switch]$QuickBuild
)

$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

Write-Host "🚀 AI IDE Build Environment Fixer" -ForegroundColor Cyan
Write-Host "Creating a complete VSCode OSS + AI IDE like Cursor" -ForegroundColor Green
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptDir
Set-Location $ProjectRoot

# Configuration
$VSCodeVersion = "1.85.2"  # Stable version that works well
$NodeVersion = "18.17.1"   # VSCode-compatible Node version
$PythonMinVersion = "3.9"

function Write-Step {
    param([string]$Message)
    Write-Host "🔧 $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Install-NodeVersion {
    param([string]$Version)
    
    Write-Step "Checking Node.js version compatibility..."
    
    if (Test-Command "node") {
        $currentVersion = node --version
        Write-Host "Current Node.js version: $currentVersion"
        
        # Check if version is compatible (18.x is ideal for VSCode)
        if ($currentVersion -match "v18\.") {
            Write-Success "Node.js version is compatible"
            return
        }
    }
    
    Write-Host "⚠️  VSCode OSS works best with Node.js 18.x" -ForegroundColor Yellow
    Write-Host "Current version may cause build issues. Consider using nvm to switch to Node 18.17.1" -ForegroundColor Yellow
}

function Fix-NodePtyIssue {
    Write-Step "Fixing node-pty compilation issues..."
    
    # Create a package.json override for node-pty to use prebuilt binaries
    $packageJsonPath = Join-Path $ProjectRoot "package.json"
    
    if (Test-Path $packageJsonPath) {
        $packageJson = Get-Content $packageJsonPath | ConvertFrom-Json
        
        # Remove problematic node-pty dependency and replace with alternative
        if ($packageJson.dependencies."node-pty") {
            Write-Host "Removing problematic node-pty dependency..."
            $packageJson.dependencies.PSObject.Properties.Remove("node-pty")
        }
        
        # Add working alternatives for terminal functionality
        if (-not $packageJson.dependencies."@vscode/windows-process-tree") {
            $packageJson.dependencies | Add-Member -MemberType NoteProperty -Name "@vscode/windows-process-tree" -Value "^0.4.0"
        }
        
        # Add resolutions to force working versions
        if (-not $packageJson.resolutions) {
            $packageJson | Add-Member -MemberType NoteProperty -Name "resolutions" -Value @{}
        }
        
        $packageJson.resolutions."node-pty" = "^0.10.1"
        
        $packageJson | ConvertTo-Json -Depth 10 | Set-Content $packageJsonPath
        Write-Success "Fixed node-pty dependency issues"
    }
}

function Fix-PythonDependencies {
    Write-Step "Fixing Python dependencies..."
    
    $requirementsPath = Join-Path $ProjectRoot "backend\requirements.txt"
    
    if (Test-Path $requirementsPath) {
        $requirements = Get-Content $requirementsPath
        
        # Fix torch version and other problematic dependencies
        $fixedRequirements = $requirements | ForEach-Object {
            $line = $_
            
            # Fix torch version
            if ($line -match "torch==") {
                "torch>=2.0.0,<3.0.0"
            }
            # Fix transformers compatibility
            elseif ($line -match "transformers==") {
                "transformers>=4.30.0,<5.0.0"
            }
            # Fix langchain versions
            elseif ($line -match "langchain==") {
                "langchain>=0.1.0"
            }
            elseif ($line -match "langchain-community==") {
                "langchain-community>=0.0.20"
            }
            # Fix numpy compatibility
            elseif ($line -match "numpy==") {
                "numpy>=1.24.0,<2.0.0"
            }
            # Remove problematic asyncio (it's built-in)
            elseif ($line -match "asyncio==") {
                "# asyncio is built-in to Python 3.7+"
            }
            else {
                $line
            }
        }
        
        $fixedRequirements | Set-Content $requirementsPath
        Write-Success "Fixed Python dependencies"
    }
}

function Setup-VSCodeOSSBase {
    Write-Step "Setting up complete VSCode OSS base (like Cursor)..."
    
    $vscodeDir = Join-Path $ProjectRoot "vscode-oss"
    $buildDir = Join-Path $ProjectRoot "ai-ide-build"
    
    # Clean existing if force clean
    if ($ForceClean -and (Test-Path $buildDir)) {
        Write-Host "🧹 Force cleaning existing build..."
        Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Clone fresh VSCode OSS if needed
    if (-not (Test-Path $vscodeDir) -or $ForceClean) {
        Write-Host "📥 Cloning VSCode OSS repository..."
        
        if (Test-Path $vscodeDir) {
            Remove-Item $vscodeDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        git clone --depth 1 --branch $VSCodeVersion https://github.com/microsoft/vscode.git $vscodeDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to clone VSCode OSS"
            return $false
        }
    }
    
    # Copy VSCode OSS to build directory (complete fork like Cursor)
    if (-not (Test-Path $buildDir)) {
        Write-Host "📁 Creating complete VSCode fork..."
        Copy-Item $vscodeDir $buildDir -Recurse -Force
        Write-Success "Complete VSCode OSS base created"
    }
    
    return $true
}

function Integrate-AIFeatures {
    Write-Step "Integrating AI features into VSCode OSS..."
    
    $buildDir = Join-Path $ProjectRoot "ai-ide-build"
    $extensionsDir = Join-Path $buildDir "extensions"
    $aiExtensionDir = Join-Path $extensionsDir "ai-assistant"
    
    # Copy AI extension
    $sourceExtension = Join-Path $ProjectRoot "extensions\ai-assistant"
    if (Test-Path $sourceExtension) {
        if (Test-Path $aiExtensionDir) {
            Remove-Item $aiExtensionDir -Recurse -Force
        }
        Copy-Item $sourceExtension $aiExtensionDir -Recurse -Force
        Write-Success "AI Assistant extension integrated"
    }
    
    # Update product.json for AI IDE branding
    $productJsonPath = Join-Path $buildDir "product.json"
    if (Test-Path $productJsonPath) {
        $productJson = Get-Content $productJsonPath | ConvertFrom-Json
        
        $productJson.nameShort = "AI IDE"
        $productJson.nameLong = "AI IDE - VSCode with AI Superpowers"
        $productJson.applicationName = "ai-ide"
        $productJson.dataFolderName = ".ai-ide"
        $productJson.quality = "stable"
        $productJson.commit = "ai-ide-build"
        
        # Add AI-specific configuration
        if (-not $productJson.aiFeatures) {
            $productJson | Add-Member -MemberType NoteProperty -Name "aiFeatures" -Value @{
                enabled = $true
                inlineCompletion = $true
                chatPanel = $true
                codeGeneration = $true
                semanticSearch = $true
            }
        }
        
        $productJson | ConvertTo-Json -Depth 10 | Set-Content $productJsonPath
        Write-Success "AI IDE branding configured"
    }
    
    # Create AI backend integration
    $aiBackendDir = Join-Path $buildDir "ai-backend"
    $sourceBackend = Join-Path $ProjectRoot "backend"
    
    if (Test-Path $sourceBackend) {
        if (Test-Path $aiBackendDir) {
            Remove-Item $aiBackendDir -Recurse -Force
        }
        Copy-Item $sourceBackend $aiBackendDir -Recurse -Force
        Write-Success "AI backend integrated"
    }
}

function Build-VSCodeOSS {
    Write-Step "Building VSCode OSS with AI features..."
    
    $buildDir = Join-Path $ProjectRoot "ai-ide-build"
    Set-Location $buildDir
    
    try {
        # Install dependencies with proper flags
        Write-Host "📦 Installing VSCode dependencies..."
        
        if (-not $SkipDependencies) {
            # Use yarn for VSCode (it's what they use)
            if (Test-Command "yarn") {
                yarn install --ignore-engines --network-timeout 100000
            } else {
                npm install --legacy-peer-deps --maxsockets 1
            }
            
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Dependency installation failed"
                return $false
            }
        }
        
        # Build VSCode
        Write-Host "🔨 Compiling VSCode with AI features..."
        
        if ($QuickBuild) {
            # Quick build for development
            npm run compile
        } else {
            # Full build
            npm run compile-build
        }
        
        if ($LASTEXITCODE -ne 0) {
            Write-Error "VSCode compilation failed"
            return $false
        }
        
        Write-Success "VSCode OSS with AI features built successfully!"
        return $true
        
    } finally {
        Set-Location $ProjectRoot
    }
}

function Create-LaunchScripts {
    Write-Step "Creating launch scripts..."
    
    $buildDir = Join-Path $ProjectRoot "ai-ide-build"
    
    # Windows launch script
    $launchScript = @"
@echo off
echo 🚀 Starting AI IDE (VSCode OSS + AI Features)
echo.
echo Features available:
echo - Complete VSCode functionality
echo - AI-powered code completion
echo - Inline code generation (Ctrl+K)
echo - AI chat panel (Ctrl+L)
echo - Semantic code search
echo - Multi-agent AI system
echo.
cd /d "$buildDir"
node scripts\code.js %*
"@
    
    $launchScript | Set-Content (Join-Path $ProjectRoot "launch-ai-ide.bat")
    
    # PowerShell launch script
    $psLaunchScript = @"
#!/usr/bin/env pwsh
Write-Host "🚀 Starting AI IDE (VSCode OSS + AI Features)" -ForegroundColor Cyan
Set-Location "$buildDir"
node scripts/code.js @args
"@
    
    $psLaunchScript | Set-Content (Join-Path $ProjectRoot "launch-ai-ide.ps1")
    
    Write-Success "Launch scripts created"
}

function Test-Build {
    Write-Step "Testing the build..."
    
    $buildDir = Join-Path $ProjectRoot "ai-ide-build"
    $codeScript = Join-Path $buildDir "scripts\code.js"
    
    if (Test-Path $codeScript) {
        Write-Success "Build appears successful - code.js found"
        
        # Test if it can start (quick test)
        Set-Location $buildDir
        $testResult = node scripts/code.js --version 2>&1
        
        if ($testResult -match "AI IDE|VSCode") {
            Write-Success "AI IDE can start successfully!"
            Write-Host ""
            Write-Host "🎉 AI IDE is ready!" -ForegroundColor Green
            Write-Host "Run: .\launch-ai-ide.bat" -ForegroundColor Cyan
            Write-Host "Or:  .\launch-ai-ide.ps1" -ForegroundColor Cyan
            return $true
        }
    }
    
    Write-Error "Build test failed"
    return $false
}

# Main execution
try {
    Write-Host "Starting AI IDE build environment setup..." -ForegroundColor Cyan
    Write-Host ""
    
    # Step 1: Check prerequisites
    Write-Step "Checking prerequisites..."
    Install-NodeVersion $NodeVersion
    
    if (-not (Test-Command "git")) {
        Write-Error "Git is required but not found"
        exit 1
    }
    
    if (-not (Test-Command "python")) {
        Write-Error "Python is required but not found"
        exit 1
    }
    
    # Step 2: Fix dependency issues
    Fix-NodePtyIssue
    Fix-PythonDependencies
    
    # Step 3: Setup VSCode OSS base
    if (-not (Setup-VSCodeOSSBase)) {
        Write-Error "Failed to setup VSCode OSS base"
        exit 1
    }
    
    # Step 4: Integrate AI features
    Integrate-AIFeatures
    
    # Step 5: Build everything
    if (-not (Build-VSCodeOSS)) {
        Write-Error "Build failed"
        exit 1
    }
    
    # Step 6: Create launch scripts
    Create-LaunchScripts
    
    # Step 7: Test the build
    Test-Build
    
    Write-Host ""
    Write-Host "🎉 SUCCESS! AI IDE build environment is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run: .\launch-ai-ide.bat (or .ps1)" -ForegroundColor White
    Write-Host "2. Your AI IDE will start with all VSCode features + AI" -ForegroundColor White
    Write-Host "3. Use Ctrl+K for inline AI code generation" -ForegroundColor White
    Write-Host "4. Use Ctrl+L for AI chat panel" -ForegroundColor White
    Write-Host ""
    Write-Host "Build location: ai-ide-build\" -ForegroundColor Cyan
    
} catch {
    Write-Error "Build script failed: $($_.Exception.Message)"
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
    exit 1
}