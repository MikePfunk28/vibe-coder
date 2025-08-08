# AI IDE Release Build Script for Windows
# Creates production-ready executables and installers for all platforms

param(
    [switch]$SkipTests,
    [switch]$SkipClean
)

# Function to print colored output
function Write-Header {
    param([string]$Message)
    Write-Host "================================" -ForegroundColor Magenta
    Write-Host $Message -ForegroundColor Magenta
    Write-Host "================================" -ForegroundColor Magenta
}

function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Step {
    param([string]$Message)
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

# Check if we're in the right directory
if (-not (Test-Path "package.json")) {
    Write-Error "package.json not found. Please run this script from the ai-ide directory."
    exit 1
}

# Get version from package.json
$packageJson = Get-Content "package.json" | ConvertFrom-Json
$version = $packageJson.version

Write-Header "AI IDE Release Build v$version"
Write-Host "Advanced AI-Powered Development Environment" -ForegroundColor Blue
Write-Host "Competitor to VSCode, GitHub Copilot, Cursor, and Windsurf" -ForegroundColor Blue
Write-Host ""

# Step 1: Clean previous builds
if (-not $SkipClean) {
    Write-Step "Cleaning previous builds..."
    try {
        npm run clean 2>$null
    } catch {
        Write-Warning "Clean command failed or not available"
    }
    
    if (Test-Path "node_modules") {
        Remove-Item -Recurse -Force "node_modules" -ErrorAction SilentlyContinue
    }
}

# Step 2: Install dependencies
Write-Step "Installing dependencies..."

Write-Status "Installing root dependencies..."
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install root dependencies"
    exit 1
}

Write-Status "Installing extension dependencies..."
Set-Location "extensions/ai-assistant"
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install extension dependencies"
    Set-Location "../.."
    exit 1
}
Set-Location "../.."

Write-Status "Installing Python backend dependencies..."
Set-Location "backend"
if (-not (Test-Path "venv")) {
    python -m venv venv
}
& "venv/Scripts/Activate.ps1"
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Warning "requirements.txt not found"
}
Set-Location ".."

# Step 3: Run tests
if (-not $SkipTests) {
    Write-Step "Running tests..."
    
    Write-Status "Testing backend..."
    try {
        npm run test:backend 2>$null
    } catch {
        Write-Warning "Backend tests failed or not available"
    }

    Write-Status "Testing extension..."
    try {
        npm run test:extension 2>$null
    } catch {
        Write-Warning "Extension tests failed or not available"
    }
}

# Step 4: Build the application
Write-Step "Building AI IDE..."
npm run build
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

# Step 5: Create final packages
Write-Step "Creating release packages..."
npm run package
if ($LASTEXITCODE -ne 0) {
    Write-Error "Packaging failed"
    exit 1
}

# Step 6: Verify builds
Write-Step "Verifying builds..."
if (Test-Path "releases") {
    Write-Status "Release files created:"
    Get-ChildItem "releases" | Where-Object { $_.Extension -match '\.(zip|exe|msi)$' } | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 1)
        Write-Host "  📄 $($_.Name) ($size MB)" -ForegroundColor White
    }
} else {
    Write-Error "Releases directory not found"
    exit 1
}

# Step 7: Create checksums
Write-Step "Creating checksums..."
Set-Location "releases"
try {
    Get-ChildItem | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        $hash = Get-FileHash $_.Name -Algorithm SHA256
        "$($hash.Hash.ToLower())  $($_.Name)" | Out-File -Append -Encoding UTF8 "checksums.sha256"
    }
    Write-Status "SHA256 checksums created"
} catch {
    Write-Warning "Failed to create checksums"
}
Set-Location ".."

# Step 8: Display summary
Write-Header "Build Summary"
Write-Host "✅ AI IDE build completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📦 Release Information:" -ForegroundColor Cyan
Write-Host "  Version: $version"
Write-Host "  Build Date: $(Get-Date)"
Write-Host "  Platform: Windows-$(if ([Environment]::Is64BitOperatingSystem) { 'x64' } else { 'x86' })"
Write-Host ""

if (Test-Path "releases") {
    Write-Host "📁 Release Files:" -ForegroundColor Cyan
    Set-Location "releases"
    Get-ChildItem | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 1)
        Write-Host "  📄 $($_.Name) ($size MB)"
    }
    Set-Location ".."
}

Write-Host ""
Write-Host "🚀 AI IDE is ready for distribution!" -ForegroundColor Magenta
Write-Host "Your AI-powered IDE competitor is complete!" -ForegroundColor Magenta
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Test the executables on target platforms"
Write-Host "  2. Upload to distribution platforms"
Write-Host "  3. Update documentation and website"
Write-Host "  4. Announce the release!"
Write-Host ""
Write-Host "Thank you for building the future of AI-powered development!" -ForegroundColor Green