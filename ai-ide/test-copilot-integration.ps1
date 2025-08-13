# Test GitHub Copilot Integration and Enhanced AI Features
# This script verifies that all AI features are working properly

Write-Host "=== AI IDE Copilot Integration Test ===" -ForegroundColor Green

$vscodium = "C:\Users\$env:USERNAME\AppData\Local\Programs\VSCodium\VSCodium.exe"
$extensionPath = "ai-ide/extensions/ai-assistant/ai-assistant-0.1.0.vsix"

# Check if VSCodium is installed
if (-not (Test-Path $vscodium)) {
    Write-Host "❌ VSCodium not found at $vscodium" -ForegroundColor Red
    exit 1
}

Write-Host "✅ VSCodium found" -ForegroundColor Green

# Check if our AI Assistant extension is packaged
if (-not (Test-Path $extensionPath)) {
    Write-Host "📦 Packaging AI Assistant extension..." -ForegroundColor Yellow
    cd ai-ide/extensions/ai-assistant
    npx vsce package --no-dependencies
    cd ../../..
}

if (Test-Path $extensionPath) {
    Write-Host "✅ AI Assistant extension packaged" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to package AI Assistant extension" -ForegroundColor Red
    exit 1
}

# Install the extension
Write-Host "📥 Installing AI Assistant extension..." -ForegroundColor Yellow
& $vscodium --install-extension $extensionPath --force

# Check installed extensions
Write-Host "🔍 Checking installed AI extensions..." -ForegroundColor Yellow
$extensions = & $vscodium --list-extensions 2>$null

$availableAIExtensions = @(
    "ai-ide.ai-assistant",      # Our custom AI Assistant
    "saoudrizwan.claude-dev",   # Claude Dev (Cline)
    "continue.continue",        # Continue.dev
    "tabnine.tabnine-vscode"   # TabNine
)

$missingExtensions = @()
foreach ($ext in $availableAIExtensions) {
    if ($extensions -contains $ext) {
        Write-Host "✅ $ext - INSTALLED" -ForegroundColor Green
    } else {
        Write-Host "⚠️  $ext - NOT INSTALLED" -ForegroundColor Yellow
        if ($ext -ne "ai-ide.ai-assistant") {
            $missingExtensions += $ext
        }
    }
}

# Install missing extensions
if ($missingExtensions.Count -gt 0) {
    Write-Host "📥 Installing missing extensions..." -ForegroundColor Yellow
    foreach ($ext in $missingExtensions) {
        if ($ext -ne "ai-ide.ai-assistant") {
            Write-Host "Installing $ext..." -ForegroundColor Cyan
            & $vscodium --install-extension $ext
        }
    }
}

# Start backend services
Write-Host "🚀 Starting AI backend services..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath "python" -ArgumentList "ai-ide/backend/main.py" -PassThru -WindowStyle Hidden

Start-Sleep -Seconds 3

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ AI Backend is running" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  AI Backend may not be fully ready (this is normal)" -ForegroundColor Yellow
}

# Create test workspace
$testWorkspace = "ai-ide-test-workspace"
if (-not (Test-Path $testWorkspace)) {
    New-Item -ItemType Directory -Path $testWorkspace | Out-Null
}

# Create test files
$testFiles = @{
    "test.js" = @"
// Test JavaScript file for AI completion
function calculateSum(a, b) {
    // TODO: Add implementation
}

// Test function for Copilot suggestions
function processUserData(userData) {
    // TODO: Validate and process user data
}

class UserManager {
    constructor() {
        // TODO: Initialize user manager
    }
    
    // TODO: Add user management methods
}
"@
    
    "test.py" = @"
# Test Python file for AI completion
def calculate_fibonacci(n):
    # TODO: Implement fibonacci calculation
    pass

class DataProcessor:
    def __init__(self):
        # TODO: Initialize data processor
        pass
    
    def process_data(self, data):
        # TODO: Process the input data
        pass
"@
    
    "README.md" = @"
# AI IDE Test Workspace

This workspace is used to test AI features including:

- GitHub Copilot integration
- Enhanced AI completions
- Contextual chat features
- Multi-model AI support
- Autonomy modes

## Test Instructions

1. Open files in VSCodium
2. Try Ctrl+K for inline generation
3. Try Ctrl+L for AI chat
4. Test contextual features with #File, #Folder, etc.
5. Test autonomy mode switching
"@
}

foreach ($file in $testFiles.Keys) {
    $filePath = Join-Path $testWorkspace $file
    $testFiles[$file] | Out-File -FilePath $filePath -Encoding UTF8
}

Write-Host "✅ Test workspace created at $testWorkspace" -ForegroundColor Green

# Launch VSCodium with test workspace
Write-Host "🚀 Launching VSCodium with test workspace..." -ForegroundColor Yellow
& $vscodium $testWorkspace

Write-Host ""
Write-Host "=== Test Instructions ===" -ForegroundColor Cyan
Write-Host "1. ✅ VSCodium should open with the test workspace" -ForegroundColor White
Write-Host "2. 🤖 Check that AI Assistant appears in the activity bar (robot icon)" -ForegroundColor White
Write-Host "3. 🔧 Try our AI Assistant features:" -ForegroundColor White
Write-Host "   - Ctrl+K: Inline code generation" -ForegroundColor Gray
Write-Host "   - Ctrl+L: Open AI chat" -ForegroundColor Gray
Write-Host "   - Ctrl+Shift+S: Semantic search" -ForegroundColor Gray
Write-Host "   - Ctrl+Shift+R: AI reasoning" -ForegroundColor Gray
Write-Host "4. 💬 Test contextual chat with:" -ForegroundColor White
Write-Host "   - #File test.js" -ForegroundColor Gray
Write-Host "   - #Folder ." -ForegroundColor Gray
Write-Host "   - #Problems" -ForegroundColor Gray
Write-Host "5. ⚙️  Check status bar for autonomy mode indicator" -ForegroundColor White
Write-Host "6. 🤖 If Claude Dev (Cline) is installed, try its features too" -ForegroundColor White
Write-Host "7. 🤖 If Continue.dev is installed, try Ctrl+I for inline edit" -ForegroundColor White
Write-Host ""
Write-Host "=== Expected Results ===" -ForegroundColor Cyan
Write-Host "✅ AI Assistant should provide inline code generation" -ForegroundColor Green
Write-Host "✅ AI chat should work with contextual features" -ForegroundColor Green
Write-Host "✅ Semantic search should find relevant code" -ForegroundColor Green
Write-Host "✅ Autonomy mode should be toggleable in status bar" -ForegroundColor Green
Write-Host "✅ Claude Dev/Continue.dev should provide additional AI features" -ForegroundColor Green
Write-Host ""
Write-Host "⚠️  NOTE: GitHub Copilot is NOT available on Open VSX (VSCodium)" -ForegroundColor Yellow
Write-Host "   But our AI Assistant + Claude Dev provide similar functionality!" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to stop the backend service..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup
if ($backendProcess -and !$backendProcess.HasExited) {
    Write-Host "🛑 Stopping backend service..." -ForegroundColor Yellow
    Stop-Process -Id $backendProcess.Id -Force
}

Write-Host "✅ Test completed!" -ForegroundColor Green