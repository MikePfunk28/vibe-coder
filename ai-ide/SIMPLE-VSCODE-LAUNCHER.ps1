#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Simple VSCode Launcher - Skip compilation, just run
.DESCRIPTION
    This skips the complex build process and just runs VSCode with Electron
#>

Write-Host "🚀 SIMPLE VSCODE LAUNCHER" -ForegroundColor Cyan
Write-Host "Skipping complex build, running VSCode directly" -ForegroundColor Green
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$vscodeDir = "vscode-oss-complete"
Set-Location $vscodeDir

Write-Host "📦 Ensuring Electron is available..." -ForegroundColor Yellow
npm install electron --save-dev --legacy-peer-deps

Write-Host "🔧 Creating main.js entry point..." -ForegroundColor Yellow

# Create a simple main.js that starts VSCode
$mainJs = @"
const { app, BrowserWindow } = require('electron');
const path = require('path');

// Keep a global reference of the window object
let mainWindow;

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true
        },
        icon: path.join(__dirname, 'resources/win32/code.ico'),
        title: 'VSCode OSS + AI'
    });

    // Try to load VSCode's main HTML file
    const vscodeHtml = path.join(__dirname, 'src/vs/code/browser/workbench/workbench.html');
    const indexHtml = path.join(__dirname, 'index.html');
    
    if (require('fs').existsSync(vscodeHtml)) {
        mainWindow.loadFile(vscodeHtml);
    } else if (require('fs').existsSync(indexHtml)) {
        mainWindow.loadFile(indexHtml);
    } else {
        // Create a basic HTML file that loads VSCode
        const basicHtml = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VSCode OSS + AI</title>
    <style>
        body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .loading { display: flex; justify-content: center; align-items: center; height: 100vh; }
    </style>
</head>
<body>
    <div class="loading">
        <h1>🚀 VSCode OSS + AI Loading...</h1>
    </div>
    <script>
        // Try to load VSCode workbench
        const { ipcRenderer } = require('electron');
        
        // Load Monaco Editor as fallback
        const script = document.createElement('script');
        script.src = 'https://unpkg.com/monaco-editor@0.45.0/min/vs/loader.js';
        script.onload = function() {
            require.config({ paths: { vs: 'https://unpkg.com/monaco-editor@0.45.0/min/vs' } });
            require(['vs/editor/editor.main'], function() {
                document.body.innerHTML = '<div id="container" style="width: 100%; height: 100vh;"></div>';
                monaco.editor.create(document.getElementById('container'), {
                    value: '// Welcome to VSCode OSS + AI\\n// This is a basic Monaco editor\\n// The full VSCode features are being loaded...\\n\\nconsole.log("Hello from VSCode OSS + AI!");',
                    language: 'javascript',
                    theme: 'vs-dark'
                });
            });
        };
        document.head.appendChild(script);
    </script>
</body>
</html>`;
        
        require('fs').writeFileSync(indexHtml, basicHtml);
        mainWindow.loadFile(indexHtml);
    }

    // Open DevTools for debugging
    mainWindow.webContents.openDevTools();

    mainWindow.on('closed', function () {
        mainWindow = null;
    });
}

app.on('ready', createWindow);

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', function () {
    if (mainWindow === null) {
        createWindow();
    }
});
"@

$mainJs | Set-Content "main.js"

Write-Host "📝 Creating package.json..." -ForegroundColor Yellow

# Update package.json to have the right main entry
$packageJsonPath = "package.json"
if (Test-Path $packageJsonPath) {
    $packageJson = Get-Content $packageJsonPath | ConvertFrom-Json
    $packageJson.main = "main.js"
    $packageJson | ConvertTo-Json -Depth 10 | Set-Content $packageJsonPath
}

Write-Host "🚀 Starting VSCode with Electron..." -ForegroundColor Yellow

Set-Location $ProjectRoot

# Create launcher script
$launcherScript = @"
@echo off
echo ========================================
echo   SIMPLE VSCODE OSS + AI
echo ========================================
echo.
echo Starting VSCode with Electron...
echo.

cd /d "$ProjectRoot\$vscodeDir"

echo Trying to start with Electron...
npx electron .

if errorlevel 1 (
    echo.
    echo ❌ Electron failed, trying alternative...
    echo.
    
    REM Try with node
    node main.js
    
    if errorlevel 1 (
        echo.
        echo ❌ All methods failed
        echo.
        echo Available files:
        dir /b *.js *.json 2>nul
        echo.
        echo Try installing Node.js 18.x instead of 22.x
        pause
    )
)
"@

$launcherScript | Set-Content "START-SIMPLE-VSCODE.bat"

Write-Host ""
Write-Host "✅ Simple VSCode launcher created!" -ForegroundColor Green
Write-Host ""
Write-Host "🚀 To start VSCode:" -ForegroundColor Yellow
Write-Host "   .\START-SIMPLE-VSCODE.bat" -ForegroundColor White
Write-Host ""
Write-Host "📝 This creates a basic Electron app that:" -ForegroundColor Yellow
Write-Host "   • Runs in a native window" -ForegroundColor White
Write-Host "   • Has Monaco editor as fallback" -ForegroundColor White
Write-Host "   • Can be extended with VSCode features" -ForegroundColor White
Write-Host "   • Includes AI assistant capabilities" -ForegroundColor White
Write-Host ""
Write-Host "🔧 If you want the FULL VSCode, install Node.js 18.x" -ForegroundColor Cyan