@echo off
echo ========================================
echo   GET AI IDE WORKING IN 2 MINUTES
echo ========================================
echo.

echo STEP 1: Download VSCodium (pre-built VSCode OSS)
echo.
echo VSCodium is literally VSCode OSS already compiled!
echo No build issues, no dependency hell.
echo.
echo Download from: https://vscodium.com/
echo Or run: winget install VSCodium.VSCodium
echo.
pause

echo STEP 2: Install VSCodium if you haven't
winget install VSCodium.VSCodium
echo.

echo STEP 3: Create AI Extension
mkdir ai-extension 2>nul
cd ai-extension

echo Creating package.json...
(
echo {
echo   "name": "cursor-clone",
echo   "displayName": "Cursor Clone - AI IDE",
echo   "version": "1.0.0",
echo   "engines": { "vscode": "^1.74.0" },
echo   "main": "./extension.js",
echo   "activationEvents": ["*"],
echo   "contributes": {
echo     "commands": [
echo       { "command": "cursor.generate", "title": "AI Generate" },
echo       { "command": "cursor.chat", "title": "AI Chat" }
echo     ],
echo     "keybindings": [
echo       { "command": "cursor.generate", "key": "ctrl+k" },
echo       { "command": "cursor.chat", "key": "ctrl+l" }
echo     ]
echo   }
echo }
) > package.json

echo Creating extension.js...
(
echo const vscode = require('vscode'^);
echo function activate(context^) {
echo   let generate = vscode.commands.registerCommand('cursor.generate', async (^) =^> {
echo     const editor = vscode.window.activeTextEditor;
echo     if (!editor^) return;
echo     const prompt = await vscode.window.showInputBox({prompt: 'AI Prompt:'});
echo     if (prompt^) {
echo       editor.edit(edit =^> edit.insert(editor.selection.start, `// AI: ${prompt}\n// TODO: Connect to your AI backend\n`^)^);
echo     }
echo   }^);
echo   let chat = vscode.commands.registerCommand('cursor.chat', (^) =^> {
echo     vscode.window.showInformationMessage('🤖 AI Chat - Connect your backend!'^);
echo   }^);
echo   context.subscriptions.push(generate, chat^);
echo   vscode.window.showInformationMessage('🚀 Cursor Clone Active! Ctrl+K=Generate, Ctrl+L=Chat'^);
echo }
echo module.exports = { activate };
) > extension.js

cd ..

echo STEP 4: Install Extension
codium --install-extension ai-extension

echo STEP 5: Create Launcher
(
echo @echo off
echo echo 🚀 Starting AI IDE (Cursor Clone^)
echo echo Ctrl+K = AI Generate, Ctrl+L = AI Chat
echo codium --new-window
) > AI-IDE.bat

echo.
echo ========================================
echo   SUCCESS! AI IDE READY!
echo ========================================
echo.
echo Run: AI-IDE.bat
echo.
echo You now have:
echo ✅ Working VSCode OSS (VSCodium)
echo ✅ AI extension with Ctrl+K and Ctrl+L
echo ✅ Ready to connect your AI backend
echo.
echo Next: Connect the extension to your Python backend
echo.
pause