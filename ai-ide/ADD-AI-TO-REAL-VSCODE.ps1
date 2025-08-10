#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Add AI Features to Real VSCode
.DESCRIPTION
    Creates AI extension for the actual VSCode installation
#>

Write-Host "🚀 Adding AI Features to REAL VSCode" -ForegroundColor Cyan
Write-Host "Using your existing VSCode installation" -ForegroundColor Green
Write-Host ""

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

# Create AI extension directory
$extensionDir = "vscode-ai-extension"
if (Test-Path $extensionDir) {
    Remove-Item $extensionDir -Recurse -Force
}
New-Item -ItemType Directory -Path $extensionDir | Out-Null

Write-Host "🤖 Creating AI IDE extension..." -ForegroundColor Yellow

# Create package.json
$packageJson = @{
    name = "ai-ide-extension"
    displayName = "AI IDE - Cursor Clone for VSCode"
    description = "Transform VSCode into an AI-powered IDE like Cursor with Ctrl+K and Ctrl+L"
    version = "1.0.0"
    publisher = "ai-ide"
    engines = @{
        vscode = "^1.74.0"
    }
    categories = @("Other", "Machine Learning", "Snippets")
    activationEvents = @("*")
    main = "./out/extension.js"
    contributes = @{
        commands = @(
            @{
                command = "ai-ide.generate"
                title = "AI: Generate Code (Ctrl+K)"
                category = "AI IDE"
            },
            @{
                command = "ai-ide.chat"
                title = "AI: Open Chat Panel (Ctrl+L)"
                category = "AI IDE"
            },
            @{
                command = "ai-ide.explain"
                title = "AI: Explain Selected Code"
                category = "AI IDE"
            }
        )
        keybindings = @(
            @{
                command = "ai-ide.generate"
                key = "ctrl+k"
                when = "editorTextFocus"
            },
            @{
                command = "ai-ide.chat"
                key = "ctrl+l"
            },
            @{
                command = "ai-ide.explain"
                key = "ctrl+e"
                when = "editorHasSelection"
            }
        )
        menus = @{
            "editor/context" = @(
                @{
                    command = "ai-ide.explain"
                    when = "editorHasSelection"
                    group = "ai-ide@1"
                }
            )
            "commandPalette" = @(
                @{
                    command = "ai-ide.generate"
                },
                @{
                    command = "ai-ide.chat"
                },
                @{
                    command = "ai-ide.explain"
                }
            )
        }
        configuration = @{
            title = "AI IDE"
            properties = @{
                "ai-ide.backendUrl" = @{
                    type = "string"
                    default = "http://localhost:8000"
                    description = "URL of your AI backend server"
                }
                "ai-ide.enableLogging" = @{
                    type = "boolean"
                    default = $true
                    description = "Enable AI IDE logging"
                }
            }
        }
    }
    scripts = @{
        compile = "tsc -p ./"
        watch = "tsc -watch -p ./"
        package = "vsce package"
    }
    devDependencies = @{
        "@types/vscode" = "^1.74.0"
        "@types/node" = "16.x"
        typescript = "^4.9.4"
    }
}

$packageJson | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $extensionDir "package.json")

# Create TypeScript source
New-Item -ItemType Directory -Path (Join-Path $extensionDir "src") -Force | Out-Null

$extensionTs = @'
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('🚀 AI IDE Extension is now active!');

    // AI Generate Command (Ctrl+K)
    const generateCommand = vscode.commands.registerCommand('ai-ide.generate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        const prompt = await vscode.window.showInputBox({
            prompt: '🤖 AI Code Generation - What would you like me to create?',
            placeHolder: 'e.g., "create a function that sorts an array", "add error handling", "optimize this code"',
            value: selectedText ? `Modify this code: ${selectedText.substring(0, 50)}...` : ''
        });

        if (prompt) {
            // Show progress
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "🤖 AI is generating code...",
                cancellable: false
            }, async (progress) => {
                progress.report({ increment: 30, message: "Analyzing request..." });
                await new Promise(resolve => setTimeout(resolve, 500));
                
                progress.report({ increment: 60, message: "Generating code..." });
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                const aiResponse = generateAICode(prompt, selectedText, editor.document.languageId);
                
                progress.report({ increment: 100, message: "Complete!" });
                
                // Insert the generated code
                await editor.edit(editBuilder => {
                    if (selection.isEmpty) {
                        editBuilder.insert(selection.start, aiResponse);
                    } else {
                        editBuilder.replace(selection, aiResponse);
                    }
                });
                
                vscode.window.showInformationMessage('✅ AI code generated! Connect to your backend for real AI power.');
            });
        }
    });

    // AI Chat Command (Ctrl+L)
    const chatCommand = vscode.commands.registerCommand('ai-ide.chat', () => {
        const panel = vscode.window.createWebviewPanel(
            'ai-ide-chat',
            '🤖 AI Chat - Cursor Clone',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
                localResourceRoots: [context.extensionUri]
            }
        );

        panel.webview.html = getChatHTML();
        
        // Handle messages from webview
        panel.webview.onDidReceiveMessage(
            async message => {
                switch (message.command) {
                    case 'sendMessage':
                        // Here you would connect to your AI backend
                        const response = await processAIMessage(message.text);
                        panel.webview.postMessage({
                            command: 'aiResponse',
                            text: response
                        });
                        break;
                }
            },
            undefined,
            context.subscriptions
        );
    });

    // AI Explain Command (Ctrl+E)
    const explainCommand = vscode.commands.registerCommand('ai-ide.explain', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (selectedText) {
            const explanation = await explainCode(selectedText, editor.document.languageId);
            
            // Show in information message with option to open detailed view
            const action = await vscode.window.showInformationMessage(
                explanation.substring(0, 100) + '...',
                { modal: false },
                'Show Full Explanation'
            );
            
            if (action === 'Show Full Explanation') {
                // Create a new document with the full explanation
                const doc = await vscode.workspace.openTextDocument({
                    content: explanation,
                    language: 'markdown'
                });
                vscode.window.showTextDocument(doc, vscode.ViewColumn.Beside);
            }
        } else {
            vscode.window.showWarningMessage('Please select some code to explain');
        }
    });

    context.subscriptions.push(generateCommand, chatCommand, explainCommand);

    // Show welcome message
    vscode.window.showInformationMessage(
        '🚀 AI IDE is now active in VSCode! Use Ctrl+K for code generation, Ctrl+L for chat, Ctrl+E to explain code',
        'Show AI Chat'
    ).then(selection => {
        if (selection === 'Show AI Chat') {
            vscode.commands.executeCommand('ai-ide.chat');
        }
    });
}

function generateAICode(prompt: string, selectedText: string, language: string): string {
    const timestamp = new Date().toISOString();
    
    return `// 🤖 AI Generated Code
// Prompt: ${prompt}
// Language: ${language}
// Generated: ${timestamp}
${selectedText ? `// Original code:\n${selectedText.split('\n').map(line => `// ${line}`).join('\n')}\n\n` : ''}
// TODO: Connect this to your AI backend at backend/main.py for real AI generation

function aiGeneratedFunction() {
    // AI would generate real code here based on: ${prompt}
    console.log('🤖 AI IDE Request: ${prompt}');
    
    ${language === 'javascript' || language === 'typescript' ? `
    // Example ${language} AI generation
    const result = processUserRequest('${prompt}');
    return result;
    ` : language === 'python' ? `
    # Example Python AI generation
    def process_user_request():
        """${prompt}"""
        return "AI generated result"
    ` : language === 'java' ? `
    // Example Java AI generation
    public static String processUserRequest() {
        // ${prompt}
        return "AI generated result";
    }
    ` : `
    // AI generated code for ${language}
    // Implement: ${prompt}
    `}
}

// 🔗 Next steps to get real AI:
// 1. Start your AI backend: python backend/main.py
// 2. Update the backend URL in settings: ai-ide.backendUrl
// 3. Replace this placeholder with real AI API calls
`;
}

async function processAIMessage(message: string): Promise<string> {
    // This is where you'd connect to your AI backend
    // For now, return a helpful placeholder response
    
    const config = vscode.workspace.getConfiguration('ai-ide');
    const backendUrl = config.get<string>('backendUrl', 'http://localhost:8000');
    
    try {
        // TODO: Replace with actual API call to your backend
        // const response = await fetch(`${backendUrl}/chat`, {
        //     method: 'POST',
        //     headers: { 'Content-Type': 'application/json' },
        //     body: JSON.stringify({ message })
        // });
        // const data = await response.json();
        // return data.response;
        
        return `AI Response to "${message}":

I received your message! To get real AI responses:

1. Start your AI backend:
   \`\`\`bash
   cd backend
   python main.py
   \`\`\`

2. Configure the backend URL in VSCode settings:
   - Open Settings (Ctrl+,)
   - Search for "ai-ide.backendUrl"
   - Set it to your backend URL (default: http://localhost:8000)

3. Your backend at \`backend/main.py\` has all the AI systems ready:
   - Multi-agent reasoning
   - Web search integration
   - Code analysis
   - And much more!

Connect me to unlock full AI power! 🚀`;
        
    } catch (error) {
        return `❌ Could not connect to AI backend at ${backendUrl}. 

Please ensure:
1. Your backend is running: \`python backend/main.py\`
2. The URL is correct in settings
3. No firewall is blocking the connection

Error: ${error}`;
    }
}

async function explainCode(code: string, language: string): Promise<string> {
    return `# 🤖 AI Code Explanation

**Language:** ${language}
**Lines:** ${code.split('\n').length}
**Characters:** ${code.length}

## Code Analysis

\`\`\`${language}
${code}
\`\`\`

## Explanation

This appears to be ${language} code. Here's what I can tell you:

- **Structure:** The code has ${code.split('\n').length} lines
- **Complexity:** ${code.length > 500 ? 'High' : code.length > 100 ? 'Medium' : 'Low'} complexity based on length
- **Language Features:** Uses ${language} syntax and conventions

## To Get Detailed AI Analysis

Connect me to your AI backend for comprehensive code analysis:

1. **Start Backend:** \`python backend/main.py\`
2. **Configure URL:** Set ai-ide.backendUrl in VSCode settings
3. **Get Real Analysis:** Your backend has advanced code analysis capabilities

## Next Steps

- Connect to backend for detailed explanations
- Use Ctrl+K to generate related code
- Use Ctrl+L to ask specific questions about this code

---
*Generated by AI IDE Extension - Connect to backend for full AI power!*`;
}

function getChatHTML(): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Chat</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            margin: 0;
            background: var(--vscode-editor-background);
            color: var(--vscode-editor-foreground);
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: var(--vscode-editor-selectionBackground);
            border-radius: 8px;
        }
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--vscode-panel-border);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: var(--vscode-editor-background);
        }
        .message {
            margin: 10px 0;
            padding: 12px;
            border-radius: 8px;
            max-width: 85%;
        }
        .user-message {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            margin-left: auto;
            text-align: right;
        }
        .ai-message {
            background: var(--vscode-editor-selectionBackground);
            margin-right: auto;
        }
        .input-container {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        input {
            flex: 1;
            padding: 12px;
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            font-size: 14px;
        }
        button {
            padding: 12px 20px;
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background: var(--vscode-button-hoverBackground);
        }
        .features {
            margin: 20px 0;
            padding: 15px;
            background: var(--vscode-editor-inactiveSelectionBackground);
            border-radius: 8px;
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 8px 0;
            border-bottom: 1px solid var(--vscode-panel-border);
        }
        .feature-list li:last-child {
            border-bottom: none;
        }
        code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: var(--vscode-editor-font-family);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 AI IDE Chat Panel</h1>
        <p>Your Cursor-like AI assistant in REAL VSCode!</p>
    </div>

    <div class="features">
        <h3>✨ AI Features in REAL VSCode:</h3>
        <ul class="feature-list">
            <li>🎯 <strong>Ctrl+K:</strong> AI inline code generation (like Cursor)</li>
            <li>💬 <strong>Ctrl+L:</strong> AI chat panel (this window)</li>
            <li>🔍 <strong>Ctrl+E:</strong> AI code explanation with detailed view</li>
            <li>🚀 <strong>Full VSCode:</strong> ALL original features included</li>
            <li>📁 <strong>Complete IDE:</strong> File Explorer, Terminal, Debug, Git, Extensions</li>
            <li>⚙️ <strong>Configurable:</strong> Set your backend URL in VSCode settings</li>
        </ul>
    </div>

    <div class="chat-container" id="messages">
        <div class="message ai-message">
            <strong>🤖 AI Assistant:</strong> Hello! I'm your AI coding assistant running in REAL VSCode! 
            <br><br>
            <strong>🎯 Available Features:</strong><br>
            • <code>Ctrl+K</code>: Generate code inline (like Cursor)<br>
            • <code>Ctrl+L</code>: Open this chat panel<br>
            • <code>Ctrl+E</code>: Explain selected code with detailed analysis<br>
            <br>
            <strong>🔗 Connect to Your AI Backend:</strong><br>
            1. Start your backend: <code>python backend/main.py</code><br>
            2. Configure URL in VSCode Settings: <code>ai-ide.backendUrl</code><br>
            3. Your backend has advanced AI systems ready!<br>
            <br>
            <strong>💡 Try This:</strong><br>
            • Select some code and press <code>Ctrl+E</code><br>
            • Press <code>Ctrl+K</code> to generate code<br>
            • Ask me questions about your code here!<br>
            <br>
            What would you like to know about your code?
        </div>
    </div>

    <div class="input-container">
        <input type="text" id="messageInput" placeholder="Ask me anything about your code..." />
        <button onclick="sendMessage()">Send</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const messages = document.getElementById('messages');
            
            if (input.value.trim()) {
                // Add user message
                const userMsg = document.createElement('div');
                userMsg.className = 'message user-message';
                userMsg.innerHTML = '<strong>You:</strong> ' + input.value;
                messages.appendChild(userMsg);
                
                // Send to extension
                vscode.postMessage({
                    command: 'sendMessage',
                    text: input.value
                });
                
                input.value = '';
                messages.scrollTop = messages.scrollHeight;
            }
        }
        
        // Listen for messages from extension
        window.addEventListener('message', event => {
            const message = event.data;
            if (message.command === 'aiResponse') {
                const messages = document.getElementById('messages');
                const aiMsg = document.createElement('div');
                aiMsg.className = 'message ai-message';
                aiMsg.innerHTML = '<strong>🤖 AI:</strong> ' + message.text.replace(/\n/g, '<br>').replace(/\`([^`]+)\`/g, '<code>$1</code>');
                messages.appendChild(aiMsg);
                messages.scrollTop = messages.scrollHeight;
            }
        });
        
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>`;
}

export function deactivate() {}
'@

$extensionTs | Set-Content (Join-Path $extensionDir "src\extension.ts")

# Create tsconfig.json
$tsconfig = @{
    compilerOptions = @{
        module = "commonjs"
        target = "ES2020"
        outDir = "out"
        lib = @("ES2020")
        sourceMap = $true
        rootDir = "src"
        strict = $true
    }
    exclude = @("node_modules", ".vscode-test")
}

$tsconfig | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $extensionDir "tsconfig.json")

Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Set-Location $extensionDir
npm install --silent

if ($LASTEXITCODE -eq 0) {
    Write-Host "🔨 Compiling TypeScript..." -ForegroundColor Yellow
    npm run compile --silent
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "📦 Installing extension in VSCode..." -ForegroundColor Yellow
        Set-Location $ProjectRoot
        
        # Install the extension
        code --install-extension $extensionDir --force
        
        Write-Host ""
        Write-Host "🎉 SUCCESS! AI Features added to REAL VSCode!" -ForegroundColor Green
        Write-Host ""
        Write-Host "🚀 To use your AI IDE:" -ForegroundColor Yellow
        Write-Host "   1. Open VSCode (it should start automatically)" -ForegroundColor White
        Write-Host "   2. Or run: code ." -ForegroundColor White
        Write-Host ""
        Write-Host "✨ AI Features in REAL VSCode:" -ForegroundColor Yellow
        Write-Host "   • Ctrl+K: AI code generation (like Cursor)" -ForegroundColor White
        Write-Host "   • Ctrl+L: AI chat panel (like Cursor)" -ForegroundColor White
        Write-Host "   • Ctrl+E: AI code explanation" -ForegroundColor White
        Write-Host "   • ALL VSCode features: File, Edit, View, Terminal, etc." -ForegroundColor White
        Write-Host ""
        Write-Host "🔗 Connect to AI Backend:" -ForegroundColor Yellow
        Write-Host "   1. Start backend: python backend/main.py" -ForegroundColor White
        Write-Host "   2. Open VSCode Settings (Ctrl+,)" -ForegroundColor White
        Write-Host "   3. Search for 'ai-ide.backendUrl'" -ForegroundColor White
        Write-Host "   4. Set to your backend URL" -ForegroundColor White
        Write-Host ""
        Write-Host "📁 Extension Location: $extensionDir" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "This is REAL VSCode with ALL features + AI!" -ForegroundColor Green
        
        # Launch VSCode
        Write-Host "🚀 Launching VSCode with AI features..." -ForegroundColor Cyan
        Start-Process "code" -ArgumentList "."
        
    } else {
        Write-Host "❌ TypeScript compilation failed" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Dependency installation failed" -ForegroundColor Red
}

Set-Location $ProjectRoot