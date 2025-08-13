import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { PocketFlowBridge } from '../services/PocketFlowBridge';

export class ContextualChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'ai-assistant.contextualChat';
    private _view?: vscode.WebviewView;
    private disposables: vscode.Disposable[] = [];

    constructor(
        private readonly context: vscode.ExtensionContext,
        private readonly pocketFlowBridge: PocketFlowBridge
    ) {}

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this.context.extensionUri]
        };

        webviewView.webview.html = this.getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(
            async (data) => {
                switch (data.type) {
                    case 'sendMessage':
                        await this.handleChatMessage(data.message);
                        break;
                    case 'addContext':
                        await this.handleContextAddition(data.contextType, data.contextData);
                        break;
                    case 'clearContext':
                        await this.clearContext();
                        break;
                }
            },
            null,
            this.disposables
        );
    }

    private async handleChatMessage(message: string) {
        if (!this._view) return;

        try {
            // Parse context references (#File, #Folder, etc.)
            const contextData = await this.parseContextReferences(message);
            
            // Show typing indicator
            this._view.webview.postMessage({
                type: 'showTyping'
            });

            // Get enhanced context
            const fullContext = await this.buildFullContext(contextData);
            
            // Use PocketFlow for reasoning with context
            const result = await this.pocketFlowBridge.executeReasoning(
                message,
                'chain-of-thought',
                fullContext
            );

            // Send response
            this._view.webview.postMessage({
                type: 'addMessage',
                role: 'assistant',
                message: result.solution,
                reasoning: result.reasoning,
                confidence: result.confidence,
                context: contextData
            });

        } catch (error) {
            console.error('Contextual chat error:', error);
            this._view.webview.postMessage({
                type: 'addMessage',
                role: 'assistant',
                message: `Sorry, I encountered an error: ${error}`,
                reasoning: [],
                confidence: 0
            });
        } finally {
            this._view.webview.postMessage({
                type: 'hideTyping'
            });
        }
    }

    private async parseContextReferences(message: string): Promise<any> {
        const contextData: any = {
            files: [],
            folders: [],
            problems: [],
            terminal: null,
            git: null,
            codebase: null
        };

        // Parse #File references
        const fileMatches = message.match(/#File\s+([^\s]+)/g);
        if (fileMatches) {
            for (const match of fileMatches) {
                const filePath = match.replace('#File ', '');
                const fileContent = await this.getFileContent(filePath);
                if (fileContent) {
                    contextData.files.push({
                        path: filePath,
                        content: fileContent
                    });
                }
            }
        }

        // Parse #Folder references
        const folderMatches = message.match(/#Folder\s+([^\s]+)/g);
        if (folderMatches) {
            for (const match of folderMatches) {
                const folderPath = match.replace('#Folder ', '');
                const folderContent = await this.getFolderContent(folderPath);
                if (folderContent) {
                    contextData.folders.push({
                        path: folderPath,
                        content: folderContent
                    });
                }
            }
        }

        // Parse #Problems
        if (message.includes('#Problems')) {
            contextData.problems = await this.getCurrentProblems();
        }

        // Parse #Terminal
        if (message.includes('#Terminal')) {
            contextData.terminal = await this.getTerminalOutput();
        }

        // Parse #Git
        if (message.includes('#Git')) {
            contextData.git = await this.getGitDiff();
        }

        // Parse #Codebase
        if (message.includes('#Codebase')) {
            contextData.codebase = await this.getCodebaseContext();
        }

        return contextData;
    }

    private async getFileContent(filePath: string): Promise<string | null> {
        try {
            // Handle relative paths
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) return null;

            const fullPath = path.isAbsolute(filePath) 
                ? filePath 
                : path.join(workspaceFolder.uri.fsPath, filePath);

            if (fs.existsSync(fullPath)) {
                return fs.readFileSync(fullPath, 'utf8');
            }

            // Try to find the file in open editors
            const openDocument = vscode.workspace.textDocuments.find(
                doc => doc.fileName.endsWith(filePath)
            );
            
            if (openDocument) {
                return openDocument.getText();
            }

            return null;
        } catch (error) {
            console.error('Error reading file:', error);
            return null;
        }
    }

    private async getFolderContent(folderPath: string): Promise<any> {
        try {
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) return null;

            const fullPath = path.isAbsolute(folderPath)
                ? folderPath
                : path.join(workspaceFolder.uri.fsPath, folderPath);

            if (!fs.existsSync(fullPath)) return null;

            const files = fs.readdirSync(fullPath, { withFileTypes: true });
            const structure: any = {
                files: [],
                directories: []
            };

            for (const file of files) {
                if (file.isFile()) {
                    structure.files.push(file.name);
                } else if (file.isDirectory()) {
                    structure.directories.push(file.name);
                }
            }

            return structure;
        } catch (error) {
            console.error('Error reading folder:', error);
            return null;
        }
    }

    private async getCurrentProblems(): Promise<any[]> {
        const problems: any[] = [];
        
        // Get diagnostics from all open documents
        for (const document of vscode.workspace.textDocuments) {
            const diagnostics = vscode.languages.getDiagnostics(document.uri);
            if (diagnostics.length > 0) {
                problems.push({
                    file: document.fileName,
                    problems: diagnostics.map(d => ({
                        line: d.range.start.line + 1,
                        column: d.range.start.character + 1,
                        severity: vscode.DiagnosticSeverity[d.severity],
                        message: d.message,
                        source: d.source
                    }))
                });
            }
        }

        return problems;
    }

    private async getTerminalOutput(): Promise<string | null> {
        // This is a simplified implementation
        // In a real implementation, you'd need to capture terminal output
        return "Terminal output capture not fully implemented yet";
    }

    private async getGitDiff(): Promise<any> {
        try {
            // Use VS Code's git extension API if available
            const gitExtension = vscode.extensions.getExtension('vscode.git');
            if (!gitExtension) return null;

            // This is a simplified implementation
            // In reality, you'd use the git extension's API
            return {
                status: "Git diff capture not fully implemented yet",
                changes: []
            };
        } catch (error) {
            console.error('Error getting git diff:', error);
            return null;
        }
    }

    private async getCodebaseContext(): Promise<any> {
        try {
            // Use semantic search to get relevant codebase context
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            if (!workspaceFolder) return null;

            // This would use the semantic search engine
            return {
                summary: "Codebase context scanning not fully implemented yet",
                relevantFiles: [],
                patterns: []
            };
        } catch (error) {
            console.error('Error getting codebase context:', error);
            return null;
        }
    }

    private async buildFullContext(contextData: any): Promise<any> {
        const editor = vscode.window.activeTextEditor;
        const baseContext = {
            currentFile: editor?.document.fileName,
            language: editor?.document.languageId,
            selection: editor?.selection.isEmpty ? undefined : editor?.document.getText(editor.selection),
            cursorPosition: editor?.selection.active
        };

        return {
            ...baseContext,
            ...contextData,
            timestamp: new Date().toISOString()
        };
    }

    private async handleContextAddition(contextType: string, contextData: any) {
        // Handle manual context addition
        console.log(`Adding context: ${contextType}`, contextData);
    }

    private async clearContext() {
        // Clear all context
        if (this._view) {
            this._view.webview.postMessage({
                type: 'contextCleared'
            });
        }
    }

    private getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Contextual AI Chat</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                }
                .context-bar {
                    padding: 8px;
                    background-color: var(--vscode-panel-background);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    font-size: 0.9em;
                }
                .context-tags {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 4px;
                    margin-top: 4px;
                }
                .context-tag {
                    background-color: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 0.8em;
                }
                .chat-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                }
                .message {
                    margin-bottom: 16px;
                    padding: 12px;
                    border-radius: 8px;
                    max-width: 90%;
                }
                .message.user {
                    background-color: var(--vscode-textBlockQuote-background);
                    margin-left: auto;
                    text-align: right;
                }
                .message.assistant {
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                }
                .message-content {
                    white-space: pre-wrap;
                }
                .context-info {
                    margin-top: 8px;
                    font-size: 0.8em;
                    opacity: 0.7;
                }
                .input-container {
                    padding: 16px;
                    border-top: 1px solid var(--vscode-panel-border);
                }
                .context-help {
                    font-size: 0.8em;
                    color: var(--vscode-descriptionForeground);
                    margin-bottom: 8px;
                }
                .input-field {
                    width: 100%;
                    padding: 8px 12px;
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    margin-bottom: 8px;
                }
                .send-button {
                    width: 100%;
                    padding: 8px 16px;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .typing-indicator {
                    display: none;
                    padding: 12px;
                    font-style: italic;
                    opacity: 0.7;
                }
            </style>
        </head>
        <body>
            <div class="context-bar">
                <div>Active Context:</div>
                <div class="context-tags" id="contextTags">
                    <span class="context-tag">No context</span>
                </div>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    <div class="message-content">Hello! I'm your contextual AI assistant. I can understand context from your workspace using these references:
                    
• #File [path] - Include specific file content
• #Folder [path] - Include folder structure  
• #Problems - Include current problems/errors
• #Terminal - Include terminal output
• #Git - Include git diff/status
• #Codebase - Include relevant codebase context

Try asking: "Help me fix the errors in #File src/main.js with #Problems context"</div>
                </div>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">AI is analyzing context...</div>
            
            <div class="input-container">
                <div class="context-help">Use #File, #Folder, #Problems, #Terminal, #Git, #Codebase for context</div>
                <textarea class="input-field" id="messageInput" placeholder="Ask me anything with context..." rows="3"></textarea>
                <button class="send-button" id="sendButton">Send</button>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                const chatContainer = document.getElementById('chatContainer');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const typingIndicator = document.getElementById('typingIndicator');
                const contextTags = document.getElementById('contextTags');

                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (message) {
                        addMessageToChat('user', message);
                        vscode.postMessage({
                            type: 'sendMessage',
                            message: message
                        });
                        messageInput.value = '';
                    }
                }

                sendButton.addEventListener('click', sendMessage);
                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && e.ctrlKey) {
                        sendMessage();
                    }
                });

                window.addEventListener('message', event => {
                    const message = event.data;
                    switch (message.type) {
                        case 'addMessage':
                            addMessageToChat(message.role, message.message, message.context);
                            updateContextTags(message.context);
                            break;
                        case 'showTyping':
                            typingIndicator.style.display = 'block';
                            scrollToBottom();
                            break;
                        case 'hideTyping':
                            typingIndicator.style.display = 'none';
                            break;
                        case 'contextCleared':
                            updateContextTags(null);
                            break;
                    }
                });

                function addMessageToChat(role, content, context) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = \`message \${role}\`;
                    
                    let messageContent = \`<div class="message-content">\${content}</div>\`;
                    
                    if (context && role === 'assistant') {
                        const contextInfo = [];
                        if (context.files?.length) contextInfo.push(\`\${context.files.length} files\`);
                        if (context.folders?.length) contextInfo.push(\`\${context.folders.length} folders\`);
                        if (context.problems?.length) contextInfo.push(\`\${context.problems.length} problems\`);
                        if (context.terminal) contextInfo.push('terminal');
                        if (context.git) contextInfo.push('git');
                        if (context.codebase) contextInfo.push('codebase');
                        
                        if (contextInfo.length > 0) {
                            messageContent += \`<div class="context-info">Context used: \${contextInfo.join(', ')}</div>\`;
                        }
                    }
                    
                    messageDiv.innerHTML = messageContent;
                    chatContainer.appendChild(messageDiv);
                    
                    scrollToBottom();
                }

                function updateContextTags(context) {
                    if (!context) {
                        contextTags.innerHTML = '<span class="context-tag">No context</span>';
                        return;
                    }

                    const tags = [];
                    if (context.files?.length) tags.push(\`Files: \${context.files.length}\`);
                    if (context.folders?.length) tags.push(\`Folders: \${context.folders.length}\`);
                    if (context.problems?.length) tags.push(\`Problems: \${context.problems.length}\`);
                    if (context.terminal) tags.push('Terminal');
                    if (context.git) tags.push('Git');
                    if (context.codebase) tags.push('Codebase');

                    if (tags.length === 0) {
                        contextTags.innerHTML = '<span class="context-tag">No context</span>';
                    } else {
                        contextTags.innerHTML = tags.map(tag => 
                            \`<span class="context-tag">\${tag}</span>\`
                        ).join('');
                    }
                }

                function scrollToBottom() {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                messageInput.focus();
            </script>
        </body>
        </html>`;
    }

    public dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}