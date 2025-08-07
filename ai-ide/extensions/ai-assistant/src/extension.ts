import * as vscode from 'vscode';
import { PocketFlowBridge } from './services/PocketFlowBridge';
import { ChatProvider } from './providers/ChatProvider';
import { SearchDashboardProvider } from './providers/SearchDashboardProvider';

let pocketFlowBridge: PocketFlowBridge;
let chatProvider: ChatProvider;
let searchDashboardProvider: SearchDashboardProvider;

export function activate(context: vscode.ExtensionContext) {
    console.log('AI Assistant extension is now active!');

    // Initialize PocketFlow bridge
    pocketFlowBridge = new PocketFlowBridge(context.extensionPath);
    
    // Initialize providers
    chatProvider = new ChatProvider(context, pocketFlowBridge);
    searchDashboardProvider = new SearchDashboardProvider(context, pocketFlowBridge);
    
    // Register webview providers
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(ChatProvider.viewType, chatProvider),
        vscode.window.registerWebviewViewProvider(SearchDashboardProvider.viewType, searchDashboardProvider)
    );
    
    // Initialize backend services
    initializeBackendServices(context);

    // Register commands
    const openChatCommand = vscode.commands.registerCommand('ai-assistant.openChat', async () => {
        await openAIChat(context);
    });

    const generateCodeCommand = vscode.commands.registerCommand('ai-assistant.generateCode', async () => {
        await generateCodeFromSelection();
    });

    const semanticSearchCommand = vscode.commands.registerCommand('ai-assistant.semanticSearch', async () => {
        await performSemanticSearch();
    });

    const reasoningCommand = vscode.commands.registerCommand('ai-assistant.reasoning', async () => {
        await performReasoning();
    });

    // New commands for advanced features
    const openSearchDashboardCommand = vscode.commands.registerCommand('ai-assistant.openSearchDashboard', async () => {
        // Focus on the search dashboard view
        vscode.commands.executeCommand('ai-assistant.searchDashboard.focus');
    });

    const insertCodeSnippetCommand = vscode.commands.registerCommand('ai-assistant.insertCodeSnippet', async (snippet) => {
        await insertCodeSnippet(snippet);
    });

    const performUnifiedSearchCommand = vscode.commands.registerCommand('ai-assistant.performUnifiedSearch', async () => {
        const query = await vscode.window.showInputBox({
            prompt: 'Enter search query for unified search (semantic + web + RAG)',
            placeHolder: 'e.g., "React hooks best practices"'
        });

        if (query) {
            // Focus search dashboard and trigger search
            await vscode.commands.executeCommand('ai-assistant.searchDashboard.focus');
            // The search will be handled by the webview
        }
    });

    context.subscriptions.push(
        openChatCommand,
        generateCodeCommand,
        semanticSearchCommand,
        reasoningCommand,
        openSearchDashboardCommand,
        insertCodeSnippetCommand,
        performUnifiedSearchCommand,
        pocketFlowBridge,
        chatProvider,
        searchDashboardProvider
    );

    // Show activation message
    vscode.window.showInformationMessage('AI Assistant extension activated with advanced UI components!');
}

async function initializeBackendServices(context: vscode.ExtensionContext) {
    try {
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Initializing AI Backend...",
            cancellable: false
        }, async (progress) => {
            progress.report({ increment: 0, message: "Starting backend services..." });
            
            await pocketFlowBridge.initialize();
            
            progress.report({ increment: 100, message: "Backend initialized!" });
        });
    } catch (error) {
        console.error('Failed to initialize backend services:', error);
        vscode.window.showErrorMessage(`Failed to initialize AI Backend: ${error}`);
    }
}

async function openAIChat(context: vscode.ExtensionContext) {
    const panel = vscode.window.createWebviewPanel(
        'ai-chat',
        'AI Assistant Chat',
        vscode.ViewColumn.Beside,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    panel.webview.html = getAIChatHtml();
    
    // Handle messages from the webview
    panel.webview.onDidReceiveMessage(
        async (message) => {
            switch (message.command) {
                case 'sendMessage':
                    await handleChatMessage(panel, message.text);
                    break;
            }
        },
        undefined,
        context.subscriptions
    );
}

async function handleChatMessage(panel: vscode.WebviewPanel, userMessage: string) {
    try {
        // Show typing indicator
        panel.webview.postMessage({
            command: 'showTyping'
        });

        // Get current editor context
        const context = getCurrentEditorContext();
        
        // Use PocketFlow for reasoning about the user's message
        const reasoningResult = await pocketFlowBridge.executeReasoning(
            `User message: ${userMessage}. Provide a helpful response for a coding assistant.`,
            'chain-of-thought',
            context
        );

        // Send response back to chat
        panel.webview.postMessage({
            command: 'addMessage',
            role: 'assistant',
            message: reasoningResult.solution,
            reasoning: reasoningResult.reasoning,
            confidence: reasoningResult.confidence
        });

    } catch (error) {
        console.error('Error handling chat message:', error);
        panel.webview.postMessage({
            command: 'addMessage',
            role: 'assistant',
            message: `Sorry, I encountered an error: ${error}`,
            reasoning: [],
            confidence: 0
        });
    } finally {
        panel.webview.postMessage({
            command: 'hideTyping'
        });
    }
}

async function generateCodeFromSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found');
        return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);
    
    if (!selectedText) {
        vscode.window.showWarningMessage('Please select some text first');
        return;
    }

    try {
        const prompt = await vscode.window.showInputBox({
            prompt: 'What would you like to generate based on the selected text?',
            placeHolder: 'e.g., "Create a unit test for this function"'
        });

        if (!prompt) return;

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Generating code...",
            cancellable: false
        }, async (progress) => {
            const context = {
                filePath: editor.document.fileName,
                selectedText: selectedText,
                language: editor.document.languageId,
                cursorPosition: editor.document.offsetAt(selection.active)
            };

            const result = await pocketFlowBridge.executeCodeGeneration(
                `${prompt}\n\nSelected code:\n${selectedText}`,
                context
            );

            // Insert generated code
            const position = selection.end;
            await editor.edit(editBuilder => {
                editBuilder.insert(position, `\n\n${result.code}\n`);
            });

            vscode.window.showInformationMessage(
                `Code generated with ${Math.round(result.confidence * 100)}% confidence`
            );
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Code generation failed: ${error}`);
    }
}

async function performSemanticSearch() {
    const query = await vscode.window.showInputBox({
        prompt: 'Enter semantic search query',
        placeHolder: 'e.g., "function that handles user authentication"'
    });

    if (!query) return;

    try {
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Performing semantic search...",
            cancellable: false
        }, async (progress) => {
            const results = await pocketFlowBridge.executeSemanticSearch(query);

            if (results.length === 0) {
                vscode.window.showInformationMessage('No results found');
                return;
            }

            // Show results in a quick pick
            const items = results.map(result => ({
                label: `${result.file}:${result.line}`,
                description: `${Math.round(result.similarity * 100)}% match`,
                detail: result.content,
                result: result
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select a search result to open'
            });

            if (selected) {
                const uri = vscode.Uri.file(selected.result.file);
                const document = await vscode.workspace.openTextDocument(uri);
                const editor = await vscode.window.showTextDocument(document);
                
                // Jump to the line
                const line = selected.result.line - 1;
                const range = new vscode.Range(line, 0, line, 0);
                editor.selection = new vscode.Selection(range.start, range.end);
                editor.revealRange(range);
            }
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Semantic search failed: ${error}`);
    }
}

async function performReasoning() {
    const problem = await vscode.window.showInputBox({
        prompt: 'Describe the problem you want me to reason about',
        placeHolder: 'e.g., "How should I structure this class hierarchy?"'
    });

    if (!problem) return;

    const mode = await vscode.window.showQuickPick([
        { label: 'Basic', value: 'basic' },
        { label: 'Chain of Thought', value: 'chain-of-thought' },
        { label: 'Deep Analysis', value: 'deep' },
        { label: 'Interleaved', value: 'interleaved' }
    ], {
        placeHolder: 'Select reasoning mode'
    });

    if (!mode) return;

    try {
        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "Reasoning...",
            cancellable: false
        }, async (progress) => {
            const context = getCurrentEditorContext();
            const result = await pocketFlowBridge.executeReasoning(
                problem,
                mode.value as any,
                context
            );

            // Show result in a new document
            const doc = await vscode.workspace.openTextDocument({
                content: `# Reasoning Result\n\n## Problem\n${problem}\n\n## Mode\n${mode.label}\n\n## Solution\n${result.solution}\n\n## Reasoning Steps\n${result.reasoning.map((step, i) => `${i + 1}. ${step}`).join('\n')}\n\n## Confidence\n${Math.round(result.confidence * 100)}%`,
                language: 'markdown'
            });

            await vscode.window.showTextDocument(doc);
        });

    } catch (error) {
        vscode.window.showErrorMessage(`Reasoning failed: ${error}`);
    }
}

function getCurrentEditorContext() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return {};

    const document = editor.document;
    const selection = editor.selection;

    return {
        filePath: document.fileName,
        language: document.languageId,
        selectedText: selection.isEmpty ? undefined : document.getText(selection),
        cursorPosition: document.offsetAt(selection.active),
        lineNumber: selection.active.line + 1,
        columnNumber: selection.active.character + 1
    };
}

function getAIChatHtml(): string {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Assistant Chat</title>
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
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
            }
            .message {
                margin-bottom: 16px;
                padding: 12px;
                border-radius: 8px;
                max-width: 80%;
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
            .reasoning-steps {
                margin-top: 8px;
                font-size: 0.9em;
                opacity: 0.8;
            }
            .confidence {
                margin-top: 4px;
                font-size: 0.8em;
                color: var(--vscode-textLink-foreground);
            }
            .input-container {
                padding: 16px;
                border-top: 1px solid var(--vscode-panel-border);
                display: flex;
                gap: 8px;
            }
            .input-field {
                flex: 1;
                padding: 8px 12px;
                border: 1px solid var(--vscode-input-border);
                border-radius: 4px;
                background-color: var(--vscode-input-background);
                color: var(--vscode-input-foreground);
            }
            .send-button {
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
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-content">Hello! I'm your AI coding assistant powered by enhanced PocketFlow. I can help you with code generation, semantic search, and complex reasoning. What would you like to work on?</div>
            </div>
        </div>
        <div class="typing-indicator" id="typingIndicator">AI is thinking...</div>
        <div class="input-container">
            <input type="text" class="input-field" id="messageInput" placeholder="Ask me anything about your code..." />
            <button class="send-button" id="sendButton">Send</button>
        </div>

        <script>
            const vscode = acquireVsCodeApi();
            const chatContainer = document.getElementById('chatContainer');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');

            function sendMessage() {
                const message = messageInput.value.trim();
                if (message) {
                    addMessageToChat('user', message);
                    vscode.postMessage({
                        command: 'sendMessage',
                        text: message
                    });
                    messageInput.value = '';
                }
            }

            sendButton.addEventListener('click', sendMessage);
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            window.addEventListener('message', event => {
                const message = event.data;
                switch (message.command) {
                    case 'addMessage':
                        addMessageToChat(message.role, message.message, message.reasoning, message.confidence);
                        break;
                    case 'showTyping':
                        typingIndicator.style.display = 'block';
                        scrollToBottom();
                        break;
                    case 'hideTyping':
                        typingIndicator.style.display = 'none';
                        break;
                }
            });

            function addMessageToChat(role, content, reasoning, confidence) {
                const messageDiv = document.createElement('div');
                messageDiv.className = \`message \${role}\`;
                
                let messageContent = \`<div class="message-content">\${content}</div>\`;
                
                if (reasoning && reasoning.length > 0) {
                    messageContent += \`<div class="reasoning-steps">Reasoning: \${reasoning.join(', ')}</div>\`;
                }
                
                if (confidence !== undefined) {
                    messageContent += \`<div class="confidence">Confidence: \${Math.round(confidence * 100)}%</div>\`;
                }
                
                messageDiv.innerHTML = messageContent;
                chatContainer.appendChild(messageDiv);
                
                scrollToBottom();
            }

            function scrollToBottom() {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            messageInput.focus();
        </script>
    </body>
    </html>`;
}

async function insertCodeSnippet(snippet: any): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showWarningMessage('No active editor found');
        return;
    }

    const position = editor.selection.active;
    await editor.edit(editBuilder => {
        editBuilder.insert(position, snippet.code || snippet);
    });

    vscode.window.showInformationMessage('Code snippet inserted successfully');
}

export function deactivate() {
    console.log('AI Assistant extension is now deactivated!');
    if (pocketFlowBridge) {
        pocketFlowBridge.dispose();
    }
    if (chatProvider) {
        chatProvider.dispose();
    }
    if (searchDashboardProvider) {
        searchDashboardProvider.dispose();
    }
}