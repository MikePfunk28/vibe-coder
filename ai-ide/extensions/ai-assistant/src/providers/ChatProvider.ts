import * as vscode from 'vscode';
import { PocketFlowBridge } from '../services/PocketFlowBridge';

export interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    agent?: string;
    reasoning?: ReasoningTrace[];
    confidence?: number;
    webSearchResults?: WebSearchResult[];
    codeSnippets?: CodeSnippet[];
    metadata?: any;
}

export interface ReasoningTrace {
    step: number;
    type: 'thought' | 'action' | 'observation' | 'conclusion';
    content: string;
    confidence?: number;
    duration?: number;
}

export interface WebSearchResult {
    title: string;
    url: string;
    snippet: string;
    relevance: number;
    source: string;
}

export interface CodeSnippet {
    language: string;
    code: string;
    description?: string;
    filePath?: string;
    insertable: boolean;
}

export interface AgentConfig {
    id: string;
    name: string;
    description: string;
    capabilities: string[];
    icon: string;
    color: string;
}

export interface ReasoningMode {
    id: string;
    name: string;
    description: string;
    icon: string;
    complexity: 'basic' | 'intermediate' | 'advanced';
}

export class ChatProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'ai-assistant.chatView';
    
    private _view?: vscode.WebviewView;
    private _context: vscode.ExtensionContext;
    private _pocketFlowBridge: PocketFlowBridge;
    private _messages: ChatMessage[] = [];
    private _currentAgent: string = 'general';
    private _currentReasoningMode: string = 'chain-of-thought';
    private _isStreaming: boolean = false;

    private readonly _agents: AgentConfig[] = [
        {
            id: 'general',
            name: 'General Assistant',
            description: 'General purpose coding assistant',
            capabilities: ['code-generation', 'explanation', 'debugging'],
            icon: 'robot',
            color: '#007ACC'
        },
        {
            id: 'code-agent',
            name: 'Code Specialist',
            description: 'Specialized in code generation and refactoring',
            capabilities: ['code-generation', 'refactoring', 'optimization'],
            icon: 'code',
            color: '#28A745'
        },
        {
            id: 'search-agent',
            name: 'Search Expert',
            description: 'Semantic search and knowledge retrieval',
            capabilities: ['semantic-search', 'web-search', 'documentation'],
            icon: 'search',
            color: '#FFC107'
        },
        {
            id: 'reasoning-agent',
            name: 'Reasoning Engine',
            description: 'Complex problem solving and analysis',
            capabilities: ['chain-of-thought', 'deep-reasoning', 'analysis'],
            icon: 'lightbulb',
            color: '#DC3545'
        },
        {
            id: 'test-agent',
            name: 'Test Engineer',
            description: 'Test generation and validation',
            capabilities: ['test-generation', 'validation', 'debugging'],
            icon: 'beaker',
            color: '#6F42C1'
        }
    ];

    private readonly _reasoningModes: ReasoningMode[] = [
        {
            id: 'basic',
            name: 'Basic',
            description: 'Quick responses with minimal reasoning',
            icon: 'zap',
            complexity: 'basic'
        },
        {
            id: 'chain-of-thought',
            name: 'Chain of Thought',
            description: 'Step-by-step reasoning process',
            icon: 'git-branch',
            complexity: 'intermediate'
        },
        {
            id: 'deep',
            name: 'Deep Analysis',
            description: 'Comprehensive analysis with detailed reasoning',
            icon: 'microscope',
            complexity: 'advanced'
        },
        {
            id: 'interleaved',
            name: 'Interleaved',
            description: 'Advanced interleaved reasoning with context windows',
            icon: 'layers',
            complexity: 'advanced'
        },
        {
            id: 'react',
            name: 'ReAct',
            description: 'Reasoning and Acting with tool usage',
            icon: 'tools',
            complexity: 'advanced'
        }
    ];

    constructor(context: vscode.ExtensionContext, pocketFlowBridge: PocketFlowBridge) {
        this._context = context;
        this._pocketFlowBridge = pocketFlowBridge;
    }

    public dispose(): void {
        // Clean up resources if needed
        this._messages = [];
        this._view = undefined;
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                this._context.extensionUri
            ]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview
        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this._handleSendMessage(data.message);
                    break;
                case 'selectAgent':
                    this._currentAgent = data.agentId;
                    this._updateAgentSelection();
                    break;
                case 'selectReasoningMode':
                    this._currentReasoningMode = data.modeId;
                    this._updateReasoningModeSelection();
                    break;
                case 'insertCode':
                    await this._insertCodeSnippet(data.snippet);
                    break;
                case 'copyCode':
                    await this._copyCodeSnippet(data.snippet);
                    break;
                case 'openWebResult':
                    await this._openWebResult(data.url);
                    break;
                case 'clearChat':
                    this._clearChat();
                    break;
                case 'exportChat':
                    await this._exportChat();
                    break;
                case 'stopGeneration':
                    this._stopGeneration();
                    break;
            }
        });

        // Initialize with welcome message
        this._addWelcomeMessage();
    }

    private async _handleSendMessage(userMessage: string): Promise<void> {
        if (this._isStreaming) {
            vscode.window.showWarningMessage('Please wait for the current response to complete.');
            return;
        }

        // Add user message to chat
        const userChatMessage: ChatMessage = {
            id: this._generateMessageId(),
            role: 'user',
            content: userMessage,
            timestamp: new Date()
        };

        this._messages.push(userChatMessage);
        this._updateChatView();

        // Start streaming response
        this._isStreaming = true;
        this._showTypingIndicator();

        try {
            // Get current editor context
            const context = this._getCurrentEditorContext();
            
            // Create assistant message placeholder
            const assistantMessage: ChatMessage = {
                id: this._generateMessageId(),
                role: 'assistant',
                content: '',
                timestamp: new Date(),
                agent: this._currentAgent,
                reasoning: [],
                confidence: 0,
                webSearchResults: [],
                codeSnippets: []
            };

            this._messages.push(assistantMessage);
            this._updateChatView();

            // Execute the task based on selected agent and reasoning mode
            await this._executeAgentTask(userMessage, context, assistantMessage);

        } catch (error) {
            console.error('Error handling chat message:', error);
            
            const errorMessage: ChatMessage = {
                id: this._generateMessageId(),
                role: 'assistant',
                content: `I encountered an error: ${error}. Please try again or contact support if the issue persists.`,
                timestamp: new Date(),
                agent: this._currentAgent,
                confidence: 0
            };

            this._messages.push(errorMessage);
            this._updateChatView();
        } finally {
            this._isStreaming = false;
            this._hideTypingIndicator();
        }
    }

    private async _executeAgentTask(
        userMessage: string,
        context: any,
        assistantMessage: ChatMessage
    ): Promise<void> {
        const agent = this._agents.find(a => a.id === this._currentAgent);
        const reasoningMode = this._reasoningModes.find(m => m.id === this._currentReasoningMode);

        if (!agent || !reasoningMode) {
            throw new Error('Invalid agent or reasoning mode selected');
        }

        // Simulate streaming by updating the message progressively
        const updateMessage = (updates: Partial<ChatMessage>) => {
            Object.assign(assistantMessage, updates);
            this._updateChatView();
        };

        // Start reasoning trace
        const reasoningTrace: ReasoningTrace[] = [];
        
        // Step 1: Understanding the request
        reasoningTrace.push({
            step: 1,
            type: 'thought',
            content: `Analyzing user request with ${agent.name} using ${reasoningMode.name} reasoning...`,
            confidence: 0.9,
            duration: 100
        });
        
        updateMessage({ reasoning: [...reasoningTrace] });
        await this._delay(500);

        try {
            // Execute based on agent capabilities
            if (agent.capabilities.includes('web-search') || userMessage.toLowerCase().includes('search')) {
                // Web search integration
                reasoningTrace.push({
                    step: 2,
                    type: 'action',
                    content: 'Performing web search for relevant information...',
                    confidence: 0.8
                });
                updateMessage({ reasoning: [...reasoningTrace] });
                
                const webResults = await this._performWebSearch(userMessage);
                assistantMessage.webSearchResults = webResults;
                
                reasoningTrace.push({
                    step: 3,
                    type: 'observation',
                    content: `Found ${webResults.length} relevant web results`,
                    confidence: 0.85
                });
                updateMessage({ reasoning: [...reasoningTrace], webSearchResults: webResults });
                await this._delay(300);
            }

            // Execute main reasoning task
            const result = await this._pocketFlowBridge.executeReasoning(
                userMessage,
                this._currentReasoningMode as any,
                {
                    ...context,
                    agent: this._currentAgent,
                    webSearchResults: assistantMessage.webSearchResults
                }
            );

            // Update reasoning trace with backend results
            if (result.reasoning && result.reasoning.length > 0) {
                result.reasoning.forEach((step, index) => {
                    reasoningTrace.push({
                        step: reasoningTrace.length + 1,
                        type: index === result.reasoning.length - 1 ? 'conclusion' : 'thought',
                        content: step,
                        confidence: result.confidence
                    });
                });
            }

            // Extract code snippets from the response
            const codeSnippets = this._extractCodeSnippets(result.solution);
            
            // Final update
            updateMessage({
                content: result.solution,
                reasoning: reasoningTrace,
                confidence: result.confidence,
                codeSnippets: codeSnippets
            });

        } catch (error) {
            reasoningTrace.push({
                step: reasoningTrace.length + 1,
                type: 'observation',
                content: `Error occurred: ${error}`,
                confidence: 0
            });
            
            updateMessage({
                content: `I encountered an error while processing your request: ${error}`,
                reasoning: reasoningTrace,
                confidence: 0
            });
        }
    }

    private async _performWebSearch(query: string): Promise<WebSearchResult[]> {
        try {
            // Use the web search agent from the backend
            const result = await this._pocketFlowBridge.executeTask({
                id: this._generateMessageId(),
                type: 'reasoning', // Will be routed to web search agent
                input: {
                    query: query,
                    search_type: 'web',
                    max_results: 5
                }
            });

            if (result.success && result.result.web_results) {
                return result.result.web_results.map((item: any) => ({
                    title: item.title || 'No title',
                    url: item.url || '',
                    snippet: item.snippet || item.content || '',
                    relevance: item.relevance || 0.5,
                    source: item.source || 'web'
                }));
            }
        } catch (error) {
            console.error('Web search failed:', error);
        }

        return [];
    }

    private _extractCodeSnippets(content: string): CodeSnippet[] {
        const codeBlocks: CodeSnippet[] = [];
        const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
        let match;

        while ((match = codeBlockRegex.exec(content)) !== null) {
            const language = match[1] || 'text';
            const code = match[2].trim();
            
            if (code) {
                codeBlocks.push({
                    language,
                    code,
                    insertable: true,
                    description: `${language} code snippet`
                });
            }
        }

        return codeBlocks;
    }

    private async _insertCodeSnippet(snippet: CodeSnippet): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const position = editor.selection.active;
        await editor.edit(editBuilder => {
            editBuilder.insert(position, snippet.code);
        });

        vscode.window.showInformationMessage('Code snippet inserted successfully');
    }

    private async _copyCodeSnippet(snippet: CodeSnippet): Promise<void> {
        await vscode.env.clipboard.writeText(snippet.code);
        vscode.window.showInformationMessage('Code snippet copied to clipboard');
    }

    private async _openWebResult(url: string): Promise<void> {
        await vscode.env.openExternal(vscode.Uri.parse(url));
    }

    private _getCurrentEditorContext(): any {
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
            columnNumber: selection.active.character + 1,
            workspaceRoot: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
        };
    }

    private _addWelcomeMessage(): void {
        const welcomeMessage: ChatMessage = {
            id: this._generateMessageId(),
            role: 'assistant',
            content: `Hello! I'm your advanced AI coding assistant with multi-agent capabilities. I can help you with:

• **Code Generation & Refactoring** - Write, improve, and optimize your code
• **Semantic Search** - Find relevant code patterns and examples
• **Deep Reasoning** - Solve complex programming problems step-by-step
• **Web Search Integration** - Get real-time information and documentation
• **Test Generation** - Create comprehensive tests for your code

**Current Configuration:**
- Agent: ${this._agents.find(a => a.id === this._currentAgent)?.name}
- Reasoning Mode: ${this._reasoningModes.find(m => m.id === this._currentReasoningMode)?.name}

You can change these settings using the controls above. What would you like to work on?`,
            timestamp: new Date(),
            agent: 'system',
            confidence: 1.0
        };

        this._messages.push(welcomeMessage);
        this._updateChatView();
    }

    private _clearChat(): void {
        this._messages = [];
        this._addWelcomeMessage();
    }

    private async _exportChat(): Promise<void> {
        const chatContent = this._messages.map(msg => {
            let content = `**${msg.role.toUpperCase()}** (${msg.timestamp.toLocaleString()})`;
            if (msg.agent && msg.agent !== 'system') {
                content += ` [${msg.agent}]`;
            }
            content += `\n${msg.content}\n`;
            
            if (msg.reasoning && msg.reasoning.length > 0) {
                content += '\n**Reasoning Trace:**\n';
                msg.reasoning.forEach(step => {
                    content += `${step.step}. [${step.type}] ${step.content}\n`;
                });
            }
            
            return content;
        }).join('\n---\n\n');

        const doc = await vscode.workspace.openTextDocument({
            content: `# AI Assistant Chat Export\n\nExported on: ${new Date().toLocaleString()}\n\n${chatContent}`,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(doc);
    }

    private _stopGeneration(): void {
        this._isStreaming = false;
        this._hideTypingIndicator();
    }

    private _showTypingIndicator(): void {
        this._view?.webview.postMessage({
            type: 'showTyping'
        });
    }

    private _hideTypingIndicator(): void {
        this._view?.webview.postMessage({
            type: 'hideTyping'
        });
    }

    private _updateChatView(): void {
        this._view?.webview.postMessage({
            type: 'updateMessages',
            messages: this._messages
        });
    }

    private _updateAgentSelection(): void {
        this._view?.webview.postMessage({
            type: 'updateAgentSelection',
            selectedAgent: this._currentAgent
        });
    }

    private _updateReasoningModeSelection(): void {
        this._view?.webview.postMessage({
            type: 'updateReasoningMode',
            selectedMode: this._currentReasoningMode
        });
    }

    private _generateMessageId(): string {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private _delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI Assistant Chat</title>
            <style>
                :root {
                    --primary-color: #007ACC;
                    --success-color: #28A745;
                    --warning-color: #FFC107;
                    --danger-color: #DC3545;
                    --info-color: #17A2B8;
                    --purple-color: #6F42C1;
                }

                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    height: 100vh;
                    font-size: var(--vscode-font-size);
                }

                .controls-container {
                    padding: 12px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    background-color: var(--vscode-sideBar-background);
                }

                .control-group {
                    margin-bottom: 8px;
                }

                .control-label {
                    font-size: 11px;
                    font-weight: 600;
                    color: var(--vscode-descriptionForeground);
                    margin-bottom: 4px;
                    display: block;
                }

                .agent-selector, .reasoning-selector {
                    display: flex;
                    gap: 4px;
                    flex-wrap: wrap;
                }

                .agent-button, .reasoning-button {
                    padding: 4px 8px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 4px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 10px;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }

                .agent-button:hover, .reasoning-button:hover {
                    background-color: var(--vscode-button-secondaryHoverBackground);
                }

                .agent-button.active, .reasoning-button.active {
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border-color: var(--vscode-focusBorder);
                }

                .agent-icon, .reasoning-icon {
                    width: 12px;
                    height: 12px;
                }

                .chat-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 16px;
                    scroll-behavior: smooth;
                }

                .message {
                    margin-bottom: 20px;
                    animation: fadeIn 0.3s ease-in;
                }

                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                .message-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 8px;
                    font-size: 12px;
                    color: var(--vscode-descriptionForeground);
                }

                .message-role {
                    font-weight: 600;
                    text-transform: uppercase;
                }

                .message-agent {
                    padding: 2px 6px;
                    border-radius: 10px;
                    font-size: 10px;
                    font-weight: 500;
                }

                .message-timestamp {
                    margin-left: auto;
                    font-size: 10px;
                }

                .message-content {
                    padding: 12px;
                    border-radius: 8px;
                    line-height: 1.5;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                }

                .message.user .message-content {
                    background-color: var(--vscode-textBlockQuote-background);
                    border-left: 3px solid var(--vscode-textBlockQuote-border);
                }

                .message.assistant .message-content {
                    background-color: var(--vscode-editor-inactiveSelectionBackground);
                    border-left: 3px solid var(--primary-color);
                }

                .message.system .message-content {
                    background-color: var(--vscode-textPreformat-background);
                    border-left: 3px solid var(--info-color);
                }

                .confidence-indicator {
                    margin-top: 8px;
                    font-size: 11px;
                    color: var(--vscode-textLink-foreground);
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }

                .confidence-bar {
                    width: 50px;
                    height: 4px;
                    background-color: var(--vscode-progressBar-background);
                    border-radius: 2px;
                    overflow: hidden;
                }

                .confidence-fill {
                    height: 100%;
                    background-color: var(--vscode-progressBar-background);
                    transition: width 0.3s ease;
                }

                .reasoning-trace {
                    margin-top: 12px;
                    border-top: 1px solid var(--vscode-panel-border);
                    padding-top: 8px;
                }

                .reasoning-header {
                    font-size: 11px;
                    font-weight: 600;
                    color: var(--vscode-descriptionForeground);
                    margin-bottom: 8px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 4px;
                }

                .reasoning-steps {
                    max-height: 200px;
                    overflow-y: auto;
                    font-size: 11px;
                }

                .reasoning-step {
                    padding: 4px 8px;
                    margin-bottom: 4px;
                    border-radius: 4px;
                    border-left: 2px solid;
                    background-color: var(--vscode-textPreformat-background);
                }

                .reasoning-step.thought { border-left-color: var(--info-color); }
                .reasoning-step.action { border-left-color: var(--warning-color); }
                .reasoning-step.observation { border-left-color: var(--success-color); }
                .reasoning-step.conclusion { border-left-color: var(--primary-color); }

                .web-results {
                    margin-top: 12px;
                    border-top: 1px solid var(--vscode-panel-border);
                    padding-top: 8px;
                }

                .web-result {
                    padding: 8px;
                    margin-bottom: 8px;
                    border-radius: 4px;
                    background-color: var(--vscode-textPreformat-background);
                    border-left: 3px solid var(--info-color);
                    cursor: pointer;
                    transition: background-color 0.2s;
                }

                .web-result:hover {
                    background-color: var(--vscode-list-hoverBackground);
                }

                .web-result-title {
                    font-weight: 600;
                    font-size: 12px;
                    color: var(--vscode-textLink-foreground);
                    margin-bottom: 4px;
                }

                .web-result-snippet {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    line-height: 1.4;
                }

                .code-snippets {
                    margin-top: 12px;
                }

                .code-snippet {
                    margin-bottom: 12px;
                    border-radius: 6px;
                    overflow: hidden;
                    border: 1px solid var(--vscode-panel-border);
                }

                .code-snippet-header {
                    padding: 8px 12px;
                    background-color: var(--vscode-sideBar-background);
                    border-bottom: 1px solid var(--vscode-panel-border);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 11px;
                }

                .code-snippet-language {
                    font-weight: 600;
                    color: var(--vscode-textLink-foreground);
                }

                .code-snippet-actions {
                    display: flex;
                    gap: 8px;
                }

                .code-action-button {
                    padding: 2px 6px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 3px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 10px;
                    transition: background-color 0.2s;
                }

                .code-action-button:hover {
                    background-color: var(--vscode-button-secondaryHoverBackground);
                }

                .code-snippet-content {
                    padding: 12px;
                    background-color: var(--vscode-textCodeBlock-background);
                    font-family: var(--vscode-editor-font-family);
                    font-size: var(--vscode-editor-font-size);
                    overflow-x: auto;
                    white-space: pre;
                }

                .typing-indicator {
                    display: none;
                    padding: 12px;
                    font-style: italic;
                    color: var(--vscode-descriptionForeground);
                    text-align: center;
                    animation: pulse 1.5s infinite;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }

                .input-container {
                    padding: 16px;
                    border-top: 1px solid var(--vscode-panel-border);
                    background-color: var(--vscode-sideBar-background);
                }

                .input-wrapper {
                    display: flex;
                    gap: 8px;
                    align-items: flex-end;
                }

                .input-field {
                    flex: 1;
                    min-height: 36px;
                    max-height: 120px;
                    padding: 8px 12px;
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 6px;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    resize: vertical;
                    font-family: var(--vscode-font-family);
                    font-size: var(--vscode-font-size);
                }

                .input-field:focus {
                    outline: none;
                    border-color: var(--vscode-focusBorder);
                }

                .send-button, .stop-button {
                    padding: 8px 16px;
                    border: none;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    transition: background-color 0.2s;
                    min-width: 60px;
                }

                .send-button {
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                }

                .send-button:hover:not(:disabled) {
                    background-color: var(--vscode-button-hoverBackground);
                }

                .send-button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }

                .stop-button {
                    background-color: var(--danger-color);
                    color: white;
                }

                .stop-button:hover {
                    background-color: #c82333;
                }

                .chat-actions {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 8px;
                }

                .chat-action-button {
                    padding: 4px 8px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 4px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 11px;
                    transition: background-color 0.2s;
                }

                .chat-action-button:hover {
                    background-color: var(--vscode-button-secondaryHoverBackground);
                }

                .hidden {
                    display: none !important;
                }

                /* Scrollbar styling */
                ::-webkit-scrollbar {
                    width: 8px;
                }

                ::-webkit-scrollbar-track {
                    background: var(--vscode-scrollbarSlider-background);
                }

                ::-webkit-scrollbar-thumb {
                    background: var(--vscode-scrollbarSlider-background);
                    border-radius: 4px;
                }

                ::-webkit-scrollbar-thumb:hover {
                    background: var(--vscode-scrollbarSlider-hoverBackground);
                }
            </style>
        </head>
        <body>
            <div class="controls-container">
                <div class="control-group">
                    <label class="control-label">Agent</label>
                    <div class="agent-selector" id="agentSelector">
                        ${this._agents.map(agent => `
                            <button class="agent-button ${agent.id === this._currentAgent ? 'active' : ''}" 
                                    data-agent="${agent.id}" 
                                    title="${agent.description}">
                                <span class="agent-icon">🤖</span>
                                ${agent.name}
                            </button>
                        `).join('')}
                    </div>
                </div>
                
                <div class="control-group">
                    <label class="control-label">Reasoning Mode</label>
                    <div class="reasoning-selector" id="reasoningSelector">
                        ${this._reasoningModes.map(mode => `
                            <button class="reasoning-button ${mode.id === this._currentReasoningMode ? 'active' : ''}" 
                                    data-mode="${mode.id}" 
                                    title="${mode.description}">
                                <span class="reasoning-icon">⚡</span>
                                ${mode.name}
                            </button>
                        `).join('')}
                    </div>
                </div>
            </div>

            <div class="chat-container" id="chatContainer">
                <!-- Messages will be inserted here -->
            </div>

            <div class="typing-indicator" id="typingIndicator">
                AI is thinking...
            </div>

            <div class="input-container">
                <div class="chat-actions">
                    <button class="chat-action-button" id="clearButton">Clear Chat</button>
                    <button class="chat-action-button" id="exportButton">Export Chat</button>
                </div>
                <div class="input-wrapper">
                    <textarea class="input-field" 
                             id="messageInput" 
                             placeholder="Ask me anything about your code..."
                             rows="1"></textarea>
                    <button class="send-button" id="sendButton">Send</button>
                    <button class="stop-button hidden" id="stopButton">Stop</button>
                </div>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                
                // DOM elements
                const chatContainer = document.getElementById('chatContainer');
                const messageInput = document.getElementById('messageInput');
                const sendButton = document.getElementById('sendButton');
                const stopButton = document.getElementById('stopButton');
                const typingIndicator = document.getElementById('typingIndicator');
                const agentSelector = document.getElementById('agentSelector');
                const reasoningSelector = document.getElementById('reasoningSelector');
                const clearButton = document.getElementById('clearButton');
                const exportButton = document.getElementById('exportButton');

                let isStreaming = false;
                let messages = [];

                // Event listeners
                sendButton.addEventListener('click', sendMessage);
                stopButton.addEventListener('click', stopGeneration);
                clearButton.addEventListener('click', clearChat);
                exportButton.addEventListener('click', exportChat);

                messageInput.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendMessage();
                    }
                });

                messageInput.addEventListener('input', () => {
                    // Auto-resize textarea
                    messageInput.style.height = 'auto';
                    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
                });

                // Agent selection
                agentSelector.addEventListener('click', (e) => {
                    if (e.target.classList.contains('agent-button')) {
                        const agentId = e.target.dataset.agent;
                        selectAgent(agentId);
                    }
                });

                // Reasoning mode selection
                reasoningSelector.addEventListener('click', (e) => {
                    if (e.target.classList.contains('reasoning-button')) {
                        const modeId = e.target.dataset.mode;
                        selectReasoningMode(modeId);
                    }
                });

                // Functions
                function sendMessage() {
                    const message = messageInput.value.trim();
                    if (message && !isStreaming) {
                        vscode.postMessage({
                            type: 'sendMessage',
                            message: message
                        });
                        messageInput.value = '';
                        messageInput.style.height = 'auto';
                        setStreaming(true);
                    }
                }

                function stopGeneration() {
                    vscode.postMessage({ type: 'stopGeneration' });
                    setStreaming(false);
                }

                function clearChat() {
                    vscode.postMessage({ type: 'clearChat' });
                }

                function exportChat() {
                    vscode.postMessage({ type: 'exportChat' });
                }

                function selectAgent(agentId) {
                    vscode.postMessage({
                        type: 'selectAgent',
                        agentId: agentId
                    });
                }

                function selectReasoningMode(modeId) {
                    vscode.postMessage({
                        type: 'selectReasoningMode',
                        modeId: modeId
                    });
                }

                function setStreaming(streaming) {
                    isStreaming = streaming;
                    sendButton.disabled = streaming;
                    sendButton.classList.toggle('hidden', streaming);
                    stopButton.classList.toggle('hidden', !streaming);
                }

                function insertCode(snippet) {
                    vscode.postMessage({
                        type: 'insertCode',
                        snippet: snippet
                    });
                }

                function copyCode(snippet) {
                    vscode.postMessage({
                        type: 'copyCode',
                        snippet: snippet
                    });
                }

                function openWebResult(url) {
                    vscode.postMessage({
                        type: 'openWebResult',
                        url: url
                    });
                }

                function renderMessages(newMessages) {
                    messages = newMessages;
                    chatContainer.innerHTML = '';
                    
                    messages.forEach(message => {
                        const messageElement = createMessageElement(message);
                        chatContainer.appendChild(messageElement);
                    });
                    
                    scrollToBottom();
                }

                function createMessageElement(message) {
                    const messageDiv = document.createElement('div');
                    messageDiv.className = \`message \${message.role}\`;
                    
                    // Message header
                    const headerDiv = document.createElement('div');
                    headerDiv.className = 'message-header';
                    
                    const roleSpan = document.createElement('span');
                    roleSpan.className = 'message-role';
                    roleSpan.textContent = message.role;
                    headerDiv.appendChild(roleSpan);
                    
                    if (message.agent && message.agent !== 'system') {
                        const agentSpan = document.createElement('span');
                        agentSpan.className = 'message-agent';
                        agentSpan.textContent = message.agent;
                        agentSpan.style.backgroundColor = getAgentColor(message.agent);
                        headerDiv.appendChild(agentSpan);
                    }
                    
                    const timestampSpan = document.createElement('span');
                    timestampSpan.className = 'message-timestamp';
                    timestampSpan.textContent = new Date(message.timestamp).toLocaleTimeString();
                    headerDiv.appendChild(timestampSpan);
                    
                    messageDiv.appendChild(headerDiv);
                    
                    // Message content
                    const contentDiv = document.createElement('div');
                    contentDiv.className = 'message-content';
                    contentDiv.textContent = message.content;
                    messageDiv.appendChild(contentDiv);
                    
                    // Confidence indicator
                    if (message.confidence !== undefined && message.role === 'assistant') {
                        const confidenceDiv = document.createElement('div');
                        confidenceDiv.className = 'confidence-indicator';
                        confidenceDiv.innerHTML = \`
                            <span>Confidence: \${Math.round(message.confidence * 100)}%</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: \${message.confidence * 100}%"></div>
                            </div>
                        \`;
                        messageDiv.appendChild(confidenceDiv);
                    }
                    
                    // Reasoning trace
                    if (message.reasoning && message.reasoning.length > 0) {
                        const reasoningDiv = document.createElement('div');
                        reasoningDiv.className = 'reasoning-trace';
                        
                        const reasoningHeader = document.createElement('div');
                        reasoningHeader.className = 'reasoning-header';
                        reasoningHeader.innerHTML = '🧠 Reasoning Trace (\${message.reasoning.length} steps)';
                        reasoningHeader.onclick = () => {
                            const steps = reasoningDiv.querySelector('.reasoning-steps');
                            steps.style.display = steps.style.display === 'none' ? 'block' : 'none';
                        };
                        reasoningDiv.appendChild(reasoningHeader);
                        
                        const stepsDiv = document.createElement('div');
                        stepsDiv.className = 'reasoning-steps';
                        
                        message.reasoning.forEach(step => {
                            const stepDiv = document.createElement('div');
                            stepDiv.className = \`reasoning-step \${step.type}\`;
                            stepDiv.innerHTML = \`<strong>\${step.step}.</strong> [\${step.type}] \${step.content}\`;
                            stepsDiv.appendChild(stepDiv);
                        });
                        
                        reasoningDiv.appendChild(stepsDiv);
                        messageDiv.appendChild(reasoningDiv);
                    }
                    
                    // Web search results
                    if (message.webSearchResults && message.webSearchResults.length > 0) {
                        const webResultsDiv = document.createElement('div');
                        webResultsDiv.className = 'web-results';
                        
                        const headerDiv = document.createElement('div');
                        headerDiv.className = 'reasoning-header';
                        headerDiv.innerHTML = \`🌐 Web Search Results (\${message.webSearchResults.length})\`;
                        webResultsDiv.appendChild(headerDiv);
                        
                        message.webSearchResults.forEach(result => {
                            const resultDiv = document.createElement('div');
                            resultDiv.className = 'web-result';
                            resultDiv.onclick = () => openWebResult(result.url);
                            
                            resultDiv.innerHTML = \`
                                <div class="web-result-title">\${result.title}</div>
                                <div class="web-result-snippet">\${result.snippet}</div>
                            \`;
                            
                            webResultsDiv.appendChild(resultDiv);
                        });
                        
                        messageDiv.appendChild(webResultsDiv);
                    }
                    
                    // Code snippets
                    if (message.codeSnippets && message.codeSnippets.length > 0) {
                        const snippetsDiv = document.createElement('div');
                        snippetsDiv.className = 'code-snippets';
                        
                        message.codeSnippets.forEach(snippet => {
                            const snippetDiv = document.createElement('div');
                            snippetDiv.className = 'code-snippet';
                            
                            snippetDiv.innerHTML = \`
                                <div class="code-snippet-header">
                                    <span class="code-snippet-language">\${snippet.language}</span>
                                    <div class="code-snippet-actions">
                                        <button class="code-action-button" onclick="copyCode(\${JSON.stringify(snippet).replace(/"/g, '&quot;')})">Copy</button>
                                        <button class="code-action-button" onclick="insertCode(\${JSON.stringify(snippet).replace(/"/g, '&quot;')})">Insert</button>
                                    </div>
                                </div>
                                <div class="code-snippet-content">\${snippet.code}</div>
                            \`;
                            
                            snippetsDiv.appendChild(snippetDiv);
                        });
                        
                        messageDiv.appendChild(snippetsDiv);
                    }
                    
                    return messageDiv;
                }

                function getAgentColor(agentId) {
                    const colors = {
                        'general': '#007ACC',
                        'code-agent': '#28A745',
                        'search-agent': '#FFC107',
                        'reasoning-agent': '#DC3545',
                        'test-agent': '#6F42C1'
                    };
                    return colors[agentId] || '#007ACC';
                }

                function scrollToBottom() {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }

                // Handle messages from extension
                window.addEventListener('message', event => {
                    const message = event.data;
                    
                    switch (message.type) {
                        case 'updateMessages':
                            renderMessages(message.messages);
                            break;
                        case 'showTyping':
                            typingIndicator.style.display = 'block';
                            scrollToBottom();
                            break;
                        case 'hideTyping':
                            typingIndicator.style.display = 'none';
                            setStreaming(false);
                            break;
                        case 'updateAgentSelection':
                            updateAgentSelection(message.selectedAgent);
                            break;
                        case 'updateReasoningMode':
                            updateReasoningModeSelection(message.selectedMode);
                            break;
                    }
                });

                function updateAgentSelection(selectedAgent) {
                    document.querySelectorAll('.agent-button').forEach(button => {
                        button.classList.toggle('active', button.dataset.agent === selectedAgent);
                    });
                }

                function updateReasoningModeSelection(selectedMode) {
                    document.querySelectorAll('.reasoning-button').forEach(button => {
                        button.classList.toggle('active', button.dataset.mode === selectedMode);
                    });
                }

                // Initialize
                messageInput.focus();
            </script>
        </body>
        </html>`;
    }
}