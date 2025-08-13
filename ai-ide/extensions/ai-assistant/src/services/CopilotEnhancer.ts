import * as vscode from 'vscode';
import { PocketFlowBridge } from './PocketFlowBridge';

export class CopilotEnhancer {
    private disposables: vscode.Disposable[] = [];
    private isEnhancedMode = false;
    private multiModelProviders: string[] = ['copilot', 'qwen-coder', 'local-llm'];

    constructor(
        private context: vscode.ExtensionContext,
        private pocketFlowBridge: PocketFlowBridge
    ) {
        this.initialize();
    }

    private initialize() {
        // Register enhanced completion provider
        const completionProvider = vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            {
                provideInlineCompletionItems: this.provideEnhancedCompletions.bind(this)
            }
        );

        // Register commands
        const toggleEnhancedMode = vscode.commands.registerCommand(
            'ai-assistant.toggleEnhancedCopilot',
            this.toggleEnhancedMode.bind(this)
        );

        const multiModelCompletion = vscode.commands.registerCommand(
            'ai-assistant.multiModelCompletion',
            this.triggerMultiModelCompletion.bind(this)
        );

        this.disposables.push(completionProvider, toggleEnhancedMode, multiModelCompletion);
    }

    private async provideEnhancedCompletions(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        if (!this.isEnhancedMode) {
            return [];
        }

        try {
            // Get current context
            const currentLine = document.lineAt(position.line).text;
            const textBeforeCursor = currentLine.substring(0, position.character);
            const textAfterCursor = currentLine.substring(position.character);
            
            // Get surrounding context (10 lines before and after)
            const startLine = Math.max(0, position.line - 10);
            const endLine = Math.min(document.lineCount - 1, position.line + 10);
            const contextRange = new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length);
            const surroundingContext = document.getText(contextRange);

            // Use PocketFlow for enhanced completion
            const completionContext = {
                filePath: document.fileName,
                language: document.languageId,
                selectedText: textBeforeCursor,
                cursorPosition: document.offsetAt(position)
            };

            const result = await this.pocketFlowBridge.executeCodeGeneration(
                `Complete the code at cursor position. Context: ${textBeforeCursor}`,
                completionContext
            );

            if (result && result.code) {
                return [{
                    insertText: result.code,
                    range: new vscode.Range(position, position),
                    command: {
                        command: 'ai-assistant.logCompletion',
                        title: 'Log Completion',
                        arguments: [result]
                    }
                }];
            }

        } catch (error) {
            console.error('Enhanced completion failed:', error);
        }

        return [];
    }

    public async toggleEnhancedMode() {
        this.isEnhancedMode = !this.isEnhancedMode;
        
        const status = this.isEnhancedMode ? 'enabled' : 'disabled';
        vscode.window.showInformationMessage(`Enhanced Copilot mode ${status}`);
        
        // Update status bar
        this.updateStatusBar();
    }

    public async triggerMultiModelCompletion() {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor found');
            return;
        }

        const position = editor.selection.active;
        const document = editor.document;
        
        // Get context
        const currentLine = document.lineAt(position.line).text;
        const textBeforeCursor = currentLine.substring(0, position.character);
        
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Getting multi-model completions...",
                cancellable: true
            }, async (progress, token) => {
                const completions: any[] = [];
                
                // Get completions from multiple models
                for (let i = 0; i < this.multiModelProviders.length; i++) {
                    if (token.isCancellationRequested) break;
                    
                    const provider = this.multiModelProviders[i];
                    progress.report({ 
                        increment: (i / this.multiModelProviders.length) * 100,
                        message: `Getting completion from ${provider}...`
                    });
                    
                    try {
                        const completion = await this.getCompletionFromProvider(
                            provider,
                            textBeforeCursor,
                            document,
                            position
                        );
                        
                        if (completion) {
                            completions.push({
                                provider: provider,
                                code: completion.code,
                                confidence: completion.confidence || 0.5
                            });
                        }
                    } catch (error) {
                        console.error(`Completion from ${provider} failed:`, error);
                    }
                }
                
                // Show completions to user
                if (completions.length > 0) {
                    await this.showMultiModelCompletions(completions, editor, position);
                } else {
                    vscode.window.showInformationMessage('No completions available');
                }
            });
            
        } catch (error) {
            vscode.window.showErrorMessage(`Multi-model completion failed: ${error}`);
        }
    }

    private async getCompletionFromProvider(
        provider: string,
        prompt: string,
        document: vscode.TextDocument,
        position: vscode.Position
    ): Promise<any> {
        const context = {
            filePath: document.fileName,
            language: document.languageId,
            prompt: prompt,
            position: position,
            provider: provider
        };

        switch (provider) {
            case 'copilot':
                // Try to get Copilot completion if available
                return await this.getCopilotCompletion(context);
            
            case 'qwen-coder':
                // Use PocketFlow with Qwen Coder
                return await this.pocketFlowBridge.executeCodeGeneration(
                    `Complete this code: ${prompt}`,
                    context
                );
            
            case 'local-llm':
                // Use local LLM through PocketFlow
                return await this.pocketFlowBridge.executeCodeGeneration(
                    `Complete this code: ${prompt}`,
                    context
                );
            
            default:
                return null;
        }
    }

    private async getCopilotCompletion(context: any): Promise<any> {
        try {
            // Check if Copilot extension is available
            const copilotExtension = vscode.extensions.getExtension('GitHub.copilot');
            if (!copilotExtension) {
                return null;
            }

            // This is a simplified approach - in reality, we'd need to use Copilot's API
            // For now, we'll simulate a Copilot-style completion
            return {
                code: `// Copilot-style completion for: ${context.prompt}`,
                confidence: 0.8
            };
        } catch (error) {
            console.error('Copilot completion failed:', error);
            return null;
        }
    }

    private async showMultiModelCompletions(
        completions: any[],
        editor: vscode.TextEditor,
        position: vscode.Position
    ) {
        // Sort by confidence
        completions.sort((a, b) => b.confidence - a.confidence);
        
        const items = completions.map((completion, index) => ({
            label: `${completion.provider} (${Math.round(completion.confidence * 100)}%)`,
            description: completion.code.substring(0, 100) + (completion.code.length > 100 ? '...' : ''),
            detail: completion.code,
            completion: completion
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a completion to insert'
        });

        if (selected) {
            await editor.edit(editBuilder => {
                editBuilder.insert(position, selected.completion.code);
            });
            
            vscode.window.showInformationMessage(
                `Inserted completion from ${selected.completion.provider}`
            );
        }
    }

    private updateStatusBar() {
        // Create or update status bar item
        const statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        
        statusBarItem.text = this.isEnhancedMode ? '$(copilot) Enhanced' : '$(copilot) Standard';
        statusBarItem.tooltip = `Copilot Enhancement: ${this.isEnhancedMode ? 'Enabled' : 'Disabled'}`;
        statusBarItem.command = 'ai-assistant.toggleEnhancedCopilot';
        statusBarItem.show();
        
        this.disposables.push(statusBarItem);
    }

    public dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}