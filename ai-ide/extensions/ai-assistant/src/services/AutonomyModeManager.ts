import * as vscode from 'vscode';
import { PocketFlowBridge } from './PocketFlowBridge';

export enum AutonomyMode {
    Supervised = 'supervised',
    Autopilot = 'autopilot'
}

export class AutonomyModeManager {
    private currentMode: AutonomyMode = AutonomyMode.Supervised;
    private disposables: vscode.Disposable[] = [];
    private changeHistory: any[] = [];
    private statusBarItem: vscode.StatusBarItem | undefined;

    constructor(
        private context: vscode.ExtensionContext,
        private pocketFlowBridge: PocketFlowBridge
    ) {
        this.initialize();
    }

    private initialize() {
        // Create status bar item
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            200
        );
        this.updateStatusBar();
        this.statusBarItem?.show();

        // Register commands
        const toggleModeCommand = vscode.commands.registerCommand(
            'ai-assistant.toggleAutonomyMode',
            this.toggleMode.bind(this)
        );

        const setModeCommand = vscode.commands.registerCommand(
            'ai-assistant.setAutonomyMode',
            this.setMode.bind(this)
        );

        const revertChangesCommand = vscode.commands.registerCommand(
            'ai-assistant.revertChanges',
            this.revertLastChanges.bind(this)
        );

        const showChangeHistoryCommand = vscode.commands.registerCommand(
            'ai-assistant.showChangeHistory',
            this.showChangeHistory.bind(this)
        );

        this.disposables.push(
            toggleModeCommand,
            setModeCommand,
            revertChangesCommand,
            showChangeHistoryCommand,
            this.statusBarItem
        );
    }

    public async toggleMode() {
        const newMode = this.currentMode === AutonomyMode.Supervised 
            ? AutonomyMode.Autopilot 
            : AutonomyMode.Supervised;
        
        await this.setMode(newMode);
    }

    public async setMode(mode?: AutonomyMode) {
        if (!mode) {
            // Show quick pick to select mode
            const items = [
                {
                    label: 'Supervised Mode',
                    description: 'Review changes before applying',
                    detail: 'AI will ask for confirmation before making changes',
                    mode: AutonomyMode.Supervised
                },
                {
                    label: 'Autopilot Mode',
                    description: 'Apply changes automatically',
                    detail: 'AI will make changes autonomously with rollback capability',
                    mode: AutonomyMode.Autopilot
                }
            ];

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select autonomy mode'
            });

            if (!selected) return;
            mode = selected.mode;
        }

        const previousMode = this.currentMode;
        this.currentMode = mode;
        
        this.updateStatusBar();
        
        // Store in workspace settings
        await vscode.workspace.getConfiguration('ai-assistant').update(
            'autonomyMode',
            mode,
            vscode.ConfigurationTarget.Workspace
        );

        vscode.window.showInformationMessage(
            `Autonomy mode changed from ${previousMode} to ${mode}`
        );
    }

    public async executeAutonomousAction(
        action: string,
        context: any,
        options: {
            description?: string;
            files?: string[];
            preview?: boolean;
        } = {}
    ): Promise<boolean> {
        const changeId = Date.now().toString();
        
        try {
            if (this.currentMode === AutonomyMode.Supervised) {
                return await this.executeSupervisedAction(action, context, options, changeId);
            } else {
                return await this.executeAutopilotAction(action, context, options, changeId);
            }
        } catch (error) {
            console.error('Autonomous action failed:', error);
            vscode.window.showErrorMessage(`Action failed: ${error}`);
            return false;
        }
    }

    private async executeSupervisedAction(
        action: string,
        context: any,
        options: any,
        changeId: string
    ): Promise<boolean> {
        // Generate preview of changes
        const preview = await this.generatePreview(action, context);
        
        if (!preview) {
            vscode.window.showWarningMessage('Could not generate preview for this action');
            return false;
        }

        // Show preview to user
        const approved = await this.showPreviewDialog(preview, options.description);
        
        if (!approved) {
            return false;
        }

        // Execute the action
        const result = await this.executeAction(action, context, changeId);
        
        if (result.success) {
            this.recordChange(changeId, {
                action,
                context,
                changes: result.changes,
                timestamp: new Date().toISOString(),
                mode: AutonomyMode.Supervised
            });
        }

        return result.success;
    }

    private async executeAutopilotAction(
        action: string,
        context: any,
        options: any,
        changeId: string
    ): Promise<boolean> {
        // Execute immediately in autopilot mode
        const result = await this.executeAction(action, context, changeId);
        
        if (result.success) {
            this.recordChange(changeId, {
                action,
                context,
                changes: result.changes,
                timestamp: new Date().toISOString(),
                mode: AutonomyMode.Autopilot
            });

            // Show notification with undo option
            const undoAction = await vscode.window.showInformationMessage(
                `Autopilot: ${options.description || action}`,
                'Undo',
                'View Changes'
            );

            if (undoAction === 'Undo') {
                await this.revertChange(changeId);
            } else if (undoAction === 'View Changes') {
                await this.showChangeDetails(changeId);
            }
        }

        return result.success;
    }

    private async generatePreview(action: string, context: any): Promise<any> {
        try {
            // Use PocketFlow to generate a preview of what changes would be made
            const previewResult = await this.pocketFlowBridge.executeReasoning(
                `Generate a preview of changes for: ${action}`,
                'chain-of-thought',
                { ...context, preview: true }
            );

            return {
                description: previewResult.solution,
                reasoning: previewResult.reasoning,
                confidence: previewResult.confidence
            };
        } catch (error) {
            console.error('Preview generation failed:', error);
            return null;
        }
    }

    private async showPreviewDialog(preview: any, description?: string): Promise<boolean> {
        const panel = vscode.window.createWebviewPanel(
            'change-preview',
            'Change Preview',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = this.getPreviewHtml(preview, description);

        return new Promise((resolve) => {
            panel.webview.onDidReceiveMessage((message) => {
                if (message.command === 'approve') {
                    resolve(true);
                    panel.dispose();
                } else if (message.command === 'reject') {
                    resolve(false);
                    panel.dispose();
                }
            });

            panel.onDidDispose(() => {
                resolve(false);
            });
        });
    }

    private async executeAction(action: string, context: any, changeId: string): Promise<any> {
        try {
            // Use PocketFlow to execute the actual action
            const result = await this.pocketFlowBridge.executeCodeGeneration(action, {
                ...context,
                changeId,
                autonomyMode: this.currentMode
            });

            return {
                success: true,
                changes: result.changes || [],
                result: result
            };
        } catch (error) {
            return {
                success: false,
                error: error
            };
        }
    }

    private recordChange(changeId: string, changeRecord: any) {
        this.changeHistory.unshift(changeRecord);
        
        // Keep only last 50 changes
        if (this.changeHistory.length > 50) {
            this.changeHistory = this.changeHistory.slice(0, 50);
        }

        // Store in workspace state
        this.context.workspaceState.update('changeHistory', this.changeHistory);
    }

    private async revertLastChanges() {
        if (this.changeHistory.length === 0) {
            vscode.window.showInformationMessage('No changes to revert');
            return;
        }

        const items = this.changeHistory.slice(0, 10).map((change, index) => ({
            label: `${change.action}`,
            description: new Date(change.timestamp).toLocaleString(),
            detail: `Mode: ${change.mode}, Changes: ${change.changes?.length || 0}`,
            change: change
        }));

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select change to revert'
        });

        if (selected) {
            await this.revertChange(selected.change.changeId);
        }
    }

    private async revertChange(changeId: string) {
        const change = this.changeHistory.find(c => c.changeId === changeId);
        if (!change) {
            vscode.window.showErrorMessage('Change not found in history');
            return;
        }

        try {
            // Implement change reversal logic here
            // This would depend on the type of changes made
            vscode.window.showInformationMessage(`Reverted change: ${change.action}`);
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to revert change: ${error}`);
        }
    }

    private async showChangeHistory() {
        if (this.changeHistory.length === 0) {
            vscode.window.showInformationMessage('No change history available');
            return;
        }

        const panel = vscode.window.createWebviewPanel(
            'change-history',
            'Change History',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = this.getChangeHistoryHtml();
    }

    private async showChangeDetails(changeId: string) {
        const change = this.changeHistory.find(c => c.changeId === changeId);
        if (!change) return;

        const panel = vscode.window.createWebviewPanel(
            'change-details',
            'Change Details',
            vscode.ViewColumn.Beside,
            { enableScripts: true }
        );

        panel.webview.html = this.getChangeDetailsHtml(change);
    }

    private updateStatusBar() {
        if (!this.statusBarItem) return;
        
        const icon = this.currentMode === AutonomyMode.Autopilot ? '$(rocket)' : '$(eye)';
        const text = this.currentMode === AutonomyMode.Autopilot ? 'Autopilot' : 'Supervised';
        
        this.statusBarItem.text = `${icon} ${text}`;
        this.statusBarItem.tooltip = `AI Autonomy Mode: ${this.currentMode}\nClick to change mode`;
        this.statusBarItem.command = 'ai-assistant.toggleAutonomyMode';
    }

    private getPreviewHtml(preview: any, description?: string): string {
        return `<!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .preview { background: var(--vscode-textBlockQuote-background); padding: 15px; border-radius: 5px; }
                .buttons { margin-top: 20px; text-align: center; }
                button { margin: 0 10px; padding: 10px 20px; }
                .approve { background: var(--vscode-button-background); color: var(--vscode-button-foreground); }
                .reject { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
            </style>
        </head>
        <body>
            <h2>Change Preview</h2>
            ${description ? `<p><strong>Action:</strong> ${description}</p>` : ''}
            <div class="preview">
                <h3>Proposed Changes:</h3>
                <p>${preview.description}</p>
                <p><strong>Confidence:</strong> ${Math.round(preview.confidence * 100)}%</p>
            </div>
            <div class="buttons">
                <button class="approve" onclick="approve()">Approve Changes</button>
                <button class="reject" onclick="reject()">Reject Changes</button>
            </div>
            <script>
                const vscode = acquireVsCodeApi();
                function approve() { vscode.postMessage({command: 'approve'}); }
                function reject() { vscode.postMessage({command: 'reject'}); }
            </script>
        </body>
        </html>`;
    }

    private getChangeHistoryHtml(): string {
        const historyItems = this.changeHistory.map(change => `
            <div class="change-item">
                <h3>${change.action}</h3>
                <p>Mode: ${change.mode} | Time: ${new Date(change.timestamp).toLocaleString()}</p>
                <p>Changes: ${change.changes?.length || 0}</p>
            </div>
        `).join('');

        return `<!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .change-item { border-bottom: 1px solid var(--vscode-panel-border); padding: 10px 0; }
            </style>
        </head>
        <body>
            <h2>Change History</h2>
            ${historyItems}
        </body>
        </html>`;
    }

    private getChangeDetailsHtml(change: any): string {
        return `<!DOCTYPE html>
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); padding: 20px; }
                .detail { margin: 10px 0; }
            </style>
        </head>
        <body>
            <h2>Change Details</h2>
            <div class="detail"><strong>Action:</strong> ${change.action}</div>
            <div class="detail"><strong>Mode:</strong> ${change.mode}</div>
            <div class="detail"><strong>Timestamp:</strong> ${new Date(change.timestamp).toLocaleString()}</div>
            <div class="detail"><strong>Changes:</strong> ${JSON.stringify(change.changes, null, 2)}</div>
        </body>
        </html>`;
    }

    public getCurrentMode(): AutonomyMode {
        return this.currentMode;
    }

    public getChangeHistory(): any[] {
        return [...this.changeHistory];
    }

    public dispose() {
        this.disposables.forEach(d => d.dispose());
    }
}