import * as vscode from 'vscode';

export class CopilotIntegration {
    private copilotExtension: vscode.Extension<any> | undefined;
    private copilotChatExtension: vscode.Extension<any> | undefined;

    constructor() {
        this.initialize();
    }

    private initialize() {
        this.copilotExtension = vscode.extensions.getExtension('GitHub.copilot');
        this.copilotChatExtension = vscode.extensions.getExtension('GitHub.copilot-chat');
    }

    public isCopilotAvailable(): boolean {
        return this.copilotExtension !== undefined && this.copilotExtension.isActive;
    }

    public isCopilotChatAvailable(): boolean {
        return this.copilotChatExtension !== undefined && this.copilotChatExtension.isActive;
    }

    public async getCopilotStatus(): Promise<{
        copilot: boolean;
        copilotChat: boolean;
        authenticated: boolean;
        enabled: boolean;
    }> {
        return {
            copilot: this.isCopilotAvailable(),
            copilotChat: this.isCopilotChatAvailable(),
            authenticated: this.copilotExtension?.isActive || false,
            enabled: vscode.workspace.getConfiguration('github.copilot').get('enable', true)
        };
    }

    public getExtensionInfo(): any {
        const info: any = {};
        if (this.copilotExtension) {
            info.copilot = {
                version: this.copilotExtension.packageJSON.version,
                active: this.copilotExtension.isActive
            };
        }
        if (this.copilotChatExtension) {
            info.copilotChat = {
                version: this.copilotChatExtension.packageJSON.version,
                active: this.copilotChatExtension.isActive
            };
        }
        return info;
    }

    public dispose() {
        // Cleanup if needed
    }
}