import * as vscode from 'vscode';
import { PocketFlowBridge } from '../services/PocketFlowBridge';

export interface SearchResult {
    id: string;
    type: 'semantic' | 'web' | 'rag';
    title: string;
    content: string;
    filePath?: string;
    lineNumber?: number;
    url?: string;
    similarity: number;
    relevance: number;
    metadata?: any;
    timestamp: Date;
}

export interface PerformanceMetric {
    id: string;
    category: 'agent' | 'reasoning' | 'improvement' | 'system';
    name: string;
    value: number;
    unit: string;
    trend: 'up' | 'down' | 'stable';
    timestamp: Date;
    details?: any;
}

export interface ReasoningTraceEntry {
    id: string;
    sessionId: string;
    step: number;
    type: 'thought' | 'action' | 'observation' | 'conclusion';
    content: string;
    confidence: number;
    duration: number;
    timestamp: Date;
    metadata?: any;
}

export interface KnowledgeBaseEntry {
    id: string;
    title: string;
    content: string;
    type: 'documentation' | 'code-example' | 'best-practice' | 'troubleshooting';
    tags: string[];
    source: string;
    lastUpdated: Date;
    relevanceScore?: number;
}

export interface MCPToolConfig {
    id: string;
    name: string;
    description: string;
    serverName: string;
    enabled: boolean;
    autoApprove: boolean;
    lastUsed?: Date;
    usageCount: number;
    successRate: number;
    configuration?: any;
}

export class SearchDashboardProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'ai-assistant.searchDashboard';
    
    private _view?: vscode.WebviewView;
    private _context: vscode.ExtensionContext;
    private _pocketFlowBridge: PocketFlowBridge;
    private _searchHistory: SearchResult[] = [];
    private _performanceMetrics: PerformanceMetric[] = [];
    private _reasoningTraces: ReasoningTraceEntry[] = [];
    private _knowledgeBase: KnowledgeBaseEntry[] = [];
    private _mcpTools: MCPToolConfig[] = [];
    private _activeTab: string = 'search';

    constructor(context: vscode.ExtensionContext, pocketFlowBridge: PocketFlowBridge) {
        this._context = context;
        this._pocketFlowBridge = pocketFlowBridge;
        this._initializeData();
    }

    public dispose(): void {
        // Clean up resources if needed
        this._searchHistory = [];
        this._performanceMetrics = [];
        this._reasoningTraces = [];
        this._knowledgeBase = [];
        this._mcpTools = [];
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
                case 'switchTab':
                    this._activeTab = data.tab;
                    this._updateView();
                    break;
                case 'performSearch':
                    await this._performUnifiedSearch(data.query, data.searchTypes);
                    break;
                case 'openSearchResult':
                    await this._openSearchResult(data.result);
                    break;
                case 'exportSearchResults':
                    await this._exportSearchResults();
                    break;
                case 'clearSearchHistory':
                    this._clearSearchHistory();
                    break;
                case 'refreshMetrics':
                    await this._refreshPerformanceMetrics();
                    break;
                case 'exportMetrics':
                    await this._exportPerformanceMetrics();
                    break;
                case 'viewReasoningTrace':
                    await this._viewReasoningTrace(data.traceId);
                    break;
                case 'exportReasoningTraces':
                    await this._exportReasoningTraces();
                    break;
                case 'clearReasoningTraces':
                    this._clearReasoningTraces();
                    break;
                case 'searchKnowledgeBase':
                    await this._searchKnowledgeBase(data.query);
                    break;
                case 'addKnowledgeEntry':
                    await this._addKnowledgeEntry(data.entry);
                    break;
                case 'updateKnowledgeEntry':
                    await this._updateKnowledgeEntry(data.entry);
                    break;
                case 'deleteKnowledgeEntry':
                    await this._deleteKnowledgeEntry(data.entryId);
                    break;
                case 'refreshMCPTools':
                    await this._refreshMCPTools();
                    break;
                case 'toggleMCPTool':
                    await this._toggleMCPTool(data.toolId, data.enabled);
                    break;
                case 'configureMCPTool':
                    await this._configureMCPTool(data.toolId, data.config);
                    break;
                case 'testMCPTool':
                    await this._testMCPTool(data.toolId);
                    break;
            }
        });

        // Initialize view
        this._updateView();
    }

    private async _initializeData(): Promise<void> {
        // Initialize with sample data - in real implementation, this would load from storage
        this._performanceMetrics = [
            {
                id: 'response_time',
                category: 'system',
                name: 'Average Response Time',
                value: 1.2,
                unit: 'seconds',
                trend: 'down',
                timestamp: new Date()
            },
            {
                id: 'accuracy_score',
                category: 'agent',
                name: 'Code Generation Accuracy',
                value: 87.5,
                unit: '%',
                trend: 'up',
                timestamp: new Date()
            },
            {
                id: 'user_satisfaction',
                category: 'system',
                name: 'User Satisfaction',
                value: 4.2,
                unit: '/5',
                trend: 'stable',
                timestamp: new Date()
            }
        ];

        this._knowledgeBase = [
            {
                id: 'kb_1',
                title: 'TypeScript Best Practices',
                content: 'Collection of TypeScript coding best practices and patterns...',
                type: 'best-practice',
                tags: ['typescript', 'best-practices', 'coding'],
                source: 'internal',
                lastUpdated: new Date()
            },
            {
                id: 'kb_2',
                title: 'React Hooks Troubleshooting',
                content: 'Common issues and solutions when working with React Hooks...',
                type: 'troubleshooting',
                tags: ['react', 'hooks', 'troubleshooting'],
                source: 'documentation',
                lastUpdated: new Date()
            }
        ];

        this._mcpTools = [
            {
                id: 'github_tool',
                name: 'GitHub Integration',
                description: 'Access GitHub repositories and issues',
                serverName: 'github-mcp',
                enabled: true,
                autoApprove: false,
                usageCount: 45,
                successRate: 92.3
            },
            {
                id: 'jira_tool',
                name: 'Jira Integration',
                description: 'Manage Jira tickets and projects',
                serverName: 'jira-mcp',
                enabled: false,
                autoApprove: false,
                usageCount: 12,
                successRate: 88.7
            }
        ];
    }

    private async _performUnifiedSearch(query: string, searchTypes: string[]): Promise<void> {
        try {
            this._view?.webview.postMessage({
                type: 'searchStarted'
            });

            const results: SearchResult[] = [];

            // Semantic search
            if (searchTypes.includes('semantic')) {
                try {
                    const semanticResults = await this._pocketFlowBridge.executeSemanticSearch(query);
                    semanticResults.forEach(result => {
                        results.push({
                            id: `semantic_${Date.now()}_${Math.random()}`,
                            type: 'semantic',
                            title: `${result.file}:${result.line}`,
                            content: result.content,
                            filePath: result.file,
                            lineNumber: result.line,
                            similarity: result.similarity,
                            relevance: result.semantic_score || result.similarity,
                            timestamp: new Date()
                        });
                    });
                } catch (error) {
                    console.error('Semantic search failed:', error);
                }
            }

            // Web search
            if (searchTypes.includes('web')) {
                try {
                    const webResult = await this._pocketFlowBridge.executeTask({
                        id: `web_search_${Date.now()}`,
                        type: 'reasoning',
                        input: {
                            query: query,
                            search_type: 'web',
                            max_results: 10
                        }
                    });

                    if (webResult.success && webResult.result.web_results) {
                        webResult.result.web_results.forEach((item: any) => {
                            results.push({
                                id: `web_${Date.now()}_${Math.random()}`,
                                type: 'web',
                                title: item.title || 'No title',
                                content: item.snippet || item.content || '',
                                url: item.url,
                                similarity: 0,
                                relevance: item.relevance || 0.5,
                                timestamp: new Date(),
                                metadata: {
                                    source: item.source || 'web'
                                }
                            });
                        });
                    }
                } catch (error) {
                    console.error('Web search failed:', error);
                }
            }

            // RAG search
            if (searchTypes.includes('rag')) {
                try {
                    const ragResult = await this._pocketFlowBridge.executeTask({
                        id: `rag_search_${Date.now()}`,
                        type: 'reasoning',
                        input: {
                            query: query,
                            search_type: 'rag',
                            max_results: 10
                        }
                    });

                    if (ragResult.success && ragResult.result.rag_results) {
                        ragResult.result.rag_results.forEach((item: any) => {
                            results.push({
                                id: `rag_${Date.now()}_${Math.random()}`,
                                type: 'rag',
                                title: item.title || 'Knowledge Base Entry',
                                content: item.content || '',
                                similarity: item.similarity || 0,
                                relevance: item.relevance || 0.5,
                                timestamp: new Date(),
                                metadata: {
                                    source: item.source || 'knowledge_base',
                                    category: item.category
                                }
                            });
                        });
                    }
                } catch (error) {
                    console.error('RAG search failed:', error);
                }
            }

            // Sort results by relevance
            results.sort((a, b) => b.relevance - a.relevance);

            // Add to search history
            this._searchHistory.unshift(...results);
            if (this._searchHistory.length > 100) {
                this._searchHistory = this._searchHistory.slice(0, 100);
            }

            this._view?.webview.postMessage({
                type: 'searchCompleted',
                results: results,
                query: query
            });

        } catch (error) {
            console.error('Unified search failed:', error);
            this._view?.webview.postMessage({
                type: 'searchError',
                error: error instanceof Error ? error.message : String(error)
            });
        }
    }

    private async _openSearchResult(result: SearchResult): Promise<void> {
        try {
            if (result.type === 'semantic' && result.filePath && result.lineNumber) {
                // Open file and navigate to line
                const uri = vscode.Uri.file(result.filePath);
                const document = await vscode.workspace.openTextDocument(uri);
                const editor = await vscode.window.showTextDocument(document);
                
                const line = result.lineNumber - 1;
                const range = new vscode.Range(line, 0, line, 0);
                editor.selection = new vscode.Selection(range.start, range.end);
                editor.revealRange(range);
                
            } else if (result.type === 'web' && result.url) {
                // Open web URL
                await vscode.env.openExternal(vscode.Uri.parse(result.url));
                
            } else if (result.type === 'rag') {
                // Show RAG content in a new document
                const doc = await vscode.workspace.openTextDocument({
                    content: `# ${result.title}\n\n${result.content}`,
                    language: 'markdown'
                });
                await vscode.window.showTextDocument(doc);
            }
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to open search result: ${error}`);
        }
    }

    private async _refreshPerformanceMetrics(): Promise<void> {
        try {
            // Get metrics from backend
            const metricsResult = await this._pocketFlowBridge.executeTask({
                id: `metrics_${Date.now()}`,
                type: 'reasoning',
                input: {
                    action: 'get_performance_metrics'
                }
            });

            if (metricsResult.success && metricsResult.result.metrics) {
                this._performanceMetrics = metricsResult.result.metrics.map((metric: any) => ({
                    id: metric.id,
                    category: metric.category,
                    name: metric.name,
                    value: metric.value,
                    unit: metric.unit,
                    trend: metric.trend || 'stable',
                    timestamp: new Date(metric.timestamp || Date.now()),
                    details: metric.details
                }));
            }

            this._updateView();
        } catch (error) {
            console.error('Failed to refresh metrics:', error);
            vscode.window.showErrorMessage(`Failed to refresh metrics: ${error}`);
        }
    }

    private async _viewReasoningTrace(traceId: string): Promise<void> {
        try {
            // Get detailed reasoning trace
            const traceResult = await this._pocketFlowBridge.executeTask({
                id: `trace_${Date.now()}`,
                type: 'reasoning',
                input: {
                    action: 'get_reasoning_trace',
                    trace_id: traceId
                }
            });

            if (traceResult.success && traceResult.result.trace) {
                const trace = traceResult.result.trace;
                
                // Create detailed trace document
                const traceContent = `# Reasoning Trace: ${traceId}

**Session ID:** ${trace.sessionId}
**Started:** ${new Date(trace.startTime).toLocaleString()}
**Duration:** ${trace.totalDuration}ms
**Confidence:** ${Math.round(trace.averageConfidence * 100)}%

## Steps

${trace.steps.map((step: any, index: number) => `
### Step ${index + 1}: ${step.type.toUpperCase()}

**Content:** ${step.content}
**Confidence:** ${Math.round(step.confidence * 100)}%
**Duration:** ${step.duration}ms
**Timestamp:** ${new Date(step.timestamp).toLocaleString()}

${step.metadata ? `**Metadata:** \`\`\`json\n${JSON.stringify(step.metadata, null, 2)}\n\`\`\`` : ''}
`).join('\n')}

## Summary

**Total Steps:** ${trace.steps.length}
**Average Confidence:** ${Math.round(trace.averageConfidence * 100)}%
**Success Rate:** ${Math.round(trace.successRate * 100)}%
`;

                const doc = await vscode.workspace.openTextDocument({
                    content: traceContent,
                    language: 'markdown'
                });
                await vscode.window.showTextDocument(doc);
            }
        } catch (error) {
            console.error('Failed to view reasoning trace:', error);
            vscode.window.showErrorMessage(`Failed to view reasoning trace: ${error}`);
        }
    }

    private async _searchKnowledgeBase(query: string): Promise<void> {
        try {
            // Filter knowledge base entries
            const filteredEntries = this._knowledgeBase.filter(entry => 
                entry.title.toLowerCase().includes(query.toLowerCase()) ||
                entry.content.toLowerCase().includes(query.toLowerCase()) ||
                entry.tags.some(tag => tag.toLowerCase().includes(query.toLowerCase()))
            );

            // Calculate relevance scores
            filteredEntries.forEach(entry => {
                let score = 0;
                const queryLower = query.toLowerCase();
                
                if (entry.title.toLowerCase().includes(queryLower)) score += 0.5;
                if (entry.content.toLowerCase().includes(queryLower)) score += 0.3;
                entry.tags.forEach(tag => {
                    if (tag.toLowerCase().includes(queryLower)) score += 0.2;
                });
                
                entry.relevanceScore = Math.min(score, 1.0);
            });

            // Sort by relevance
            filteredEntries.sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));

            this._view?.webview.postMessage({
                type: 'knowledgeBaseSearchResults',
                results: filteredEntries,
                query: query
            });
        } catch (error) {
            console.error('Knowledge base search failed:', error);
        }
    }

    private async _refreshMCPTools(): Promise<void> {
        try {
            // Get MCP tools from backend
            const toolsResult = await this._pocketFlowBridge.executeTask({
                id: `mcp_tools_${Date.now()}`,
                type: 'reasoning',
                input: {
                    action: 'get_mcp_tools'
                }
            });

            if (toolsResult.success && toolsResult.result.tools) {
                this._mcpTools = toolsResult.result.tools.map((tool: any) => ({
                    id: tool.id,
                    name: tool.name,
                    description: tool.description,
                    serverName: tool.serverName,
                    enabled: tool.enabled,
                    autoApprove: tool.autoApprove,
                    lastUsed: tool.lastUsed ? new Date(tool.lastUsed) : undefined,
                    usageCount: tool.usageCount || 0,
                    successRate: tool.successRate || 0,
                    configuration: tool.configuration
                }));
            }

            this._updateView();
        } catch (error) {
            console.error('Failed to refresh MCP tools:', error);
        }
    }

    private async _toggleMCPTool(toolId: string, enabled: boolean): Promise<void> {
        try {
            const result = await this._pocketFlowBridge.executeTask({
                id: `toggle_mcp_${Date.now()}`,
                type: 'reasoning',
                input: {
                    action: 'toggle_mcp_tool',
                    tool_id: toolId,
                    enabled: enabled
                }
            });

            if (result.success) {
                const tool = this._mcpTools.find(t => t.id === toolId);
                if (tool) {
                    tool.enabled = enabled;
                    this._updateView();
                }
                vscode.window.showInformationMessage(
                    `MCP tool ${enabled ? 'enabled' : 'disabled'} successfully`
                );
            } else {
                throw new Error(result.error || 'Failed to toggle MCP tool');
            }
        } catch (error) {
            console.error('Failed to toggle MCP tool:', error);
            vscode.window.showErrorMessage(`Failed to toggle MCP tool: ${error}`);
        }
    }

    private async _testMCPTool(toolId: string): Promise<void> {
        try {
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Testing MCP tool...",
                cancellable: false
            }, async (progress) => {
                const result = await this._pocketFlowBridge.executeTask({
                    id: `test_mcp_${Date.now()}`,
                    type: 'reasoning',
                    input: {
                        action: 'test_mcp_tool',
                        tool_id: toolId
                    }
                });

                if (result.success) {
                    vscode.window.showInformationMessage(
                        `MCP tool test successful: ${result.result.message || 'Tool is working correctly'}`
                    );
                } else {
                    vscode.window.showErrorMessage(
                        `MCP tool test failed: ${result.error || 'Unknown error'}`
                    );
                }
            });
        } catch (error) {
            console.error('Failed to test MCP tool:', error);
            vscode.window.showErrorMessage(`Failed to test MCP tool: ${error}`);
        }
    }

    // Export and utility methods
    private async _exportSearchResults(): Promise<void> {
        const content = this._searchHistory.map(result => {
            return `## ${result.title} (${result.type})
**Relevance:** ${Math.round(result.relevance * 100)}%
**Timestamp:** ${result.timestamp.toLocaleString()}
${result.filePath ? `**File:** ${result.filePath}:${result.lineNumber}` : ''}
${result.url ? `**URL:** ${result.url}` : ''}

${result.content}

---
`;
        }).join('\n');

        const doc = await vscode.workspace.openTextDocument({
            content: `# Search Results Export\n\nExported on: ${new Date().toLocaleString()}\n\n${content}`,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(doc);
    }

    private _clearSearchHistory(): void {
        this._searchHistory = [];
        this._updateView();
        vscode.window.showInformationMessage('Search history cleared');
    }

    private async _exportPerformanceMetrics(): Promise<void> {
        const content = this._performanceMetrics.map(metric => {
            return `## ${metric.name}
**Category:** ${metric.category}
**Value:** ${metric.value} ${metric.unit}
**Trend:** ${metric.trend}
**Timestamp:** ${metric.timestamp.toLocaleString()}
${metric.details ? `**Details:** ${JSON.stringify(metric.details, null, 2)}` : ''}

---
`;
        }).join('\n');

        const doc = await vscode.workspace.openTextDocument({
            content: `# Performance Metrics Export\n\nExported on: ${new Date().toLocaleString()}\n\n${content}`,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(doc);
    }

    private async _exportReasoningTraces(): Promise<void> {
        const content = this._reasoningTraces.map(trace => {
            return `## Trace ${trace.id}
**Session:** ${trace.sessionId}
**Step:** ${trace.step}
**Type:** ${trace.type}
**Confidence:** ${Math.round(trace.confidence * 100)}%
**Duration:** ${trace.duration}ms
**Timestamp:** ${trace.timestamp.toLocaleString()}

${trace.content}

---
`;
        }).join('\n');

        const doc = await vscode.workspace.openTextDocument({
            content: `# Reasoning Traces Export\n\nExported on: ${new Date().toLocaleString()}\n\n${content}`,
            language: 'markdown'
        });

        await vscode.window.showTextDocument(doc);
    }

    private _clearReasoningTraces(): void {
        this._reasoningTraces = [];
        this._updateView();
        vscode.window.showInformationMessage('Reasoning traces cleared');
    }

    private async _addKnowledgeEntry(entry: Partial<KnowledgeBaseEntry>): Promise<void> {
        const newEntry: KnowledgeBaseEntry = {
            id: `kb_${Date.now()}`,
            title: entry.title || 'New Entry',
            content: entry.content || '',
            type: entry.type || 'documentation',
            tags: entry.tags || [],
            source: 'user',
            lastUpdated: new Date()
        };

        this._knowledgeBase.unshift(newEntry);
        this._updateView();
        vscode.window.showInformationMessage('Knowledge base entry added');
    }

    private async _updateKnowledgeEntry(entry: KnowledgeBaseEntry): Promise<void> {
        const index = this._knowledgeBase.findIndex(e => e.id === entry.id);
        if (index !== -1) {
            this._knowledgeBase[index] = { ...entry, lastUpdated: new Date() };
            this._updateView();
            vscode.window.showInformationMessage('Knowledge base entry updated');
        }
    }

    private async _deleteKnowledgeEntry(entryId: string): Promise<void> {
        const index = this._knowledgeBase.findIndex(e => e.id === entryId);
        if (index !== -1) {
            this._knowledgeBase.splice(index, 1);
            this._updateView();
            vscode.window.showInformationMessage('Knowledge base entry deleted');
        }
    }

    private async _configureMCPTool(toolId: string, config: any): Promise<void> {
        try {
            const result = await this._pocketFlowBridge.executeTask({
                id: `config_mcp_${Date.now()}`,
                type: 'reasoning',
                input: {
                    action: 'configure_mcp_tool',
                    tool_id: toolId,
                    configuration: config
                }
            });

            if (result.success) {
                const tool = this._mcpTools.find(t => t.id === toolId);
                if (tool) {
                    tool.configuration = config;
                    this._updateView();
                }
                vscode.window.showInformationMessage('MCP tool configured successfully');
            } else {
                throw new Error(result.error || 'Failed to configure MCP tool');
            }
        } catch (error) {
            console.error('Failed to configure MCP tool:', error);
            vscode.window.showErrorMessage(`Failed to configure MCP tool: ${error}`);
        }
    }

    private _updateView(): void {
        if (!this._view) return;

        this._view.webview.postMessage({
            type: 'updateData',
            data: {
                activeTab: this._activeTab,
                searchHistory: this._searchHistory.slice(0, 50), // Limit for performance
                performanceMetrics: this._performanceMetrics,
                reasoningTraces: this._reasoningTraces.slice(0, 100),
                knowledgeBase: this._knowledgeBase,
                mcpTools: this._mcpTools
            }
        });
    }

    private _getHtmlForWebview(webview: vscode.Webview): string {
        return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Search & Dashboard</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    color: var(--vscode-foreground);
                    background-color: var(--vscode-editor-background);
                    margin: 0;
                    padding: 0;
                    font-size: var(--vscode-font-size);
                }

                .tab-container {
                    display: flex;
                    border-bottom: 1px solid var(--vscode-panel-border);
                    background-color: var(--vscode-sideBar-background);
                }

                .tab-button {
                    padding: 8px 12px;
                    border: none;
                    background: none;
                    color: var(--vscode-foreground);
                    cursor: pointer;
                    border-bottom: 2px solid transparent;
                    font-size: 11px;
                    transition: all 0.2s;
                }

                .tab-button:hover {
                    background-color: var(--vscode-list-hoverBackground);
                }

                .tab-button.active {
                    border-bottom-color: var(--vscode-focusBorder);
                    background-color: var(--vscode-tab-activeBackground);
                }

                .tab-content {
                    padding: 16px;
                    height: calc(100vh - 40px);
                    overflow-y: auto;
                }

                .search-container {
                    margin-bottom: 16px;
                }

                .search-input-group {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 8px;
                }

                .search-input {
                    flex: 1;
                    padding: 8px 12px;
                    border: 1px solid var(--vscode-input-border);
                    border-radius: 4px;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                }

                .search-button {
                    padding: 8px 16px;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }

                .search-types {
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                }

                .search-type-checkbox {
                    display: flex;
                    align-items: center;
                    gap: 4px;
                    font-size: 11px;
                }

                .search-results {
                    margin-top: 16px;
                }

                .search-result {
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    border-left: 3px solid;
                    background-color: var(--vscode-textPreformat-background);
                    cursor: pointer;
                    transition: background-color 0.2s;
                }

                .search-result:hover {
                    background-color: var(--vscode-list-hoverBackground);
                }

                .search-result.semantic { border-left-color: #007ACC; }
                .search-result.web { border-left-color: #28A745; }
                .search-result.rag { border-left-color: #FFC107; }

                .result-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 4px;
                }

                .result-title {
                    font-weight: 600;
                    font-size: 12px;
                }

                .result-type {
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 10px;
                    background-color: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                }

                .result-content {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    line-height: 1.4;
                    margin-bottom: 4px;
                }

                .result-metadata {
                    font-size: 10px;
                    color: var(--vscode-descriptionForeground);
                    display: flex;
                    justify-content: space-between;
                }

                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 12px;
                    margin-bottom: 16px;
                }

                .metric-card {
                    padding: 12px;
                    border-radius: 6px;
                    background-color: var(--vscode-textPreformat-background);
                    border-left: 3px solid var(--vscode-focusBorder);
                }

                .metric-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }

                .metric-name {
                    font-weight: 600;
                    font-size: 12px;
                }

                .metric-trend {
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 10px;
                }

                .metric-trend.up { background-color: #28A745; color: white; }
                .metric-trend.down { background-color: #DC3545; color: white; }
                .metric-trend.stable { background-color: #6C757D; color: white; }

                .metric-value {
                    font-size: 18px;
                    font-weight: 700;
                    color: var(--vscode-textLink-foreground);
                }

                .metric-unit {
                    font-size: 12px;
                    color: var(--vscode-descriptionForeground);
                }

                .knowledge-entry {
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    background-color: var(--vscode-textPreformat-background);
                    border-left: 3px solid var(--vscode-textLink-foreground);
                }

                .entry-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }

                .entry-title {
                    font-weight: 600;
                    font-size: 12px;
                }

                .entry-type {
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 10px;
                    background-color: var(--vscode-badge-background);
                    color: var(--vscode-badge-foreground);
                }

                .entry-content {
                    font-size: 11px;
                    line-height: 1.4;
                    margin-bottom: 8px;
                }

                .entry-tags {
                    display: flex;
                    gap: 4px;
                    flex-wrap: wrap;
                }

                .entry-tag {
                    font-size: 9px;
                    padding: 2px 4px;
                    border-radius: 8px;
                    background-color: var(--vscode-textBlockQuote-background);
                    color: var(--vscode-textBlockQuote-foreground);
                }

                .mcp-tool {
                    padding: 12px;
                    margin-bottom: 8px;
                    border-radius: 6px;
                    background-color: var(--vscode-textPreformat-background);
                    border-left: 3px solid;
                }

                .mcp-tool.enabled { border-left-color: #28A745; }
                .mcp-tool.disabled { border-left-color: #DC3545; }

                .tool-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 8px;
                }

                .tool-name {
                    font-weight: 600;
                    font-size: 12px;
                }

                .tool-status {
                    font-size: 10px;
                    padding: 2px 6px;
                    border-radius: 10px;
                }

                .tool-status.enabled {
                    background-color: #28A745;
                    color: white;
                }

                .tool-status.disabled {
                    background-color: #DC3545;
                    color: white;
                }

                .tool-description {
                    font-size: 11px;
                    color: var(--vscode-descriptionForeground);
                    margin-bottom: 8px;
                }

                .tool-stats {
                    display: flex;
                    gap: 16px;
                    font-size: 10px;
                    color: var(--vscode-descriptionForeground);
                }

                .tool-actions {
                    margin-top: 8px;
                    display: flex;
                    gap: 8px;
                }

                .tool-action-button {
                    padding: 4px 8px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 3px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 10px;
                }

                .action-button {
                    padding: 6px 12px;
                    border: 1px solid var(--vscode-button-border);
                    border-radius: 4px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    cursor: pointer;
                    font-size: 11px;
                    margin-right: 8px;
                    margin-bottom: 8px;
                }

                .action-button:hover {
                    background-color: var(--vscode-button-secondaryHoverBackground);
                }

                .hidden {
                    display: none !important;
                }

                .loading {
                    text-align: center;
                    padding: 20px;
                    color: var(--vscode-descriptionForeground);
                }

                .empty-state {
                    text-align: center;
                    padding: 40px 20px;
                    color: var(--vscode-descriptionForeground);
                }

                .section-header {
                    font-weight: 600;
                    font-size: 14px;
                    margin-bottom: 12px;
                    padding-bottom: 8px;
                    border-bottom: 1px solid var(--vscode-panel-border);
                }
            </style>
        </head>
        <body>
            <div class="tab-container">
                <button class="tab-button active" data-tab="search">🔍 Search</button>
                <button class="tab-button" data-tab="dashboard">📊 Dashboard</button>
                <button class="tab-button" data-tab="reasoning">🧠 Reasoning</button>
                <button class="tab-button" data-tab="knowledge">📚 Knowledge</button>
                <button class="tab-button" data-tab="mcp">🔧 MCP Tools</button>
            </div>

            <!-- Search Tab -->
            <div class="tab-content" id="searchTab">
                <div class="section-header">Unified Search</div>
                <div class="search-container">
                    <div class="search-input-group">
                        <input type="text" class="search-input" id="searchInput" placeholder="Search across semantic, web, and knowledge base...">
                        <button class="search-button" id="searchButton">Search</button>
                    </div>
                    <div class="search-types">
                        <label class="search-type-checkbox">
                            <input type="checkbox" id="semanticSearch" checked> Semantic
                        </label>
                        <label class="search-type-checkbox">
                            <input type="checkbox" id="webSearch" checked> Web
                        </label>
                        <label class="search-type-checkbox">
                            <input type="checkbox" id="ragSearch" checked> RAG
                        </label>
                    </div>
                </div>
                
                <div class="action-button" onclick="exportSearchResults()">Export Results</div>
                <div class="action-button" onclick="clearSearchHistory()">Clear History</div>
                
                <div class="search-results" id="searchResults">
                    <div class="empty-state">Enter a search query to get started</div>
                </div>
            </div>

            <!-- Dashboard Tab -->
            <div class="tab-content hidden" id="dashboardTab">
                <div class="section-header">Performance Dashboard</div>
                
                <div class="action-button" onclick="refreshMetrics()">Refresh Metrics</div>
                <div class="action-button" onclick="exportMetrics()">Export Metrics</div>
                
                <div class="metrics-grid" id="metricsGrid">
                    <div class="empty-state">Loading metrics...</div>
                </div>
            </div>

            <!-- Reasoning Tab -->
            <div class="tab-content hidden" id="reasoningTab">
                <div class="section-header">Reasoning Trace Explorer</div>
                
                <div class="action-button" onclick="exportReasoningTraces()">Export Traces</div>
                <div class="action-button" onclick="clearReasoningTraces()">Clear Traces</div>
                
                <div id="reasoningTraces">
                    <div class="empty-state">No reasoning traces available</div>
                </div>
            </div>

            <!-- Knowledge Tab -->
            <div class="tab-content hidden" id="knowledgeTab">
                <div class="section-header">Knowledge Base</div>
                
                <div class="search-container">
                    <div class="search-input-group">
                        <input type="text" class="search-input" id="knowledgeSearchInput" placeholder="Search knowledge base...">
                        <button class="search-button" onclick="searchKnowledgeBase()">Search</button>
                    </div>
                </div>
                
                <div class="action-button" onclick="addKnowledgeEntry()">Add Entry</div>
                
                <div id="knowledgeEntries">
                    <div class="empty-state">No knowledge base entries</div>
                </div>
            </div>

            <!-- MCP Tools Tab -->
            <div class="tab-content hidden" id="mcpTab">
                <div class="section-header">MCP Tool Management</div>
                
                <div class="action-button" onclick="refreshMCPTools()">Refresh Tools</div>
                
                <div id="mcpTools">
                    <div class="empty-state">Loading MCP tools...</div>
                </div>
            </div>

            <script>
                const vscode = acquireVsCodeApi();
                
                let currentData = {
                    activeTab: 'search',
                    searchHistory: [],
                    performanceMetrics: [],
                    reasoningTraces: [],
                    knowledgeBase: [],
                    mcpTools: []
                };

                // Tab switching
                document.querySelectorAll('.tab-button').forEach(button => {
                    button.addEventListener('click', () => {
                        const tab = button.dataset.tab;
                        switchTab(tab);
                    });
                });

                function switchTab(tab) {
                    // Update tab buttons
                    document.querySelectorAll('.tab-button').forEach(btn => {
                        btn.classList.toggle('active', btn.dataset.tab === tab);
                    });
                    
                    // Update tab content
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.toggle('hidden', content.id !== tab + 'Tab');
                    });
                    
                    vscode.postMessage({ type: 'switchTab', tab: tab });
                }

                // Search functionality
                document.getElementById('searchButton').addEventListener('click', performSearch);
                document.getElementById('searchInput').addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') performSearch();
                });

                function performSearch() {
                    const query = document.getElementById('searchInput').value.trim();
                    if (!query) return;
                    
                    const searchTypes = [];
                    if (document.getElementById('semanticSearch').checked) searchTypes.push('semantic');
                    if (document.getElementById('webSearch').checked) searchTypes.push('web');
                    if (document.getElementById('ragSearch').checked) searchTypes.push('rag');
                    
                    if (searchTypes.length === 0) {
                        alert('Please select at least one search type');
                        return;
                    }
                    
                    document.getElementById('searchResults').innerHTML = '<div class="loading">Searching...</div>';
                    
                    vscode.postMessage({
                        type: 'performSearch',
                        query: query,
                        searchTypes: searchTypes
                    });
                }

                function openSearchResult(result) {
                    vscode.postMessage({
                        type: 'openSearchResult',
                        result: result
                    });
                }

                function exportSearchResults() {
                    vscode.postMessage({ type: 'exportSearchResults' });
                }

                function clearSearchHistory() {
                    vscode.postMessage({ type: 'clearSearchHistory' });
                }

                function refreshMetrics() {
                    vscode.postMessage({ type: 'refreshMetrics' });
                }

                function exportMetrics() {
                    vscode.postMessage({ type: 'exportMetrics' });
                }

                function viewReasoningTrace(traceId) {
                    vscode.postMessage({
                        type: 'viewReasoningTrace',
                        traceId: traceId
                    });
                }

                function exportReasoningTraces() {
                    vscode.postMessage({ type: 'exportReasoningTraces' });
                }

                function clearReasoningTraces() {
                    vscode.postMessage({ type: 'clearReasoningTraces' });
                }

                function searchKnowledgeBase() {
                    const query = document.getElementById('knowledgeSearchInput').value.trim();
                    if (!query) return;
                    
                    vscode.postMessage({
                        type: 'searchKnowledgeBase',
                        query: query
                    });
                }

                function addKnowledgeEntry() {
                    const title = prompt('Entry title:');
                    if (!title) return;
                    
                    const content = prompt('Entry content:');
                    if (!content) return;
                    
                    const tags = prompt('Tags (comma-separated):');
                    const tagArray = tags ? tags.split(',').map(t => t.trim()) : [];
                    
                    vscode.postMessage({
                        type: 'addKnowledgeEntry',
                        entry: {
                            title: title,
                            content: content,
                            tags: tagArray,
                            type: 'documentation'
                        }
                    });
                }

                function refreshMCPTools() {
                    vscode.postMessage({ type: 'refreshMCPTools' });
                }

                function toggleMCPTool(toolId, enabled) {
                    vscode.postMessage({
                        type: 'toggleMCPTool',
                        toolId: toolId,
                        enabled: enabled
                    });
                }

                function testMCPTool(toolId) {
                    vscode.postMessage({
                        type: 'testMCPTool',
                        toolId: toolId
                    });
                }

                // Render functions
                function renderSearchResults(results) {
                    const container = document.getElementById('searchResults');
                    
                    if (results.length === 0) {
                        container.innerHTML = '<div class="empty-state">No results found</div>';
                        return;
                    }
                    
                    container.innerHTML = results.map(result => \`
                        <div class="search-result \${result.type}" onclick="openSearchResult(\${JSON.stringify(result).replace(/"/g, '&quot;')})">
                            <div class="result-header">
                                <div class="result-title">\${result.title}</div>
                                <div class="result-type">\${result.type}</div>
                            </div>
                            <div class="result-content">\${result.content.substring(0, 200)}...</div>
                            <div class="result-metadata">
                                <span>Relevance: \${Math.round(result.relevance * 100)}%</span>
                                <span>\${result.timestamp ? new Date(result.timestamp).toLocaleTimeString() : ''}</span>
                            </div>
                        </div>
                    \`).join('');
                }

                function renderMetrics(metrics) {
                    const container = document.getElementById('metricsGrid');
                    
                    if (metrics.length === 0) {
                        container.innerHTML = '<div class="empty-state">No metrics available</div>';
                        return;
                    }
                    
                    container.innerHTML = metrics.map(metric => \`
                        <div class="metric-card">
                            <div class="metric-header">
                                <div class="metric-name">\${metric.name}</div>
                                <div class="metric-trend \${metric.trend}">\${metric.trend}</div>
                            </div>
                            <div class="metric-value">
                                \${metric.value} <span class="metric-unit">\${metric.unit}</span>
                            </div>
                        </div>
                    \`).join('');
                }

                function renderKnowledgeBase(entries) {
                    const container = document.getElementById('knowledgeEntries');
                    
                    if (entries.length === 0) {
                        container.innerHTML = '<div class="empty-state">No knowledge base entries</div>';
                        return;
                    }
                    
                    container.innerHTML = entries.map(entry => \`
                        <div class="knowledge-entry">
                            <div class="entry-header">
                                <div class="entry-title">\${entry.title}</div>
                                <div class="entry-type">\${entry.type}</div>
                            </div>
                            <div class="entry-content">\${entry.content.substring(0, 200)}...</div>
                            <div class="entry-tags">
                                \${entry.tags.map(tag => \`<span class="entry-tag">\${tag}</span>\`).join('')}
                            </div>
                        </div>
                    \`).join('');
                }

                function renderMCPTools(tools) {
                    const container = document.getElementById('mcpTools');
                    
                    if (tools.length === 0) {
                        container.innerHTML = '<div class="empty-state">No MCP tools configured</div>';
                        return;
                    }
                    
                    container.innerHTML = tools.map(tool => \`
                        <div class="mcp-tool \${tool.enabled ? 'enabled' : 'disabled'}">
                            <div class="tool-header">
                                <div class="tool-name">\${tool.name}</div>
                                <div class="tool-status \${tool.enabled ? 'enabled' : 'disabled'}">
                                    \${tool.enabled ? 'Enabled' : 'Disabled'}
                                </div>
                            </div>
                            <div class="tool-description">\${tool.description}</div>
                            <div class="tool-stats">
                                <span>Usage: \${tool.usageCount}</span>
                                <span>Success Rate: \${Math.round(tool.successRate)}%</span>
                                \${tool.lastUsed ? \`<span>Last Used: \${new Date(tool.lastUsed).toLocaleDateString()}</span>\` : ''}
                            </div>
                            <div class="tool-actions">
                                <button class="tool-action-button" onclick="toggleMCPTool('\${tool.id}', \${!tool.enabled})">
                                    \${tool.enabled ? 'Disable' : 'Enable'}
                                </button>
                                <button class="tool-action-button" onclick="testMCPTool('\${tool.id}')">Test</button>
                            </div>
                        </div>
                    \`).join('');
                }

                // Handle messages from extension
                window.addEventListener('message', event => {
                    const message = event.data;
                    
                    switch (message.type) {
                        case 'updateData':
                            currentData = message.data;
                            renderSearchResults(currentData.searchHistory);
                            renderMetrics(currentData.performanceMetrics);
                            renderKnowledgeBase(currentData.knowledgeBase);
                            renderMCPTools(currentData.mcpTools);
                            break;
                        case 'searchCompleted':
                            renderSearchResults(message.results);
                            break;
                        case 'searchError':
                            document.getElementById('searchResults').innerHTML = 
                                \`<div class="empty-state">Search failed: \${message.error}</div>\`;
                            break;
                        case 'knowledgeBaseSearchResults':
                            renderKnowledgeBase(message.results);
                            break;
                    }
                });
            </script>
        </body>
        </html>`;
    }
}