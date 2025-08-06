import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as vscode from 'vscode';

export interface PocketFlowTask {
    id: string;
    type: 'code_generation' | 'code_analysis' | 'semantic_search' | 'reasoning' | 'qwen_coder';
    input: any;
    context?: any;
}

export interface QwenCoderRequest {
    prompt: string;
    task_type: 'completion' | 'generation' | 'refactoring' | 'debugging' | 'documentation' | 'explanation' | 'optimization';
    language: string;
    max_tokens?: number;
    temperature?: number;
    stream?: boolean;
    include_explanation?: boolean;
    context?: {
        filePath?: string;
        selectedText?: string;
        cursorPosition?: number;
        surroundingCode?: string;
        projectContext?: any;
        imports?: string[];
        functions?: string[];
        classes?: string[];
    };
}

export interface QwenCoderResponse {
    code: string;
    language: string;
    confidence: number;
    explanation?: string;
    suggestions?: string[];
    metadata?: any;
    execution_time: number;
    model_info?: any;
}

export interface PocketFlowResult {
    taskId: string;
    success: boolean;
    result: any;
    error?: string;
    executionTime: number;
    reasoning_trace?: any[];
    execution_metrics?: any;
}

export class PocketFlowBridge {
    private pythonProcess: ChildProcess | null = null;
    private extensionPath: string;
    private isInitialized: boolean = false;
    private taskQueue: Map<string, (result: PocketFlowResult) => void> = new Map();
    private outputChannel: vscode.OutputChannel;

    constructor(extensionPath: string) {
        this.extensionPath = extensionPath;
        this.outputChannel = vscode.window.createOutputChannel('AI IDE Backend');
    }

    public async initialize(): Promise<void> {
        try {
            this.outputChannel.appendLine('Initializing PocketFlow bridge...');
            
            // Find the Python executable and backend script
            const pythonPath = await this.findPythonExecutable();
            const backendScript = await this.findBackendScript();

            if (!pythonPath || !backendScript) {
                throw new Error('Python executable or backend script not found');
            }

            this.outputChannel.appendLine(`Using Python: ${pythonPath}`);
            this.outputChannel.appendLine(`Using backend: ${backendScript}`);

            // Start the Python process
            this.pythonProcess = spawn(pythonPath, [backendScript], {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: path.dirname(backendScript),
                env: { ...process.env, PYTHONPATH: path.dirname(backendScript) }
            });

            // Set up communication handlers
            this.setupProcessHandlers();

            // Wait a moment for the process to start
            await new Promise(resolve => setTimeout(resolve, 2000));

            this.isInitialized = true;
            this.outputChannel.appendLine('PocketFlow bridge initialized successfully');
            
        } catch (error) {
            this.outputChannel.appendLine(`Failed to initialize PocketFlow bridge: ${error}`);
            throw error;
        }
    }

    public async executeTask(task: PocketFlowTask): Promise<PocketFlowResult> {
        if (!this.isInitialized || !this.pythonProcess) {
            throw new Error('PocketFlow bridge is not initialized');
        }

        return new Promise((resolve, reject) => {
            const taskId = task.id || this.generateTaskId();
            
            // Store the resolver for this task
            this.taskQueue.set(taskId, resolve);

            // Send task to Python process
            const taskMessage = JSON.stringify({
                id: taskId,
                type: task.type,
                input: task.input,
                context: task.context
            }) + '\n';

            this.outputChannel.appendLine(`Sending task: ${taskId} (${task.type})`);
            this.pythonProcess!.stdin!.write(taskMessage);

            // Set timeout for task execution
            setTimeout(() => {
                if (this.taskQueue.has(taskId)) {
                    this.taskQueue.delete(taskId);
                    reject(new Error(`Task ${taskId} timed out`));
                }
            }, 60000); // 60 second timeout for complex tasks
        });
    }

    public async executeCodeGeneration(
        prompt: string,
        context: {
            filePath?: string;
            selectedText?: string;
            cursorPosition?: number;
            language?: string;
        }
    ): Promise<{
        code: string;
        language: string;
        confidence: number;
        reasoning_trace?: any[];
    }> {
        const task: PocketFlowTask = {
            id: this.generateTaskId(),
            type: 'code_generation',
            input: { prompt },
            context
        };

        const result = await this.executeTask(task);
        
        if (!result.success) {
            throw new Error(result.error || 'Code generation failed');
        }

        return {
            code: result.result.code || '',
            language: result.result.language || 'python',
            confidence: result.result.confidence || 0.8,
            reasoning_trace: result.reasoning_trace
        };
    }

    public async executeSemanticSearch(
        query: string,
        options: {
            fileTypes?: string[];
            maxResults?: number;
        } = {}
    ): Promise<Array<{
        file: string;
        line: number;
        content: string;
        similarity: number;
        semantic_score?: number;
    }>> {
        const task: PocketFlowTask = {
            id: this.generateTaskId(),
            type: 'semantic_search',
            input: { query, options }
        };

        const result = await this.executeTask(task);
        
        if (!result.success) {
            throw new Error(result.error || 'Semantic search failed');
        }

        return result.result.matches || [];
    }

    public async executeReasoning(
        problem: string,
        mode: 'basic' | 'chain-of-thought' | 'deep' | 'interleaved',
        context?: any
    ): Promise<{
        solution: string;
        reasoning: string[];
        confidence: number;
        trace_id?: string;
    }> {
        const task: PocketFlowTask = {
            id: this.generateTaskId(),
            type: 'reasoning',
            input: { problem, mode },
            context
        };

        const result = await this.executeTask(task);
        
        if (!result.success) {
            throw new Error(result.error || 'Reasoning failed');
        }

        return {
            solution: result.result.solution || '',
            reasoning: result.result.reasoning || [],
            confidence: result.result.confidence || 0.8,
            trace_id: result.result.trace_id
        };
    }

    // Qwen Coder 3 specific methods
    public async executeQwenCoderTask(request: QwenCoderRequest): Promise<QwenCoderResponse> {
        const task: PocketFlowTask = {
            id: this.generateTaskId(),
            type: 'code_generation', // Use existing code_generation type but with enhanced input
            input: {
                prompt: request.prompt,
                task_type: request.task_type,
                max_tokens: request.max_tokens || 2048,
                temperature: request.temperature || 0.3,
                stream: request.stream || false,
                include_explanation: request.include_explanation || false
            },
            context: {
                language: request.language,
                filePath: request.context?.filePath,
                selectedText: request.context?.selectedText,
                cursorPosition: request.context?.cursorPosition,
                surroundingCode: request.context?.surroundingCode,
                projectContext: request.context?.projectContext,
                imports: request.context?.imports,
                functions: request.context?.functions,
                classes: request.context?.classes
            }
        };

        const result = await this.executeTask(task);
        
        if (!result.success) {
            throw new Error(result.error || 'Qwen Coder task failed');
        }

        return {
            code: result.result.code || '',
            language: result.result.language || request.language,
            confidence: result.result.confidence || 0.8,
            explanation: result.result.explanation,
            suggestions: result.result.suggestions,
            metadata: result.result.metadata,
            execution_time: result.result.execution_time || result.executionTime,
            model_info: result.result.model_info
        };
    }

    public async completeCode(
        code: string,
        language: string,
        context?: {
            filePath?: string;
            selectedText?: string;
            surroundingCode?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: code,
            task_type: 'completion',
            language,
            temperature: 0.2, // Lower temperature for completion
            context
        });
    }

    public async generateCode(
        prompt: string,
        language: string,
        options?: {
            includeExplanation?: boolean;
            maxTokens?: number;
            temperature?: number;
            context?: {
                filePath?: string;
                selectedText?: string;
                projectContext?: any;
            };
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt,
            task_type: 'generation',
            language,
            max_tokens: options?.maxTokens,
            temperature: options?.temperature,
            include_explanation: options?.includeExplanation,
            context: options?.context
        });
    }

    public async refactorCode(
        code: string,
        language: string,
        refactoringRequest: string,
        context?: {
            filePath?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: refactoringRequest,
            task_type: 'refactoring',
            language,
            include_explanation: true,
            context: {
                selectedText: code,
                filePath: context?.filePath
            }
        });
    }

    public async debugCode(
        code: string,
        language: string,
        issueDescription: string,
        context?: {
            filePath?: string;
            surroundingCode?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: issueDescription,
            task_type: 'debugging',
            language,
            include_explanation: true,
            context: {
                selectedText: code,
                filePath: context?.filePath,
                surroundingCode: context?.surroundingCode
            }
        });
    }

    public async documentCode(
        code: string,
        language: string,
        documentationRequest?: string,
        context?: {
            filePath?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: documentationRequest || 'Generate comprehensive documentation for this code',
            task_type: 'documentation',
            language,
            include_explanation: true,
            context: {
                selectedText: code,
                filePath: context?.filePath
            }
        });
    }

    public async explainCode(
        code: string,
        language: string,
        explanationRequest?: string,
        context?: {
            filePath?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: explanationRequest || 'Explain this code in detail',
            task_type: 'explanation',
            language,
            include_explanation: true,
            context: {
                selectedText: code,
                filePath: context?.filePath
            }
        });
    }

    public async optimizeCode(
        code: string,
        language: string,
        optimizationRequest?: string,
        context?: {
            filePath?: string;
        }
    ): Promise<QwenCoderResponse> {
        return this.executeQwenCoderTask({
            prompt: optimizationRequest || 'Optimize this code for better performance',
            task_type: 'optimization',
            language,
            include_explanation: true,
            context: {
                selectedText: code,
                filePath: context?.filePath
            }
        });
    }

    private setupProcessHandlers(): void {
        if (!this.pythonProcess) return;

        // Handle stdout (results from Python)
        this.pythonProcess.stdout!.on('data', (data: Buffer) => {
            const lines = data.toString().split('\n');
            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const result: PocketFlowResult = JSON.parse(line);
                        this.outputChannel.appendLine(`Received result for task: ${result.taskId}`);
                        
                        const resolver = this.taskQueue.get(result.taskId);
                        if (resolver) {
                            resolver(result);
                            this.taskQueue.delete(result.taskId);
                        }
                    } catch (error) {
                        this.outputChannel.appendLine(`Failed to parse result: ${error}`);
                        this.outputChannel.appendLine(`Raw data: ${line}`);
                    }
                }
            }
        });

        // Handle stderr (errors from Python)
        this.pythonProcess.stderr!.on('data', (data: Buffer) => {
            const errorMsg = data.toString();
            this.outputChannel.appendLine(`Backend stderr: ${errorMsg}`);
            
            // Don't show every stderr message as an error popup, just log it
            if (errorMsg.includes('ERROR') || errorMsg.includes('CRITICAL')) {
                vscode.window.showErrorMessage(`AI Backend Error: ${errorMsg.substring(0, 100)}...`);
            }
        });

        // Handle process exit
        this.pythonProcess.on('exit', (code: number | null) => {
            this.outputChannel.appendLine(`Backend process exited with code ${code}`);
            this.isInitialized = false;
            this.pythonProcess = null;
            
            // Reject all pending tasks
            for (const [taskId, resolver] of this.taskQueue) {
                resolver({
                    taskId,
                    success: false,
                    result: null,
                    error: 'Backend process exited unexpectedly',
                    executionTime: 0
                });
            }
            this.taskQueue.clear();

            // Show notification to user
            vscode.window.showWarningMessage(
                'AI Backend process stopped. Some features may not work until restart.',
                'Restart Backend'
            ).then(selection => {
                if (selection === 'Restart Backend') {
                    this.initialize().catch(err => {
                        vscode.window.showErrorMessage(`Failed to restart backend: ${err}`);
                    });
                }
            });
        });
    }

    private async findPythonExecutable(): Promise<string | null> {
        // Try common Python executable names and paths
        const pythonNames = ['python', 'python3', 'py'];
        
        for (const name of pythonNames) {
            try {
                // Test if python executable works
                const testProcess = spawn(name, ['--version'], { stdio: 'pipe' });
                const result = await new Promise<boolean>((resolve) => {
                    testProcess.on('exit', (code) => resolve(code === 0));
                    testProcess.on('error', () => resolve(false));
                    setTimeout(() => resolve(false), 5000); // 5 second timeout
                });
                
                if (result) {
                    return name;
                }
            } catch (error) {
                // Continue to next option
            }
        }

        // Check if there's a virtual environment in the workspace
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (workspaceRoot) {
            const venvPaths = [
                path.join(workspaceRoot, '.venv', 'Scripts', 'python.exe'),
                path.join(workspaceRoot, '.venv', 'bin', 'python'),
                path.join(workspaceRoot, 'venv', 'Scripts', 'python.exe'),
                path.join(workspaceRoot, 'venv', 'bin', 'python')
            ];

            for (const venvPath of venvPaths) {
                if (fs.existsSync(venvPath)) {
                    return venvPath;
                }
            }
        }

        return null;
    }

    private async findBackendScript(): Promise<string | null> {
        // Look for the backend script
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        
        const possiblePaths = [
            // In the AI IDE project structure
            path.join(this.extensionPath, '..', '..', 'backend', 'main.py'),
            // In the workspace root
            workspaceRoot ? path.join(workspaceRoot, 'ai-ide', 'backend', 'main.py') : '',
            workspaceRoot ? path.join(workspaceRoot, 'backend', 'main.py') : '',
            // Fallback to existing flow.py
            workspaceRoot ? path.join(workspaceRoot, 'flow.py') : '',
            workspaceRoot ? path.join(workspaceRoot, 'main.py') : ''
        ].filter(p => p); // Remove empty paths

        for (const scriptPath of possiblePaths) {
            if (fs.existsSync(scriptPath)) {
                return scriptPath;
            }
        }

        return null;
    }

    private generateTaskId(): string {
        return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    public getStatus(): {
        initialized: boolean;
        processRunning: boolean;
        pendingTasks: number;
    } {
        return {
            initialized: this.isInitialized,
            processRunning: this.pythonProcess !== null && !this.pythonProcess.killed,
            pendingTasks: this.taskQueue.size
        };
    }

    public dispose(): void {
        if (this.pythonProcess) {
            this.pythonProcess.kill();
            this.pythonProcess = null;
        }
        this.isInitialized = false;
        this.taskQueue.clear();
        this.outputChannel.dispose();
    }
}