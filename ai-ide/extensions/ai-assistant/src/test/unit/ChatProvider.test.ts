/**
 * Unit Tests for ChatProvider
 */

import * as assert from 'assert';
import * as sinon from 'sinon';
import * as vscode from 'vscode';
import { ChatProvider } from '../../providers/ChatProvider';

describe('ChatProvider', () => {
    let chatProvider: ChatProvider;
    let mockContext: vscode.ExtensionContext;
    let mockWebviewPanel: any;
    let createWebviewPanelStub: sinon.SinonStub;

    beforeEach(() => {
        // Mock extension context
        mockContext = {
            subscriptions: [],
            extensionUri: vscode.Uri.file('/test/extension'),
            extensionPath: '/test/extension'
        } as vscode.ExtensionContext;

        // Mock webview panel
        mockWebviewPanel = {
            webview: {
                html: '',
                postMessage: sinon.stub(),
                onDidReceiveMessage: sinon.stub(),
                options: {},
                cspSource: 'vscode-webview:'
            },
            dispose: sinon.stub(),
            onDidDispose: sinon.stub(),
            reveal: sinon.stub(),
            visible: true,
            active: true
        };

        createWebviewPanelStub = sinon.stub(vscode.window, 'createWebviewPanel');
        createWebviewPanelStub.returns(mockWebviewPanel);

        chatProvider = new ChatProvider(mockContext);
    });

    afterEach(() => {
        sinon.restore();
    });

    describe('initialization', () => {
        it('should initialize with extension context', () => {
            assert.ok(chatProvider);
            assert.strictEqual(chatProvider.context, mockContext);
        });

        it('should have required methods', () => {
            assert.strictEqual(typeof chatProvider.createChatPanel, 'function');
            assert.strictEqual(typeof chatProvider.handleMessage, 'function');
            assert.strictEqual(typeof chatProvider.sendMessageToWebview, 'function');
        });
    });

    describe('createChatPanel', () => {
        it('should create webview panel with correct configuration', () => {
            const panel = chatProvider.createChatPanel();

            assert.ok(createWebviewPanelStub.calledOnce);
            
            const args = createWebviewPanelStub.getCall(0).args;
            assert.strictEqual(args[0], 'aiChat'); // viewType
            assert.strictEqual(args[1], 'AI Chat'); // title
            assert.strictEqual(args[2], vscode.ViewColumn.Two); // viewColumn
        });

        it('should set webview options correctly', () => {
            const panel = chatProvider.createChatPanel();

            const args = createWebviewPanelStub.getCall(0).args;
            const options = args[3];
            
            assert.ok(options.enableScripts);
            assert.ok(options.retainContextWhenHidden);
        });

        it('should set webview HTML content', () => {
            const panel = chatProvider.createChatPanel();

            assert.ok(mockWebviewPanel.webview.html.length > 0);
            assert.ok(mockWebviewPanel.webview.html.includes('AI Chat Interface'));
        });

        it('should setup message listener', () => {
            const panel = chatProvider.createChatPanel();

            assert.ok(mockWebviewPanel.webview.onDidReceiveMessage.calledOnce);
        });
    });

    describe('handleMessage', () => {
        beforeEach(() => {
            chatProvider.createChatPanel();
        });

        it('should handle user message', async () => {
            const message = {
                type: 'user_message',
                content: 'Hello AI',
                context: { file: 'test.py' }
            };

            const handleMessageSpy = sinon.spy(chatProvider, 'handleMessage');
            await chatProvider.handleMessage(message);

            assert.ok(handleMessageSpy.calledOnce);
            assert.deepStrictEqual(handleMessageSpy.getCall(0).args[0], message);
        });

        it('should handle code generation request', async () => {
            const message = {
                type: 'generate_code',
                prompt: 'Create a Python function',
                context: { 
                    language: 'python',
                    file: 'main.py',
                    cursor_position: 100
                }
            };

            // Mock the bridge response
            const bridgeStub = sinon.stub(chatProvider, 'sendToBridge');
            bridgeStub.resolves({
                content: 'def example_function():\n    pass',
                confidence: 0.9,
                reasoning: ['Analyzed request', 'Generated function']
            });

            await chatProvider.handleMessage(message);

            assert.ok(bridgeStub.calledOnce);
            assert.ok(bridgeStub.calledWith('/api/code/generate', {
                prompt: message.prompt,
                context: message.context
            }));
        });

        it('should handle semantic search request', async () => {
            const message = {
                type: 'semantic_search',
                query: 'find file operations',
                context: { project_path: '/test/project' }
            };

            const bridgeStub = sinon.stub(chatProvider, 'sendToBridge');
            bridgeStub.resolves({
                results: [
                    { file: 'file_utils.py', relevance: 0.9 },
                    { file: 'io_handler.py', relevance: 0.8 }
                ]
            });

            await chatProvider.handleMessage(message);

            assert.ok(bridgeStub.calledOnce);
            assert.ok(bridgeStub.calledWith('/api/search/semantic', {
                query: message.query,
                context: message.context
            }));
        });

        it('should handle reasoning request', async () => {
            const message = {
                type: 'reasoning',
                problem: 'How to optimize this algorithm?',
                context: { 
                    code: 'def bubble_sort(arr): ...',
                    performance_goal: 'O(n log n)'
                }
            };

            const bridgeStub = sinon.stub(chatProvider, 'sendToBridge');
            bridgeStub.resolves({
                reasoning_trace: [
                    'Analyzing current algorithm complexity',
                    'Identifying optimization opportunities',
                    'Suggesting merge sort implementation'
                ],
                solution: 'Use merge sort for O(n log n) complexity'
            });

            await chatProvider.handleMessage(message);

            assert.ok(bridgeStub.calledOnce);
            assert.ok(bridgeStub.calledWith('/api/reasoning/analyze', {
                problem: message.problem,
                context: message.context
            }));
        });

        it('should handle unknown message type', async () => {
            const message = {
                type: 'unknown_type',
                data: 'test'
            };

            const postMessageSpy = sinon.spy(mockWebviewPanel.webview, 'postMessage');
            await chatProvider.handleMessage(message);

            assert.ok(postMessageSpy.calledOnce);
            const sentMessage = postMessageSpy.getCall(0).args[0];
            assert.strictEqual(sentMessage.type, 'error');
            assert.ok(sentMessage.message.includes('Unknown message type'));
        });
    });

    describe('sendMessageToWebview', () => {
        beforeEach(() => {
            chatProvider.createChatPanel();
        });

        it('should send message to webview', () => {
            const message = {
                type: 'ai_response',
                content: 'Hello user!',
                timestamp: Date.now()
            };

            chatProvider.sendMessageToWebview(message);

            assert.ok(mockWebviewPanel.webview.postMessage.calledOnce);
            assert.deepStrictEqual(
                mockWebviewPanel.webview.postMessage.getCall(0).args[0],
                message
            );
        });

        it('should handle webview disposal', () => {
            chatProvider.sendMessageToWebview({ type: 'test' });

            // Simulate webview disposal
            mockWebviewPanel.webview.postMessage.throws(new Error('Webview disposed'));

            // Should not throw error
            assert.doesNotThrow(() => {
                chatProvider.sendMessageToWebview({ type: 'test' });
            });
        });
    });

    describe('error handling', () => {
        it('should handle bridge communication errors', async () => {
            chatProvider.createChatPanel();

            const bridgeStub = sinon.stub(chatProvider, 'sendToBridge');
            bridgeStub.rejects(new Error('Bridge connection failed'));

            const message = {
                type: 'generate_code',
                prompt: 'Test prompt'
            };

            await chatProvider.handleMessage(message);

            // Should send error message to webview
            assert.ok(mockWebviewPanel.webview.postMessage.calledOnce);
            const sentMessage = mockWebviewPanel.webview.postMessage.getCall(0).args[0];
            assert.strictEqual(sentMessage.type, 'error');
        });

        it('should handle malformed messages', async () => {
            chatProvider.createChatPanel();

            const malformedMessage = null;

            await chatProvider.handleMessage(malformedMessage);

            // Should send error message to webview
            assert.ok(mockWebviewPanel.webview.postMessage.calledOnce);
            const sentMessage = mockWebviewPanel.webview.postMessage.getCall(0).args[0];
            assert.strictEqual(sentMessage.type, 'error');
        });
    });

    describe('context management', () => {
        it('should maintain conversation context', async () => {
            chatProvider.createChatPanel();

            const message1 = {
                type: 'user_message',
                content: 'What is Python?',
                context: { file: 'test.py' }
            };

            const message2 = {
                type: 'user_message',
                content: 'Show me an example',
                context: { file: 'test.py' }
            };

            const bridgeStub = sinon.stub(chatProvider, 'sendToBridge');
            bridgeStub.resolves({ content: 'Response' });

            await chatProvider.handleMessage(message1);
            await chatProvider.handleMessage(message2);

            // Second message should include conversation context
            const secondCallArgs = bridgeStub.getCall(1).args[1];
            assert.ok(secondCallArgs.conversation_context);
        });

        it('should clear context on request', () => {
            chatProvider.createChatPanel();

            const clearMessage = {
                type: 'clear_context'
            };

            chatProvider.handleMessage(clearMessage);

            // Context should be cleared
            assert.strictEqual(chatProvider.getConversationContext().length, 0);
        });
    });
});