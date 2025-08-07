/**
 * Unit Tests for PocketFlowBridge Service
 */

import * as assert from 'assert';
import * as sinon from 'sinon';
import { PocketFlowBridge } from '../../services/PocketFlowBridge';

describe('PocketFlowBridge', () => {
    let bridge: PocketFlowBridge;
    let axiosStub: sinon.SinonStub;

    beforeEach(() => {
        bridge = new PocketFlowBridge();
        
        // Mock axios
        const axios = require('axios');
        axiosStub = sinon.stub(axios, 'post');
    });

    afterEach(() => {
        sinon.restore();
    });

    describe('initialization', () => {
        it('should initialize with default configuration', () => {
            assert.ok(bridge);
            assert.strictEqual(typeof bridge.sendRequest, 'function');
            assert.strictEqual(typeof bridge.startFlow, 'function');
        });

        it('should set correct backend URL', () => {
            const expectedUrl = 'http://localhost:8000';
            assert.strictEqual(bridge.getBackendUrl(), expectedUrl);
        });
    });

    describe('sendRequest', () => {
        it('should send POST request to correct endpoint', async () => {
            const mockResponse = {
                data: {
                    success: true,
                    content: 'Test response',
                    confidence: 0.9
                }
            };
            axiosStub.resolves(mockResponse);

            const endpoint = '/api/test';
            const data = { test: 'data' };

            const result = await bridge.sendRequest(endpoint, data);

            assert.ok(axiosStub.calledOnce);
            assert.ok(axiosStub.calledWith('http://localhost:8000/api/test', data));
            assert.deepStrictEqual(result, mockResponse.data);
        });

        it('should handle request timeout', async () => {
            axiosStub.rejects(new Error('timeout'));

            try {
                await bridge.sendRequest('/api/test', {});
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error instanceof Error);
                assert.ok(error.message.includes('timeout'));
            }
        });

        it('should retry failed requests', async () => {
            axiosStub.onFirstCall().rejects(new Error('Network error'));
            axiosStub.onSecondCall().resolves({
                data: { success: true, content: 'Retry successful' }
            });

            const result = await bridge.sendRequest('/api/test', {}, { retries: 1 });

            assert.strictEqual(axiosStub.callCount, 2);
            assert.strictEqual(result.content, 'Retry successful');
        });
    });

    describe('startFlow', () => {
        it('should start PocketFlow workflow', async () => {
            const mockResponse = {
                data: {
                    flow_id: 'test-flow-123',
                    status: 'started',
                    nodes: ['analyze', 'generate', 'validate']
                }
            };
            axiosStub.resolves(mockResponse);

            const flowConfig = {
                type: 'code_generation',
                context: { language: 'python' }
            };

            const result = await bridge.startFlow(flowConfig);

            assert.ok(axiosStub.calledOnce);
            assert.ok(axiosStub.calledWith('http://localhost:8000/api/flow/start', flowConfig));
            assert.strictEqual(result.flow_id, 'test-flow-123');
            assert.strictEqual(result.status, 'started');
        });

        it('should handle flow start failure', async () => {
            axiosStub.rejects(new Error('Flow start failed'));

            try {
                await bridge.startFlow({ type: 'test' });
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error instanceof Error);
                assert.ok(error.message.includes('Flow start failed'));
            }
        });
    });

    describe('getFlowStatus', () => {
        it('should get flow status', async () => {
            const mockResponse = {
                data: {
                    flow_id: 'test-flow-123',
                    status: 'completed',
                    progress: 100,
                    result: { content: 'Flow completed successfully' }
                }
            };
            axiosStub.resolves(mockResponse);

            const result = await bridge.getFlowStatus('test-flow-123');

            assert.ok(axiosStub.calledOnce);
            assert.ok(axiosStub.calledWith('http://localhost:8000/api/flow/status/test-flow-123'));
            assert.strictEqual(result.status, 'completed');
            assert.strictEqual(result.progress, 100);
        });
    });

    describe('error handling', () => {
        it('should handle network errors gracefully', async () => {
            axiosStub.rejects(new Error('ECONNREFUSED'));

            try {
                await bridge.sendRequest('/api/test', {});
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error instanceof Error);
                assert.ok(error.message.includes('ECONNREFUSED'));
            }
        });

        it('should handle HTTP error responses', async () => {
            const errorResponse = {
                response: {
                    status: 500,
                    data: { error: 'Internal server error' }
                }
            };
            axiosStub.rejects(errorResponse);

            try {
                await bridge.sendRequest('/api/test', {});
                assert.fail('Should have thrown an error');
            } catch (error) {
                assert.ok(error.response);
                assert.strictEqual(error.response.status, 500);
            }
        });
    });

    describe('request configuration', () => {
        it('should set correct headers', async () => {
            const mockResponse = { data: { success: true } };
            axiosStub.resolves(mockResponse);

            await bridge.sendRequest('/api/test', {});

            const callArgs = axiosStub.getCall(0).args;
            const config = callArgs[2];
            
            assert.ok(config.headers);
            assert.strictEqual(config.headers['Content-Type'], 'application/json');
        });

        it('should set request timeout', async () => {
            const mockResponse = { data: { success: true } };
            axiosStub.resolves(mockResponse);

            await bridge.sendRequest('/api/test', {}, { timeout: 5000 });

            const callArgs = axiosStub.getCall(0).args;
            const config = callArgs[2];
            
            assert.strictEqual(config.timeout, 5000);
        });
    });
});