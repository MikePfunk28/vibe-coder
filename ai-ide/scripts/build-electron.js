#!/usr/bin/env node

/**
 * AI IDE Electron Build Script
 * Creates standalone executable with embedded backend services
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const archiver = require('archiver');

const PROJECT_ROOT = path.dirname(__dirname);
const DIST_DIR = path.join(PROJECT_ROOT, 'dist');
const ELECTRON_DIR = path.join(PROJECT_ROOT, 'electron');

// Configuration
const config = {
  appName: 'AI IDE',
  appId: 'dev.ai-ide.app',
  version: require(path.join(PROJECT_ROOT, 'package.json')).version || '0.1.0',
  author: 'AI IDE Team',
  description: 'Advanced AI-powered development environment',
  platforms: ['win32', 'darwin', 'linux'],
  architectures: ['x64', 'arm64']
};

class ElectronBuilder {
  constructor() {
    this.ensureDirectories();
    this.installDependencies();
  }

  ensureDirectories() {
    console.log('📁 Creating directories...');
    [DIST_DIR, ELECTRON_DIR].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  installDependencies() {
    console.log('📦 Installing Electron dependencies...');
    
    // Create package.json for Electron app
    const electronPackageJson = {
      name: 'ai-ide-electron',
      version: config.version,
      description: config.description,
      main: 'main.js',
      author: config.author,
      license: 'MIT',
      dependencies: {
        'electron': '^27.0.0',
        'electron-builder': '^24.6.4',
        'electron-store': '^8.1.0',
        'node-pty': '^1.0.0',
        'ws': '^8.14.2',
        'express': '^4.18.2',
        'cors': '^2.8.5'
      },
      devDependencies: {
        'electron-builder': '^24.6.4'
      },
      build: {
        appId: config.appId,
        productName: config.appName,
        directories: {
          output: '../dist'
        },
        files: [
          'main.js',
          'preload.js',
          'renderer/**/*',
          'backend/**/*',
          'extensions/**/*',
          'node_modules/**/*'
        ],
        extraResources: [
          {
            from: '../backend',
            to: 'backend',
            filter: ['**/*', '!**/__pycache__', '!**/venv', '!**/node_modules']
          },
          {
            from: '../extensions/ai-assistant/out',
            to: 'extensions/ai-assistant'
          }
        ],
        win: {
          target: [
            { target: 'nsis', arch: ['x64', 'arm64'] },
            { target: 'portable', arch: ['x64', 'arm64'] }
          ],
          icon: 'assets/icon.ico'
        },
        mac: {
          target: [
            { target: 'dmg', arch: ['x64', 'arm64'] },
            { target: 'zip', arch: ['x64', 'arm64'] }
          ],
          icon: 'assets/icon.icns',
          category: 'public.app-category.developer-tools'
        },
        linux: {
          target: [
            { target: 'AppImage', arch: ['x64', 'arm64'] },
            { target: 'deb', arch: ['x64', 'arm64'] },
            { target: 'rpm', arch: ['x64', 'arm64'] }
          ],
          icon: 'assets/icon.png',
          category: 'Development'
        },
        nsis: {
          oneClick: false,
          allowToChangeInstallationDirectory: true,
          createDesktopShortcut: true,
          createStartMenuShortcut: true
        }
      }
    };

    fs.writeFileSync(
      path.join(ELECTRON_DIR, 'package.json'),
      JSON.stringify(electronPackageJson, null, 2)
    );

    // Install dependencies
    try {
      execSync('npm install', { 
        cwd: ELECTRON_DIR, 
        stdio: 'inherit' 
      });
    } catch (error) {
      console.error('❌ Failed to install dependencies:', error.message);
      process.exit(1);
    }
  }

  createMainProcess() {
    console.log('⚡ Creating Electron main process...');
    
    const mainJs = `
const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const Store = require('electron-store');
const express = require('express');
const cors = require('cors');

// Configuration store
const store = new Store();

class AIIDEApp {
  constructor() {
    this.mainWindow = null;
    this.backendProcess = null;
    this.backendServer = null;
    this.isQuitting = false;
  }

  async initialize() {
    await app.whenReady();
    this.createWindow();
    this.setupBackend();
    this.setupIPC();
    this.setupAppEvents();
  }

  createWindow() {
    console.log('Creating main window...');
    
    // Get window state from store
    const windowState = store.get('windowState', {
      width: 1400,
      height: 900,
      x: undefined,
      y: undefined
    });

    this.mainWindow = new BrowserWindow({
      width: windowState.width,
      height: windowState.height,
      x: windowState.x,
      y: windowState.y,
      minWidth: 1000,
      minHeight: 700,
      icon: path.join(__dirname, 'assets', 'icon.png'),
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, 'preload.js'),
        webSecurity: false // Allow loading local resources
      },
      titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
      show: false
    });

    // Load the AI IDE interface
    this.mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

    // Show window when ready
    this.mainWindow.once('ready-to-show', () => {
      this.mainWindow.show();
      
      // Open DevTools in development
      if (process.env.NODE_ENV === 'development') {
        this.mainWindow.webContents.openDevTools();
      }
    });

    // Save window state on close
    this.mainWindow.on('close', (event) => {
      if (!this.isQuitting) {
        event.preventDefault();
        this.mainWindow.hide();
        return;
      }

      const bounds = this.mainWindow.getBounds();
      store.set('windowState', bounds);
    });

    // Handle external links
    this.mainWindow.webContents.setWindowOpenHandler(({ url }) => {
      shell.openExternal(url);
      return { action: 'deny' };
    });
  }

  async setupBackend() {
    console.log('Setting up backend services...');
    
    try {
      // Start embedded Python backend
      const backendPath = path.join(__dirname, 'backend');
      const pythonExecutable = this.findPythonExecutable();
      
      if (!pythonExecutable) {
        throw new Error('Python executable not found');
      }

      // Set up Python environment
      const env = {
        ...process.env,
        PYTHONPATH: backendPath,
        AI_IDE_ENV: 'electron',
        AI_IDE_DATA_DIR: path.join(app.getPath('userData'), 'ai-ide-data')
      };

      // Ensure data directory exists
      fs.mkdirSync(env.AI_IDE_DATA_DIR, { recursive: true });

      // Start backend process
      this.backendProcess = spawn(pythonExecutable, [
        path.join(backendPath, 'main.py'),
        '--port', '8000',
        '--host', '127.0.0.1'
      ], {
        env,
        cwd: backendPath,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.backendProcess.stdout.on('data', (data) => {
        console.log('Backend:', data.toString());
      });

      this.backendProcess.stderr.on('data', (data) => {
        console.error('Backend Error:', data.toString());
      });

      this.backendProcess.on('close', (code) => {
        console.log('Backend process exited with code:', code);
        if (!this.isQuitting && code !== 0) {
          this.showBackendError();
        }
      });

      // Wait for backend to start
      await this.waitForBackend();
      
    } catch (error) {
      console.error('Failed to start backend:', error);
      this.showBackendError(error.message);
    }
  }

  findPythonExecutable() {
    const possiblePaths = [
      'python3',
      'python',
      path.join(__dirname, 'backend', 'venv', 'bin', 'python'),
      path.join(__dirname, 'backend', 'venv', 'Scripts', 'python.exe')
    ];

    for (const pythonPath of possiblePaths) {
      try {
        execSync(\`\${pythonPath} --version\`, { stdio: 'ignore' });
        return pythonPath;
      } catch (error) {
        continue;
      }
    }

    return null;
  }

  async waitForBackend(maxAttempts = 30) {
    const http = require('http');
    
    for (let i = 0; i < maxAttempts; i++) {
      try {
        await new Promise((resolve, reject) => {
          const req = http.get('http://127.0.0.1:8000/health', (res) => {
            resolve();
          });
          req.on('error', reject);
          req.setTimeout(1000, () => req.destroy());
        });
        
        console.log('Backend is ready!');
        return;
      } catch (error) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    throw new Error('Backend failed to start within timeout');
  }

  setupIPC() {
    // Handle backend communication
    ipcMain.handle('backend-request', async (event, { method, url, data }) => {
      try {
        const response = await this.makeBackendRequest(method, url, data);
        return { success: true, data: response };
      } catch (error) {
        return { success: false, error: error.message };
      }
    });

    // Handle file operations
    ipcMain.handle('show-open-dialog', async (event, options) => {
      const result = await dialog.showOpenDialog(this.mainWindow, options);
      return result;
    });

    ipcMain.handle('show-save-dialog', async (event, options) => {
      const result = await dialog.showSaveDialog(this.mainWindow, options);
      return result;
    });

    // Handle settings
    ipcMain.handle('get-setting', (event, key) => {
      return store.get(key);
    });

    ipcMain.handle('set-setting', (event, key, value) => {
      store.set(key, value);
    });
  }

  async makeBackendRequest(method, url, data) {
    const http = require('http');
    const querystring = require('querystring');
    
    return new Promise((resolve, reject) => {
      const options = {
        hostname: '127.0.0.1',
        port: 8000,
        path: url,
        method: method.toUpperCase(),
        headers: {
          'Content-Type': 'application/json'
        }
      };

      const req = http.request(options, (res) => {
        let responseData = '';
        
        res.on('data', (chunk) => {
          responseData += chunk;
        });
        
        res.on('end', () => {
          try {
            const parsed = JSON.parse(responseData);
            resolve(parsed);
          } catch (error) {
            resolve(responseData);
          }
        });
      });

      req.on('error', reject);
      
      if (data && (method === 'POST' || method === 'PUT')) {
        req.write(JSON.stringify(data));
      }
      
      req.end();
    });
  }

  setupAppEvents() {
    app.on('window-all-closed', () => {
      if (process.platform !== 'darwin') {
        this.quit();
      }
    });

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) {
        this.createWindow();
      } else {
        this.mainWindow.show();
      }
    });

    app.on('before-quit', () => {
      this.isQuitting = true;
    });

    app.on('will-quit', (event) => {
      if (this.backendProcess && !this.backendProcess.killed) {
        event.preventDefault();
        this.quit();
      }
    });
  }

  showBackendError(message = 'Failed to start AI IDE backend services') {
    dialog.showErrorBox('AI IDE Error', message + '\\n\\nPlease check that Python 3.11+ is installed and try again.');
  }

  quit() {
    this.isQuitting = true;
    
    if (this.backendProcess && !this.backendProcess.killed) {
      console.log('Stopping backend process...');
      this.backendProcess.kill('SIGTERM');
      
      setTimeout(() => {
        if (!this.backendProcess.killed) {
          this.backendProcess.kill('SIGKILL');
        }
        app.quit();
      }, 5000);
    } else {
      app.quit();
    }
  }
}

// Initialize app
const aiIdeApp = new AIIDEApp();
aiIdeApp.initialize().catch(console.error);

// Handle protocol for deep linking
app.setAsDefaultProtocolClient('ai-ide');
`;

    fs.writeFileSync(path.join(ELECTRON_DIR, 'main.js'), mainJs);
  }

  createRenderer() {
    console.log('🎨 Creating renderer interface...');
    
    const rendererDir = path.join(ELECTRON_DIR, 'renderer');
    fs.mkdirSync(rendererDir, { recursive: true });

    // Create CSS styles
    const stylesCSS = `
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1e1e1e;
    color: #cccccc;
    height: 100vh;
    overflow: hidden;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: #2d2d30;
    border-bottom: 1px solid #3e3e42;
    -webkit-app-region: drag;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-left h1 {
    font-size: 16px;
    font-weight: 600;
}

.version {
    font-size: 12px;
    color: #888;
    background: #404040;
    padding: 2px 6px;
    border-radius: 3px;
}

.header-right {
    display: flex;
    gap: 4px;
    -webkit-app-region: no-drag;
}

.header-btn {
    background: none;
    border: none;
    color: #cccccc;
    padding: 4px 8px;
    cursor: pointer;
    border-radius: 3px;
    font-size: 14px;
}

.header-btn:hover {
    background: #404040;
}

.main-content {
    display: flex;
    height: calc(100vh - 48px);
}

.sidebar {
    width: 200px;
    background: #252526;
    border-right: 1px solid #3e3e42;
}

.nav {
    padding: 16px 0;
}

.nav-btn {
    display: block;
    width: 100%;
    padding: 12px 16px;
    background: none;
    border: none;
    color: #cccccc;
    text-align: left;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
}

.nav-btn:hover {
    background: #2a2d2e;
}

.nav-btn.active {
    background: #094771;
    color: #ffffff;
}

.content {
    flex: 1;
    overflow: hidden;
}

.view {
    display: none;
    height: 100%;
    padding: 20px;
    overflow-y: auto;
}

.view.active {
    display: block;
}

.chat-container, .search-container, .reasoning-container {
    height: 100%;
    display: flex;
    flex-direction: column;
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    margin-bottom: 16px;
}

.chat-input-container, .search-input-container, .reasoning-input-container {
    display: flex;
    gap: 8px;
}

#chat-input, #reasoning-input {
    flex: 1;
    padding: 12px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    color: #cccccc;
    font-family: inherit;
    resize: vertical;
    min-height: 80px;
}

#search-input {
    flex: 1;
    padding: 12px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    color: #cccccc;
    font-family: inherit;
}

button {
    padding: 12px 24px;
    background: #0e639c;
    border: none;
    border-radius: 6px;
    color: white;
    cursor: pointer;
    font-weight: 500;
}

button:hover {
    background: #1177bb;
}

.search-results, .reasoning-results {
    flex: 1;
    margin-top: 16px;
    padding: 16px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    overflow-y: auto;
}

.agents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
    margin-top: 16px;
}

.agent-card {
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.agent-card h3 {
    margin-bottom: 8px;
    color: #ffffff;
}

.agent-card p {
    margin-bottom: 16px;
    color: #888;
    font-size: 14px;
}

.settings-section {
    margin-bottom: 24px;
    padding: 16px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
}

.settings-section h3 {
    margin-bottom: 12px;
    color: #ffffff;
}

.settings-section label {
    display: block;
    margin-bottom: 12px;
}

.settings-section label span {
    display: block;
    margin-bottom: 4px;
    font-size: 14px;
}

.settings-section input[type="text"] {
    width: 100%;
    padding: 8px 12px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    color: #cccccc;
}

.settings-section input[type="checkbox"] {
    margin-right: 8px;
}

.message {
    margin-bottom: 16px;
    padding: 12px;
    border-radius: 6px;
}

.message.user {
    background: #094771;
    margin-left: 20%;
}

.message.ai {
    background: #2d2d30;
    margin-right: 20%;
}

.message-content {
    white-space: pre-wrap;
}

.search-result {
    margin-bottom: 16px;
    padding: 12px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
}

.search-result h4 {
    color: #ffffff;
    margin-bottom: 8px;
}

.search-result p {
    color: #cccccc;
    font-size: 14px;
}

.reasoning-step {
    margin-bottom: 16px;
    padding: 12px;
    background: #2d2d30;
    border-left: 4px solid #0e639c;
    border-radius: 0 6px 6px 0;
}

.reasoning-step h4 {
    color: #ffffff;
    margin-bottom: 8px;
}
`;

    fs.writeFileSync(path.join(rendererDir, 'styles.css'), stylesCSS);

    // Create JavaScript app logic
    const appJS = `
class AIIDEApp {
    constructor() {
        this.currentView = 'chat';
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupChat();
        this.setupSearch();
        this.setupReasoning();
        this.setupAgents();
        this.setupSettings();
        this.loadSettings();
    }

    setupNavigation() {
        const navBtns = document.querySelectorAll('.nav-btn');
        navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const view = btn.dataset.view;
                this.switchView(view);
            });
        });
    }

    switchView(viewName) {
        // Update navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === viewName);
        });

        // Update content
        document.querySelectorAll('.view').forEach(view => {
            view.classList.toggle('active', view.id === viewName + '-view');
        });

        this.currentView = viewName;
    }

    setupChat() {
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const chatMessages = document.getElementById('chat-messages');

        const sendMessage = async () => {
            const message = chatInput.value.trim();
            if (!message) return;

            this.addChatMessage('user', message);
            chatInput.value = '';

            try {
                const response = await window.aiIdeAPI.sendToAgent('general', message);
                if (response.success) {
                    this.addChatMessage('ai', response.data.response || 'No response received');
                } else {
                    this.addChatMessage('ai', 'Error: ' + response.error);
                }
            } catch (error) {
                this.addChatMessage('ai', 'Error: ' + error.message);
            }
        };

        sendBtn.addEventListener('click', sendMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                sendMessage();
            }
        });
    }

    addChatMessage(type, content) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = \`message \${type}\`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    setupSearch() {
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');
        const searchResults = document.getElementById('search-results');

        const performSearch = async () => {
            const query = searchInput.value.trim();
            if (!query) return;

            searchResults.innerHTML = '<p>Searching...</p>';

            try {
                // Perform unified search (semantic + web + RAG)
                const [semanticResults, webResults, ragResults] = await Promise.all([
                    window.aiIdeAPI.semanticSearch(query),
                    window.aiIdeAPI.webSearch(query),
                    window.aiIdeAPI.ragQuery(query)
                ]);

                this.displaySearchResults(searchResults, {
                    semantic: semanticResults.success ? semanticResults.data : [],
                    web: webResults.success ? webResults.data : [],
                    rag: ragResults.success ? ragResults.data : []
                });
            } catch (error) {
                searchResults.innerHTML = \`<p>Error: \${error.message}</p>\`;
            }
        };

        searchBtn.addEventListener('click', performSearch);
        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                performSearch();
            }
        });
    }

    displaySearchResults(container, results) {
        container.innerHTML = '';

        // Display semantic results
        if (results.semantic.length > 0) {
            const section = document.createElement('div');
            section.innerHTML = '<h3>Code Search Results</h3>';
            results.semantic.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'search-result';
                resultDiv.innerHTML = \`
                    <h4>\${result.file_path}</h4>
                    <p>\${result.snippet}</p>
                    <small>Similarity: \${(result.similarity_score * 100).toFixed(1)}%</small>
                \`;
                section.appendChild(resultDiv);
            });
            container.appendChild(section);
        }

        // Display web results
        if (results.web.length > 0) {
            const section = document.createElement('div');
            section.innerHTML = '<h3>Web Search Results</h3>';
            results.web.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'search-result';
                resultDiv.innerHTML = \`
                    <h4><a href="\${result.url}" target="_blank">\${result.title}</a></h4>
                    <p>\${result.snippet}</p>
                \`;
                section.appendChild(resultDiv);
            });
            container.appendChild(section);
        }

        // Display RAG results
        if (results.rag.length > 0) {
            const section = document.createElement('div');
            section.innerHTML = '<h3>Knowledge Base Results</h3>';
            results.rag.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'search-result';
                resultDiv.innerHTML = \`
                    <h4>\${result.title}</h4>
                    <p>\${result.content}</p>
                \`;
                section.appendChild(resultDiv);
            });
            container.appendChild(section);
        }

        if (results.semantic.length === 0 && results.web.length === 0 && results.rag.length === 0) {
            container.innerHTML = '<p>No results found.</p>';
        }
    }

    setupReasoning() {
        const reasoningInput = document.getElementById('reasoning-input');
        const reasonBtn = document.getElementById('reason-btn');
        const reasoningResults = document.getElementById('reasoning-results');

        const performReasoning = async () => {
            const problem = reasoningInput.value.trim();
            if (!problem) return;

            reasoningResults.innerHTML = '<p>Analyzing...</p>';

            try {
                const response = await window.aiIdeAPI.reason(problem, 'deep');
                if (response.success) {
                    this.displayReasoningResults(reasoningResults, response.data);
                } else {
                    reasoningResults.innerHTML = \`<p>Error: \${response.error}</p>\`;
                }
            } catch (error) {
                reasoningResults.innerHTML = \`<p>Error: \${error.message}</p>\`;
            }
        };

        reasonBtn.addEventListener('click', performReasoning);
        reasoningInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                performReasoning();
            }
        });
    }

    displayReasoningResults(container, results) {
        container.innerHTML = '';

        if (results.steps) {
            results.steps.forEach((step, index) => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'reasoning-step';
                stepDiv.innerHTML = \`
                    <h4>Step \${index + 1}: \${step.title}</h4>
                    <p>\${step.content}</p>
                \`;
                container.appendChild(stepDiv);
            });
        }

        if (results.conclusion) {
            const conclusionDiv = document.createElement('div');
            conclusionDiv.className = 'reasoning-step';
            conclusionDiv.style.borderLeftColor = '#28a745';
            conclusionDiv.innerHTML = \`
                <h4>Conclusion</h4>
                <p>\${results.conclusion}</p>
            \`;
            container.appendChild(conclusionDiv);
        }
    }

    setupAgents() {
        const agentCards = document.querySelectorAll('.agent-card');
        agentCards.forEach(card => {
            const btn = card.querySelector('.agent-btn');
            btn.addEventListener('click', () => {
                const agentType = card.dataset.agent;
                this.openAgentChat(agentType);
            });
        });
    }

    openAgentChat(agentType) {
        // Switch to chat view and set agent context
        this.switchView('chat');
        const chatInput = document.getElementById('chat-input');
        chatInput.placeholder = \`Chat with \${agentType} agent...\`;
        chatInput.focus();
    }

    setupSettings() {
        const saveBtn = document.getElementById('save-settings');
        saveBtn.addEventListener('click', () => {
            this.saveSettings();
        });
    }

    async loadSettings() {
        try {
            const lmStudioUrl = await window.electronAPI.getSetting('lmStudioUrl') || 'http://localhost:1234';
            const modelName = await window.electronAPI.getSetting('modelName') || 'qwen-coder-3';
            const webSearchEnabled = await window.electronAPI.getSetting('webSearchEnabled') !== false;
            const semanticSearchEnabled = await window.electronAPI.getSetting('semanticSearchEnabled') !== false;

            document.getElementById('lm-studio-url').value = lmStudioUrl;
            document.getElementById('model-name').value = modelName;
            document.getElementById('web-search-enabled').checked = webSearchEnabled;
            document.getElementById('semantic-search-enabled').checked = semanticSearchEnabled;
        } catch (error) {
            console.error('Failed to load settings:', error);
        }
    }

    async saveSettings() {
        try {
            const lmStudioUrl = document.getElementById('lm-studio-url').value;
            const modelName = document.getElementById('model-name').value;
            const webSearchEnabled = document.getElementById('web-search-enabled').checked;
            const semanticSearchEnabled = document.getElementById('semantic-search-enabled').checked;

            await window.electronAPI.setSetting('lmStudioUrl', lmStudioUrl);
            await window.electronAPI.setSetting('modelName', modelName);
            await window.electronAPI.setSetting('webSearchEnabled', webSearchEnabled);
            await window.electronAPI.setSetting('semanticSearchEnabled', semanticSearchEnabled);

            alert('Settings saved successfully!');
        } catch (error) {
            alert('Failed to save settings: ' + error.message);
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIIDEApp();
});
`;

    fs.writeFileSync(path.join(rendererDir, 'app.js'), appJS);
  }

  createPreloadScript() {
    console.log('🔧 Creating preload script...');
    
    const preloadJs = `
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Backend communication
  backendRequest: (method, url, data) => 
    ipcRenderer.invoke('backend-request', { method, url, data }),
  
  // File operations
  showOpenDialog: (options) => 
    ipcRenderer.invoke('show-open-dialog', options),
  
  showSaveDialog: (options) => 
    ipcRenderer.invoke('show-save-dialog', options),
  
  // Settings
  getSetting: (key) => 
    ipcRenderer.invoke('get-setting', key),
  
  setSetting: (key, value) => 
    ipcRenderer.invoke('set-setting', key, value),
  
  // Platform info
  platform: process.platform,
  
  // Version info
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron
  }
});

// AI IDE specific API
contextBridge.exposeInMainWorld('aiIdeAPI', {
  // Agent communication
  sendToAgent: async (agentType, message, context = {}) => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: \`/api/agents/\${agentType}/chat\`,
      data: { message, context }
    });
  },
  
  // Semantic search
  semanticSearch: async (query, options = {}) => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: '/api/search/semantic',
      data: { query, ...options }
    });
  },
  
  // Web search
  webSearch: async (query, options = {}) => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: '/api/search/web',
      data: { query, ...options }
    });
  },
  
  // RAG queries
  ragQuery: async (query, context = {}) => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: '/api/rag/query',
      data: { query, context }
    });
  },
  
  // Code generation
  generateCode: async (prompt, context = {}) => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: '/api/code/generate',
      data: { prompt, context }
    });
  },
  
  // Reasoning
  reason: async (problem, mode = 'fast') => {
    return await ipcRenderer.invoke('backend-request', {
      method: 'POST',
      url: '/api/reasoning/analyze',
      data: { problem, mode }
    });
  }
});
`;

    fs.writeFileSync(path.join(ELECTRON_DIR, 'preload.js'), preloadJs);
  }

  createRenderer() {
    console.log('🎨 Creating renderer interface...');
    
    const rendererDir = path.join(ELECTRON_DIR, 'renderer');
    fs.mkdirSync(rendererDir, { recursive: true });

    // Create main HTML file
    const indexHtml = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI IDE</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div id="app">
        <header class="header">
            <div class="header-left">
                <h1>AI IDE</h1>
                <span class="version">v${config.version}</span>
            </div>
            <div class="header-right">
                <button id="settings-btn" class="header-btn">⚙️</button>
                <button id="minimize-btn" class="header-btn">−</button>
                <button id="close-btn" class="header-btn">×</button>
            </div>
        </header>
        
        <div class="main-content">
            <aside class="sidebar">
                <nav class="nav">
                    <button class="nav-btn active" data-view="chat">💬 AI Chat</button>
                    <button class="nav-btn" data-view="search">🔍 Search</button>
                    <button class="nav-btn" data-view="reasoning">🧠 Reasoning</button>
                    <button class="nav-btn" data-view="agents">🤖 Agents</button>
                    <button class="nav-btn" data-view="settings">⚙️ Settings</button>
                </nav>
            </aside>
            
            <main class="content">
                <div id="chat-view" class="view active">
                    <div class="chat-container">
                        <div id="chat-messages" class="chat-messages"></div>
                        <div class="chat-input-container">
                            <textarea id="chat-input" placeholder="Ask AI anything..."></textarea>
                            <button id="send-btn">Send</button>
                        </div>
                    </div>
                </div>
                
                <div id="search-view" class="view">
                    <div class="search-container">
                        <div class="search-input-container">
                            <input type="text" id="search-input" placeholder="Search code, web, or knowledge base...">
                            <button id="search-btn">Search</button>
                        </div>
                        <div id="search-results" class="search-results"></div>
                    </div>
                </div>
                
                <div id="reasoning-view" class="view">
                    <div class="reasoning-container">
                        <div class="reasoning-input-container">
                            <textarea id="reasoning-input" placeholder="Describe a problem for AI reasoning..."></textarea>
                            <button id="reason-btn">Analyze</button>
                        </div>
                        <div id="reasoning-results" class="reasoning-results"></div>
                    </div>
                </div>
                
                <div id="agents-view" class="view">
                    <div class="agents-container">
                        <h2>AI Agents</h2>
                        <div class="agents-grid">
                            <div class="agent-card" data-agent="code">
                                <h3>Code Agent</h3>
                                <p>Specialized in code generation and analysis</p>
                                <button class="agent-btn">Chat</button>
                            </div>
                            <div class="agent-card" data-agent="search">
                                <h3>Search Agent</h3>
                                <p>Expert in finding relevant information</p>
                                <button class="agent-btn">Chat</button>
                            </div>
                            <div class="agent-card" data-agent="reasoning">
                                <h3>Reasoning Agent</h3>
                                <p>Deep problem analysis and solution</p>
                                <button class="agent-btn">Chat</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div id="settings-view" class="view">
                    <div class="settings-container">
                        <h2>Settings</h2>
                        <div class="settings-section">
                            <h3>AI Models</h3>
                            <label>
                                <span>LM Studio URL:</span>
                                <input type="text" id="lm-studio-url" value="http://localhost:1234">
                            </label>
                            <label>
                                <span>Model Name:</span>
                                <input type="text" id="model-name" value="qwen-coder-3">
                            </label>
                        </div>
                        <div class="settings-section">
                            <h3>Search Settings</h3>
                            <label>
                                <input type="checkbox" id="web-search-enabled" checked>
                                <span>Enable Web Search</span>
                            </label>
                            <label>
                                <input type="checkbox" id="semantic-search-enabled" checked>
                                <span>Enable Semantic Search</span>
                            </label>
                        </div>
                        <button id="save-settings">Save Settings</button>
                    </div>
                </div>
            </main>
        </div>
    </div>
    
    <script src="app.js"></script>
</body>
</html>`;
        
createAssets() {
    console.log('🎨 Creating application assets...');
    
    const assetsDir = path.join(ELECTRON_DIR, 'assets');
    fs.mkdirSync(assetsDir, { recursive: true });

    // Create simple icon files (placeholder - replace with actual icons)
    const iconSVG = `
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <rect width="256" height="256" fill="#0e639c"/>
  <circle cx="128" cy="128" r="80" fill="#ffffff"/>
  <text x="128" y="140" text-anchor="middle" fill="#0e639c" font-family="Arial" font-size="48" font-weight="bold">AI</text>
</svg>`;

    fs.writeFileSync(path.join(assetsDir, 'icon.svg'), iconSVG);
    
    // Note: In production, you would convert SVG to ICO, ICNS, and PNG formats
    console.log('📝 Note: Icon files created as placeholders. Convert to proper formats for production.');
  }

  copyBackendFiles() {
    console.log('📦 Copying backend files...');
    
    const backendSrc = path.join(PROJECT_ROOT, 'backend');
    const backendDest = path.join(ELECTRON_DIR, 'backend');
    
    if (fs.existsSync(backendSrc)) {
      this.copyDirectory(backendSrc, backendDest, {
        exclude: ['__pycache__', 'venv', 'node_modules', '.git', '*.pyc']
      });
    } else {
      console.warn('⚠️ Backend directory not found, creating placeholder...');
      fs.mkdirSync(backendDest, { recursive: true });
      
      // Create minimal backend placeholder
      const placeholderMain = `
#!/usr/bin/env python3
"""
AI IDE Backend Service
Placeholder implementation for Electron packaging
"""

import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

class AIIDEHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok'}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # Placeholder response
        response = {'success': True, 'data': {'response': 'AI IDE backend is running'}}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

if __name__ == '__main__':
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    server = HTTPServer(('127.0.0.1', port), AIIDEHandler)
    print(f'AI IDE Backend running on port {port}')
    server.serve_forever()
`;
      
      fs.writeFileSync(path.join(backendDest, 'main.py'), placeholderMain);
    }
  }

  copyDirectory(src, dest, options = {}) {
    const { exclude = [] } = options;
    
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }

    const items = fs.readdirSync(src);
    
    for (const item of items) {
      const srcPath = path.join(src, item);
      const destPath = path.join(dest, item);
      
      // Check if item should be excluded
      if (exclude.some(pattern => {
        if (pattern.includes('*')) {
          return item.match(new RegExp(pattern.replace('*', '.*')));
        }
        return item === pattern;
      })) {
        continue;
      }

      const stat = fs.statSync(srcPath);
      
      if (stat.isDirectory()) {
        this.copyDirectory(srcPath, destPath, options);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  async buildElectronApp() {
    console.log('🔨 Building Electron application...');
    
    try {
      // Build for current platform first
      const currentPlatform = process.platform;
      const currentArch = process.arch;
      
      console.log(`Building for ${currentPlatform}-${currentArch}...`);
      
      execSync(`npx electron-builder --${currentPlatform} --${currentArch}`, {
        cwd: ELECTRON_DIR,
        stdio: 'inherit'
      });
      
      console.log('✅ Electron build completed!');
      
    } catch (error) {
      console.error('❌ Electron build failed:', error.message);
      throw error;
    }
  }

  async buildAllPlatforms() {
    console.log('🌍 Building for all platforms...');
    
    const builds = [
      { platform: 'win32', arch: 'x64' },
      { platform: 'win32', arch: 'arm64' },
      { platform: 'darwin', arch: 'x64' },
      { platform: 'darwin', arch: 'arm64' },
      { platform: 'linux', arch: 'x64' },
      { platform: 'linux', arch: 'arm64' }
    ];

    for (const build of builds) {
      try {
        console.log(`Building ${build.platform}-${build.arch}...`);
        
        execSync(`npx electron-builder --${build.platform} --${build.arch}`, {
          cwd: ELECTRON_DIR,
          stdio: 'inherit'
        });
        
        console.log(`✅ ${build.platform}-${build.arch} build completed`);
        
      } catch (error) {
        console.error(`❌ ${build.platform}-${build.arch} build failed:`, error.message);
        // Continue with other builds
      }
    }
  }

  createPortableVersion() {
    console.log('📦 Creating portable version...');
    
    const portableDir = path.join(DIST_DIR, 'portable');
    fs.mkdirSync(portableDir, { recursive: true });

    // Copy the built application
    const builtAppDir = this.findBuiltApp();
    if (builtAppDir) {
      const portableAppDir = path.join(portableDir, 'AI-IDE-Portable');
      this.copyDirectory(builtAppDir, portableAppDir);
      
      // Create portable launcher script
      const launcherScript = process.platform === 'win32' ? 
        this.createWindowsLauncher(portableAppDir) :
        this.createUnixLauncher(portableAppDir);
      
      const launcherName = process.platform === 'win32' ? 'AI-IDE-Portable.bat' : 'AI-IDE-Portable.sh';
      fs.writeFileSync(path.join(portableDir, launcherName), launcherScript);
      
      if (process.platform !== 'win32') {
        // Make launcher executable on Unix systems
        execSync(`chmod +x "${path.join(portableDir, launcherName)}"`);
      }
      
      console.log('✅ Portable version created');
    } else {
      console.error('❌ Could not find built application for portable version');
    }
  }

  findBuiltApp() {
    const possiblePaths = [
      path.join(DIST_DIR, 'win-unpacked'),
      path.join(DIST_DIR, 'mac'),
      path.join(DIST_DIR, 'linux-unpacked')
    ];

    for (const appPath of possiblePaths) {
      if (fs.existsSync(appPath)) {
        return appPath;
      }
    }

    return null;
  }

  createWindowsLauncher(appDir) {
    return `@echo off
cd /d "%~dp0"
set AI_IDE_PORTABLE=1
set AI_IDE_DATA_DIR=%~dp0data
if not exist "%AI_IDE_DATA_DIR%" mkdir "%AI_IDE_DATA_DIR%"
start "" "AI-IDE-Portable\\AI IDE.exe"
`;
  }

  createUnixLauncher(appDir) {
    return `#!/bin/bash
cd "$(dirname "$0")"
export AI_IDE_PORTABLE=1
export AI_IDE_DATA_DIR="$(pwd)/data"
mkdir -p "$AI_IDE_DATA_DIR"
./AI-IDE-Portable/ai-ide
`;
  }

  createInstaller() {
    console.log('📦 Creating installer packages...');
    
    try {
      // Create NSIS installer for Windows
      if (process.platform === 'win32') {
        execSync('npx electron-builder --win --publish=never', {
          cwd: ELECTRON_DIR,
          stdio: 'inherit'
        });
      }
      
      // Create DMG for macOS
      if (process.platform === 'darwin') {
        execSync('npx electron-builder --mac --publish=never', {
          cwd: ELECTRON_DIR,
          stdio: 'inherit'
        });
      }
      
      // Create AppImage/DEB/RPM for Linux
      if (process.platform === 'linux') {
        execSync('npx electron-builder --linux --publish=never', {
          cwd: ELECTRON_DIR,
          stdio: 'inherit'
        });
      }
      
      console.log('✅ Installer packages created');
      
    } catch (error) {
      console.error('❌ Installer creation failed:', error.message);
    }
  }

  async build() {
    console.log('🚀 Starting AI IDE Electron build process...');
    
    try {
      // Create all necessary files
      this.createMainProcess();
      this.createPreloadScript();
      this.createRenderer();
      this.createAssets();
      this.copyBackendFiles();
      
      // Build the application
      await this.buildElectronApp();
      
      // Create portable version
      this.createPortableVersion();
      
      // Create installers
      this.createInstaller();
      
      console.log('🎉 AI IDE build completed successfully!');
      console.log(`📁 Output directory: ${DIST_DIR}`);
      
      // List created files
      this.listOutputFiles();
      
    } catch (error) {
      console.error('❌ Build failed:', error.message);
      process.exit(1);
    }
  }

  listOutputFiles() {
    console.log('\n📋 Created files:');
    
    if (fs.existsSync(DIST_DIR)) {
      const files = fs.readdirSync(DIST_DIR);
      files.forEach(file => {
        const filePath = path.join(DIST_DIR, file);
        const stat = fs.statSync(filePath);
        const size = stat.isFile() ? `(${(stat.size / 1024 / 1024).toFixed(1)} MB)` : '';
        console.log(`  📄 ${file} ${size}`);
      });
    }
  }
}

// Main execution
if (require.main === module) {
  const builder = new ElectronBuilder();
  
  // Parse command line arguments
  const args = process.argv.slice(2);
  const buildAll = args.includes('--all-platforms');
  
  if (buildAll) {
    builder.buildAllPlatforms().catch(console.error);
  } else {
    builder.build().catch(console.error);
  }
}

module.exports = ElectronBuilder;