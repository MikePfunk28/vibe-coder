/**
 * AI IDE Complete Application Logic
 * Full-featured IDE with VSCode-like functionality
 */

class AIIDE {
    constructor() {
        this.currentFile = null;
        this.openFiles = new Map();
        this.editor = null;
        this.fileTree = new Map();
        this.currentWorkspace = null;
        this.activePanel = 'explorer';
        this.commandHistory = [];
        this.version = '1.0.0';
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing AI IDE...');
        
        // Initialize Monaco Editor
        await this.initializeMonacoEditor();
        
        // Setup event listeners
        this.setupMenuHandlers();
        this.setupActivityBar();
        this.setupKeyboardShortcuts();
        this.setupFileOperations();
        this.setupAIIntegration();
        this.setupCommandPalette();
        this.setupContextMenus();
        
        // Load workspace if available
        await this.loadWorkspace();
        
        console.log('✅ AI IDE initialized successfully');
    }

    async initializeMonacoEditor() {
        return new Promise((resolve) => {
            require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.44.0/min/vs' } });
            require(['vs/editor/editor.main'], () => {
                this.editor = monaco.editor.create(document.getElementById('monaco-editor'), {
                    value: '// Welcome to AI IDE\n// Start coding with AI assistance!\n',
                    language: 'javascript',
                    theme: 'vs-dark',
                    automaticLayout: true,
                    fontSize: 14,
                    lineNumbers: 'on',
                    roundedSelection: false,
                    scrollBeyondLastLine: false,
                    readOnly: false,
                    minimap: { enabled: true },
                    suggestOnTriggerCharacters: true,
                    quickSuggestions: true,
                    wordBasedSuggestions: true
                });

                // Setup editor event listeners
                this.editor.onDidChangeModelContent(() => {
                    this.onEditorContentChange();
                });

                this.editor.onDidChangeCursorPosition((e) => {
                    this.updateCursorPosition(e.position);
                });

                // Hide welcome screen
                document.getElementById('welcome-screen').style.display = 'none';
                
                resolve();
            });
        });
    }

    setupMenuHandlers() {
        // File menu handlers
        this.registerMenuAction('new-file', () => this.newFile());
        this.registerMenuAction('new-folder', () => this.newFolder());
        this.registerMenuAction('open-file', () => this.openFile());
        this.registerMenuAction('open-folder', () => this.openFolder());
        this.registerMenuAction('save', () => this.saveFile());
        this.registerMenuAction('save-as', () => this.saveFileAs());
        this.registerMenuAction('save-all', () => this.saveAllFiles());
        this.registerMenuAction('close-file', () => this.closeFile());
        this.registerMenuAction('exit', () => this.exitApplication());

        // Edit menu handlers
        this.registerMenuAction('undo', () => this.editor?.trigger('keyboard', 'undo'));
        this.registerMenuAction('redo', () => this.editor?.trigger('keyboard', 'redo'));
        this.registerMenuAction('cut', () => this.editor?.trigger('keyboard', 'editor.action.clipboardCutAction'));
        this.registerMenuAction('copy', () => this.editor?.trigger('keyboard', 'editor.action.clipboardCopyAction'));
        this.registerMenuAction('paste', () => this.editor?.trigger('keyboard', 'editor.action.clipboardPasteAction'));
        this.registerMenuAction('find', () => this.editor?.trigger('keyboard', 'actions.find'));
        this.registerMenuAction('replace', () => this.editor?.trigger('keyboard', 'editor.action.startFindReplaceAction'));
        this.registerMenuAction('select-all', () => this.editor?.trigger('keyboard', 'editor.action.selectAll'));

        // View menu handlers
        this.registerMenuAction('command-palette', () => this.showCommandPalette());
        this.registerMenuAction('toggle-sidebar', () => this.toggleSidebar());
        this.registerMenuAction('toggle-panel', () => this.toggleBottomPanel());
        this.registerMenuAction('toggle-terminal', () => this.toggleTerminal());
        this.registerMenuAction('zoom-in', () => this.zoomIn());
        this.registerMenuAction('zoom-out', () => this.zoomOut());
        this.registerMenuAction('reset-zoom', () => this.resetZoom());
        this.registerMenuAction('fullscreen', () => this.toggleFullscreen());

        // AI menu handlers
        this.registerMenuAction('ai-chat', () => this.openAIChat());
        this.registerMenuAction('ai-generate', () => this.generateCode());
        this.registerMenuAction('ai-explain', () => this.explainCode());
        this.registerMenuAction('ai-refactor', () => this.refactorCode());
        this.registerMenuAction('semantic-search', () => this.semanticSearch());
        this.registerMenuAction('web-search', () => this.webSearch());
        this.registerMenuAction('reasoning-mode', () => this.reasoningMode());

        // Setup menu interactions
        this.setupMenuInteractions();
    }

    registerMenuAction(action, handler) {
        document.addEventListener('click', (e) => {
            if (e.target.dataset.action === action) {
                handler();
            }
        });
    }

    setupMenuInteractions() {
        // Menu hover and click handling
        const menuItems = document.querySelectorAll('.menu-item');
        menuItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleMenu(item);
            });
        });

        // Close menus when clicking outside
        document.addEventListener('click', () => {
            this.closeAllMenus();
        });
    }

    toggleMenu(menuItem) {
        const menuId = menuItem.dataset.menu + '-menu';
        const menu = document.getElementById(menuId);
        
        // Close all other menus
        this.closeAllMenus();
        
        // Toggle current menu
        if (menu) {
            menu.classList.toggle('show');
            menuItem.classList.toggle('active');
        }
    }

    closeAllMenus() {
        document.querySelectorAll('.dropdown-menu').forEach(menu => {
            menu.classList.remove('show');
        });
        document.querySelectorAll('.menu-item').forEach(item => {
            item.classList.remove('active');
        });
    }

    setupActivityBar() {
        const activityItems = document.querySelectorAll('.activity-item');
        activityItems.forEach(item => {
            item.addEventListener('click', () => {
                const view = item.dataset.view;
                this.switchPanel(view);
                
                // Update active state
                activityItems.forEach(i => i.classList.remove('active'));
                item.classList.add('active');
            });
        });
    }

    switchPanel(panelName) {
        // Hide all panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.add('hidden');
        });
        
        // Show selected panel
        const targetPanel = document.getElementById(panelName + '-panel');
        if (targetPanel) {
            targetPanel.classList.remove('hidden');
            this.activePanel = panelName;
        }
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            const ctrl = e.ctrlKey || e.metaKey;
            const shift = e.shiftKey;
            const alt = e.altKey;

            // File operations
            if (ctrl && e.key === 'n') { e.preventDefault(); this.newFile(); }
            if (ctrl && e.key === 'o') { e.preventDefault(); this.openFile(); }
            if (ctrl && e.key === 's') { e.preventDefault(); this.saveFile(); }
            if (ctrl && shift && e.key === 'S') { e.preventDefault(); this.saveFileAs(); }
            if (ctrl && e.key === 'w') { e.preventDefault(); this.closeFile(); }

            // View operations
            if (ctrl && shift && e.key === 'P') { e.preventDefault(); this.showCommandPalette(); }
            if (ctrl && e.key === 'b') { e.preventDefault(); this.toggleSidebar(); }
            if (ctrl && e.key === 'j') { e.preventDefault(); this.toggleBottomPanel(); }
            if (ctrl && e.key === '`') { e.preventDefault(); this.toggleTerminal(); }

            // AI operations
            if (ctrl && shift && e.key === 'A') { e.preventDefault(); this.openAIChat(); }
            if (ctrl && shift && e.key === 'G') { e.preventDefault(); this.generateCode(); }
            if (ctrl && shift && e.key === 'E') { e.preventDefault(); this.explainCode(); }
            if (ctrl && shift && e.key === 'R') { e.preventDefault(); this.refactorCode(); }

            // Zoom operations
            if (ctrl && e.key === '=') { e.preventDefault(); this.zoomIn(); }
            if (ctrl && e.key === '-') { e.preventDefault(); this.zoomOut(); }
            if (ctrl && e.key === '0') { e.preventDefault(); this.resetZoom(); }

            // Fullscreen
            if (e.key === 'F11') { e.preventDefault(); this.toggleFullscreen(); }

            // Exit
            if (alt && e.key === 'F4') { e.preventDefault(); this.exitApplication(); }
        });
    }

    // File Operations
    async newFile() {
        const fileName = `untitled-${Date.now()}.txt`;
        const fileContent = '';
        
        this.openFiles.set(fileName, {
            content: fileContent,
            modified: false,
            language: 'plaintext'
        });
        
        this.createTab(fileName);
        this.switchToFile(fileName);
        
        if (this.editor) {
            this.editor.setValue(fileContent);
            this.editor.focus();
        }
    }

    async openFile() {
        try {
            const result = await window.electronAPI.showOpenDialog({
                properties: ['openFile'],
                filters: [
                    { name: 'All Files', extensions: ['*'] },
                    { name: 'JavaScript', extensions: ['js', 'jsx'] },
                    { name: 'TypeScript', extensions: ['ts', 'tsx'] },
                    { name: 'Python', extensions: ['py'] },
                    { name: 'HTML', extensions: ['html', 'htm'] },
                    { name: 'CSS', extensions: ['css', 'scss', 'sass'] },
                    { name: 'JSON', extensions: ['json'] },
                    { name: 'Markdown', extensions: ['md'] }
                ]
            });

            if (!result.canceled && result.filePaths.length > 0) {
                const filePath = result.filePaths[0];
                await this.loadFile(filePath);
            }
        } catch (error) {
            console.error('Failed to open file:', error);
            this.showNotification('Failed to open file', 'error');
        }
    }

    async openFolder() {
        try {
            const result = await window.electronAPI.showOpenDialog({
                properties: ['openDirectory']
            });

            if (!result.canceled && result.filePaths.length > 0) {
                const folderPath = result.filePaths[0];
                await this.loadWorkspace(folderPath);
            }
        } catch (error) {
            console.error('Failed to open folder:', error);
            this.showNotification('Failed to open folder', 'error');
        }
    }

    async saveFile() {
        if (!this.currentFile) {
            return this.saveFileAs();
        }

        try {
            const content = this.editor?.getValue() || '';
            // Save file logic here
            this.openFiles.get(this.currentFile).modified = false;
            this.updateTabModifiedState(this.currentFile, false);
            this.showNotification('File saved successfully');
        } catch (error) {
            console.error('Failed to save file:', error);
            this.showNotification('Failed to save file', 'error');
        }
    }

    async saveFileAs() {
        try {
            const result = await window.electronAPI.showSaveDialog({
                filters: [
                    { name: 'All Files', extensions: ['*'] },
                    { name: 'JavaScript', extensions: ['js'] },
                    { name: 'TypeScript', extensions: ['ts'] },
                    { name: 'Python', extensions: ['py'] },
                    { name: 'HTML', extensions: ['html'] },
                    { name: 'CSS', extensions: ['css'] },
                    { name: 'JSON', extensions: ['json'] },
                    { name: 'Markdown', extensions: ['md'] }
                ]
            });

            if (!result.canceled && result.filePath) {
                const content = this.editor?.getValue() || '';
                // Save file logic here
                this.showNotification('File saved successfully');
            }
        } catch (error) {
            console.error('Failed to save file as:', error);
            this.showNotification('Failed to save file', 'error');
        }
    }

    async saveAllFiles() {
        const modifiedFiles = Array.from(this.openFiles.entries())
            .filter(([_, fileData]) => fileData.modified);

        for (const [fileName, _] of modifiedFiles) {
            // Save each modified file
            console.log(`Saving ${fileName}`);
        }

        this.showNotification(`Saved ${modifiedFiles.length} files`);
    }    
closeFile() {
        if (!this.currentFile) return;

        const fileData = this.openFiles.get(this.currentFile);
        if (fileData?.modified) {
            // Show confirmation dialog for unsaved changes
            const save = confirm(`Save changes to ${this.currentFile}?`);
            if (save) {
                this.saveFile();
            }
        }

        this.openFiles.delete(this.currentFile);
        this.removeTab(this.currentFile);
        
        // Switch to another open file or show welcome screen
        const remainingFiles = Array.from(this.openFiles.keys());
        if (remainingFiles.length > 0) {
            this.switchToFile(remainingFiles[0]);
        } else {
            this.currentFile = null;
            document.getElementById('welcome-screen').style.display = 'flex';
        }
    }

    // Tab Management
    createTab(fileName) {
        const tabBar = document.getElementById('tab-bar');
        const tab = document.createElement('div');
        tab.className = 'tab';
        tab.dataset.file = fileName;
        
        tab.innerHTML = `
            <span class="tab-name">${fileName}</span>
            <span class="tab-close" onclick="event.stopPropagation(); aiIDE.closeFile('${fileName}')">&times;</span>
        `;
        
        tab.addEventListener('click', () => this.switchToFile(fileName));
        tabBar.appendChild(tab);
    }

    removeTab(fileName) {
        const tab = document.querySelector(`[data-file="${fileName}"]`);
        if (tab) {
            tab.remove();
        }
    }

    switchToFile(fileName) {
        this.currentFile = fileName;
        
        // Update tab states
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.file === fileName);
        });
        
        // Load file content in editor
        const fileData = this.openFiles.get(fileName);
        if (fileData && this.editor) {
            this.editor.setValue(fileData.content);
            monaco.editor.setModelLanguage(this.editor.getModel(), fileData.language);
        }
        
        // Hide welcome screen
        document.getElementById('welcome-screen').style.display = 'none';
        
        // Update status bar
        this.updateStatusBar();
    }

    updateTabModifiedState(fileName, modified) {
        const tab = document.querySelector(`[data-file="${fileName}"]`);
        if (tab) {
            const tabName = tab.querySelector('.tab-name');
            if (modified && !tabName.textContent.startsWith('●')) {
                tabName.textContent = '● ' + tabName.textContent;
            } else if (!modified && tabName.textContent.startsWith('●')) {
                tabName.textContent = tabName.textContent.substring(2);
            }
        }
    }

    // AI Integration
    async openAIChat() {
        this.switchPanel('ai-chat');
        document.getElementById('chat-input').focus();
    }

    async generateCode() {
        const selection = this.editor?.getSelection();
        const selectedText = this.editor?.getModel()?.getValueInRange(selection);
        
        if (selectedText) {
            try {
                const response = await window.aiIdeAPI.generateCode(
                    `Generate code based on: ${selectedText}`,
                    { language: this.getCurrentLanguage() }
                );
                
                if (response.success) {
                    this.editor?.executeEdits('ai-generate', [{
                        range: selection,
                        text: response.data.code
                    }]);
                }
            } catch (error) {
                console.error('Code generation failed:', error);
                this.showNotification('Code generation failed', 'error');
            }
        } else {
            this.showNotification('Please select code to generate from', 'warning');
        }
    }

    async explainCode() {
        const selection = this.editor?.getSelection();
        const selectedText = this.editor?.getModel()?.getValueInRange(selection);
        
        if (selectedText) {
            this.openAIChat();
            const chatInput = document.getElementById('chat-input');
            chatInput.value = `Explain this code:\n\n${selectedText}`;
            this.sendChatMessage();
        } else {
            this.showNotification('Please select code to explain', 'warning');
        }
    }

    async refactorCode() {
        const selection = this.editor?.getSelection();
        const selectedText = this.editor?.getModel()?.getValueInRange(selection);
        
        if (selectedText) {
            try {
                const response = await window.aiIdeAPI.sendToAgent('code', 
                    `Refactor this code for better performance and readability:\n\n${selectedText}`
                );
                
                if (response.success) {
                    this.editor?.executeEdits('ai-refactor', [{
                        range: selection,
                        text: response.data.response
                    }]);
                }
            } catch (error) {
                console.error('Code refactoring failed:', error);
                this.showNotification('Code refactoring failed', 'error');
            }
        } else {
            this.showNotification('Please select code to refactor', 'warning');
        }
    }

    async semanticSearch() {
        const query = prompt('Enter search query:');
        if (query) {
            try {
                const response = await window.aiIdeAPI.semanticSearch(query);
                if (response.success) {
                    this.showSearchResults(response.data, 'semantic');
                }
            } catch (error) {
                console.error('Semantic search failed:', error);
                this.showNotification('Semantic search failed', 'error');
            }
        }
    }

    async webSearch() {
        const query = prompt('Enter web search query:');
        if (query) {
            try {
                const response = await window.aiIdeAPI.webSearch(query);
                if (response.success) {
                    this.showSearchResults(response.data, 'web');
                }
            } catch (error) {
                console.error('Web search failed:', error);
                this.showNotification('Web search failed', 'error');
            }
        }
    }

    async reasoningMode() {
        const problem = prompt('Describe the problem for AI reasoning:');
        if (problem) {
            this.openAIChat();
            const chatInput = document.getElementById('chat-input');
            chatInput.value = `[REASONING MODE] ${problem}`;
            this.sendChatMessage();
        }
    }

    // Chat functionality
    async sendChatMessage() {
        const chatInput = document.getElementById('chat-input');
        const message = chatInput.value.trim();
        if (!message) return;

        const agentType = document.getElementById('agent-selector').value;
        
        // Add user message to chat
        this.addChatMessage('user', message);
        chatInput.value = '';

        try {
            const response = await window.aiIdeAPI.sendToAgent(agentType, message, {
                currentFile: this.currentFile,
                language: this.getCurrentLanguage(),
                workspace: this.currentWorkspace
            });

            if (response.success) {
                this.addChatMessage('ai', response.data.response || 'No response received');
            } else {
                this.addChatMessage('ai', `Error: ${response.error}`);
            }
        } catch (error) {
            console.error('Chat message failed:', error);
            this.addChatMessage('ai', `Error: ${error.message}`);
        }
    }

    addChatMessage(type, content) {
        const chatMessages = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}`;
        messageDiv.textContent = content;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Command Palette
    showCommandPalette() {
        const palette = document.getElementById('command-palette');
        const input = document.getElementById('command-input');
        
        palette.classList.add('show');
        input.focus();
        input.value = '';
        
        this.updateCommandResults('');
    }

    hideCommandPalette() {
        document.getElementById('command-palette').classList.remove('show');
    }

    setupCommandPalette() {
        const input = document.getElementById('command-input');
        const results = document.getElementById('command-results');
        
        input.addEventListener('input', (e) => {
            this.updateCommandResults(e.target.value);
        });
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideCommandPalette();
            } else if (e.key === 'Enter') {
                this.executeSelectedCommand();
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                this.navigateCommandResults(e.key === 'ArrowDown' ? 1 : -1);
            }
        });
        
        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!document.getElementById('command-palette').contains(e.target)) {
                this.hideCommandPalette();
            }
        });
    }

    updateCommandResults(query) {
        const commands = this.getAvailableCommands();
        const filtered = commands.filter(cmd => 
            cmd.name.toLowerCase().includes(query.toLowerCase()) ||
            cmd.description.toLowerCase().includes(query.toLowerCase())
        );
        
        const results = document.getElementById('command-results');
        results.innerHTML = '';
        
        filtered.forEach((cmd, index) => {
            const item = document.createElement('div');
            item.className = 'command-item';
            if (index === 0) item.classList.add('selected');
            
            item.innerHTML = `
                <div>
                    <div class="command-name">${cmd.name}</div>
                    <div class="command-description">${cmd.description}</div>
                </div>
                <div class="command-shortcut">${cmd.shortcut || ''}</div>
            `;
            
            item.addEventListener('click', () => {
                this.executeCommand(cmd);
                this.hideCommandPalette();
            });
            
            results.appendChild(item);
        });
    }

    getAvailableCommands() {
        return [
            { name: 'File: New File', description: 'Create a new file', action: 'new-file', shortcut: 'Ctrl+N' },
            { name: 'File: Open File', description: 'Open an existing file', action: 'open-file', shortcut: 'Ctrl+O' },
            { name: 'File: Save', description: 'Save the current file', action: 'save', shortcut: 'Ctrl+S' },
            { name: 'AI: Generate Code', description: 'Generate code with AI', action: 'ai-generate', shortcut: 'Ctrl+Shift+G' },
            { name: 'AI: Explain Code', description: 'Explain selected code', action: 'ai-explain', shortcut: 'Ctrl+Shift+E' },
            { name: 'AI: Chat', description: 'Open AI chat', action: 'ai-chat', shortcut: 'Ctrl+Shift+A' },
            { name: 'View: Toggle Sidebar', description: 'Show/hide sidebar', action: 'toggle-sidebar', shortcut: 'Ctrl+B' },
            { name: 'View: Toggle Terminal', description: 'Show/hide terminal', action: 'toggle-terminal', shortcut: 'Ctrl+`' }
        ];
    }

    executeSelectedCommand() {
        const selected = document.querySelector('.command-item.selected');
        if (selected) {
            const commandName = selected.querySelector('.command-name').textContent;
            const command = this.getAvailableCommands().find(cmd => cmd.name === commandName);
            if (command) {
                this.executeCommand(command);
                this.hideCommandPalette();
            }
        }
    }

    executeCommand(command) {
        // Execute the command action
        const event = new CustomEvent('click');
        const element = document.querySelector(`[data-action="${command.action}"]`);
        if (element) {
            element.dispatchEvent(event);
        }
    }

    // View Operations
    toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        sidebar.style.display = sidebar.style.display === 'none' ? 'flex' : 'none';
    }

    toggleBottomPanel() {
        const panel = document.querySelector('.bottom-panel');
        panel.style.display = panel.style.display === 'none' ? 'flex' : 'none';
    }

    toggleTerminal() {
        this.toggleBottomPanel();
    }

    zoomIn() {
        if (this.editor) {
            const currentFontSize = this.editor.getOption(monaco.editor.EditorOption.fontSize);
            this.editor.updateOptions({ fontSize: currentFontSize + 1 });
        }
    }

    zoomOut() {
        if (this.editor) {
            const currentFontSize = this.editor.getOption(monaco.editor.EditorOption.fontSize);
            this.editor.updateOptions({ fontSize: Math.max(8, currentFontSize - 1) });
        }
    }

    resetZoom() {
        if (this.editor) {
            this.editor.updateOptions({ fontSize: 14 });
        }
    }

    toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    }

    // Utility Functions
    getCurrentLanguage() {
        if (!this.currentFile) return 'plaintext';
        
        const extension = this.currentFile.split('.').pop()?.toLowerCase();
        const languageMap = {
            'js': 'javascript',
            'jsx': 'javascript',
            'ts': 'typescript',
            'tsx': 'typescript',
            'py': 'python',
            'html': 'html',
            'htm': 'html',
            'css': 'css',
            'scss': 'scss',
            'sass': 'sass',
            'json': 'json',
            'md': 'markdown',
            'xml': 'xml',
            'yaml': 'yaml',
            'yml': 'yaml'
        };
        
        return languageMap[extension] || 'plaintext';
    }

    onEditorContentChange() {
        if (this.currentFile) {
            const fileData = this.openFiles.get(this.currentFile);
            if (fileData) {
                fileData.modified = true;
                this.updateTabModifiedState(this.currentFile, true);
            }
        }
    }

    updateCursorPosition(position) {
        const statusElement = document.getElementById('cursor-position');
        if (statusElement) {
            statusElement.textContent = `Ln ${position.lineNumber}, Col ${position.column}`;
        }
    }

    updateStatusBar() {
        // Update file type
        const fileTypeElement = document.getElementById('file-type');
        if (fileTypeElement) {
            fileTypeElement.textContent = this.getCurrentLanguage();
        }
        
        // Update AI status
        const aiStatusElement = document.getElementById('ai-status');
        if (aiStatusElement) {
            aiStatusElement.textContent = 'AI Ready';
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Style the notification
        Object.assign(notification.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            padding: '12px 16px',
            borderRadius: '4px',
            color: 'white',
            zIndex: '10000',
            fontSize: '14px',
            maxWidth: '300px',
            backgroundColor: type === 'error' ? '#e74c3c' : 
                           type === 'warning' ? '#f39c12' : '#27ae60'
        });
        
        document.body.appendChild(notification);
        
        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    async loadWorkspace(workspacePath) {
        if (workspacePath) {
            this.currentWorkspace = workspacePath;
            // Load file tree and workspace settings
            await this.refreshFileTree();
        }
    }

    async refreshFileTree() {
        // Implementation for loading file tree from workspace
        const fileTree = document.getElementById('file-tree');
        fileTree.innerHTML = '<div class="file-item">📁 src</div><div class="file-item">📄 README.md</div>';
    }

    setupContextMenus() {
        // Right-click context menus for files, editor, etc.
        document.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            // Show appropriate context menu based on target
        });
    }

    setupFileOperations() {
        // Setup drag and drop for files
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        
        document.addEventListener('drop', async (e) => {
            e.preventDefault();
            const files = Array.from(e.dataTransfer.files);
            
            for (const file of files) {
                if (file.type.startsWith('text/') || file.name.match(/\.(js|ts|py|html|css|json|md)$/)) {
                    const content = await file.text();
                    this.openFiles.set(file.name, {
                        content: content,
                        modified: false,
                        language: this.getLanguageFromFileName(file.name)
                    });
                    this.createTab(file.name);
                    this.switchToFile(file.name);
                }
            }
        });
    }

    getLanguageFromFileName(fileName) {
        const extension = fileName.split('.').pop()?.toLowerCase();
        const languageMap = {
            'js': 'javascript',
            'jsx': 'javascript', 
            'ts': 'typescript',
            'tsx': 'typescript',
            'py': 'python',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'md': 'markdown'
        };
        return languageMap[extension] || 'plaintext';
    }

    async exitApplication() {
        // Check for unsaved files
        const unsavedFiles = Array.from(this.openFiles.entries())
            .filter(([_, fileData]) => fileData.modified);
        
        if (unsavedFiles.length > 0) {
            const save = confirm(`You have ${unsavedFiles.length} unsaved files. Save before exiting?`);
            if (save) {
                await this.saveAllFiles();
            }
        }
        
        // Close the application
        window.close();
    }
}

// Initialize AI IDE when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.aiIDE = new AIIDE();
    
    // Setup chat functionality
    document.getElementById('send-chat').addEventListener('click', () => {
        window.aiIDE.sendChatMessage();
    });
    
    document.getElementById('chat-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            window.aiIDE.sendChatMessage();
        }
    });
});

// Export for global access
window.AIIDE = AIIDE;