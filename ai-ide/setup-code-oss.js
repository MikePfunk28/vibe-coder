#!/usr/bin/env node

/**
 * AI IDE Code OSS Setup Script
 * Downloads and configures VSCode OSS as the base, just like Cursor
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const https = require('https');

class CodeOSSSetup {
    constructor() {
        this.projectRoot = __dirname;
        this.codeOSSDir = path.join(this.projectRoot, 'code-oss');
        this.aiIdeDir = path.join(this.projectRoot, 'ai-ide-build');
        this.version = '1.85.0'; // Latest stable VSCode version
    }

    async setup() {
        console.log('🚀 Setting up AI IDE with Code OSS base...');
        console.log('📦 This will create a complete VSCode-based IDE like Cursor');
        console.log('');

        try {
            // Step 1: Clone VSCode OSS
            await this.cloneCodeOSS();
            
            // Step 2: Configure for AI IDE
            await this.configureAIIDE();
            
            // Step 3: Add AI extensions
            await this.addAIExtensions();
            
            // Step 4: Integrate backend
            await this.integrateBackend();
            
            // Step 5: Setup build system
            await this.setupBuildSystem();
            
            // Step 6: Create branding
            await this.createBranding();
            
            console.log('🎉 AI IDE setup completed successfully!');
            console.log('📁 Your VSCode-based AI IDE is ready in:', this.aiIdeDir);
            
        } catch (error) {
            console.error('❌ Setup failed:', error.message);
            throw error;
        }
    }

    async cloneCodeOSS() {
        console.log('📥 Cloning VSCode OSS repository...');
        
        if (fs.existsSync(this.codeOSSDir)) {
            console.log('🔄 Code OSS already exists, updating...');
            execSync('git pull', { cwd: this.codeOSSDir, stdio: 'inherit' });
        } else {
            console.log('⬇️ Downloading VSCode OSS...');
            execSync(`git clone --depth 1 --branch ${this.version} https://github.com/microsoft/vscode.git ${this.codeOSSDir}`, {
                stdio: 'inherit'
            });
        }
        
        console.log('✅ VSCode OSS downloaded successfully');
    }

    async configureAIIDE() {
        console.log('🔧 Configuring AI IDE based on Code OSS...');
        
        // Copy Code OSS to our build directory
        if (fs.existsSync(this.aiIdeDir)) {
            fs.rmSync(this.aiIdeDir, { recursive: true, force: true });
        }
        
        console.log('📁 Copying Code OSS to AI IDE build directory...');
        this.copyDirectory(this.codeOSSDir, this.aiIdeDir, {
            exclude: ['.git', 'node_modules', 'out', '.build']
        });
        
        // Update package.json for AI IDE
        await this.updatePackageJson();
        
        // Update product.json for branding
        await this.updateProductJson();
        
        console.log('✅ AI IDE configuration completed');
    }

    async updatePackageJson() {
        const packageJsonPath = path.join(this.aiIdeDir, 'package.json');
        const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
        
        // Update for AI IDE
        packageJson.name = 'ai-ide';
        packageJson.displayName = 'AI IDE';
        packageJson.description = 'Advanced AI-Powered Development Environment based on VSCode OSS';
        packageJson.version = '1.0.0';
        packageJson.author = 'AI IDE Team';
        packageJson.homepage = 'https://github.com/your-username/ai-ide';
        
        // Add AI-specific dependencies
        packageJson.dependencies = {
            ...packageJson.dependencies,
            'playwright': '^1.40.0',
            'axios': '^1.6.2',
            'ws': '^8.14.2'
        };
        
        // Add AI-specific scripts
        packageJson.scripts = {
            ...packageJson.scripts,
            'build:ai-ide': 'node scripts/build-ai-ide.js',
            'package:ai-ide': 'node scripts/package-ai-ide.js',
            'dev:ai-ide': 'yarn watch & node backend/main.py'
        };
        
        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
        console.log('✅ Updated package.json for AI IDE');
    }

    async updateProductJson() {
        const productJsonPath = path.join(this.aiIdeDir, 'product.json');
        const productJson = JSON.parse(fs.readFileSync(productJsonPath, 'utf8'));
        
        // Update branding for AI IDE
        productJson.nameShort = 'AI IDE';
        productJson.nameLong = 'AI IDE - Advanced AI-Powered Development Environment';
        productJson.applicationName = 'ai-ide';
        productJson.dataFolderName = '.ai-ide';
        productJson.serverDataFolderName = '.ai-ide-server';
        productJson.downloadUrl = 'https://github.com/your-username/ai-ide';
        productJson.updateUrl = 'https://github.com/your-username/ai-ide/releases';
        productJson.webUrl = 'https://ai-ide.dev';
        productJson.documentationUrl = 'https://docs.ai-ide.dev';
        productJson.keyboardShortcutsUrlMac = 'https://docs.ai-ide.dev/shortcuts/mac';
        productJson.keyboardShortcutsUrlLinux = 'https://docs.ai-ide.dev/shortcuts/linux';
        productJson.keyboardShortcutsUrlWin = 'https://docs.ai-ide.dev/shortcuts/windows';
        
        // Remove Microsoft-specific URLs and telemetry
        delete productJson.telemetryOptOutUrl;
        delete productJson.privacyStatementUrl;
        delete productJson.licenseUrl;
        delete productJson.aiConfig;
        delete productJson.msftInternalDomains;
        
        // Add AI IDE specific configuration
        productJson.aiIdeConfig = {
            enableAI: true,
            enableWebSearch: true,
            enablePlaywright: true,
            enableMultiAgent: true,
            enableSemanticSearch: true,
            enableRAG: true,
            defaultAIProvider: 'lm-studio',
            backendPort: 8000
        };
        
        // Update extension gallery to allow all extensions
        productJson.extensionsGallery = {
            serviceUrl: 'https://marketplace.visualstudio.com/_apis/public/gallery',
            cacheUrl: 'https://vscode.blob.core.windows.net/gallery/index',
            itemUrl: 'https://marketplace.visualstudio.com/items',
            controlUrl: '',
            recommendationsUrl: ''
        };
        
        fs.writeFileSync(productJsonPath, JSON.stringify(productJson, null, 2));
        console.log('✅ Updated product.json for AI IDE branding');
    }

    async addAIExtensions() {
        console.log('🤖 Adding AI extensions to Code OSS...');
        
        const extensionsDir = path.join(this.aiIdeDir, 'extensions');
        const aiExtensionsDir = path.join(extensionsDir, 'ai-assistant');
        
        // Copy our AI assistant extension
        const sourceExtensionDir = path.join(this.projectRoot, 'extensions', 'ai-assistant');
        if (fs.existsSync(sourceExtensionDir)) {
            this.copyDirectory(sourceExtensionDir, aiExtensionsDir);
            console.log('✅ AI Assistant extension added');
        }
        
        // Create AI-specific built-in extensions
        await this.createBuiltInAIExtensions(extensionsDir);
        
        console.log('✅ AI extensions integrated successfully');
    }

    async createBuiltInAIExtensions(extensionsDir) {
        // AI Chat Extension (built-in)
        const aiChatDir = path.join(extensionsDir, 'ai-chat');
        fs.mkdirSync(aiChatDir, { recursive: true });
        
        const aiChatPackage = {
            name: 'ai-chat',
            displayName: 'AI Chat',
            description: 'Built-in AI chat functionality',
            version: '1.0.0',
            publisher: 'ai-ide',
            engines: { vscode: '^1.85.0' },
            categories: ['Other'],
            activationEvents: ['*'],
            main: './out/extension.js',
            contributes: {
                commands: [
                    {
                        command: 'ai-chat.open',
                        title: 'Open AI Chat',
                        category: 'AI'
                    }
                ],
                views: {
                    explorer: [
                        {
                            id: 'ai-chat',
                            name: 'AI Assistant',
                            when: 'true'
                        }
                    ]
                },
                keybindings: [
                    {
                        command: 'ai-chat.open',
                        key: 'ctrl+shift+a',
                        mac: 'cmd+shift+a'
                    }
                ]
            }
        };
        
        fs.writeFileSync(path.join(aiChatDir, 'package.json'), JSON.stringify(aiChatPackage, null, 2));
        
        // AI Semantic Search Extension (built-in)
        const aiSearchDir = path.join(extensionsDir, 'ai-semantic-search');
        fs.mkdirSync(aiSearchDir, { recursive: true });
        
        const aiSearchPackage = {
            name: 'ai-semantic-search',
            displayName: 'AI Semantic Search',
            description: 'Built-in semantic code search',
            version: '1.0.0',
            publisher: 'ai-ide',
            engines: { vscode: '^1.85.0' },
            categories: ['Other'],
            activationEvents: ['*'],
            main: './out/extension.js',
            contributes: {
                commands: [
                    {
                        command: 'ai-search.semantic',
                        title: 'Semantic Search',
                        category: 'AI'
                    }
                ],
                keybindings: [
                    {
                        command: 'ai-search.semantic',
                        key: 'ctrl+shift+f',
                        mac: 'cmd+shift+f'
                    }
                ]
            }
        };
        
        fs.writeFileSync(path.join(aiSearchDir, 'package.json'), JSON.stringify(aiSearchPackage, null, 2));
        
        console.log('✅ Built-in AI extensions created');
    }

    async integrateBackend() {
        console.log('🐍 Integrating AI backend with Code OSS...');
        
        const backendDir = path.join(this.aiIdeDir, 'ai-backend');
        const sourceBackendDir = path.join(this.projectRoot, 'backend');
        
        // Copy backend to Code OSS
        if (fs.existsSync(sourceBackendDir)) {
            this.copyDirectory(sourceBackendDir, backendDir);
        }
        
        // Create backend integration script
        const backendIntegrationScript = `
const { spawn } = require('child_process');
const path = require('path');

class AIBackendIntegration {
    constructor() {
        this.backendProcess = null;
        this.isRunning = false;
    }

    async start() {
        if (this.isRunning) return;
        
        const backendPath = path.join(__dirname, '..', 'ai-backend');
        const pythonExecutable = this.findPython();
        
        if (!pythonExecutable) {
            console.error('Python not found for AI backend');
            return;
        }
        
        console.log('🚀 Starting AI backend...');
        
        this.backendProcess = spawn(pythonExecutable, [
            path.join(backendPath, 'main.py'),
            '--port', '8000'
        ], {
            cwd: backendPath,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        this.backendProcess.stdout.on('data', (data) => {
            console.log('AI Backend:', data.toString());
        });
        
        this.backendProcess.stderr.on('data', (data) => {
            console.error('AI Backend Error:', data.toString());
        });
        
        this.isRunning = true;
        console.log('✅ AI backend started successfully');
    }
    
    findPython() {
        const candidates = ['python3', 'python'];
        for (const candidate of candidates) {
            try {
                require('child_process').execSync(\`\${candidate} --version\`, { stdio: 'ignore' });
                return candidate;
            } catch (error) {
                continue;
            }
        }
        return null;
    }
    
    async stop() {
        if (this.backendProcess && !this.backendProcess.killed) {
            this.backendProcess.kill();
            this.isRunning = false;
            console.log('🛑 AI backend stopped');
        }
    }
}

module.exports = { AIBackendIntegration };
`;
        
        fs.writeFileSync(path.join(this.aiIdeDir, 'src', 'ai-backend-integration.js'), backendIntegrationScript);
        
        console.log('✅ AI backend integrated with Code OSS');
    }

    async setupBuildSystem() {
        console.log('🔨 Setting up AI IDE build system...');
        
        const scriptsDir = path.join(this.aiIdeDir, 'scripts');
        if (!fs.existsSync(scriptsDir)) {
            fs.mkdirSync(scriptsDir, { recursive: true });
        }
        
        // Create AI IDE specific build script
        const buildScript = `
#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

console.log('🚀 Building AI IDE...');

// Install dependencies
console.log('📦 Installing dependencies...');
execSync('yarn install', { stdio: 'inherit' });

// Build VSCode base
console.log('🔨 Building VSCode base...');
execSync('yarn run compile', { stdio: 'inherit' });

// Build AI extensions
console.log('🤖 Building AI extensions...');
const extensionsDir = path.join(__dirname, '..', 'extensions');
const aiExtensions = ['ai-assistant', 'ai-chat', 'ai-semantic-search'];

for (const extension of aiExtensions) {
    const extensionPath = path.join(extensionsDir, extension);
    if (require('fs').existsSync(extensionPath)) {
        console.log(\`Building \${extension}...\`);
        execSync('npm run compile', { cwd: extensionPath, stdio: 'inherit' });
    }
}

// Setup AI backend
console.log('🐍 Setting up AI backend...');
const backendPath = path.join(__dirname, '..', 'ai-backend');
if (require('fs').existsSync(backendPath)) {
    execSync('pip install -r requirements.txt', { cwd: backendPath, stdio: 'inherit' });
}

console.log('✅ AI IDE build completed!');
`;
        
        fs.writeFileSync(path.join(scriptsDir, 'build-ai-ide.js'), buildScript);
        
        // Create packaging script
        const packageScript = `
#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

console.log('📦 Packaging AI IDE...');

// Build first
execSync('node scripts/build-ai-ide.js', { stdio: 'inherit' });

// Package for current platform
console.log('🔨 Creating executable...');
execSync('yarn run electron-builder', { stdio: 'inherit' });

console.log('✅ AI IDE packaged successfully!');
`;
        
        fs.writeFileSync(path.join(scriptsDir, 'package-ai-ide.js'), packageScript);
        
        // Make scripts executable
        if (process.platform !== 'win32') {
            execSync(`chmod +x ${path.join(scriptsDir, 'build-ai-ide.js')}`);
            execSync(`chmod +x ${path.join(scriptsDir, 'package-ai-ide.js')}`);
        }
        
        console.log('✅ Build system configured');
    }

    async createBranding() {
        console.log('🎨 Creating AI IDE branding...');
        
        // Update icons and branding
        const resourcesDir = path.join(this.aiIdeDir, 'resources');
        
        // Create AI IDE specific icons (placeholder)
        const iconSVG = `
<svg width="256" height="256" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="aiGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#007ACC;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#0099FF;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="256" height="256" fill="url(#aiGradient)" rx="32"/>
  <circle cx="128" cy="100" r="40" fill="white" opacity="0.9"/>
  <circle cx="100" cy="140" r="20" fill="white" opacity="0.7"/>
  <circle cx="156" cy="140" r="20" fill="white" opacity="0.7"/>
  <path d="M 80 180 Q 128 200 176 180" stroke="white" stroke-width="8" fill="none" opacity="0.8"/>
  <text x="128" y="220" text-anchor="middle" fill="white" font-family="Arial" font-size="24" font-weight="bold">AI</text>
</svg>`;
        
        fs.writeFileSync(path.join(resourcesDir, 'ai-ide-icon.svg'), iconSVG);
        
        // Update README for AI IDE
        const readmeContent = `
# AI IDE - Advanced AI-Powered Development Environment

Built on VSCode OSS, enhanced with cutting-edge AI capabilities.

## Features

🤖 **Multi-Agent AI System**
- Specialized AI agents for different development tasks
- Chain-of-thought reasoning for complex problems
- Self-improving AI with Darwin-Gödel model

🔍 **Advanced Search & Intelligence**
- Semantic code search with similarity ranking
- Web search integration with Playwright
- RAG system for enhanced code assistance

⚡ **Intelligent Code Assistance**
- Local LM Studio integration with Qwen Coder 3
- Context-aware code completion and generation
- Real-time code analysis and suggestions

🌐 **Web-Enabled Capabilities**
- Automatic Stack Overflow integration
- Real-time documentation scraping
- GitHub repository analysis

## Building

\`\`\`bash
# Setup (run once)
node setup-code-oss.js

# Build AI IDE
cd ai-ide-build
yarn run build:ai-ide

# Package executable
yarn run package:ai-ide
\`\`\`

## Running

\`\`\`bash
cd ai-ide-build
yarn run dev:ai-ide
\`\`\`

---

**AI IDE** - The VSCode-based AI development environment that competes with Cursor and Windsurf.
`;
        
        fs.writeFileSync(path.join(this.aiIdeDir, 'README-AI-IDE.md'), readmeContent);
        
        console.log('✅ AI IDE branding created');
    }

    copyDirectory(src, dest, options = {}) {
        const { exclude = [] } = options;
        
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }

        const items = fs.readdirSync(src);
        
        for (const item of items) {
            if (exclude.includes(item)) continue;
            
            const srcPath = path.join(src, item);
            const destPath = path.join(dest, item);
            
            const stat = fs.statSync(srcPath);
            
            if (stat.isDirectory()) {
                this.copyDirectory(srcPath, destPath, options);
            } else {
                fs.copyFileSync(srcPath, destPath);
            }
        }
    }
}

// CLI Interface
if (require.main === module) {
    const setup = new CodeOSSSetup();
    setup.setup().catch(console.error);
}

module.exports = CodeOSSSetup;