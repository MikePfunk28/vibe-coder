#!/usr/bin/env node

/**
 * AI IDE Builder - Code OSS Based
 * Creates a complete AI IDE using VSCode OSS as the foundation
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class AIIDEBuilder {
    constructor() {
        this.projectRoot = __dirname;
        this.config = this.loadConfig();
        this.codeOSSDir = path.join(this.projectRoot, 'code-oss');
        this.buildDir = path.join(this.projectRoot, 'ai-ide-build');
        this.distDir = path.join(this.projectRoot, 'dist');
    }

    loadConfig() {
        const configPath = path.join(this.projectRoot, 'build-config-oss.json');
        return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }

    async build() {
        console.log('🚀 Building AI IDE based on VSCode OSS...');
        console.log(`📦 Version: ${this.config.version}`);
        console.log(`🔧 Base VSCode: ${this.config.baseVersion}`);
        console.log('');

        try {
            // Step 1: Setup Code OSS base
            await this.setupCodeOSS();
            
            // Step 2: Integrate AI features
            await this.integrateAIFeatures();
            
            // Step 3: Build the application
            await this.buildApplication();
            
            // Step 4: Create executables
            await this.createExecutables();
            
            // Step 5: Package for distribution
            await this.packageForDistribution();
            
            console.log('🎉 AI IDE build completed successfully!');
            this.displayBuildSummary();
            
        } catch (error) {
            console.error('❌ Build failed:', error.message);
            throw error;
        }
    }

    async setupCodeOSS() {
        console.log('📥 Setting up VSCode OSS base...');
        
        // Run the Code OSS setup if not already done
        if (!fs.existsSync(this.buildDir)) {
            console.log('🔧 Running Code OSS setup...');
            execSync('node setup-code-oss.js', { stdio: 'inherit' });
        }
        
        // Verify build directory exists
        if (!fs.existsSync(this.buildDir)) {
            throw new Error('Code OSS setup failed - build directory not found');
        }
        
        console.log('✅ VSCode OSS base ready');
    }

    async integrateAIFeatures() {
        console.log('🤖 Integrating AI features...');
        
        // Update product configuration with AI features
        await this.updateProductConfiguration();
        
        // Add AI-specific built-in extensions
        await this.addAIExtensions();
        
        // Integrate AI backend
        await this.integrateAIBackend();
        
        // Add AI-specific UI components
        await this.addAIUIComponents();
        
        // Configure AI settings
        await this.configureAISettings();
        
        console.log('✅ AI features integrated');
    }

    async updateProductConfiguration() {
        const productJsonPath = path.join(this.buildDir, 'product.json');
        const productJson = JSON.parse(fs.readFileSync(productJsonPath, 'utf8'));
        
        // Add AI IDE specific configuration
        productJson.aiIdeConfig = this.config.aiFeatures;
        
        // Update branding
        Object.assign(productJson, this.config.branding);
        
        // Configure extension marketplace
        productJson.extensionsGallery = this.config.extensions.marketplace;
        
        fs.writeFileSync(productJsonPath, JSON.stringify(productJson, null, 2));
        console.log('✅ Product configuration updated');
    }

    async addAIExtensions() {
        console.log('🧩 Adding AI extensions...');
        
        const extensionsDir = path.join(this.buildDir, 'extensions');
        
        // Copy our AI extensions
        const sourceExtensionsDir = path.join(this.projectRoot, 'extensions');
        if (fs.existsSync(sourceExtensionsDir)) {
            const extensions = fs.readdirSync(sourceExtensionsDir);
            for (const extension of extensions) {
                const sourcePath = path.join(sourceExtensionsDir, extension);
                const destPath = path.join(extensionsDir, extension);
                
                if (fs.statSync(sourcePath).isDirectory()) {
                    this.copyDirectory(sourcePath, destPath);
                    console.log(`✅ Added extension: ${extension}`);
                }
            }
        }
        
        // Create additional built-in AI extensions
        await this.createBuiltInAIExtensions(extensionsDir);
    }

    async createBuiltInAIExtensions(extensionsDir) {
        const builtinExtensions = [
            {
                name: 'ai-web-search',
                displayName: 'AI Web Search',
                description: 'Web search integration with Playwright',
                commands: [
                    { command: 'ai-web-search.search', title: 'Search Web', category: 'AI' }
                ],
                keybindings: [
                    { command: 'ai-web-search.search', key: 'ctrl+shift+w', mac: 'cmd+shift+w' }
                ]
            },
            {
                name: 'ai-reasoning',
                displayName: 'AI Reasoning',
                description: 'Advanced AI reasoning and problem solving',
                commands: [
                    { command: 'ai-reasoning.analyze', title: 'Analyze Problem', category: 'AI' }
                ],
                keybindings: [
                    { command: 'ai-reasoning.analyze', key: 'ctrl+shift+r', mac: 'cmd+shift+r' }
                ]
            },
            {
                name: 'ai-multi-agent',
                displayName: 'AI Multi-Agent',
                description: 'Multi-agent AI system coordination',
                commands: [
                    { command: 'ai-multi-agent.select', title: 'Select Agent', category: 'AI' }
                ],
                views: {
                    explorer: [
                        { id: 'ai-agents', name: 'AI Agents', when: 'true' }
                    ]
                }
            }
        ];

        for (const extension of builtinExtensions) {
            const extensionDir = path.join(extensionsDir, extension.name);
            fs.mkdirSync(extensionDir, { recursive: true });
            
            const packageJson = {
                name: extension.name,
                displayName: extension.displayName,
                description: extension.description,
                version: '1.0.0',
                publisher: 'ai-ide',
                engines: { vscode: '^1.85.0' },
                categories: ['Other'],
                activationEvents: ['*'],
                main: './out/extension.js',
                contributes: {
                    commands: extension.commands || [],
                    keybindings: extension.keybindings || [],
                    views: extension.views || {}
                }
            };
            
            fs.writeFileSync(
                path.join(extensionDir, 'package.json'),
                JSON.stringify(packageJson, null, 2)
            );
            
            // Create basic extension implementation
            const srcDir = path.join(extensionDir, 'src');
            fs.mkdirSync(srcDir, { recursive: true });
            
            const extensionTs = `
import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('${extension.displayName} extension activated');
    
    // Register commands
    ${extension.commands?.map(cmd => `
    const ${cmd.command.replace(/[.-]/g, '_')} = vscode.commands.registerCommand('${cmd.command}', () => {
        vscode.window.showInformationMessage('${cmd.title} executed!');
        // TODO: Implement ${cmd.title} functionality
    });
    context.subscriptions.push(${cmd.command.replace(/[.-]/g, '_')});
    `).join('') || ''}
}

export function deactivate() {}
`;
            
            fs.writeFileSync(path.join(srcDir, 'extension.ts'), extensionTs);
            
            // Create tsconfig.json
            const tsConfig = {
                compilerOptions: {
                    module: 'commonjs',
                    target: 'ES2020',
                    outDir: 'out',
                    lib: ['ES2020'],
                    sourceMap: true,
                    rootDir: 'src',
                    strict: true
                },
                exclude: ['node_modules', '.vscode-test']
            };
            
            fs.writeFileSync(
                path.join(extensionDir, 'tsconfig.json'),
                JSON.stringify(tsConfig, null, 2)
            );
            
            console.log(`✅ Created built-in extension: ${extension.name}`);
        }
    }

    async integrateAIBackend() {
        console.log('🐍 Integrating AI backend...');
        
        const backendDir = path.join(this.buildDir, 'ai-backend');
        const sourceBackendDir = path.join(this.projectRoot, 'backend');
        
        if (fs.existsSync(sourceBackendDir)) {
            this.copyDirectory(sourceBackendDir, backendDir);
            
            // Create backend startup integration
            const backendIntegration = `
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

class AIBackendManager {
    constructor() {
        this.backendProcess = null;
        this.isRunning = false;
        this.port = ${this.config.backend.port};
    }

    async start() {
        if (this.isRunning) return;
        
        const backendPath = path.join(__dirname, '..', 'ai-backend');
        const pythonExecutable = this.findPython();
        
        if (!pythonExecutable) {
            console.error('❌ Python not found for AI backend');
            return false;
        }
        
        // Check if virtual environment exists
        const venvPath = path.join(backendPath, 'venv');
        if (!fs.existsSync(venvPath)) {
            console.log('📦 Creating Python virtual environment...');
            try {
                require('child_process').execSync(\`\${pythonExecutable} -m venv venv\`, { 
                    cwd: backendPath, 
                    stdio: 'inherit' 
                });
            } catch (error) {
                console.error('❌ Failed to create virtual environment:', error.message);
                return false;
            }
        }
        
        // Install requirements
        const requirementsPath = path.join(backendPath, 'requirements.txt');
        if (fs.existsSync(requirementsPath)) {
            console.log('📦 Installing Python dependencies...');
            const activateScript = process.platform === 'win32' ? 
                'venv\\\\Scripts\\\\activate && pip install -r requirements.txt' :
                'source venv/bin/activate && pip install -r requirements.txt';
            
            try {
                require('child_process').execSync(activateScript, { 
                    cwd: backendPath, 
                    stdio: 'inherit',
                    shell: true 
                });
            } catch (error) {
                console.warn('⚠️ Failed to install some dependencies:', error.message);
            }
        }
        
        console.log('🚀 Starting AI backend...');
        
        const pythonScript = process.platform === 'win32' ?
            path.join(venvPath, 'Scripts', 'python.exe') :
            path.join(venvPath, 'bin', 'python');
        
        this.backendProcess = spawn(pythonScript, [
            path.join(backendPath, 'main.py'),
            '--port', this.port.toString(),
            '--host', '127.0.0.1'
        ], {
            cwd: backendPath,
            stdio: ['pipe', 'pipe', 'pipe']
        });
        
        this.backendProcess.stdout.on('data', (data) => {
            console.log('🤖 AI Backend:', data.toString().trim());
        });
        
        this.backendProcess.stderr.on('data', (data) => {
            console.error('🤖 AI Backend Error:', data.toString().trim());
        });
        
        this.backendProcess.on('close', (code) => {
            console.log(\`🤖 AI Backend exited with code: \${code}\`);
            this.isRunning = false;
        });
        
        this.isRunning = true;
        console.log(\`✅ AI backend started on port \${this.port}\`);
        return true;
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
            this.backendProcess.kill('SIGTERM');
            this.isRunning = false;
            console.log('🛑 AI backend stopped');
        }
    }
    
    isBackendRunning() {
        return this.isRunning;
    }
}

module.exports = { AIBackendManager };
`;
            
            fs.writeFileSync(
                path.join(this.buildDir, 'src', 'vs', 'workbench', 'contrib', 'ai', 'aiBackendManager.js'),
                backendIntegration
            );
            
            console.log('✅ AI backend integrated');
        }
    }

    async addAIUIComponents() {
        console.log('🎨 Adding AI UI components...');
        
        // Add AI-specific UI components to VSCode
        const workbenchDir = path.join(this.buildDir, 'src', 'vs', 'workbench');
        const aiContribDir = path.join(workbenchDir, 'contrib', 'ai');
        
        fs.mkdirSync(aiContribDir, { recursive: true });
        
        // Create AI chat panel
        const aiChatPanel = `
import { Disposable } from 'vs/base/common/lifecycle';
import { IViewletService } from 'vs/workbench/services/viewlet/browser/viewlet';
import { IInstantiationService } from 'vs/platform/instantiation/common/instantiation';

export class AIChatPanel extends Disposable {
    constructor(
        @IViewletService private readonly viewletService: IViewletService,
        @IInstantiationService private readonly instantiationService: IInstantiationService
    ) {
        super();
        this.initializeAIChat();
    }
    
    private initializeAIChat(): void {
        // Initialize AI chat functionality
        console.log('AI Chat Panel initialized');
    }
    
    public sendMessage(message: string): void {
        // Send message to AI backend
        fetch('http://127.0.0.1:${this.config.backend.port}/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message })
        }).then(response => response.json())
          .then(data => {
              console.log('AI Response:', data);
              // Handle AI response
          }).catch(error => {
              console.error('AI Chat Error:', error);
          });
    }
}
`;
        
        fs.writeFileSync(path.join(aiContribDir, 'aiChatPanel.ts'), aiChatPanel);
        
        console.log('✅ AI UI components added');
    }

    async configureAISettings() {
        console.log('⚙️ Configuring AI settings...');
        
        // Add AI-specific settings to VSCode configuration
        const configurationDir = path.join(this.buildDir, 'src', 'vs', 'workbench', 'services', 'configuration');
        const aiConfigPath = path.join(configurationDir, 'common', 'aiConfiguration.ts');
        
        const aiConfiguration = `
export interface IAIConfiguration {
    enableAI: boolean;
    enableWebSearch: boolean;
    enablePlaywright: boolean;
    enableMultiAgent: boolean;
    enableSemanticSearch: boolean;
    enableRAG: boolean;
    defaultAIProvider: string;
    backendPort: number;
    lmStudioUrl: string;
    openaiApiKey: string;
}

export const defaultAIConfiguration: IAIConfiguration = ${JSON.stringify(this.config.aiFeatures, null, 4)};
`;
        
        fs.writeFileSync(aiConfigPath, aiConfiguration);
        
        console.log('✅ AI settings configured');
    }

    async buildApplication() {
        console.log('🔨 Building AI IDE application...');
        
        process.chdir(this.buildDir);
        
        // Install dependencies
        console.log('📦 Installing dependencies...');
        execSync('yarn install', { stdio: 'inherit' });
        
        // Build VSCode with AI enhancements
        console.log('🔨 Compiling AI IDE...');
        execSync('yarn run compile', { stdio: 'inherit' });
        
        // Build AI extensions
        console.log('🧩 Building AI extensions...');
        const extensionsDir = path.join(this.buildDir, 'extensions');
        const extensions = fs.readdirSync(extensionsDir).filter(item => 
            fs.statSync(path.join(extensionsDir, item)).isDirectory() && 
            item.startsWith('ai-')
        );
        
        for (const extension of extensions) {
            const extensionPath = path.join(extensionsDir, extension);
            const packageJsonPath = path.join(extensionPath, 'package.json');
            
            if (fs.existsSync(packageJsonPath)) {
                console.log(`Building extension: ${extension}`);
                try {
                    execSync('npm install && npm run compile', { 
                        cwd: extensionPath, 
                        stdio: 'inherit' 
                    });
                } catch (error) {
                    console.warn(`Failed to build extension ${extension}:`, error.message);
                }
            }
        }
        
        process.chdir(this.projectRoot);
        console.log('✅ AI IDE application built');
    }

    async createExecutables() {
        console.log('📦 Creating executables...');
        
        process.chdir(this.buildDir);
        
        // Create electron-builder configuration
        const electronBuilderConfig = {
            appId: 'dev.ai-ide.app',
            productName: this.config.name,
            directories: {
                output: '../dist'
            },
            files: [
                '**/*',
                '!**/node_modules/*/{CHANGELOG.md,README.md,README,readme.md,readme}',
                '!**/node_modules/*/{test,__tests__,tests,powered-test,example,examples}',
                '!**/node_modules/*.d.ts',
                '!**/node_modules/.bin',
                '!**/*.{iml,o,hprof,orig,pyc,pyo,rbc,swp,csproj,sln,xproj}',
                '!.editorconfig',
                '!**/._*',
                '!**/{.DS_Store,.git,.hg,.svn,CVS,RCS,SCCS,.gitignore,.gitattributes}',
                '!**/{__pycache__,thumbs.db,.flowconfig,.idea,.vs,.nyc_output}',
                '!**/{appveyor.yml,.travis.yml,circle.yml}',
                '!**/{npm-debug.log,yarn.lock,.yarn-integrity,.yarn-metadata.json}'
            ],
            extraResources: [
                {
                    from: 'ai-backend',
                    to: 'ai-backend',
                    filter: ['**/*', '!**/venv', '!**/__pycache__', '!**/*.pyc']
                }
            ],
            ...this.config.buildTargets
        };
        
        fs.writeFileSync('electron-builder.json', JSON.stringify(electronBuilderConfig, null, 2));
        
        // Build executables
        execSync('npx electron-builder', { stdio: 'inherit' });
        
        process.chdir(this.projectRoot);
        console.log('✅ Executables created');
    }

    async packageForDistribution() {
        console.log('📦 Packaging for distribution...');
        
        // Create portable versions and additional packages
        const distFiles = fs.readdirSync(this.distDir);
        console.log('📋 Created distribution files:');
        
        distFiles.forEach(file => {
            const filePath = path.join(this.distDir, file);
            const stats = fs.statSync(filePath);
            const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
            console.log(`  📄 ${file} (${sizeMB} MB)`);
        });
        
        console.log('✅ Packaging completed');
    }

    displayBuildSummary() {
        console.log('\n🎉 AI IDE Build Summary:');
        console.log(`📦 Name: ${this.config.name}`);
        console.log(`🔢 Version: ${this.config.version}`);
        console.log(`🔧 Base VSCode: ${this.config.baseVersion}`);
        console.log(`📁 Build Directory: ${this.buildDir}`);
        console.log(`📁 Distribution: ${this.distDir}`);
        console.log('\n🚀 Your VSCode-based AI IDE is ready!');
        console.log('🎯 Competes directly with Cursor and Windsurf!');
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
    const builder = new AIIDEBuilder();
    builder.build().catch(console.error);
}

module.exports = AIIDEBuilder;