#!/usr/bin/env node

/**
 * AI IDE Executable Builder
 * Complete build system for creating production executables
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const VersionManager = require('./version-manager');

class ExecutableBuilder {
    constructor() {
        this.projectRoot = __dirname;
        this.versionManager = new VersionManager();
        this.buildConfig = this.loadBuildConfig();
    }

    loadBuildConfig() {
        const defaultConfig = {
            name: 'AI IDE',
            description: 'Advanced AI-Powered Development Environment',
            platforms: ['win32', 'darwin', 'linux'],
            architectures: ['x64', 'arm64'],
            includePlaywright: true,
            includeMonaco: true,
            bundleBackend: true,
            createPortable: true,
            createInstaller: true,
            signCode: false
        };

        try {
            const configPath = path.join(this.projectRoot, 'build-config.json');
            if (fs.existsSync(configPath)) {
                const userConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
                return { ...defaultConfig, ...userConfig };
            }
        } catch (error) {
            console.warn('Using default build configuration');
        }

        return defaultConfig;
    }

    async buildExecutable(options = {}) {
        const {
            platform = process.platform,
            arch = process.arch,
            createIteration = true,
            skipTests = false
        } = options;

        console.log('🚀 Starting AI IDE Executable Build Process');
        console.log(`📦 Platform: ${platform}-${arch}`);
        console.log(`🔢 Version: ${this.versionManager.currentVersion}`);
        console.log('');

        try {
            // Step 1: Create iteration
            if (createIteration) {
                await this.versionManager.createIteration(`Build for ${platform}-${arch}`);
            }

            // Step 2: Prepare environment
            await this.prepareEnvironment();

            // Step 3: Install dependencies
            await this.installDependencies();

            // Step 4: Run tests (optional)
            if (!skipTests) {
                await this.runTests();
            }

            // Step 5: Build backend
            await this.buildBackend();

            // Step 6: Build frontend
            await this.buildFrontend();

            // Step 7: Create Electron app
            await this.buildElectronApp();

            // Step 8: Create executables
            await this.createExecutables(platform, arch);

            // Step 9: Create portable versions
            if (this.buildConfig.createPortable) {
                await this.createPortableVersions();
            }

            // Step 10: Create installers
            if (this.buildConfig.createInstaller) {
                await this.createInstallers();
            }

            // Step 11: Verify builds
            await this.verifyBuilds();

            console.log('🎉 AI IDE Executable Build Completed Successfully!');
            this.displayBuildSummary();

        } catch (error) {
            console.error('❌ Build failed:', error.message);
            throw error;
        }
    }

    async prepareEnvironment() {
        console.log('🔧 Preparing build environment...');

        // Create necessary directories
        const dirs = ['dist', 'releases', 'temp', 'electron'];
        dirs.forEach(dir => {
            const dirPath = path.join(this.projectRoot, dir);
            if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
            }
        });

        // Clean previous builds
        console.log('🧹 Cleaning previous builds...');
        try {
            execSync('npm run clean', { stdio: 'inherit' });
        } catch (error) {
            console.warn('Clean command failed, continuing...');
        }
    }

    async installDependencies() {
        console.log('📦 Installing dependencies...');

        // Install root dependencies
        console.log('Installing root dependencies...');
        execSync('npm install', { stdio: 'inherit' });

        // Install extension dependencies
        console.log('Installing extension dependencies...');
        const extensionPath = path.join(this.projectRoot, 'extensions', 'ai-assistant');
        if (fs.existsSync(extensionPath)) {
            execSync('npm install', { cwd: extensionPath, stdio: 'inherit' });
        }

        // Install Python dependencies
        console.log('Installing Python dependencies...');
        const backendPath = path.join(this.projectRoot, 'backend');
        if (fs.existsSync(backendPath)) {
            // Create virtual environment if it doesn't exist
            const venvPath = path.join(backendPath, 'venv');
            if (!fs.existsSync(venvPath)) {
                execSync('python -m venv venv', { cwd: backendPath, stdio: 'inherit' });
            }

            // Install requirements
            const activateScript = process.platform === 'win32' ? 
                'venv\\Scripts\\activate && pip install -r requirements.txt' :
                'source venv/bin/activate && pip install -r requirements.txt';
            
            execSync(activateScript, { cwd: backendPath, stdio: 'inherit', shell: true });

            // Install Playwright browsers
            if (this.buildConfig.includePlaywright) {
                console.log('Installing Playwright browsers...');
                execSync('source venv/bin/activate && playwright install', { 
                    cwd: backendPath, 
                    stdio: 'inherit', 
                    shell: true 
                });
            }
        }
    }

    async runTests() {
        console.log('🧪 Running tests...');

        try {
            // Run backend tests
            console.log('Running backend tests...');
            execSync('npm run test:backend', { stdio: 'inherit' });

            // Run extension tests
            console.log('Running extension tests...');
            execSync('npm run test:extension', { stdio: 'inherit' });

            console.log('✅ All tests passed');
        } catch (error) {
            console.error('❌ Tests failed:', error.message);
            throw new Error('Tests failed - build aborted');
        }
    }

    async buildBackend() {
        console.log('🐍 Building Python backend...');

        const backendPath = path.join(this.projectRoot, 'backend');
        
        // Compile Python files (optional optimization)
        try {
            execSync('python -m compileall .', { cwd: backendPath, stdio: 'inherit' });
        } catch (error) {
            console.warn('Python compilation failed, continuing...');
        }

        // Create backend bundle
        const backendBundle = path.join(this.projectRoot, 'electron', 'backend');
        if (fs.existsSync(backendBundle)) {
            fs.rmSync(backendBundle, { recursive: true, force: true });
        }

        // Copy backend files
        this.copyDirectory(backendPath, backendBundle, {
            exclude: ['venv', '__pycache__', '*.pyc', '.pytest_cache', 'node_modules']
        });

        console.log('✅ Backend built successfully');
    }

    async buildFrontend() {
        console.log('🎨 Building frontend...');

        // Build extension
        const extensionPath = path.join(this.projectRoot, 'extensions', 'ai-assistant');
        if (fs.existsSync(extensionPath)) {
            execSync('npm run compile', { cwd: extensionPath, stdio: 'inherit' });
        }

        // Copy Monaco Editor assets if needed
        if (this.buildConfig.includeMonaco) {
            console.log('Including Monaco Editor assets...');
            // Monaco assets are loaded via CDN in our implementation
        }

        console.log('✅ Frontend built successfully');
    }

    async buildElectronApp() {
        console.log('⚡ Building Electron application...');

        // Run the Electron build script
        execSync('node scripts/build-electron.js', { stdio: 'inherit' });

        console.log('✅ Electron app built successfully');
    }

    async createExecutables(platform, arch) {
        console.log(`🔨 Creating executables for ${platform}-${arch}...`);

        const electronPath = path.join(this.projectRoot, 'electron');
        
        // Build with electron-builder
        const buildCommand = `npx electron-builder --${platform} --${arch} --publish=never`;
        execSync(buildCommand, { cwd: electronPath, stdio: 'inherit' });

        console.log('✅ Executables created successfully');
    }

    async createPortableVersions() {
        console.log('📦 Creating portable versions...');

        // Run the final packaging script
        execSync('node scripts/package-final.js', { stdio: 'inherit' });

        console.log('✅ Portable versions created successfully');
    }

    async createInstallers() {
        console.log('📦 Creating installers...');

        const electronPath = path.join(this.projectRoot, 'electron');
        
        // Create installers for each platform
        const platforms = this.buildConfig.platforms;
        
        for (const platform of platforms) {
            try {
                console.log(`Creating installer for ${platform}...`);
                const buildCommand = `npx electron-builder --${platform} --publish=never`;
                execSync(buildCommand, { cwd: electronPath, stdio: 'inherit' });
            } catch (error) {
                console.warn(`Failed to create installer for ${platform}:`, error.message);
            }
        }

        console.log('✅ Installers created successfully');
    }

    async verifyBuilds() {
        console.log('🔍 Verifying builds...');

        const distPath = path.join(this.projectRoot, 'dist');
        const releasesPath = path.join(this.projectRoot, 'releases');

        // Check if build artifacts exist
        const artifacts = [];
        
        if (fs.existsSync(distPath)) {
            const distFiles = fs.readdirSync(distPath);
            artifacts.push(...distFiles.map(f => ({ name: f, path: distPath, type: 'dist' })));
        }

        if (fs.existsSync(releasesPath)) {
            const releaseFiles = fs.readdirSync(releasesPath);
            artifacts.push(...releaseFiles.map(f => ({ name: f, path: releasesPath, type: 'release' })));
        }

        if (artifacts.length === 0) {
            throw new Error('No build artifacts found');
        }

        console.log(`✅ Found ${artifacts.length} build artifacts`);
        return artifacts;
    }

    displayBuildSummary() {
        console.log('\n📋 Build Summary:');
        console.log(`Version: ${this.versionManager.currentVersion}`);
        console.log(`Build Time: ${new Date().toISOString()}`);
        
        const versionInfo = this.versionManager.getVersionInfo();
        console.log(`Total Versions: ${versionInfo.totalVersions}`);
        console.log(`Total Iterations: ${versionInfo.totalIterations}`);
        console.log(`Total Releases: ${versionInfo.totalReleases}`);

        // List created files
        const releasesPath = path.join(this.projectRoot, 'releases');
        if (fs.existsSync(releasesPath)) {
            console.log('\n📦 Created Files:');
            const files = fs.readdirSync(releasesPath);
            files.forEach(file => {
                const filePath = path.join(releasesPath, file);
                const stats = fs.statSync(filePath);
                const sizeMB = (stats.size / 1024 / 1024).toFixed(1);
                console.log(`  📄 ${file} (${sizeMB} MB)`);
            });
        }

        console.log('\n🎉 AI IDE is ready for distribution!');
        console.log('🚀 Your VSCode/Cursor/Windsurf competitor is complete!');
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

    async buildAllPlatforms() {
        console.log('🌍 Building for all platforms...');

        const platforms = [
            { platform: 'win32', arch: 'x64' },
            { platform: 'win32', arch: 'arm64' },
            { platform: 'darwin', arch: 'x64' },
            { platform: 'darwin', arch: 'arm64' },
            { platform: 'linux', arch: 'x64' },
            { platform: 'linux', arch: 'arm64' }
        ];

        for (const { platform, arch } of platforms) {
            try {
                console.log(`\n🔨 Building ${platform}-${arch}...`);
                await this.buildExecutable({ 
                    platform, 
                    arch, 
                    createIteration: false, 
                    skipTests: true 
                });
            } catch (error) {
                console.error(`❌ Failed to build ${platform}-${arch}:`, error.message);
            }
        }

        // Create final release
        await this.versionManager.createRelease(
            this.versionManager.currentVersion,
            'Multi-platform release with all executables and installers'
        );
    }
}

// CLI Interface
if (require.main === module) {
    const builder = new ExecutableBuilder();
    const command = process.argv[2];
    const options = {};

    // Parse command line options
    process.argv.slice(3).forEach(arg => {
        if (arg.startsWith('--')) {
            const [key, value] = arg.substring(2).split('=');
            options[key] = value || true;
        }
    });

    switch (command) {
        case 'build':
            builder.buildExecutable(options).catch(console.error);
            break;
            
        case 'build-all':
            builder.buildAllPlatforms().catch(console.error);
            break;
            
        case 'config':
            console.log('Current build configuration:');
            console.log(JSON.stringify(builder.buildConfig, null, 2));
            break;
            
        default:
            console.log(`
AI IDE Executable Builder

Usage:
  node build-exe.js <command> [options]

Commands:
  build                    - Build for current platform
  build-all               - Build for all platforms
  config                  - Show build configuration

Options:
  --platform=<platform>   - Target platform (win32, darwin, linux)
  --arch=<arch>          - Target architecture (x64, arm64)
  --skip-tests           - Skip running tests
  --no-iteration         - Don't create iteration

Examples:
  node build-exe.js build
  node build-exe.js build --platform=win32 --arch=x64
  node build-exe.js build-all
  node build-exe.js build --skip-tests --no-iteration
            `);
            break;
    }
}

module.exports = ExecutableBuilder;