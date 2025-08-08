#!/usr/bin/env node

/**
 * AI IDE Final Packaging Script
 * Creates production-ready executables and installers
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const archiver = require('archiver');

const PROJECT_ROOT = path.dirname(__dirname);
const DIST_DIR = path.join(PROJECT_ROOT, 'dist');
const RELEASE_DIR = path.join(PROJECT_ROOT, 'releases');

class FinalPackager {
  constructor() {
    this.version = this.getVersion();
    this.ensureDirectories();
  }

  getVersion() {
    try {
      const packageJson = require(path.join(PROJECT_ROOT, 'package.json'));
      return packageJson.version || '1.0.0';
    } catch (error) {
      return '1.0.0';
    }
  }

  ensureDirectories() {
    [DIST_DIR, RELEASE_DIR].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  async createProductionBuilds() {
    console.log('🏭 Creating production builds...');

    // First, build the Electron app
    const ElectronBuilder = require('./build-electron.js');
    const electronBuilder = new ElectronBuilder();
    
    try {
      await electronBuilder.build();
      console.log('✅ Electron build completed');
    } catch (error) {
      console.error('❌ Electron build failed:', error);
      throw error;
    }
  }

  createWindowsExecutable() {
    console.log('🪟 Creating Windows executable...');
    
    const winDir = path.join(DIST_DIR, 'win-unpacked');
    if (!fs.existsSync(winDir)) {
      console.warn('⚠️ Windows build not found, skipping...');
      return;
    }

    // Create portable Windows version
    const portableWinDir = path.join(RELEASE_DIR, `AI-IDE-v${this.version}-Windows-Portable`);
    this.copyDirectory(winDir, portableWinDir);

    // Create launcher script
    const launcherScript = `@echo off
title AI IDE - Advanced AI-Powered Development Environment
echo Starting AI IDE...
echo.
echo AI IDE v${this.version}
echo Advanced AI-Powered Development Environment
echo Competitor to VSCode, GitHub Copilot, Cursor, and Windsurf
echo.

REM Set portable mode
set AI_IDE_PORTABLE=1
set AI_IDE_DATA_DIR=%~dp0data
set AI_IDE_CONFIG_DIR=%~dp0config

REM Create data directories
if not exist "%AI_IDE_DATA_DIR%" mkdir "%AI_IDE_DATA_DIR%"
if not exist "%AI_IDE_CONFIG_DIR%" mkdir "%AI_IDE_CONFIG_DIR%"

REM Start the application
start "" "%~dp0AI IDE.exe"

REM Optional: Keep console open for debugging
REM pause
`;

    fs.writeFileSync(path.join(portableWinDir, 'Start-AI-IDE.bat'), launcherScript);

    // Create README
    this.createReadme(portableWinDir, 'windows');

    // Create ZIP archive
    this.createZipArchive(portableWinDir, `AI-IDE-v${this.version}-Windows-Portable.zip`);

    console.log('✅ Windows executable created');
  }

  createMacExecutable() {
    console.log('🍎 Creating macOS executable...');
    
    const macDir = path.join(DIST_DIR, 'mac');
    if (!fs.existsSync(macDir)) {
      console.warn('⚠️ macOS build not found, skipping...');
      return;
    }

    // Create portable macOS version
    const portableMacDir = path.join(RELEASE_DIR, `AI-IDE-v${this.version}-macOS-Portable`);
    this.copyDirectory(macDir, portableMacDir);

    // Create launcher script
    const launcherScript = `#!/bin/bash
echo "AI IDE v${this.version}"
echo "Advanced AI-Powered Development Environment"
echo "Competitor to VSCode, GitHub Copilot, Cursor, and Windsurf"
echo ""

# Set portable mode
export AI_IDE_PORTABLE=1
export AI_IDE_DATA_DIR="$(dirname "$0")/data"
export AI_IDE_CONFIG_DIR="$(dirname "$0")/config"

# Create data directories
mkdir -p "$AI_IDE_DATA_DIR"
mkdir -p "$AI_IDE_CONFIG_DIR"

# Start the application
open "$(dirname "$0")/AI IDE.app"
`;

    fs.writeFileSync(path.join(portableMacDir, 'Start-AI-IDE.sh'), launcherScript);
    execSync(`chmod +x "${path.join(portableMacDir, 'Start-AI-IDE.sh')}"`);

    // Create README
    this.createReadme(portableMacDir, 'macos');

    // Create ZIP archive
    this.createZipArchive(portableMacDir, `AI-IDE-v${this.version}-macOS-Portable.zip`);

    console.log('✅ macOS executable created');
  }

  createLinuxExecutable() {
    console.log('🐧 Creating Linux executable...');
    
    const linuxDir = path.join(DIST_DIR, 'linux-unpacked');
    if (!fs.existsSync(linuxDir)) {
      console.warn('⚠️ Linux build not found, skipping...');
      return;
    }

    // Create portable Linux version
    const portableLinuxDir = path.join(RELEASE_DIR, `AI-IDE-v${this.version}-Linux-Portable`);
    this.copyDirectory(linuxDir, portableLinuxDir);

    // Create launcher script
    const launcherScript = `#!/bin/bash
echo "AI IDE v${this.version}"
echo "Advanced AI-Powered Development Environment"
echo "Competitor to VSCode, GitHub Copilot, Cursor, and Windsurf"
echo ""

# Set portable mode
export AI_IDE_PORTABLE=1
export AI_IDE_DATA_DIR="$(dirname "$0")/data"
export AI_IDE_CONFIG_DIR="$(dirname "$0")/config"

# Create data directories
mkdir -p "$AI_IDE_DATA_DIR"
mkdir -p "$AI_IDE_CONFIG_DIR"

# Start the application
"$(dirname "$0")/ai-ide"
`;

    fs.writeFileSync(path.join(portableLinuxDir, 'Start-AI-IDE.sh'), launcherScript);
    execSync(`chmod +x "${path.join(portableLinuxDir, 'Start-AI-IDE.sh')}"`);

    // Create README
    this.createReadme(portableLinuxDir, 'linux');

    // Create TAR.GZ archive
    this.createTarArchive(portableLinuxDir, `AI-IDE-v${this.version}-Linux-Portable.tar.gz`);

    console.log('✅ Linux executable created');
  }

  createReadme(targetDir, platform) {
    const readme = `# AI IDE v${this.version}

## Advanced AI-Powered Development Environment

AI IDE is a cutting-edge development environment that combines the power of VSCodium with advanced AI capabilities, making it a strong competitor to VSCode, GitHub Copilot, Cursor, and Windsurf.

### Features

🤖 **Multi-Agent AI System**
- Specialized agents for code generation, search, reasoning, and testing
- Chain-of-thought reasoning for complex problem solving
- ReAct framework for dynamic tool usage

🔍 **Advanced Search Capabilities**
- Semantic similarity search with interleaved context windows
- Web search integration for real-time information
- RAG (Retrieval-Augmented Generation) system

🧠 **Self-Improving AI**
- Darwin-Gödel model that can rewrite and optimize its own code
- Mini-benchmarking system for performance validation
- Reinforcement learning for user preference adaptation

⚡ **Intelligent Code Assistance**
- LM Studio integration with Qwen Coder 3
- Context-aware code completion and generation
- PocketFlow workflow management

🔧 **Extensible Architecture**
- Model Context Protocol (MCP) integration
- LangChain orchestration for complex AI workflows
- Tool and API integration framework

### Installation

#### ${platform.charAt(0).toUpperCase() + platform.slice(1)} Installation

1. Extract this archive to your desired location
2. Run the launcher script:
   ${platform === 'windows' ? '- Double-click `Start-AI-IDE.bat`' : '- Run `./Start-AI-IDE.sh` in terminal'}
3. The application will start in portable mode

#### Requirements

- ${platform === 'windows' ? 'Windows 10 or later' : platform === 'macos' ? 'macOS 10.15 or later' : 'Linux with glibc 2.17 or later'}
- Python 3.11+ (for AI backend services)
- 4GB RAM minimum, 8GB recommended
- 2GB free disk space

#### Optional: LM Studio Setup

For enhanced AI capabilities:
1. Download and install LM Studio from https://lmstudio.ai/
2. Download the Qwen Coder 3 model
3. Start LM Studio server on localhost:1234
4. Configure AI IDE settings to use your local model

### Usage

1. **AI Chat**: Interact with AI agents for coding assistance
2. **Semantic Search**: Find code patterns and examples intelligently  
3. **Reasoning Mode**: Get deep analysis for complex problems
4. **Agent Selection**: Choose specialized agents for specific tasks

### Configuration

The application runs in portable mode by default. All data is stored in:
- Data: \`./data/\`
- Configuration: \`./config/\`

### Support

For issues, documentation, and updates:
- GitHub: [Your Repository URL]
- Documentation: [Your Docs URL]
- Support: [Your Support Email]

### License

[Your License Information]

---

**AI IDE v${this.version}** - Revolutionizing Development with AI
`;

    fs.writeFileSync(path.join(targetDir, 'README.md'), readme);
  }

  copyDirectory(src, dest) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }

    const items = fs.readdirSync(src);
    
    for (const item of items) {
      const srcPath = path.join(src, item);
      const destPath = path.join(dest, item);
      
      const stat = fs.statSync(srcPath);
      
      if (stat.isDirectory()) {
        this.copyDirectory(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  createZipArchive(sourceDir, filename) {
    return new Promise((resolve, reject) => {
      const output = fs.createWriteStream(path.join(RELEASE_DIR, filename));
      const archive = archiver('zip', { zlib: { level: 9 } });

      output.on('close', () => {
        console.log(`📦 Created ${filename} (${(archive.pointer() / 1024 / 1024).toFixed(1)} MB)`);
        resolve();
      });

      archive.on('error', reject);
      archive.pipe(output);
      archive.directory(sourceDir, path.basename(sourceDir));
      archive.finalize();
    });
  }

  createTarArchive(sourceDir, filename) {
    try {
      execSync(`tar -czf "${path.join(RELEASE_DIR, filename)}" -C "${path.dirname(sourceDir)}" "${path.basename(sourceDir)}"`, {
        stdio: 'inherit'
      });
      console.log(`📦 Created ${filename}`);
    } catch (error) {
      console.error(`❌ Failed to create ${filename}:`, error.message);
    }
  }

  createInstallers() {
    console.log('📦 Creating installer packages...');

    // Copy installer files from dist to releases
    const installerFiles = [
      'AI IDE Setup *.exe',
      '*.dmg',
      '*.AppImage',
      '*.deb',
      '*.rpm'
    ];

    installerFiles.forEach(pattern => {
      try {
        const files = fs.readdirSync(DIST_DIR).filter(file => {
          const regex = new RegExp(pattern.replace('*', '.*'));
          return regex.test(file);
        });

        files.forEach(file => {
          const src = path.join(DIST_DIR, file);
          const dest = path.join(RELEASE_DIR, file);
          fs.copyFileSync(src, dest);
          console.log(`📦 Copied installer: ${file}`);
        });
      } catch (error) {
        // Ignore if files don't exist
      }
    });
  }

  createReleaseNotes() {
    console.log('📝 Creating release notes...');

    const releaseNotes = `# AI IDE v${this.version} Release Notes

## 🚀 What's New

AI IDE v${this.version} is a revolutionary AI-powered development environment that brings together the best of modern AI technologies to enhance your coding experience.

### 🎯 Key Features

- **Multi-Agent AI System**: Specialized AI agents for different development tasks
- **Advanced Code Intelligence**: Semantic search, code generation, and intelligent completion
- **Self-Improving AI**: Darwin-Gödel model that learns and improves over time
- **Web-Enabled Reasoning**: Real-time information retrieval and analysis
- **Extensible Architecture**: MCP integration and custom tool support

### 🆚 Competitive Advantages

**vs VSCode + GitHub Copilot:**
- Multi-agent system vs single AI assistant
- Self-improving capabilities vs static model
- Advanced reasoning vs simple completion

**vs Cursor:**
- More comprehensive AI integration
- Web search and RAG capabilities
- Specialized agent architecture

**vs Windsurf:**
- Open architecture with local model support
- Advanced context management
- Reinforcement learning adaptation

### 📦 Available Downloads

- **Windows**: \`AI-IDE-v${this.version}-Windows-Portable.zip\`
- **macOS**: \`AI-IDE-v${this.version}-macOS-Portable.zip\`
- **Linux**: \`AI-IDE-v${this.version}-Linux-Portable.tar.gz\`

### 🔧 Installation

1. Download the appropriate package for your platform
2. Extract to your desired location
3. Run the launcher script
4. Optionally configure LM Studio for enhanced AI capabilities

### 📋 System Requirements

- **OS**: Windows 10+, macOS 10.15+, or Linux with glibc 2.17+
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Python**: 3.11+ for AI backend services

### 🐛 Known Issues

- First startup may take longer while initializing AI models
- Some features require internet connection for web search
- LM Studio integration requires separate installation

### 🔮 Coming Soon

- Plugin marketplace
- Cloud synchronization
- Team collaboration features
- Mobile companion app

---

**Download AI IDE v${this.version}** and experience the future of AI-powered development!
`;

    fs.writeFileSync(path.join(RELEASE_DIR, `RELEASE-NOTES-v${this.version}.md`), releaseNotes);
  }

  async packageAll() {
    console.log('🎯 Starting final packaging process...');
    console.log(`📦 Version: ${this.version}`);
    console.log(`📁 Release directory: ${RELEASE_DIR}`);

    try {
      // Create production builds
      await this.createProductionBuilds();

      // Create platform-specific executables
      this.createWindowsExecutable();
      this.createMacExecutable();
      this.createLinuxExecutable();

      // Copy installers
      this.createInstallers();

      // Create release notes
      this.createReleaseNotes();

      console.log('🎉 Final packaging completed successfully!');
      console.log('\n📋 Release files created:');
      
      const releaseFiles = fs.readdirSync(RELEASE_DIR);
      releaseFiles.forEach(file => {
        const filePath = path.join(RELEASE_DIR, file);
        const stat = fs.statSync(filePath);
        if (stat.isFile()) {
          const size = `(${(stat.size / 1024 / 1024).toFixed(1)} MB)`;
          console.log(`  📄 ${file} ${size}`);
        } else {
          console.log(`  📁 ${file}/`);
        }
      });

      console.log(`\n✅ AI IDE v${this.version} is ready for distribution!`);
      console.log('🚀 Your AI-powered IDE competitor to VSCode, Copilot, Cursor, and Windsurf is complete!');

    } catch (error) {
      console.error('❌ Packaging failed:', error.message);
      process.exit(1);
    }
  }
}

// Main execution
if (require.main === module) {
  const packager = new FinalPackager();
  packager.packageAll().catch(console.error);
}

module.exports = FinalPackager;