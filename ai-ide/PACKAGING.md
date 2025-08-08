# AI IDE Packaging Guide

This guide explains how to build and package AI IDE for distribution as a competitor to VSCode, GitHub Copilot, Cursor, and Windsurf.

## Overview

AI IDE uses Electron to create cross-platform desktop applications with embedded Python backend services. The packaging system creates:

- **Portable executables** for Windows, macOS, and Linux
- **Installer packages** (NSIS, DMG, AppImage, DEB, RPM)
- **ZIP/TAR archives** for easy distribution

## Prerequisites

### System Requirements

- **Node.js** 18.0.0 or later
- **npm** 8.0.0 or later
- **Python** 3.11 or later
- **Git** for version control

### Platform-Specific Requirements

#### Windows
- **Visual Studio Build Tools** or Visual Studio Community
- **Windows SDK** 10.0.17763.0 or later
- **NSIS** 3.0 or later (for installer creation)

#### macOS
- **Xcode Command Line Tools**
- **macOS** 10.15 or later
- **Apple Developer Account** (for code signing)

#### Linux
- **build-essential** package
- **libnss3-dev**, **libatk-bridge2.0-dev**, **libdrm2**, **libxcomposite1**, **libxdamage1**, **libxrandr2**, **libgbm1**, **libxss1**, **libasound2**

## Quick Start

### 1. Install Dependencies

```bash
# Install all dependencies (root, extension, and backend)
npm run install:all
```

### 2. Build for Current Platform

```bash
# Build for your current platform
npm run build
```

### 3. Create Release Packages

```bash
# Create final release packages
npm run package
```

### 4. Build for All Platforms (Advanced)

```bash
# Build for all supported platforms
npm run build:all
```

## Detailed Build Process

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ai-ide.git
cd ai-ide

# Install dependencies
npm install
cd extensions/ai-assistant && npm install && cd ../..
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..
```

### Step 2: Development Build

```bash
# Build for development
npm run build

# Run in development mode
npm run dev
```

### Step 3: Production Build

#### Windows
```powershell
# Run the Windows build script
.\scripts\build-release.ps1
```

#### macOS/Linux
```bash
# Run the Unix build script
./scripts/build-release.sh
```

### Step 4: Manual Packaging (Advanced)

```bash
# Build Electron app only
node scripts/build-electron.js

# Create final packages only
node scripts/package-final.js
```

## Output Structure

After successful packaging, you'll find the following in the `releases/` directory:

```
releases/
├── AI-IDE-v1.0.0-Windows-Portable.zip
├── AI-IDE-v1.0.0-macOS-Portable.zip
├── AI-IDE-v1.0.0-Linux-Portable.tar.gz
├── AI IDE Setup 1.0.0.exe                    # Windows installer
├── AI IDE-1.0.0.dmg                          # macOS installer
├── AI IDE-1.0.0.AppImage                     # Linux AppImage
├── ai-ide_1.0.0_amd64.deb                    # Debian package
├── ai-ide-1.0.0.x86_64.rpm                   # RPM package
├── checksums.sha256                          # File checksums
└── RELEASE-NOTES-v1.0.0.md                   # Release notes
```

## Portable Versions

Each portable version includes:

- **Application files**: The complete AI IDE application
- **Launcher script**: Platform-specific startup script
- **README.md**: Installation and usage instructions
- **Data directory**: For portable user data storage
- **Config directory**: For portable configuration

### Windows Portable
- Extract `AI-IDE-v1.0.0-Windows-Portable.zip`
- Run `Start-AI-IDE.bat`

### macOS Portable
- Extract `AI-IDE-v1.0.0-macOS-Portable.zip`
- Run `Start-AI-IDE.sh`

### Linux Portable
- Extract `AI-IDE-v1.0.0-Linux-Portable.tar.gz`
- Run `./Start-AI-IDE.sh`

## Installer Packages

### Windows (NSIS)
- **File**: `AI IDE Setup 1.0.0.exe`
- **Features**: Custom installation directory, desktop shortcut, start menu entry
- **Uninstaller**: Included in installation

### macOS (DMG)
- **File**: `AI IDE-1.0.0.dmg`
- **Features**: Drag-and-drop installation, background image
- **Code Signing**: Configure in `package.json` build section

### Linux (Multiple Formats)
- **AppImage**: `AI IDE-1.0.0.AppImage` - Universal Linux binary
- **DEB**: `ai-ide_1.0.0_amd64.deb` - Debian/Ubuntu package
- **RPM**: `ai-ide-1.0.0.x86_64.rpm` - Red Hat/Fedora package

## Configuration

### Build Configuration

Edit `package.json` to customize build settings:

```json
{
  "build": {
    "appId": "dev.ai-ide.app",
    "productName": "AI IDE",
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": [
        { "target": "nsis", "arch": ["x64", "arm64"] },
        { "target": "portable", "arch": ["x64", "arm64"] }
      ]
    }
  }
}
```

### Code Signing

#### Windows
```json
{
  "build": {
    "win": {
      "certificateFile": "path/to/certificate.p12",
      "certificatePassword": "password"
    }
  }
}
```

#### macOS
```json
{
  "build": {
    "mac": {
      "identity": "Developer ID Application: Your Name (TEAM_ID)"
    }
  }
}
```

## Troubleshooting

### Common Issues

#### Build Fails on Windows
- Ensure Visual Studio Build Tools are installed
- Check that Python is in PATH
- Verify Node.js version compatibility

#### macOS Code Signing Issues
- Ensure Apple Developer account is active
- Check certificate validity
- Verify provisioning profiles

#### Linux Dependencies Missing
```bash
# Ubuntu/Debian
sudo apt-get install build-essential libnss3-dev libatk-bridge2.0-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install nss-devel atk-devel
```

#### Large Bundle Size
- Review included files in `package.json`
- Exclude unnecessary backend files
- Optimize Python dependencies

### Debug Mode

Enable debug output:

```bash
# Set debug environment variable
export DEBUG=electron-builder

# Run build with debug info
npm run build
```

## Distribution

### Release Checklist

- [ ] Version number updated in `package.json`
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Release notes created
- [ ] Checksums generated
- [ ] Code signed (if applicable)
- [ ] Tested on target platforms

### Upload Platforms

- **GitHub Releases**: Attach ZIP/TAR files and installers
- **Microsoft Store**: Submit MSIX package
- **Mac App Store**: Submit signed DMG
- **Snap Store**: Submit snap package
- **Flathub**: Submit Flatpak
- **AUR**: Submit PKGBUILD for Arch Linux

## Performance Optimization

### Bundle Size Reduction

1. **Exclude unnecessary files**:
   ```json
   {
     "build": {
       "files": [
         "!backend/**/__pycache__",
         "!backend/**/venv",
         "!**/*.pyc"
       ]
     }
   }
   ```

2. **Compress resources**:
   ```json
   {
     "build": {
       "compression": "maximum"
     }
   }
   ```

3. **Use asar packaging**:
   ```json
   {
     "build": {
       "asar": true
     }
   }
   ```

### Startup Performance

1. **Lazy load modules** in main process
2. **Preload critical resources** in renderer
3. **Optimize Python backend** startup time
4. **Cache embeddings** and models

## Security Considerations

### Code Signing
- Always sign releases for distribution
- Use timestamping for long-term validity
- Verify signatures before release

### Sandboxing
- Enable context isolation in renderer
- Disable node integration where possible
- Validate all IPC communications

### Updates
- Implement secure auto-update mechanism
- Use HTTPS for update checks
- Verify update signatures

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [windows-latest, macos-latest, ubuntu-latest]
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - run: npm run install:all
      - run: npm run build
      - run: npm run package
      
      - uses: actions/upload-artifact@v3
        with:
          name: releases-${{ matrix.os }}
          path: releases/
```

## Support

For packaging issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review [Electron Builder documentation](https://www.electron.build/)
3. Open an issue on GitHub
4. Contact the development team

---

**AI IDE Packaging System** - Creating the future of AI-powered development tools!