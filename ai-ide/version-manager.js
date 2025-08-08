#!/usr/bin/env node

/**
 * AI IDE Version Manager
 * Handles versioning, releases, and iterations
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class VersionManager {
    constructor() {
        this.packagePath = path.join(__dirname, 'package.json');
        this.changelogPath = path.join(__dirname, 'CHANGELOG.md');
        this.versionHistoryPath = path.join(__dirname, 'VERSION_HISTORY.json');
        
        this.loadCurrentVersion();
        this.loadVersionHistory();
    }

    loadCurrentVersion() {
        try {
            const packageJson = JSON.parse(fs.readFileSync(this.packagePath, 'utf8'));
            this.currentVersion = packageJson.version;
            this.packageJson = packageJson;
        } catch (error) {
            console.error('Failed to load package.json:', error);
            this.currentVersion = '1.0.0';
        }
    }

    loadVersionHistory() {
        try {
            if (fs.existsSync(this.versionHistoryPath)) {
                this.versionHistory = JSON.parse(fs.readFileSync(this.versionHistoryPath, 'utf8'));
            } else {
                this.versionHistory = {
                    versions: [],
                    releases: [],
                    iterations: []
                };
            }
        } catch (error) {
            console.error('Failed to load version history:', error);
            this.versionHistory = { versions: [], releases: [], iterations: [] };
        }
    }

    saveVersionHistory() {
        try {
            fs.writeFileSync(this.versionHistoryPath, JSON.stringify(this.versionHistory, null, 2));
        } catch (error) {
            console.error('Failed to save version history:', error);
        }
    }

    parseVersion(version) {
        const parts = version.split('.');
        return {
            major: parseInt(parts[0]) || 0,
            minor: parseInt(parts[1]) || 0,
            patch: parseInt(parts[2]) || 0
        };
    }

    formatVersion(versionObj) {
        return `${versionObj.major}.${versionObj.minor}.${versionObj.patch}`;
    }

    incrementVersion(type = 'patch') {
        const current = this.parseVersion(this.currentVersion);
        
        switch (type) {
            case 'major':
                current.major++;
                current.minor = 0;
                current.patch = 0;
                break;
            case 'minor':
                current.minor++;
                current.patch = 0;
                break;
            case 'patch':
            default:
                current.patch++;
                break;
        }
        
        return this.formatVersion(current);
    }

    async createNewVersion(type = 'patch', description = '') {
        const oldVersion = this.currentVersion;
        const newVersion = this.incrementVersion(type);
        
        console.log(`🚀 Creating new version: ${oldVersion} → ${newVersion}`);
        
        // Update package.json
        this.packageJson.version = newVersion;
        fs.writeFileSync(this.packagePath, JSON.stringify(this.packageJson, null, 2));
        
        // Create version entry
        const versionEntry = {
            version: newVersion,
            previousVersion: oldVersion,
            type: type,
            description: description,
            timestamp: new Date().toISOString(),
            features: [],
            bugfixes: [],
            breaking: [],
            gitCommit: this.getGitCommit()
        };
        
        this.versionHistory.versions.push(versionEntry);
        this.currentVersion = newVersion;
        
        // Update changelog
        this.updateChangelog(versionEntry);
        
        // Save history
        this.saveVersionHistory();
        
        console.log(`✅ Version ${newVersion} created successfully`);
        return newVersion;
    }

    async createIteration(description = '') {
        const iterationNumber = this.versionHistory.iterations.length + 1;
        const iterationId = `${this.currentVersion}-iter.${iterationNumber}`;
        
        console.log(`🔄 Creating iteration: ${iterationId}`);
        
        const iteration = {
            id: iterationId,
            version: this.currentVersion,
            number: iterationNumber,
            description: description,
            timestamp: new Date().toISOString(),
            changes: [],
            gitCommit: this.getGitCommit(),
            buildNumber: this.generateBuildNumber()
        };
        
        this.versionHistory.iterations.push(iteration);
        this.saveVersionHistory();
        
        console.log(`✅ Iteration ${iterationId} created successfully`);
        return iteration;
    }

    async createRelease(version = null, releaseNotes = '') {
        const releaseVersion = version || this.currentVersion;
        const releaseId = `release-${releaseVersion}`;
        
        console.log(`📦 Creating release: ${releaseVersion}`);
        
        const release = {
            id: releaseId,
            version: releaseVersion,
            releaseNotes: releaseNotes,
            timestamp: new Date().toISOString(),
            gitCommit: this.getGitCommit(),
            buildNumber: this.generateBuildNumber(),
            artifacts: [],
            status: 'draft'
        };
        
        // Build the release
        await this.buildRelease(release);
        
        this.versionHistory.releases.push(release);
        this.saveVersionHistory();
        
        console.log(`✅ Release ${releaseVersion} created successfully`);
        return release;
    }

    async buildRelease(release) {
        console.log('🔨 Building release artifacts...');
        
        try {
            // Run the build process
            execSync('npm run package', { stdio: 'inherit' });
            
            // List created artifacts
            const releasesDir = path.join(__dirname, 'releases');
            if (fs.existsSync(releasesDir)) {
                const artifacts = fs.readdirSync(releasesDir)
                    .filter(file => file.includes(release.version))
                    .map(file => ({
                        name: file,
                        path: path.join(releasesDir, file),
                        size: fs.statSync(path.join(releasesDir, file)).size,
                        type: this.getArtifactType(file)
                    }));
                
                release.artifacts = artifacts;
                release.status = 'built';
                
                console.log(`📦 Built ${artifacts.length} artifacts:`);
                artifacts.forEach(artifact => {
                    const sizeMB = (artifact.size / 1024 / 1024).toFixed(1);
                    console.log(`  - ${artifact.name} (${sizeMB} MB)`);
                });
            }
            
        } catch (error) {
            console.error('❌ Build failed:', error.message);
            release.status = 'failed';
            release.buildError = error.message;
        }
    }

    getArtifactType(filename) {
        if (filename.endsWith('.exe')) return 'windows-installer';
        if (filename.endsWith('.dmg')) return 'macos-installer';
        if (filename.endsWith('.AppImage')) return 'linux-appimage';
        if (filename.endsWith('.deb')) return 'linux-deb';
        if (filename.endsWith('.rpm')) return 'linux-rpm';
        if (filename.includes('Portable.zip')) return 'windows-portable';
        if (filename.includes('Portable.tar.gz')) return 'linux-portable';
        return 'unknown';
    }

    generateBuildNumber() {
        const timestamp = Date.now();
        const random = Math.floor(Math.random() * 1000);
        return `${timestamp}-${random}`;
    }

    getGitCommit() {
        try {
            return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
        } catch (error) {
            return 'unknown';
        }
    }

    updateChangelog(versionEntry) {
        const changelogEntry = `
## [${versionEntry.version}] - ${new Date(versionEntry.timestamp).toISOString().split('T')[0]}

### ${versionEntry.type.charAt(0).toUpperCase() + versionEntry.type.slice(1)} Release

${versionEntry.description}

### Features
${versionEntry.features.length > 0 ? versionEntry.features.map(f => `- ${f}`).join('\n') : '- No new features'}

### Bug Fixes
${versionEntry.bugfixes.length > 0 ? versionEntry.bugfixes.map(f => `- ${f}`).join('\n') : '- No bug fixes'}

### Breaking Changes
${versionEntry.breaking.length > 0 ? versionEntry.breaking.map(f => `- ${f}`).join('\n') : '- No breaking changes'}

---
`;

        if (fs.existsSync(this.changelogPath)) {
            const existingChangelog = fs.readFileSync(this.changelogPath, 'utf8');
            const newChangelog = `# AI IDE Changelog\n\n${changelogEntry}\n${existingChangelog.replace('# AI IDE Changelog\n\n', '')}`;
            fs.writeFileSync(this.changelogPath, newChangelog);
        } else {
            fs.writeFileSync(this.changelogPath, `# AI IDE Changelog\n\n${changelogEntry}`);
        }
    }

    listVersions() {
        console.log('📋 Version History:');
        console.log(`Current Version: ${this.currentVersion}`);
        console.log('');
        
        if (this.versionHistory.versions.length > 0) {
            console.log('Previous Versions:');
            this.versionHistory.versions.slice(-10).reverse().forEach(version => {
                const date = new Date(version.timestamp).toLocaleDateString();
                console.log(`  ${version.version} (${version.type}) - ${date}`);
                if (version.description) {
                    console.log(`    ${version.description}`);
                }
            });
        }
        
        console.log('');
        console.log(`Total Versions: ${this.versionHistory.versions.length}`);
        console.log(`Total Iterations: ${this.versionHistory.iterations.length}`);
        console.log(`Total Releases: ${this.versionHistory.releases.length}`);
    }

    listReleases() {
        console.log('📦 Release History:');
        
        if (this.versionHistory.releases.length > 0) {
            this.versionHistory.releases.forEach(release => {
                const date = new Date(release.timestamp).toLocaleDateString();
                console.log(`\n${release.version} - ${date} [${release.status}]`);
                if (release.releaseNotes) {
                    console.log(`  ${release.releaseNotes}`);
                }
                if (release.artifacts.length > 0) {
                    console.log(`  Artifacts: ${release.artifacts.length}`);
                    release.artifacts.forEach(artifact => {
                        const sizeMB = (artifact.size / 1024 / 1024).toFixed(1);
                        console.log(`    - ${artifact.name} (${sizeMB} MB)`);
                    });
                }
            });
        } else {
            console.log('No releases found.');
        }
    }

    async rollback(targetVersion) {
        console.log(`🔄 Rolling back to version ${targetVersion}...`);
        
        const versionExists = this.versionHistory.versions.find(v => v.version === targetVersion);
        if (!versionExists) {
            console.error(`❌ Version ${targetVersion} not found in history`);
            return false;
        }
        
        // Update package.json
        this.packageJson.version = targetVersion;
        fs.writeFileSync(this.packagePath, JSON.stringify(this.packageJson, null, 2));
        
        this.currentVersion = targetVersion;
        
        console.log(`✅ Rolled back to version ${targetVersion}`);
        return true;
    }

    getVersionInfo() {
        return {
            current: this.currentVersion,
            history: this.versionHistory,
            totalVersions: this.versionHistory.versions.length,
            totalIterations: this.versionHistory.iterations.length,
            totalReleases: this.versionHistory.releases.length,
            lastRelease: this.versionHistory.releases[this.versionHistory.releases.length - 1],
            lastIteration: this.versionHistory.iterations[this.versionHistory.iterations.length - 1]
        };
    }
}

// CLI Interface
if (require.main === module) {
    const versionManager = new VersionManager();
    const command = process.argv[2];
    const args = process.argv.slice(3);

    switch (command) {
        case 'bump':
            const type = args[0] || 'patch';
            const description = args[1] || '';
            versionManager.createNewVersion(type, description);
            break;
            
        case 'iterate':
            const iterDescription = args[0] || '';
            versionManager.createIteration(iterDescription);
            break;
            
        case 'release':
            const releaseVersion = args[0];
            const releaseNotes = args[1] || '';
            versionManager.createRelease(releaseVersion, releaseNotes);
            break;
            
        case 'list':
            versionManager.listVersions();
            break;
            
        case 'releases':
            versionManager.listReleases();
            break;
            
        case 'rollback':
            const targetVersion = args[0];
            if (targetVersion) {
                versionManager.rollback(targetVersion);
            } else {
                console.error('Please specify target version');
            }
            break;
            
        case 'info':
            console.log(JSON.stringify(versionManager.getVersionInfo(), null, 2));
            break;
            
        default:
            console.log(`
AI IDE Version Manager

Usage:
  node version-manager.js <command> [options]

Commands:
  bump [major|minor|patch] [description]  - Create new version
  iterate [description]                   - Create new iteration
  release [version] [notes]              - Create release
  list                                   - List version history
  releases                              - List releases
  rollback <version>                    - Rollback to version
  info                                  - Show version info

Examples:
  node version-manager.js bump patch "Bug fixes"
  node version-manager.js bump minor "New AI features"
  node version-manager.js iterate "Performance improvements"
  node version-manager.js release 1.0.0 "Initial release"
            `);
            break;
    }
}

module.exports = VersionManager;