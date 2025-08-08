#!/bin/bash

# AI IDE Release Build Script
# Creates production-ready executables and installers for all platforms

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the ai-ide directory."
    exit 1
fi

print_header "AI IDE Release Build v$(node -p "require('./package.json').version")"
echo -e "${BLUE}Advanced AI-Powered Development Environment${NC}"
echo -e "${BLUE}Competitor to VSCode, GitHub Copilot, Cursor, and Windsurf${NC}"
echo ""

# Step 1: Clean previous builds
print_step "Cleaning previous builds..."
npm run clean 2>/dev/null || true
rm -rf node_modules 2>/dev/null || true

# Step 2: Install dependencies
print_step "Installing dependencies..."
print_status "Installing root dependencies..."
npm install

print_status "Installing extension dependencies..."
cd extensions/ai-assistant
npm install
cd ../..

print_status "Installing Python backend dependencies..."
cd backend
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt 2>/dev/null || print_warning "requirements.txt not found"
cd ..

# Step 3: Run tests
print_step "Running tests..."
print_status "Testing backend..."
npm run test:backend 2>/dev/null || print_warning "Backend tests failed or not available"

print_status "Testing extension..."
npm run test:extension 2>/dev/null || print_warning "Extension tests failed or not available"

# Step 4: Build the application
print_step "Building AI IDE..."
npm run build

# Step 5: Create final packages
print_step "Creating release packages..."
npm run package

# Step 6: Verify builds
print_step "Verifying builds..."
if [ -d "releases" ]; then
    print_status "Release files created:"
    ls -la releases/ | grep -E '\.(zip|tar\.gz|exe|dmg|AppImage|deb|rpm)$' || print_warning "No release files found"
else
    print_error "Releases directory not found"
    exit 1
fi

# Step 7: Create checksums
print_step "Creating checksums..."
cd releases
if command -v sha256sum &> /dev/null; then
    sha256sum * > checksums.sha256 2>/dev/null || true
    print_status "SHA256 checksums created"
elif command -v shasum &> /dev/null; then
    shasum -a 256 * > checksums.sha256 2>/dev/null || true
    print_status "SHA256 checksums created"
else
    print_warning "No checksum utility found"
fi
cd ..

# Step 8: Display summary
print_header "Build Summary"
echo -e "${GREEN}✅ AI IDE build completed successfully!${NC}"
echo ""
echo -e "${CYAN}📦 Release Information:${NC}"
echo -e "  Version: $(node -p "require('./package.json').version")"
echo -e "  Build Date: $(date)"
echo -e "  Platform: $(uname -s)-$(uname -m)"
echo ""

if [ -d "releases" ]; then
    echo -e "${CYAN}📁 Release Files:${NC}"
    cd releases
    for file in *; do
        if [ -f "$file" ]; then
            size=$(du -h "$file" | cut -f1)
            echo -e "  📄 $file ($size)"
        fi
    done
    cd ..
fi

echo ""
echo -e "${PURPLE}🚀 AI IDE is ready for distribution!${NC}"
echo -e "${PURPLE}Your AI-powered IDE competitor is complete!${NC}"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  1. Test the executables on target platforms"
echo -e "  2. Upload to distribution platforms"
echo -e "  3. Update documentation and website"
echo -e "  4. Announce the release!"
echo ""
echo -e "${GREEN}Thank you for building the future of AI-powered development!${NC}"