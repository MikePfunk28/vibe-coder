# Package AI Assistant Extension
Write-Host "Compiling TypeScript..." -ForegroundColor Green

# Compile without tests
npx tsc --skipLibCheck --noEmit false

if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful!" -ForegroundColor Green
    
    # Package the extension
    Write-Host "Packaging extension..." -ForegroundColor Yellow
    npx vsce package --no-dependencies
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Extension packaged successfully!" -ForegroundColor Green
        Write-Host "VSIX file created in current directory" -ForegroundColor Cyan
    } else {
        Write-Host "Extension packaging failed!" -ForegroundColor Red
    }
} else {
    Write-Host "Compilation failed!" -ForegroundColor Red
}