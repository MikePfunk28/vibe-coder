#!/usr/bin/env pwsh
# Setup build environment with proper PATH for VSCode OSS build

Write-Host "Setting up build environment for Mike-AI-IDE..."

# Set comprehensive PATH that includes all necessary tools
$newPath = @(
    "C:\Program Files\nodejs\"
    "C:\Users\mikep\AppData\Roaming\nvm\v22.17.0"
    "C:\Users\mikep\AppData\Roaming\npm"
    "C:\WINDOWS\System32"
    "C:\WINDOWS"
    "C:\WINDOWS\System32\Wbem"
    "C:\Windows\System32\WindowsPowerShell\v1.0\"
    "C:\Windows\System32\PowerShell\7.5.1"
    "C:\Program Files\PowerShell\7"
    $env:PATH
) -join ";"

# Set environment variables for this session and all child processes
[Environment]::SetEnvironmentVariable("PATH", $newPath, "Process")
$env:PATH = $newPath

Write-Host "PATH updated. Testing tools..."
Write-Host "Node.js: $(node --version)"
Write-Host "Yarn: $(yarn --version)"
Write-Host "NPM: $(npm --version)"

Write-Host "Environment ready. You can now run yarn commands."