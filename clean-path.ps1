#!/usr/bin/env pwsh
# PATH Cleanup Script
# Backs up current PATH, removes duplicates, and sets clean PATH from PATH.md

param(
    [switch]$Restore,
    [switch]$ShowCurrent
)

$BackupFile = "path-backup.json"

function Show-CurrentPath {
    Write-Host "=== CURRENT PATH ANALYSIS ===" -ForegroundColor Cyan
    $currentPath = $env:PATH -split ";"
    Write-Host "Total PATH entries: $($currentPath.Count)" -ForegroundColor Yellow
    
    # Find duplicates
    $duplicates = $currentPath | Group-Object | Where-Object { $_.Count -gt 1 }
    if ($duplicates) {
        Write-Host "`nDUPLICATE ENTRIES:" -ForegroundColor Red
        foreach ($dup in $duplicates) {
            Write-Host "  '$($dup.Name)' appears $($dup.Count) times" -ForegroundColor Red
        }
    }
    
    Write-Host "`nCURRENT PATH:" -ForegroundColor Green
    $currentPath | ForEach-Object { Write-Host "  $_" }
}

function Backup-CurrentPath {
    Write-Host "Backing up current PATH..." -ForegroundColor Yellow
    
    $backup = @{
        Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
        SystemPath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        CurrentSessionPath = $env:PATH
    }
    
    $backup | ConvertTo-Json | Set-Content $BackupFile
    Write-Host "PATH backed up to $BackupFile" -ForegroundColor Green
}

function Restore-Path {
    if (-not (Test-Path $BackupFile)) {
        Write-Host "No backup file found at $BackupFile" -ForegroundColor Red
        return
    }
    
    Write-Host "Restoring PATH from backup..." -ForegroundColor Yellow
    $backup = Get-Content $BackupFile | ConvertFrom-Json
    
    Write-Host "Backup from: $($backup.Timestamp)" -ForegroundColor Cyan
    
    # Restore user PATH
    if ($backup.UserPath) {
        [Environment]::SetEnvironmentVariable("PATH", $backup.UserPath, "User")
        Write-Host "User PATH restored" -ForegroundColor Green
    }
    
    # Note: We don't restore system PATH as that requires admin rights
    Write-Host "System PATH not restored (requires admin rights)" -ForegroundColor Yellow
    
    # Set current session PATH
    $env:PATH = $backup.CurrentSessionPath
    Write-Host "Current session PATH restored" -ForegroundColor Green
}

function Set-CleanPath {
    Write-Host "Reading desired PATH from PATH.md..." -ForegroundColor Yellow
    
    if (-not (Test-Path "PATH.md")) {
        Write-Host "PATH.md not found!" -ForegroundColor Red
        return
    }
    
    $pathContent = Get-Content "PATH.md" -Raw
    $lines = $pathContent -split "`n" | Where-Object { $_.Trim() -ne "" }
    
    if ($lines.Count -lt 2) {
        Write-Host "PATH.md should contain two lines: User PATH and System PATH" -ForegroundColor Red
        return
    }
    
    # Parse user path (first line)
    $userPathLine = $lines[0].Trim()
    if ($userPathLine.StartsWith("PATH=")) {
        $userPathLine = $userPathLine.Substring(5)
    }
    
    # Parse system path (second line)
    $systemPathLine = $lines[1].Trim()
    if ($systemPathLine.StartsWith("%PATH%;")) {
        $systemPathLine = $systemPathLine.Substring(7)
    }
    
    # Expand variables and clean paths
    $userPaths = $userPathLine -split ";" | ForEach-Object {
        $path = $_.Trim()
        # Remove problematic quote characters
        $path = $path -replace '"', ''
        # Remove ${env:PATH} references
        $path = $path -replace '\$\{env:PATH\}', ''
        # Replace %UP% with user profile
        $path = $path -replace "%UP%", $env:USERPROFILE
        # Replace %HF% with a reasonable default
        $path = $path -replace "%HF%", "$env:USERPROFILE\AppData\Roaming\Python\Python313\Scripts"
        return $path
    } | Where-Object { $_ -ne "" -and $_ -ne "%PATH%" }
    
    $systemPaths = $systemPathLine -split ";" | ForEach-Object {
        $path = $_.Trim()
        # Remove problematic quote characters
        $path = $path -replace '"', ''
        # Replace system variables
        $path = $path -replace "%SystemRoot%", $env:SystemRoot
        $path = $path -replace "%SYSTEMROOT%", $env:SystemRoot
        $path = $path -replace "%NVM_HOME%", "$env:USERPROFILE\AppData\Roaming\nvm"
        $path = $path -replace "%NVM_SYMLINK%", "C:\Program Files\nodejs"
        return $path
    } | Where-Object { $_ -ne "" -and $_ -ne "%PATH%" }
    
    # Combine and remove duplicates while preserving order
    $allPaths = @()
    $seen = @{}
    
    # Add user paths first
    foreach ($path in $userPaths) {
        # Remove quotes and clean up the path
        $cleanPath = $path.Trim('"').Trim()
        if (-not $seen.ContainsKey($cleanPath.ToLower()) -and $cleanPath -ne "" -and $cleanPath -ne '${env:PATH}') {
            $allPaths += $cleanPath
            $seen[$cleanPath.ToLower()] = $true
        }
    }
    
    # Add system paths
    foreach ($path in $systemPaths) {
        # Remove quotes and clean up the path
        $cleanPath = $path.Trim('"').Trim()
        if (-not $seen.ContainsKey($cleanPath.ToLower()) -and $cleanPath -ne "" -and $cleanPath -ne '${env:PATH}') {
            $allPaths += $cleanPath
            $seen[$cleanPath.ToLower()] = $true
        }
    }
    
    # Create clean PATH
    $cleanPath = $allPaths -join ";"
    
    Write-Host "`nCLEAN PATH SUMMARY:" -ForegroundColor Cyan
    Write-Host "Original PATH entries: $((($env:PATH -split ";").Count))" -ForegroundColor Red
    Write-Host "Clean PATH entries: $($allPaths.Count)" -ForegroundColor Green
    Write-Host "Duplicates removed: $((($env:PATH -split ";").Count) - $allPaths.Count)" -ForegroundColor Yellow
    
    # Set the clean PATH for current session
    $env:PATH = $cleanPath
    
    # Set user PATH (this persists)
    $userOnlyPath = $userPaths -join ";"
    [Environment]::SetEnvironmentVariable("PATH", $userOnlyPath, "User")
    
    Write-Host "`nPATH cleaned successfully!" -ForegroundColor Green
    Write-Host "Current session PATH updated" -ForegroundColor Green
    Write-Host "User PATH updated (persistent)" -ForegroundColor Green
    Write-Host "Note: System PATH changes require administrator privileges" -ForegroundColor Yellow
    
    Write-Host "`nNEW CLEAN PATH:" -ForegroundColor Green
    $allPaths | ForEach-Object { Write-Host "  $_" }
}

# Main execution
if ($ShowCurrent) {
    Show-CurrentPath
    exit
}

if ($Restore) {
    Restore-Path
    exit
}

# Default action: backup and clean
Write-Host "=== PATH CLEANUP SCRIPT ===" -ForegroundColor Cyan
Write-Host "This script will:"
Write-Host "1. Backup your current PATH"
Write-Host "2. Remove duplicates"
Write-Host "3. Set clean PATH from PATH.md"
Write-Host ""

$confirm = Read-Host "Continue? (y/N)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit
}

Backup-CurrentPath
Set-CleanPath

Write-Host "`n=== CLEANUP COMPLETE ===" -ForegroundColor Cyan
Write-Host "To restore your original PATH, run: .\clean-path.ps1 -Restore" -ForegroundColor Yellow
Write-Host "To view current PATH analysis, run: .\clean-path.ps1 -ShowCurrent" -ForegroundColor Yellow