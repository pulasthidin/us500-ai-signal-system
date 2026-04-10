<#
.SYNOPSIS
    Push full backup (including data, models, db) to private repo.
    Called automatically by "git pushall" alias.
.DESCRIPTION
    Uses a temporary worktree so the main working directory is NEVER disturbed.
    Safe to run even while the trading system is live (files locked).
    
    1. Pushes current branch to origin (public)
    2. Creates a temp worktree for the full-backup branch
    3. Copies all data/model/db/log files into the worktree
    4. Commits and pushes to backup remote (private)
    5. Cleans up the temp worktree
#>

param(
    [switch]$BackupOnly
)

$repoRoot = (git rev-parse --show-toplevel 2>&1).ToString().Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Not inside a git repository" -ForegroundColor Red
    exit 1
}

Set-Location $repoRoot

$currentBranch = (git rev-parse --abbrev-ref HEAD 2>&1).ToString().Trim()
$backupBranch  = "full-backup"
$worktreePath  = Join-Path $env:TEMP "us500-backup-worktree"
$gitignoreBackup = Join-Path $repoRoot ".gitignore.backup"

if (-not (Test-Path $gitignoreBackup)) {
    Write-Host "ERROR: .gitignore.backup not found" -ForegroundColor Red
    exit 1
}

# ── Step 1: Push to public (origin) ──────────────────────────
if (-not $BackupOnly) {
    Write-Host ""
    Write-Host "=== Pushing to PUBLIC (origin/$currentBranch) ===" -ForegroundColor Cyan
    $output = git push origin $currentBranch 2>&1
    Write-Host ($output -join "`n")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Public push failed or already up to date" -ForegroundColor Yellow
    } else {
        Write-Host "Public push OK" -ForegroundColor Green
    }
}

# ── Step 2: Prepare backup branch and worktree ───────────────
Write-Host ""
Write-Host "=== Preparing PRIVATE backup ===" -ForegroundColor Magenta

# Clean up any leftover worktree
if (Test-Path $worktreePath) {
    Remove-Item $worktreePath -Recurse -Force -ErrorAction SilentlyContinue
}
git worktree prune 2>&1 | Out-Null

# Create orphan backup branch if it doesn't exist
$branchExists = git branch --list $backupBranch 2>&1
if (-not $branchExists) {
    Write-Host "Creating backup branch..." -ForegroundColor Gray
    git checkout --orphan $backupBranch 2>&1 | Out-Null
    git reset --hard 2>&1 | Out-Null
    git commit --allow-empty -m "init: backup branch" 2>&1 | Out-Null
    git checkout $currentBranch 2>&1 | Out-Null
}

# Create worktree for the backup branch
Write-Host "Creating temporary worktree..." -ForegroundColor Gray
git worktree add $worktreePath $backupBranch 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create worktree" -ForegroundColor Red
    exit 1
}

try {
    # ── Step 3: Copy .gitignore.backup as .gitignore ─────────
    Copy-Item $gitignoreBackup (Join-Path $worktreePath ".gitignore") -Force

    # ── Step 4: Sync all files from main repo to worktree ────
    Write-Host "Syncing files to backup..." -ForegroundColor Gray
    
    # Use robocopy to mirror files (handles locked files gracefully)
    $excludeDirs = @(
        ".git", "venv", ".venv", "env", "__pycache__",
        ".pytest_cache", ".pytest_tmp", "htmlcov",
        ".claude", ".vscode", ".idea", ".cursor",
        "node_modules", ".github"
    )
    $excludeFiles = @(".env", ".DS_Store", "Thumbs.db", "desktop.ini", ".coverage")
    
    $xdArgs = ($excludeDirs | ForEach-Object { $_ }) -join " "
    $xfArgs = ($excludeFiles | ForEach-Object { $_ }) -join " "
    
    $robocopyArgs = @(
        $repoRoot, $worktreePath,
        "/MIR", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS",
        "/XD", ".git", "venv", ".venv", "env", "__pycache__",
              ".pytest_cache", ".pytest_tmp", "htmlcov",
              ".claude", ".vscode", ".idea", ".cursor",
              "node_modules", ".github",
        "/XF", ".env", ".DS_Store", "Thumbs.db", "desktop.ini", ".coverage"
    )
    
    & robocopy @robocopyArgs 2>&1 | Out-Null
    # robocopy exit codes < 8 are success
    
    # Restore the backup .gitignore (robocopy may have overwritten it)
    Copy-Item $gitignoreBackup (Join-Path $worktreePath ".gitignore") -Force

    # ── Step 5: Commit in the worktree ───────────────────────
    Write-Host "Committing backup..." -ForegroundColor Gray
    Push-Location $worktreePath
    
    git add -A 2>&1 | Out-Null
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $commitMsg = "backup: full snapshot $timestamp"
    
    $commitOutput = git commit -m $commitMsg 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "No new changes to backup" -ForegroundColor Yellow
    } else {
        Write-Host "Backup commit created" -ForegroundColor Green
    }

    # ── Step 6: Push to private ──────────────────────────────
    Write-Host "Pushing to private repo..." -ForegroundColor Gray
    $pushOutput = git push backup $backupBranch --force 2>&1
    Write-Host ($pushOutput -join "`n")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Private backup push failed!" -ForegroundColor Red
    } else {
        Write-Host "Private backup push OK" -ForegroundColor Green
    }
    
    Pop-Location

} finally {
    # ── Step 7: Clean up worktree ────────────────────────────
    Write-Host ""
    Write-Host "Cleaning up..." -ForegroundColor Gray
    
    Set-Location $repoRoot
    if (Test-Path $worktreePath) {
        Remove-Item $worktreePath -Recurse -Force -ErrorAction SilentlyContinue
    }
    git worktree prune 2>&1 | Out-Null

    Write-Host ""
    Write-Host "=== Done! ===" -ForegroundColor Green
    if (-not $BackupOnly) {
        Write-Host "  Public:  origin/$currentBranch" -ForegroundColor Cyan
    }
    Write-Host "  Private: backup/$backupBranch" -ForegroundColor Magenta
}
