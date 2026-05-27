param(
    [string]$PythonExe = "C:\Program Files\Python313\python.exe"
)

chcp 65001
Set-Location C:\Users\Salix\Desktop\PyPTO\pypto_phase-fence-review-fixes-test
Set-ExecutionPolicy -Scope Process Bypass

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PRE_COMMIT_HOME = "$PWD\.pre-commit-cache"

# Skip only the known local blocker.
$env:SKIP = "check-english-only"

# Prefer Python 3.13 when pre-commit creates Python hook envs.
$env:PY_PYTHON = "3.13"

$ErrorActionPreference = "Stop"

$prFiles = @()
try {
    $prFiles = git diff --name-only "upstream/main...HEAD"
} catch {
    Write-Warning "Failed to diff against upstream/main. Falling back to local modified files only."
}

$localFiles = git diff --name-only
$files = @($prFiles + $localFiles) |
    Where-Object { $_ -and (Test-Path $_) } |
    Select-Object -Unique

if (-not $files -or $files.Count -eq 0) {
    Write-Host "No changed files to run pre-commit on."
    exit 0
}

Write-Host "Running pre-commit on $($files.Count) file(s)..."
& $PythonExe -m pre_commit run --files @files
