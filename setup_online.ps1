# Restore the public research archive and publisher-hosted inputs on Windows.
param([switch]$Plan)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
Push-Location $ProjectRoot
try {
    if ($Plan) {
        $Downloads = Get-Content -LiteralPath (Join-Path $ProjectRoot 'artifact-downloads.json') -Raw | ConvertFrom-Json
        $Downloads | Select-Object group,name,url | Format-Table -AutoSize
        exit 0
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot 'setup_env.ps1')
    if ($LASTEXITCODE -ne 0) { throw 'Environment setup failed.' }
    & .\.venv\Scripts\python.exe scripts/download_artifacts.py
    if ($LASTEXITCODE -ne 0) { throw 'Artifact restoration failed. Resolve the reported error and rerun.' }
    & .\.venv\Scripts\python.exe scripts/verify_datasets.py
    if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
    & .\.venv\Scripts\python.exe -m hybridad.cli --help
    if ($LASTEXITCODE -ne 0) { throw 'CLI import check failed.' }
    Write-Host 'Setup complete. See README.md for prepare/evaluate commands. This does not train or evaluate new images.'
} finally { Pop-Location }
