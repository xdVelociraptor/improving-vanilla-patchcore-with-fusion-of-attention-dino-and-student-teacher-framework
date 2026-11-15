# Restore a local research archive into this separate checkout and run its quick self-check.
param([string]$Archive = (Join-Path $env:USERPROFILE "OneDrive\Desktop\kutty dinosaur"))
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArchivePython = Join-Path $Archive ".venv\Scripts\python.exe"
if ((Test-Path -LiteralPath (Join-Path $Root "runs_check")) -or
    (Test-Path -LiteralPath (Join-Path $Root "logs\self_check.json"))) {
    throw "The quick check needs a fresh runs_check folder and logs\self_check.json. Preserve or move the existing items before continuing."
}
if (-not (Test-Path -LiteralPath $ArchivePython -PathType Leaf)) {
    throw "The original archive environment was not found. Use -Archive with the full path to the original project."
}
# -B prevents Python bytecode writes anywhere in the source archive.
& $ArchivePython -B (Join-Path $Root "scripts\restore_from_archive.py") --archive $Archive
if ($LASTEXITCODE -ne 0) { throw "Archive copy failed; existing files were not overwritten." }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = (Join-Path $Root "src")
& $Py (Join-Path $Root "check_package.py") --quick
if ($LASTEXITCODE -ne 0) { throw "The quick project check failed. Review the output and logs\self_check.json." }
Write-Host "Ready. Run .\hybridad.cmd --help or see README.md for experiment commands."
