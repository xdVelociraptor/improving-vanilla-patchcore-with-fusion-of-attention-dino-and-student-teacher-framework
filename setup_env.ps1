# One-command environment setup for this folder (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1          create or reuse the project .venv
#   powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Rebuild rebuild .venv from requirements.lock.txt (downloads)
# The original environment was built on the original machine with Python 3.12.10. On another machine, or if the folder was
# moved and the environment no longer starts, use -Rebuild (needs Python 3.12 and internet access).
param([switch]$Rebuild)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

function Test-Env {
    if (-not (Test-Path $Py)) { return $false }
    & $Py -c "import torch, torchvision, faiss, sklearn, scipy, pandas, yaml, tifffile, matplotlib" 2>$null
    return ($LASTEXITCODE -eq 0)
}

if ($Rebuild -or -not (Test-Env)) {
    if ((Test-Path $Venv) -and -not $Rebuild) {
        Write-Host "The existing .venv does not work on this machine. Re-run with -Rebuild to recreate it."; exit 1
    }
    if (Test-Path $Venv) { Remove-Item $Venv -Recurse -Force }
    $base = $null
    if (Get-Command py -ErrorAction SilentlyContinue) { $base = @("py", "-3.12") } else { $base = @("python") }
    & $base[0] $base[1..($base.Length - 1)] -m venv $Venv
    & $Py -m pip install --upgrade pip
    & $Py -m pip install -r (Join-Path $Root "requirements.lock.txt") --extra-index-url https://download.pytorch.org/whl/cu128
}
# (Re)register this folder's src/ as the hybridad package (also needed after moving the folder).
& $Py -m pip install -e $Root --no-deps --no-build-isolation --no-index -q
& $Py -m hybridad.cli --help
exit $LASTEXITCODE
