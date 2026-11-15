#!/usr/bin/env bash
# Adds AD pixel metrics as AD categories finish training (CPU-only, RAM-gated, below-normal priority).
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src
PY=./.venv/Scripts/python.exe; LOG=logs/localization_chain.log
while [ "$(ls runs_rerun/G_full/mvtec_ad/*/seed0/localization.json 2>/dev/null | wc -l)" -lt 15 ]; do
  $PY scripts/localize_ad.py --experiment G_full --threads 1 --min-free-gb 6 >> $LOG 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null; do
    powershell -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.CommandLine -match 'localize_ad') -and \$_.Name -match 'python' } | ForEach-Object { try { (Get-Process -Id \$_.ProcessId).PriorityClass = 'BelowNormal' } catch {} }" >/dev/null 2>&1
    sleep 30
  done
  $PY -m hybridad.cli report >> $LOG 2>&1
  sleep 1800
done
echo "AD_LOCALIZATION_COMPLETE $(date)" >> $LOG
