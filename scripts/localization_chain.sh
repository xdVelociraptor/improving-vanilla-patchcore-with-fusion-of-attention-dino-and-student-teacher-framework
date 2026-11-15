#!/usr/bin/env bash
# CPU-only localization chain (this session). GPU is reserved for the training lanes.
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src
PY=./.venv/Scripts/python.exe
LOG=logs/localization_chain.log
lower() {  # keep every localization/evaluator python process at below-normal priority
  while true; do
    powershell -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.CommandLine -match 'localize_stream|localize_ad|runpy') -and \$_.Name -match 'python' } | ForEach-Object { try { (Get-Process -Id \$_.ProcessId).PriorityClass = 'BelowNormal' } catch {} }" >/dev/null 2>&1
    sleep 30
  done
}
lower & LP=$!
$PY scripts/localize_stream.py score --experiment G_full --category juice_bottle pushpins screw_bag splicing_connectors --threads 1 >> $LOG 2>&1
echo "LOCO_SCORE_DONE $(date)" >> $LOG
$PY scripts/localize_ad.py --experiment G_full --threads 1 --min-free-gb 6 >> $LOG 2>&1
echo "AD_LOCALIZE_PASS_DONE $(date)" >> $LOG
kill $LP 2>/dev/null
