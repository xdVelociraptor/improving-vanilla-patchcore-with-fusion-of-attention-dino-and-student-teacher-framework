#!/usr/bin/env bash
# leather DINO retry #5 (2026-09-28), user-approved fix: the unchanged pipeline under a per-process CUDA memory cap (0.8).
# Diagnosis (scripts/diagnose_leather.py -> logs/diagnose_leather.jsonl): reserved memory grew 4.3 -> 16.8 GB by step 860
# while live memory stayed at 1.73 GB (peak 3.93 GB). The hard-mining warm-up changes temporary sizes every step, and with
# the Windows WDDM system-memory fallback cudaMalloc never fails, so the cache was never freed until system RAM ran out.
# Runs ALONE on the GPU. On success it restarts the ablation lane (also capped); on failure it stops and reports.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; LOG=logs/resume_queue.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
: > logs/leather_capped.jsonl
MARK="=== leather retry #5 (memory cap 0.8) $(date '+%F %T')"
echo "$MARK" >> logs/ad_repair.log
log "leather retry #5 (memory cap 0.8) started"
$PY scripts/run_capped.py --fraction 0.8 --mem-log logs/leather_capped.jsonl -- \
  prepare --experiment experiments/C_dino.yaml --dataset mvtec_ad --category leather >> logs/ad_repair.log 2>&1
rc=$?
if [ $rc -eq 0 ] && sed -n "/^$MARK\$/,\$p" logs/ad_repair.log | grep -q "prepared 1/1" \
   && ! sed -n "/^$MARK\$/,\$p" logs/ad_repair.log | grep -q "FAILED"; then
  log "leather retry #5 done"; echo "LEATHER_RETRY5_DONE $(date)" >> $LOG
else
  log "leather retry #5 FAILED (rc=$rc); ablation lane NOT restarted"; echo "LEATHER_RETRY5_FAILED $(date)" >> $LOG
  exit 1
fi
log "restarting the AD ablation lane (memory cap 0.8)"
bash scripts/lane_main.sh
