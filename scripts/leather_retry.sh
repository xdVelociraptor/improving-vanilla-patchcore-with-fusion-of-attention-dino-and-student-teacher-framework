#!/usr/bin/env bash
# leather DINO retry #4 (2026-09-28). Three prior runs hit CUDA OOM at iteration ~850-900; the last ran ALONE and its
# GPU memory jumped from 4.66 GB to 7.85 GB around iteration ~550-850 (logs/leather_gpu_mem.log), consistent with
# caching-allocator fragmentation during the loss warm-up (hard-mining temporaries change size every iteration).
# Mitigation WITHOUT changing math or code: PyTorch allocator settings supported on Windows
# (expandable_segments is not). Runs ALONE on the GPU after the current ablation job exits; then restarts the lane.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; LOG=logs/resume_queue.log
log() { echo "$(date '+%F %T') $*" >> $LOG; }
WAIT_PID=${1:-}
log "leather retry #4 queued; waiting for pid $WAIT_PID"
if [ -n "$WAIT_PID" ]; then while tasklist //FI "PID eq $WAIT_PID" 2>/dev/null | grep -q "$WAIT_PID"; do sleep 30; done; fi
( while true; do echo "$(date '+%T') retry4 $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';') | total $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> logs/leather_gpu_mem.log; sleep 30; done ) & MEM=$!
log "leather retry #4 started with PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128"
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
  $PY -m hybridad.cli prepare --experiment experiments/C_dino.yaml --dataset mvtec_ad --category leather >> logs/ad_repair.log 2>&1
kill $MEM 2>/dev/null
if tail -50 logs/ad_repair.log | grep -q "leather\] FAILED"; then log "leather retry #4 FAILED"; else log "leather retry #4 done"; fi
echo "LEATHER_RETRY4_DONE $(date)" >> $LOG
log "restarting the AD ablation lane"
bash scripts/lane_main.sh
