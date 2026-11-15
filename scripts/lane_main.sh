#!/usr/bin/env bash
# Single sequential GPU lane (the 8 GB GPU cannot share DINO training without spilling to system RAM).
# Per category: local ST, then DINO. LOCO first (completes a dataset for the full model), then AD,
# then the no-penalty ST ablation, the DINO head/attention ablations and the published reproductions.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; LOG=logs/lane_main.log
# Every job runs the unchanged CLI under a per-process CUDA memory cap (user decision 2026-09-28; see scripts/run_capped.py).
CLI="$PY scripts/run_capped.py --fraction 0.8 --"
WAIT_PID=${1:-}
if [ -n "$WAIT_PID" ]; then
  while tasklist //FI "PID eq $WAIT_PID" 2>/dev/null | grep -q "$WAIT_PID"; do sleep 30; done
fi
run() { $CLI prepare --experiment "experiments/$1.yaml" --dataset $2 --category $3 >> $LOG 2>&1; echo "$(date +%H:%M:%S) done $1 $2/$3" >> $LOG; }
run C_dino mvtec_loco breakfast_box          # no-op if the running job finished; resumes otherwise
run B_local_st mvtec_loco breakfast_box
for c in juice_bottle pushpins screw_bag splicing_connectors; do run B_local_st mvtec_loco $c; run C_dino mvtec_loco $c; done
echo "LOCO_REF_DONE $(date)" >> $LOG
run C_dino mvtec_ad bottle
for c in cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
  run B_local_st mvtec_ad $c; run C_dino mvtec_ad $c
done
echo "AD_REF_DONE $(date)" >> $LOG
for v in C_dino_compact C_dino_identity C_dino_softmax B_local_st_nopenalty; do run $v mvtec_ad bottle; run $v mvtec_loco breakfast_box; done
for v in C_dino_compact B_local_st_nopenalty C_dino_identity C_dino_softmax; do
  for c in juice_bottle pushpins screw_bag splicing_connectors; do run $v mvtec_loco $c; done
  for c in cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do run $v mvtec_ad $c; done
done
echo "ABLATIONS_DONE $(date)" >> $LOG
$CLI prepare --experiment experiments/R_patchcore_published.yaml experiments/R_dino_published.yaml --dataset mvtec_ad >> $LOG 2>&1
echo "LANE_MAIN_DONE $(date)" >> $LOG
