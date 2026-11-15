#!/usr/bin/env bash
# Runs the full evaluation once the MVTec AD reference phase has finished AND every one of the 15 categories has a
# full-model calibration (so a failed category, e.g. leather's 2026-09-27 CUDA OOM, is repaired first, not silently missing).
cd "$(dirname "$0")/.."
until grep -q AD_REF_DONE logs/lane_main.log; do sleep 120; done
CATS="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
while true; do
  missing=0
  for c in $CATS; do
    [ -f runs_rerun/G_full/mvtec_ad/$c/seed0/calibration.json ] && [ -f runs_rerun/G_full_mean/mvtec_ad/$c/seed0/calibration.json ] || missing=1
  done
  [ $missing -eq 0 ] && break
  sleep 120
done
echo "AD_EVAL_START $(date)" >> logs/eval_all.log
bash scripts/eval_all.sh
