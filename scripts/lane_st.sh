#!/usr/bin/env bash
# Local student-teacher worker ($1 = worker index 0..2). Latency-bound at batch 1, so 3 run in parallel.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; W=$1; N=${2:-3}; LOG=logs/lane_st_$W.log
ALL=("mvtec_loco breakfast_box" "mvtec_loco juice_bottle" "mvtec_loco pushpins" "mvtec_loco screw_bag" "mvtec_loco splicing_connectors"
     "mvtec_ad bottle" "mvtec_ad cable" "mvtec_ad capsule" "mvtec_ad carpet" "mvtec_ad grid" "mvtec_ad hazelnut" "mvtec_ad leather"
     "mvtec_ad metal_nut" "mvtec_ad pill" "mvtec_ad screw" "mvtec_ad tile" "mvtec_ad toothbrush" "mvtec_ad transistor"
     "mvtec_ad wood" "mvtec_ad zipper")
for exp in B_local_st B_local_st_nopenalty; do
  for i in "${!ALL[@]}"; do
    if [ $((i % N)) -eq $W ]; then set -- ${ALL[$i]}
      $PY -m hybridad.cli prepare --experiment experiments/$exp.yaml --dataset $1 --category $2 >> $LOG 2>&1
      echo "$(date +%H:%M:%S) done $exp $1/$2" >> $LOG
    fi
  done
done
if [ $W -eq 0 ]; then $PY -m hybridad.cli prepare --experiment experiments/R_patchcore_published.yaml --dataset mvtec_ad >> $LOG 2>&1; fi
echo "LANE_ST_${W}_DONE $(date)" >> $LOG
