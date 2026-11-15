#!/usr/bin/env bash
# DINO lane (GPU compute-bound, one job at a time). Priority: all LOCO, then AD, then ablations, then reproduction.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; LOG=logs/lane_dino.log
LOCO="breakfast_box juice_bottle pushpins screw_bag splicing_connectors"
AD="bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper"
run() { $PY -m hybridad.cli prepare --experiment "experiments/$1.yaml" --dataset $2 --category $3 >> $LOG 2>&1; echo "$(date +%H:%M:%S) done $1 $2/$3" >> $LOG; }
for c in $LOCO; do run C_dino mvtec_loco $c; done
for c in $AD; do run C_dino mvtec_ad $c; done
echo "DINO_REF_DONE $(date)" >> $LOG
for v in C_dino_compact C_dino_identity C_dino_softmax; do run $v mvtec_ad bottle; run $v mvtec_loco breakfast_box; done
for v in C_dino_compact C_dino_identity C_dino_softmax; do
  for c in juice_bottle pushpins screw_bag splicing_connectors; do run $v mvtec_loco $c; done
  for c in cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do run $v mvtec_ad $c; done
done
echo "DINO_ABLATIONS_DONE $(date)" >> $LOG
$PY -m hybridad.cli prepare --experiment experiments/R_dino_published.yaml --dataset mvtec_ad >> $LOG 2>&1
echo "LANE_DINO_DONE $(date)" >> $LOG
