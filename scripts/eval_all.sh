#!/usr/bin/env bash
# Evaluate every pre-registered experiment on whatever is complete (missing categories are flagged, never filled).
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src; LOG=logs/eval_all.log
echo "EVAL_START $(date)" >> $LOG
HEAD="experiments/A_patchcore.yaml experiments/B_local_st.yaml experiments/C_dino.yaml experiments/D_pc_st.yaml experiments/E_pc_dino.yaml experiments/F_three.yaml experiments/G_full.yaml experiments/H_composition.yaml"
REST="experiments/G_full_mean.yaml experiments/G_full_or.yaml experiments/G_full_softmax.yaml experiments/G_full_identity.yaml experiments/G_full_compact.yaml experiments/C_dino_softmax.yaml experiments/C_dino_identity.yaml experiments/C_dino_compact.yaml experiments/C_dino_pool_ref.yaml experiments/C_dino_pool_top0.5.yaml experiments/C_dino_pool_top5.yaml experiments/B_local_st_top1.yaml experiments/B_local_st_nopenalty.yaml experiments/A_patchcore_reweight.yaml"
$PY -m hybridad.cli evaluate --experiment $HEAD --dataset all >> $LOG 2>&1
$PY -m hybridad.cli evaluate --experiment $REST --dataset all --no-plots >> $LOG 2>&1
# reproductions: only when their lanes finished (evaluating an untrained reproduction would train it first)
grep -q LANE_ST_0_DONE logs/lane_st_0.log 2>/dev/null && $PY -m hybridad.cli evaluate --experiment experiments/R_patchcore_published.yaml --dataset mvtec_ad >> $LOG 2>&1
grep -q LANE_DINO_DONE logs/lane_dino.log 2>/dev/null && $PY -m hybridad.cli evaluate --experiment experiments/R_dino_published.yaml --dataset mvtec_ad >> $LOG 2>&1
$PY -m hybridad.cli report >> $LOG 2>&1
echo "EVAL_DONE $(date)" >> $LOG
