#!/usr/bin/env bash
# Interim MVTec AD evaluation (14/15 categories; leather's DINO failed twice with CUDA OOM and is flagged missing).
# CPU only, below-normal priority. The final 15-category evaluation still runs later via scripts/ad_eval_waiter.sh.
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe; export PYTHONPATH=src CUDA_VISIBLE_DEVICES=-1; LOG=logs/eval_ad_partial.log
echo "AD_PARTIAL_EVAL_START $(date)" >> $LOG
HEAD="experiments/A_patchcore.yaml experiments/B_local_st.yaml experiments/C_dino.yaml experiments/D_pc_st.yaml experiments/E_pc_dino.yaml experiments/F_three.yaml experiments/G_full.yaml experiments/H_composition.yaml"
REST="experiments/G_full_mean.yaml experiments/G_full_or.yaml experiments/C_dino_pool_ref.yaml experiments/C_dino_pool_top0.5.yaml experiments/C_dino_pool_top5.yaml experiments/B_local_st_top1.yaml experiments/A_patchcore_reweight.yaml"
$PY -m hybridad.cli evaluate --experiment $HEAD --dataset mvtec_ad >> $LOG 2>&1
$PY -m hybridad.cli evaluate --experiment $REST --dataset mvtec_ad --no-plots >> $LOG 2>&1
echo "AD_PARTIAL_EVAL_DONE $(date)" >> $LOG
