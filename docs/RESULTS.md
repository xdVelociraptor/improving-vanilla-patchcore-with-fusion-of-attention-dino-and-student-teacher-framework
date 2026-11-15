# Results: AUROC (generated from saved outputs)

**Historical results.** Generated in the original research archive. Aggregate tables are copied under `results/`; empty tables indicate runs with no category results. References below to raw runs, pre-registration records and access logs refer to the original archive. Every number below is read from files under `runs/`; experiments or categories without saved outputs are listed as missing.

* Pre-registration file: present, created 2026-09-26T06:29:27.736629+00:00
* The results under `runs/` were produced by the pre-registered pipeline (code hash a682c370e767840a; history v1-v3 in runs/PREREGISTRATION_CHANGELOG.md). This package reports AUROC-based metrics only: the pre-registered evaluation also computed threshold-dependent metrics, which are omitted here, and the code in this package is a derivative with that computation removed (code hash 60fb73f4ee59e47f), so re-running it labels its outputs EXPLORATORY.
* Test-set evaluations recorded for the experiments in this package (docs/TEST_ACCESS_LOG_excerpt.jsonl): 142 (138 confirmatory, 4 exploratory)
  * mvtec_ad: 79 of them (79 confirmatory, 0 exploratory), 2026-09-25 to 2026-09-28 (UTC). The earlier confirmatory ones are progress looks at the same pre-registered configurations as categories finished; the tables below come from the latest evaluation.
  * mvtec_loco: 63 of them (59 confirmatory, 4 exploratory), 2026-09-25 to 2026-09-28 (UTC). The earlier confirmatory ones are progress looks at the same pre-registered configurations as categories finished; the tables below come from the latest evaluation. The exploratory ones are the evaluations of X3 (docs/X3.md).
* Interim looks, disclosed (MVTec AD test labels joined before the final 2026-09-28 evaluation; local times, from docs/TEST_ACCESS_LOG_excerpt.jsonl): 2026-09-26 04:03-11:05, 24 evaluations during the reference phase under pre-registration v2 (before the v3 resume bug fix at 14:29); 2026-09-27 06:19-15:46, per-category G_full localization as categories finished (leather on 2026-09-28 14:52); 2026-09-27 14:47-14:48, a 13-category CPU interim evaluation of G_full, G_full_mean and A_patchcore (runs/interim/); 2026-09-27 17:57-18:10, a 14-category evaluation with leather missing for the DINO experiments (scripts/eval_ad_partial.sh). No configuration, model or fusion rule was selected, tuned or changed because of any of them. The v2->v3 change was a resume bug fix (runs/PREREGISTRATION_CHANGELOG.md), and the only later change was a runtime CUDA memory cap used to train leather's DINO, which changes neither the method nor its math (docs/METHODS.md, 'Run incidents'). The final 15-category evaluation therefore keeps its confirmatory label, with these looks stated here.
* Earlier, separate projects on this machine had already evaluated on the official MVTec AD and LOCO test splits (see docs/METHODS.md, 'Test-set history'); no untouched real-anomaly confirmation set exists.
* The post-test EXPLORATORY model X3 is described in docs/X3.md; it is NOT part of the tables below.
* AUROC: image-level, from continuous fused scores; 95 % CI: stratified percentile bootstrap (2,000 resamples). Pooled AUROC ranks standardized fused scores across categories (labelled secondary number).

## mvtec_ad

| experiment | seed | status | categories | macro AUROC | pooled AUROC | branch AUROC: PatchCore | branch AUROC: local ST | branch AUROC: DINO | branch AUROC: composition |
|---|---|---|---|---|---|---|---|---|---|
| A_patchcore | seed0 | confirmatory | 15 | 0.990 | 0.965 | 0.990 | n/a | n/a | n/a |
| B_local_st | seed0 | confirmatory | 15 | 0.963 | 0.936 | n/a | 0.963 | n/a | n/a |
| C_dino | seed0 | confirmatory | 15 | 0.997 | 0.993 | n/a | n/a | 0.997 | n/a |
| D_pc_st | seed0 | confirmatory | 15 | 0.990 | 0.968 | 0.990 | 0.963 | n/a | n/a |
| E_pc_dino | seed0 | confirmatory | 15 | 0.998 | 0.989 | 0.990 | n/a | 0.997 | n/a |
| F_three | seed0 | confirmatory | 15 | 0.996 | 0.984 | 0.990 | 0.963 | 0.997 | n/a |
| G_full | seed0 | confirmatory | 15 | 0.994 | 0.982 | 0.990 | 0.963 | 0.997 | 0.871 |
| H_composition | seed0 | confirmatory | 10 (5 n/a by design) | 0.871 | 0.840 | n/a | n/a | n/a | 0.871 |
| G_full_mean | seed0 | confirmatory | 15 | 0.997 | 0.985 | 0.990 | 0.963 | 0.997 | 0.871 |
| G_full_or | seed0 | confirmatory | 15 | 0.994 | 0.983 | 0.990 | 0.963 | 0.997 | 0.871 |
| C_dino_pool_ref | seed0 | confirmatory | 15 | 0.997 | 0.994 | n/a | n/a | 0.997 | n/a |
| C_dino_pool_top0.5 | seed0 | confirmatory | 15 | 0.997 | 0.991 | n/a | n/a | 0.997 | n/a |
| C_dino_pool_top5 | seed0 | confirmatory | 15 | 0.994 | 0.988 | n/a | n/a | 0.994 | n/a |
| B_local_st_top1 | seed0 | confirmatory | 15 | 0.967 | 0.937 | n/a | 0.967 | n/a | n/a |
| A_patchcore_reweight | seed0 | confirmatory | 15 | 0.989 | 0.961 | 0.989 | n/a | n/a | n/a |

### G_full per category (seed0)

| category | n normal | n anomalous | AUROC | 95% CI | branch AUROC: PatchCore | branch AUROC: local ST | branch AUROC: DINO | branch AUROC: composition |
|---|---|---|---|---|---|---|---|---|
| bottle | 20 | 63 | 1.000 | [1.000, 1.000] | 1.000 | 0.998 | 1.000 | 1.000 |
| cable | 58 | 92 | 0.990 | [0.975, 1.000] | 0.995 | 0.842 | 0.997 | 0.837 |
| capsule | 23 | 109 | 0.989 | [0.972, 0.999] | 0.966 | 0.915 | 0.989 | 0.876 |
| carpet | 28 | 89 | 0.997 | [0.990, 1.000] | 0.989 | 0.997 | 1.000 | n/a |
| grid | 21 | 57 | 1.000 | [1.000, 1.000] | 0.973 | 1.000 | 1.000 | n/a |
| hazelnut | 40 | 70 | 1.000 | [1.000, 1.000] | 1.000 | 0.927 | 1.000 | 0.710 |
| leather | 32 | 92 | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | n/a |
| metal_nut | 22 | 93 | 0.995 | [0.981, 1.000] | 0.999 | 0.994 | 1.000 | 0.855 |
| pill | 26 | 141 | 0.997 | [0.990, 1.000] | 0.967 | 0.984 | 0.996 | 0.882 |
| screw | 41 | 119 | 0.987 | [0.973, 0.996] | 0.983 | 0.931 | 0.984 | 0.851 |
| tile | 33 | 84 | 1.000 | [1.000, 1.000] | 0.989 | 1.000 | 1.000 | n/a |
| toothbrush | 12 | 30 | 1.000 | [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.925 |
| transistor | 60 | 40 | 0.964 | [0.921, 0.993] | 1.000 | 0.871 | 0.995 | 0.804 |
| wood | 19 | 60 | 0.999 | [0.995, 1.000] | 0.991 | 0.992 | 0.996 | n/a |
| zipper | 32 | 119 | 0.997 | [0.992, 1.000] | 0.994 | 0.996 | 0.999 | 0.976 |

## mvtec_loco

| experiment | seed | status | categories | macro AUROC | pooled AUROC | logical AUROC | structural AUROC | balanced mean | branch AUROC: PatchCore | branch AUROC: local ST | branch AUROC: DINO | branch AUROC: composition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_patchcore | seed0 | confirmatory | 5 | 0.813 | 0.807 | 0.793 | 0.850 | 0.822 | 0.813 | n/a | n/a | n/a |
| B_local_st | seed0 | confirmatory | 5 | 0.793 | 0.792 | 0.719 | 0.894 | 0.806 | n/a | 0.793 | n/a | n/a |
| C_dino | seed0 | confirmatory | 5 | 0.789 | 0.790 | 0.721 | 0.888 | 0.804 | n/a | n/a | 0.789 | n/a |
| D_pc_st | seed0 | confirmatory | 5 | 0.829 | 0.823 | 0.781 | 0.900 | 0.840 | 0.813 | 0.793 | n/a | n/a |
| E_pc_dino | seed0 | confirmatory | 5 | 0.818 | 0.821 | 0.764 | 0.894 | 0.829 | 0.813 | n/a | 0.789 | n/a |
| F_three | seed0 | confirmatory | 5 | 0.821 | 0.822 | 0.758 | 0.909 | 0.834 | 0.813 | 0.793 | 0.789 | n/a |
| G_full | seed0 | confirmatory | 5 | 0.860 | 0.854 | 0.828 | 0.912 | 0.870 | 0.813 | 0.793 | 0.789 | 0.732 |
| H_composition | seed0 | confirmatory | 5 | 0.732 | 0.717 | 0.787 | 0.675 | 0.731 | n/a | n/a | n/a | 0.732 |
| G_full_mean | seed0 | confirmatory | 5 | 0.870 | 0.870 | 0.837 | 0.925 | 0.881 | 0.813 | 0.793 | 0.789 | 0.732 |
| G_full_or | seed0 | confirmatory | 5 | 0.860 | 0.854 | 0.831 | 0.910 | 0.870 | 0.813 | 0.793 | 0.789 | 0.732 |
| G_full_identity | seed0 | confirmatory | 1 (missing 4) | 0.959 | 0.959 | 0.982 | 0.937 | 0.960 | 0.780 | 0.761 | 0.910 | 0.848 |
| G_full_compact | seed0 | confirmatory | 1 (missing 4) | 0.956 | 0.956 | 0.985 | 0.929 | 0.957 | 0.780 | 0.761 | 0.911 | 0.848 |
| C_dino_identity | seed0 | confirmatory | 1 (missing 4) | 0.910 | 0.910 | 0.881 | 0.936 | 0.909 | n/a | n/a | 0.910 | n/a |
| C_dino_compact | seed0 | confirmatory | 1 (missing 4) | 0.911 | 0.911 | 0.877 | 0.943 | 0.910 | n/a | n/a | 0.911 | n/a |
| C_dino_pool_ref | seed0 | confirmatory | 5 | 0.674 | 0.669 | 0.639 | 0.730 | 0.684 | n/a | n/a | 0.674 | n/a |
| C_dino_pool_top0.5 | seed0 | confirmatory | 5 | 0.784 | 0.789 | 0.715 | 0.885 | 0.800 | n/a | n/a | 0.784 | n/a |
| C_dino_pool_top5 | seed0 | confirmatory | 5 | 0.796 | 0.794 | 0.739 | 0.879 | 0.809 | n/a | n/a | 0.796 | n/a |
| B_local_st_top1 | seed0 | confirmatory | 5 | 0.789 | 0.783 | 0.713 | 0.891 | 0.802 | n/a | 0.789 | n/a | n/a |
| A_patchcore_reweight | seed0 | confirmatory | 5 | 0.809 | 0.802 | 0.790 | 0.845 | 0.818 | 0.809 | n/a | n/a | n/a |

### G_full per category (seed0)

| category | n normal | n anomalous | AUROC | 95% CI | logical AUROC | structural AUROC | branch AUROC: PatchCore | branch AUROC: local ST | branch AUROC: DINO | branch AUROC: composition |
|---|---|---|---|---|---|---|---|---|---|---|
| breakfast_box | 102 | 173 | 0.874 | [0.831, 0.915] | 0.937 | 0.816 | 0.780 | 0.761 | 0.767 | 0.848 |
| juice_bottle | 94 | 236 | 0.991 | [0.983, 0.996] | 0.986 | 0.997 | 0.988 | 0.947 | 0.922 | 0.818 |
| pushpins | 138 | 172 | 0.767 | [0.717, 0.816] | 0.696 | 0.847 | 0.747 | 0.783 | 0.679 | 0.623 |
| screw_bag | 122 | 219 | 0.759 | [0.711, 0.806] | 0.643 | 0.952 | 0.681 | 0.710 | 0.700 | 0.662 |
| splicing_connectors | 119 | 193 | 0.907 | [0.873, 0.939] | 0.876 | 0.946 | 0.871 | 0.766 | 0.880 | 0.709 |

## Evaluated with zero completed categories (branches not trained yet)

B_local_st_nopenalty / mvtec_ad / seed0, B_local_st_nopenalty / mvtec_loco / seed0, C_dino_compact / mvtec_ad / seed0, C_dino_identity / mvtec_ad / seed0, C_dino_softmax / mvtec_ad / seed0, C_dino_softmax / mvtec_loco / seed0, G_full_compact / mvtec_ad / seed0, G_full_identity / mvtec_ad / seed0, G_full_softmax / mvtec_ad / seed0, G_full_softmax / mvtec_loco / seed0

## Localization (pre-registered secondary metric)

MVTec AD: pixel AUROC and AU-PRO@0.3 (re-implementation of the official algorithm, checked against a brute-force sweep). MVTec LOCO: official MVTec evaluator v2.0, AU-sPRO (mean of logical and structural) at integration limit 0.05, run with three documented Windows compatibility shims (os.nice no-op, '/' path separators, de-duplicated initial curve points). The composition branch has no map and is excluded.

| experiment | dataset | category | pixel AUROC | AU-PRO@0.3 | AU-sPRO@0.05 (logical / structural / mean) |
|---|---|---|---|---|---|
| G_full | mvtec_ad | bottle | 0.991 | 0.967 | n/a |
| G_full | mvtec_ad | cable | 0.988 | 0.953 | n/a |
| G_full | mvtec_ad | capsule | 0.993 | 0.981 | n/a |
| G_full | mvtec_ad | carpet | 0.994 | 0.985 | n/a |
| G_full | mvtec_ad | grid | 0.996 | 0.984 | n/a |
| G_full | mvtec_ad | hazelnut | 0.989 | 0.968 | n/a |
| G_full | mvtec_ad | leather | 0.994 | 0.983 | n/a |
| G_full | mvtec_ad | metal_nut | 0.986 | 0.972 | n/a |
| G_full | mvtec_ad | pill | 0.986 | 0.962 | n/a |
| G_full | mvtec_ad | screw | 0.996 | 0.982 | n/a |
| G_full | mvtec_ad | tile | 0.984 | 0.952 | n/a |
| G_full | mvtec_ad | toothbrush | 0.988 | 0.945 | n/a |
| G_full | mvtec_ad | transistor | 0.963 | 0.908 | n/a |
| G_full | mvtec_ad | wood | 0.967 | 0.931 | n/a |
| G_full | mvtec_ad | zipper | 0.993 | 0.980 | n/a |
| G_full | mvtec_loco | breakfast_box | n/a | n/a | 0.442 / 0.798 / 0.620 |
| G_full | mvtec_loco | juice_bottle | n/a | n/a | 0.722 / 0.857 / 0.789 |
| G_full | mvtec_loco | pushpins | n/a | n/a | 0.569 / 0.632 / 0.600 |
| G_full | mvtec_loco | screw_bag | n/a | n/a | 0.425 / 0.847 / 0.636 |
| G_full | mvtec_loco | splicing_connectors | n/a | n/a | 0.630 / 0.853 / 0.741 |
