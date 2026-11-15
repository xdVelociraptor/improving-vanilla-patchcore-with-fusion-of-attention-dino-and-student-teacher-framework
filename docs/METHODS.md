# Methods note

This note records what was built, which sources were checked, where this project follows a
reference implementation and where it deliberately departs from it, and what cannot be claimed.
Measured results are in `docs/RESULTS.md` (generated from saved outputs only).

## 1. What is measured and reported

| Item | How it is handled |
|---|---|
| Image-level detection, anomalous = 1 | Every output uses this label convention; higher score = more anomalous. |
| Reported metrics | Image-level AUROC from continuous scores: per category with stratified bootstrap 95 % CIs, macro and pooled over categories, per branch, and for MVTec LOCO separately for logical and structural anomalies. Localization: pixel AUROC and AU-PRO (MVTec AD), AU-sPRO (MVTec LOCO). |
| No metric manipulation | No noise injection, score corruption, favourable test subsets or deliberately weakened detectors exist anywhere in the code. |

The two datasets are always evaluated and reported separately; no headline number combines them.

This package reports AUROC-based metrics only. The pre-registered evaluation also computed threshold-dependent
metrics; they are omitted here, and the code in this package is a derivative with that computation removed (section 8).
Work on MVTec LOCO after its confirmatory results (the exploratory model X3, `docs/X3.md`) is kept separate from the
pre-registered results.

## 2. Environment (measured on this machine)

Windows 11, AMD Ryzen 9 8940HX (16 cores), 23 GB RAM, NVIDIA RTX 5070 Laptop GPU (8 GB, sm_120),
Python 3.12.10, torch 2.11.0+cu128, faiss-cpu 1.15.1. `requirements.lock.txt` is the pinned
`pip freeze` of the original research archive's environment; `hybridad env` writes `docs/environment.json`.

The original archive included the Python environment, datasets, pretrained weights, trained models, and saved scores. This source repository does not include those large inputs. See `README.md` and `DATA_AND_LICENSES.md` for setup, download, and verification instructions. `check_package.py` requires the full original archive and its saved scores.

## 3. Sources, versions and provenance

| Source | Pinned version | Author code? | Use |
|---|---|---|---|
| amazon-science/patchcore-inspection | fcaa92f124fb | yes | PatchCore reference; our port is tested op-for-op against it |
| guojiajeremy/Dinomaly | 1f252be03a91 | yes | ViTill decoder, loss, optimizer, schedule, pooling |
| facebookresearch/dinov2 | 7764ea0f912e | yes | DINOv2 ViT-B/14-reg4 model code |
| open-edge-platform/anomalib (efficient_ad) | d716bb94f865 | **no** (third-party reimplementation) | PDN architecture, local ST objective |
| Tokichan/CSAD | d46cf856b99b | yes | cited reference only, **not reproduced** |
| MVTec LOCO AD evaluation | v2.0 (README) | yes (MVTec) | official AU-sPRO; copied from a previous local project, not re-downloaded |

| Weights / data | SHA-256 | Origin |
|---|---|---|
| wide_resnet50_2-95faca4d.pth | 95faca4d11227ddd… | torchvision IMAGENET1K_V1 (the file the PatchCore author code loads) |
| dinov2_vitb14_reg4_pretrain.pth | 73182a088cf94833… | dl.fbaipublicfiles.com (Meta, official) |
| pretrained_teacher_small.pth (PDN-S) | a16ded54719674435… | anomalib GitHub release. **Third-party**, not published by the EfficientAD authors; how it was distilled was not verified |
| imagenette2.tgz | 6cbfac238434d89f… | fast.ai S3, the penalty dataset anomalib uses |

Full checksums: `src/hybridad/provenance.py`. A weight whose checksum differs is refused at load time.

Papers checked for this project: PatchCore (arXiv 2106.08265 v2), EfficientAD (2303.14535 v3), Dinomaly
(2405.14325), and AeCSAD (IEEE Access 2026, 10.1109/ACCESS.2026.3670056). CSAD (2408.15628) and PSAD
(2312.13783) are cited for the composition idea only. No reported score from any cited paper is transferred to this model.

## 4. Data protocol and leakage prevention

* Official category structure and splits. Immutable manifests (`manifests/<dataset>/<category>.csv` + `.meta.json`)
  hold the relative image ID, role, labels (evaluation only), file SHA-256 and decoded-pixel SHA-256. Rebuilding
  must reproduce the same bytes or the run stops; loading re-verifies the hash.
* **MVTec AD:** a deterministic hash-ordered 20 % of `train/good` (minimum 10) is held out for calibration.
* **MVTec LOCO:** `train/good` is used for fitting, and `validation/good` is the calibration pool.
* Calibration pools with at least 40 images are split into two disjoint halves: `calib_scale` (branch normalizers)
  and a second half (role name `calib_threshold`, kept for manifest compatibility), used only for the OR fusion rule's
  per-branch offsets. Smaller pools (toothbrush: 12 images) are `calib_shared` and flagged.
* Exact and re-encoded duplicate checks run across roles. The earlier-role copy is removed, so test content never
  enters a memory or a normalizer. None were found in either dataset.
* Memories, teacher statistics, PCA, clustering and descriptor scales are fitted on `fit` images only. The pipeline
  refuses to score a fitting image, so self-matching cannot occur. The composition memory uses leave-one-out when
  computing its own scales.
* Test labels and masks are read only by `evaluate.py` / `localization.py`. They never feed preprocessing, checkpoint
  selection (checkpoints are the final iteration of a fixed schedule), normalization, fusion or early stopping. The
  true anomaly type never selects a branch; category identity only routes to that category's models.

### Test-set history (disclosure)

Before this project, a separate local project evaluated extensively on the official
MVTec AD and MVTec LOCO **test** splits, and its notes state that one of its branches "used assigned training
anomalies". No independent real-anomaly set exists on this machine, so no result here can be confirmed on untouched
data. This project's own test accesses are appended to `runs/TEST_ACCESS_LOG.jsonl`; the original research archive contains an excerpt
for these experiments as `runs/TEST_ACCESS_LOG_excerpt.jsonl`.

### Normal-only calibration

Predetermined training schedules; branch normalizers (and, for the OR rule, per-branch offsets) are fitted on held-out
normal images only. No anomaly label is used before the test evaluation.

## 5. Branches

### 5.1 Protected PatchCore (`branches/patchcore.py`)
Faithful port of the author code, verified by `tests/test_patchcore_reference.py`: embeddings agree with the vendored
author `PatchCore._embed` to rtol 1e-5, and coreset selections are identical under the same seeds.
* WRN-50-2, layer2 + layer3, 3x3 aggregation, 1024 -> 1024, approximate greedy coreset 10 % (the author script's
  "IM224 Baseline"), 10 starting points, 128-d random projection, coreset seed 0, k = 1.
* **Distance:** the author code uses `faiss.IndexFlatL2`, which returns **squared** Euclidean distances; the paper
  writes Euclidean. We keep the code's squared convention. `sqrt` is monotone, so max-aggregated rankings are unaffected.
* **Aggregation:** max over patches. The paper's neighbourhood re-weighting (Eq. 7) is **absent from the author code**.
  It exists here only as the separately named score `paper_reweighted_b9`; b = 9 is an assumption, because the paper
  does not state b.
* The author run script min-max normalizes scores over the TEST set. That is not used here.
* **Search:** faiss exact search is the preferred backend. The chunked exact torch search, with float64 refinement of
  the selected neighbours, is the fallback and is tested against brute force and faiss.
* **Preprocessing:** MVTec AD uses the published 256 -> centre-crop 224. MVTec LOCO uses an ADAPTATION: the whole scene
  letterboxed to 256x256, and patches whose cell is < 50 % real image are excluded from the memory and from the max.
  The LOCO memory is built for this adaptation.
* **Paired baseline vs reproduction:** the paired baseline (`A_patchcore`) uses the fit split, like every hybrid.
  `R_patchcore_published` fits on the full official training set, reports AUROC only, and is never
  used as the paired comparison.
* **Invariance:** branch models and score files are shared across experiments, so `score_patchcore` is byte-identical
  in A and G (tested). This is an implementation invariant; it does NOT imply the fused structural performance
  cannot decrease.

### 5.2 Local student–teacher (`branches/local_st.py`)
An ADAPTATION of EfficientAD's local pathway, not a reproduction of complete EfficientAD. The autoencoder, and the
student's second 384 channels that imitate it, are omitted.
* PDN-S teacher (pretrained, frozen, 33x33 receptive field at stride 4, asserted in tests) and a randomly initialized
  PDN-S student with the same geometry. A random teacher is never substituted: a missing or mismatched teacher
  file stops the run.
* Teacher channel mean and std are computed on the fit images (the paper: "all training images").
* **Loss:** hard-feature loss at p_hard = 0.999 plus the ImageNette penalty (Resize 512 -> RandomGrayscale 0.3 ->
  CenterCrop 256, batch 1).
* **Optimization:** Adam(1e-4, wd 1e-5), batch 1, 70,000 iterations.
* **Code vs paper, LR decay:** the paper decays the LR to 1e-5 after 66,500 iterations. anomalib returns a StepLR with
  no `interval`, so Lightning steps it once per epoch and the decay never triggers within 70k iterations. We follow
  the paper and step per iteration.
* **Code vs paper, penalty data:** the paper uses ImageNet and anomalib uses ImageNette. We follow anomalib.
  `B_local_st_nopenalty` is a labelled ablation without the penalty.
* **Map normalization:** anomalib computes the map quantiles q_a / q_b on its validation loader, which defaults to the
  test split for MVTec. They are not used here. For a single map they are a positive affine transform, so rankings
  and our calibrated scores are unchanged.
* **Map and score:** the map is the channel mean of the squared normalized discrepancy, padded by 4 and bilinearly
  upsampled. The image score is its max (reference). Top-1 % tail pooling is the separate ablation `B_local_st_top1`.
* **Preprocessing:** MVTec AD uses a 256x256 resize (anomalib). LOCO uses a letterbox ADAPTATION; padding is
  excluded from the teacher statistics, the loss and pooling.
* Complementarity with PatchCore is not assumed; it is measured by comparing AUROC with and without the branch
  (experiments A, D, F and G).

### 5.3 DINOv2 global-attention reconstruction (`branches/dino_recon.py`)
* Frozen DINOv2 ViT-B/14 with 4 registers (model code from facebookresearch/dinov2). The token layout
  [CLS, 4 registers, patches] is asserted, and only patch tokens are reshaped into the grid.
* Layers are zero-based block indices [2..9], grouped as [2-5] and [6-9].
* **Decoder:** input is the mean of the 8 layers over all tokens, as in the reference. Then the noisy bottleneck
  (MLP, dropout 0.2 on input, hidden and output), then 8 blocks (768-d, 12 heads, LayerNorm eps 1e-8) with
  LinearAttention2 (phi = elu + 1). No decoder positional embedding, no encoder-to-output shortcut; only the
  blocks' internal residuals remain.
* **Loss:** global cosine hard-mined loss; the easiest 90 % of points have their gradient scaled by 0.1, with p
  warmed 0 -> 0.9 over 1,000 iterations.
* **Optimization:** StableAdamW + AMSGrad (lr 2e-3, wd 1e-4, eps 1e-8), WarmCosine to 2e-4 with 100 warm-up
  iterations, grad clip 0.1, batch 16.
* **Code vs paper:**
  * Iterations: the paper states 10,000 for MVTec in the multi-class setting. The per-class author script
    (`dinomaly_mvtec_sep.py`) runs **5,000** (its save_name says "it10k"). We train per category and use 5,000.
  * Positional embedding: Dinomaly loads the reg4 model with `interpolate_antialias=False, offset=0.1`, while the
    official DINOv2 hub `*_reg` setting is `antialias=True, offset=0.0`. We follow Dinomaly
    (`pos_interp: dinomaly`); `official` is a named option.
  * Optimizer eps: 1e-8 in the per-class script, 1e-10 in the unified script.
* **Numeric precision (user-approved deviation, 2026-09-26):** Dinomaly pins torch 1.12, where TF32 matmuls are off by
  default, so fp32 is the faithful setting (measured about 108 min per category). At the project owner's request, to fit the
  full suite into the time available, every DINO variant uses **TF32 matmuls** (`matmul_precision: tf32`,
  measured about 69 min before caching). The same precision is used for all variants, so ablations stay
  comparable. This is a documented numeric deviation from the reference.
* **Exact speed-ups:** (bitwise-equality tests in `tests/test_speedups_exact.py`)
  * The frozen encoder's outputs for the fitting images are computed once and memory-mapped (training uses no
    augmentation). They are shared by all decoder variants.
  * Local ST penalty crops are preloaded, with grayscale applied after cropping. This is pixel-wise identical
    to the reference order.
  * Local ST runs in 3 parallel processes beside DINO training.
  * The plan change is recorded in `runs/PREREGISTRATION_CHANGELOG.md`: v1 was superseded before any affected
    evaluation.
* **Resume:** interrupted training resumes from checkpoints every 1,000 iterations. A device bug in restoring RNG
  state was fixed in v3 and has a regression test (`tests/test_resume.py`).
* **Memory:** batch 16 at 1,029 tokens peaks at 10.8 GiB, more than this GPU's 8 GiB. Windows spilled it to shared
  memory, at about 32 s/iteration. Activation checkpointing of the decoder blocks, which is exact recomputation
  (tested), brings the peak to 3.9 GiB at about 1.3 s/iteration. Batch size and head size were not reduced.
* **Full-image ADAPTATION:** MVTec AD uses a 448 resize without centre crop; LOCO uses a 448 letterbox. Both give
  a 32x32 grid.
  * Padding (cells < 50 % real) is excluded from the loss (masked global cosine and masked hard-mining pool), pooling
    and maps.
  * The encoder's attention still sees padding tokens, so `padding_diagnostic` rescores normal calibration images with
    a black fill instead of the mean-colour fill, and compares border and interior map levels.
* **Pooling:** the adapted image score is the top 1 % of valid grid cells. Named ablations: the reference pooling
  location (upsample -> 256 -> Gaussian -> top 1 %), top 0.5 %, top 5 %.
* The published preprocessing (448 -> centre crop 392) exists only in `R_dino_published`.
* Seeds: runs use the suite seed (0 for the first runs); Dinomaly's script calls `setup_seed(1)`. Seeds are recorded per run.
* **Ablations:**
  * `softmax`: standard attention via fused SDPA.
  * `identity`: the matched no-token-mixing control, with the same qkv/proj parameters and an identity attention
    matrix.
  * `compact`: 4 blocks at width 384, a PROPOSED engineering choice, not a published setting.
  * Everything else (encoder, data, iterations, preprocessing) is identical, so a different encoder is never used
    as an "attention ablation".
  * AeCSAD's own attention ablation reports gains that are modest and non-uniform, with logical AUROC on screw_bag
    decreasing. Attention is therefore tested, not assumed.
* A context-aware decoder is not a logical-reasoning detector, and attention weights are not anomaly probabilities.

### 5.4 Composition evidence (`branches/composition.py`)
This project's own design, "Part-Occupancy Composition Memory". It is inspired by CSAD's patch histograms and
PSAD's part-level reasoning but reproduces neither. Novelty is not claimed.
* **Clusters:** frozen DINOv2 final-layer patch tokens -> PCA-64 -> k-means with K = 10 "visual clusters",
  soft-assigned at tau = 0.05. All of this is fitted on fit images only. Clusters are called visual clusters or
  occupancy descriptors: semantic part consistency has not been verified.
* **Descriptor groups (kept separate):**
  * mass: the fraction of VALID area per cluster, so quantity is not lost to normalized appearance;
  * appearance: the cluster-conditional mean feature;
  * layout: a 4x4 occupancy grid over the valid region, only where position is meaningful. It is disabled for
    screw, hazelnut and screw_bag;
  * adjacency: cluster co-occurrence over neighbouring valid cells, which is translation- and rotation-invariant;
  * counts: connected components, DISABLED by default because occupancy is not an instance count.
* **Scoring:** group distances (Hellinger-style L1; mass-weighted 1 - cosine for appearance) are each scaled by the
  median leave-one-out nearest-neighbour distance among fit images. The score is the mean joint distance to the 3
  nearest normal FIT images. Every normal assembly is kept; none is averaged into a template.
* **Absent components:** handled explicitly with presence flags. The appearance fallback is fixed in advance to
  the maximum leave-one-out value. No NaNs are produced.
* **Failures:** too few valid cells or non-finite descriptors are recorded per image. They become a maximal alarm
  under the pre-registered missing-score policy, never a normal pass.
* Disabled for texture categories by configuration fixed before any test evaluation.
* Qualitative look (4 normal calibration images of breakfast_box, figure in the local research archive; contains dataset imagery): clusters visibly align with nectarine, tangerines, granola, banana chips/almonds, the tray divider and the background. This is an eyeball check on normal images only, not a semantic-consistency verification, so the terms remain "visual clusters" / "occupancy descriptors".
* No localization map: the composition score is image-level evidence only.
* Additional pretrained segmentation models: none are used, so there is no pretrained external supervision, no
  pseudo-labels, no synthetic anomalies and no manual annotation in this branch.

## 6. Normalization and fusion
* **Branch scaling:** z = (s - median) / (Q95 - median + eps), fitted on calib_scale normals. Near-constant branches
  fall back to MAD, then std; a truly constant branch is flagged degenerate and excluded from fusion. z is not a
  probability. Test-set min/max normalization is never used.
* **Fusion rules:**
  * max: the pre-registered default for the full model, because the branches are hypothesised to fire on different
    anomalies. This is a design choice, not a validated result.
  * mean: equal weight.
  * OR: an ablation. Each standardized score minus a per-branch offset fitted on the second held-out normal half
    (alpha 0.10 / n_branches), then the maximum.
* The calibration record binds the active branches, variant IDs, score-file hashes, normalizers, fusion rule and OR
  offsets. Any change invalidates it (`verify_binding`).

## 7. Metrics
* **Image-level AUROC** from continuous fused scores (NaN with a reason if only one class is present), with a
  stratified percentile bootstrap 95 % CI (2,000 resamples, seed 0).
* **Per-branch AUROC** for every score an experiment uses, from the raw branch scores.
* **MVTec LOCO:** logical-vs-normal and structural-vs-normal AUROC (each reuses all normals and excludes the other
  anomaly type) and their balanced mean.
* **Aggregation:** per-category values, macro means over categories (primary) and a labelled pooled AUROC that ranks
  standardized fused scores across categories (secondary).
* **ROC curves:** `figures/roc_curves.png` in each run folder, with every curve's AUROC in the legend.
* **Localization:**
  * MVTec AD: pixel AUROC and AU-PRO up to a pixel false-positive rate of 0.3, a re-implementation of the official
    algorithm checked against a brute-force sweep.
  * LOCO: AU-sPRO via the official v2.0 script, run through Windows compatibility shims; the vendored code is
    unmodified.
  * Unobserved pixels (outside a centre crop) are reported, not hidden.

## 8. Pre-registration and test access
`hybridad preregister` freezes the experiment suite (a spec hash per experiment, dataset and category, plus a
source-code hash). Evaluation refuses non-matching configurations unless `--exploratory` is given, and every output
then says EXPLORATORY. Every test evaluation is appended to `runs/TEST_ACCESS_LOG.jsonl`. A test-based change requires
fresh independent confirmation before any confirmatory claim, and none is available here (see "Test-set history").

**Pre-registration history** (full reasons in `runs/PREREGISTRATION_CHANGELOG.md`):
| Version | When | Why | Code hash |
|---|---|---|---|
| v1 | 2026-09-26 01:23 | initial freeze, before any real test evaluation | 135b7716… |
| v2 | 01:55 | the project owner approved TF32 for every DINO variant, plus exact speed-ups; superseded v1 before any affected evaluation | baf601f7… |
| v3 | 14:29 | bug fix only: resuming an interrupted DINO training failed on RNG-state device; no config change, and no evaluated model used that path | a682c370… |

Results keep the confirmatory status recorded when they were evaluated: MVTec LOCO A–H and their fusion/pooling
variants, and MVTec AD A and H, under v2. `src/hybridad_x/`, `scripts/` and the docs are outside the code hash
(`PRESENTATION_ONLY` modules plus everything outside `src/hybridad`), so exploratory work there never changes it.

**Code in this package.** The results under `runs/` were produced by the pre-registered pipeline (code hash
a682c370…). The code shipped here is a derivative with the threshold-dependent metrics removed from evaluation,
calibration and reporting, so its code hash differs and re-running it labels outputs EXPLORATORY. Saved outputs were
post-processed the same way; every AUROC value kept was recomputed with the package code from the saved predictions
and matches the original exactly.

**Exploratory (post-test) results in this package:** X3, a higher-resolution design specified before it was built
or evaluated (`docs/X3.md`). It is reported separately and is never confirmatory. The pre-registered code hash was
verified unchanged after the exploratory work (a682c370…).

## 9. Known limitations
* **Small calibration sets:** normalizers are fitted on small held-out normal sets (toothbrush shares 12 images
  between both halves).
* **One seed so far:** the first real runs use seed 0. The multi-seed commands exist but are compute-bound.
* **Compute:** see README section 5 for the measured timings. The full 24-experiment suite on all 20 categories
  exceeds 10 GPU-hours on this laptop, so results delivered at a deadline are flagged as partial where incomplete.
* **Unverified cluster semantics:** composition clusters are not semantically verified, and count evidence is off.
* **Test history:** no untouched real-anomaly confirmation data exists.
* **Run incidents (lost time only; no effect on any result):**
  * 01:57–03:45 on 2026-09-26: three training jobs overloaded the 8 GB GPU, so memory spilled and everything slowed
    about 100×. Only one GPU job runs at a time since then.
  * 14:31–16:34 on 2026-09-26: system standby froze training. Keep the laptop plugged in with sleep disabled.
  * A full stop requested by the project owner on 2026-09-27; everything was resumed from checkpoints.
  * Leather's DINO training failed four times around step 850–900 (2026-09-26 to 2026-09-28). A diagnostic re-run
    (`scripts/diagnose_leather.py`, `logs/diagnose_leather.jsonl`) showed a memory-management failure, not a lack of
    memory. Live GPU memory stayed at 1.73 GB, with a peak of 3.93 GB. Memory reserved by PyTorch's cache grew from
    4.3 GB to 16.8 GB by step 860 on the 8 GB card, with no allocation retries.
    * **Cause:** during the 1,000-step hard-mining warm-up, the per-step temporary buffers change size every step, so
      the cache keeps opening new blocks. On Windows (WDDM) the driver satisfies them from system memory instead of
      failing, so the cache is never freed until system memory runs out.
    * **Other categories:** the growth is the same for every category, since the buffer sizes depend only on the step.
      The likely reason the other categories survived is that more system memory was free when they ran.
    * **Fix (the project owner's decision, 2026-09-28):** the unchanged pipeline runs under a per-process CUDA memory cap of 0.8
      (`scripts/run_capped.py`, `torch.cuda.set_per_process_memory_fraction`). The cap forces the cache to release
      memory and retry. It changes neither the method nor its math, and the code hash is unchanged. Strict bitwise
      identity would also require that no cuBLAS/cuDNN/SDPA kernel choice depends on free workspace, which is
      unlikely here.
    * **Scope:** leather retry #5 and every later job in `scripts/lane_main.sh` run under this cap.
    * **Rejected alternative:** a one-line bitwise-identical rewrite of the hard-mining gradient hook (`torch.where`).
      It was not applied because it would have required pre-registration v4.
