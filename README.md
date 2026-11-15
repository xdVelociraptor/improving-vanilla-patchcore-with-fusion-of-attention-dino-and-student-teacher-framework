# improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework

Research code for image and pixel anomaly detection on the original MVTec AD and MVTec LOCO AD. The pipeline combines PatchCore, a local student–teacher, DINOv2 reconstruction, and composition evidence.

- [Historical report PDF](docs/REPORT.pdf) (describes the original full archive; see this README for the source-only release)
- [Methods and caveats](docs/METHODS.md)
- [Historical AUROC results](docs/RESULTS.md) and [machine-readable summary tables](results/)
- [X3 exploratory experiments](docs/X3.md)
- [Architecture diagrams](architecture.html)
- [Interactive results viewer](visualiser.html) and [pre-registration changelog](runs/PREREGISTRATION_CHANGELOG.md)
- [Model metadata and excluded artifacts](work/README.md)
- [Dataset, model and code sources and terms](DATA_AND_LICENSES.md)

## Headline results (seed 0, image-level AUROC)

| Dataset | Model | Status | macro AUROC | pooled AUROC | logical / structural AUROC |
|---|---|---|---|---|---|
| MVTec AD, 15 categories | G_full (all four scores, max fusion) | confirmatory | 0.994 | 0.982 | – |
| MVTec LOCO, 5 categories | G_full (all four scores, max fusion) | confirmatory | 0.860 | 0.854 | 0.828 / 0.912 |
| MVTec LOCO, 5 categories | X3_mean | EXPLORATORY | 0.885 | 0.880 | 0.844 / 0.947 |

For historical localization, per-category AUROC and confidence intervals, see [the results](docs/RESULTS.md).

## One-command setup on the original Windows PC

If your complete original project folder is still at `%USERPROFILE%\OneDrive\Desktop\kutty dinosaur`, download and extract this repository into a **separate** folder, then double-click `setup_local_archive.cmd`. The setup copies the original datasets, weights, trained models, score files, saved results, and Python environment into the checkout without modifying the original. It then runs the quick self-check. The original folder must have a working Python 3.12 environment and the files listed above. Allow roughly 27 GB plus 2 GB spare disk space and time for the copy.

If the archive is in a different folder, run this in PowerShell from the checkout:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_local_archive.ps1 -Archive "C:\path\to\original project"
```

Setup stops rather than overwriting a different existing file. If a previous quick check has left `runs_check/` or `logs/self_check.json`, preserve or move those items before rerunning setup. After it passes, run `.\hybridad.cmd --help` and use `hybridad.cmd` / `x3.cmd` as shown in the experiment guide. These wrappers select this checkout's source. On this original machine, the quick check verified weights, datasets, scores, and sample AUROC reproductions.

This convenience setup uses files already present on the original Windows PC; GitHub does not contain the datasets, trained checkpoints, scores, or Python environment. To use the repository on a different machine without the archive, follow the manual installation and publisher download guide below, and retrain or restore your artifacts separately. The historical full-archive check is not expected to pass without those saved artifacts.

## Install and check the code

Use Python 3.12. For a CPU or platform-default setup, create and activate a virtual environment, then install the package and test dependencies:

```shell
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test,faiss]"
python -m hybridad.cli --help
python -m pytest -q
```

The pytest suite can run without the original images or pretrained weights. Weight-dependent tests skip when their inputs are absent. The tests marked `data` check the copied split manifests and do not load the images. The CLI `smoke` workflow exercises real branches on synthetic images and can still require pretrained weights.

If FAISS is unavailable on your platform, install with `python -m pip install -e ".[test]"` and run `python -m pytest -q -m "not reference"`. This omits the PatchCore comparison that needs FAISS. Install FAISS for the reference experiment configuration below.

For the original Python 3.12.10 and PyTorch CUDA 12.8 environment on Windows, run `.\setup_env.ps1`. The original package versions are listed in `requirements.lock.txt`. `bash setup_env.sh` attempts the same pinned environment on Linux where matching wheels are available. Use the package's default dependencies above if you do not need the original CUDA build.

## Reproduce an experiment

1. Download the original datasets and pretrained weights using [the source and layout guide](DATA_AND_LICENSES.md). Keep the archives as separate inputs. Leave image names and directories unchanged.
2. From the project root, use the following command to verify the tracked image checksums. To use data stored elsewhere, pass `--ad-root` and `--loco-root`, and configure the same locations for the pipeline via `configs/local.yaml` (copy `configs/local.example.yaml`) or `HYBRIDAD_MVTEC_AD` and `HYBRIDAD_MVTEC_LOCO`.

   ```shell
   python scripts/verify_datasets.py
   ```

3. Train, score and evaluate one category:

   ```shell
   python -m hybridad.cli manifests --dataset mvtec_ad --category bottle
   python -m hybridad.cli prepare --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle
   python -m hybridad.cli evaluate --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle --exploratory
   python -m hybridad.cli localize --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle
   ```

The included split manifests give the exact image IDs, file digests and split roles used in the archive. The pipeline also checks the pretrained weight digests. Each new experiment writes to `runs_rerun/` and uses `work/` for large artifacts. Without archived scores and checkpoints, this workflow scores and trains from scratch. The local student–teacher penalty step needs Imagenette unless using the explicit no-penalty configuration.

The historical confirmatory figures were produced by the preregistered code. This source release is the later AUROC-only derivative. New runs are exploratory and will not re-create the original preregistration or produce a new confirmation; see [the original evaluation history](docs/RESULTS.md). The historical aggregate results in this snapshot are sufficient to inspect previously reported metrics; the original saved scores, preregistration, and audit logs remain in the local archive. `check.cmd`, report generation, and the saved-score reproduction path require that fuller archive.

For X3, run `python scripts/x3_run.py prepare` and then `python scripts/x3_run.py evaluate` after placing the MVTec LOCO inputs and the required weights. X3 was designed after looking at LOCO test results and is always exploratory.

## Experiments (configs in `configs/experiments/`)

| Name | Branches | Fusion |
|---|---|---|
| A_patchcore | PatchCore | – |
| B_local_st | local ST | – |
| C_dino | DINO reconstruction | – |
| D_pc_st | PatchCore + local ST | max |
| E_pc_dino | PatchCore + DINO | max |
| F_three | PatchCore + local ST + DINO | max |
| **G_full** | all four scores | **max** (pre-registered) |
| H_composition | composition only (diagnostic) | – |
| G_full_mean / G_full_or | full model | mean / OR rule |
| C_dino_softmax / _identity / _compact, G_full_softmax / _identity / _compact | attention and head ablations | max |
| C_dino_pool_ref / _top0.5 / _top5, B_local_st_top1, A_patchcore_reweight | pooling / aggregation variants (no retraining) | – |
| B_local_st_nopenalty | local ST without penalty data | – |
| R_patchcore_published, R_dino_published | published-protocol reproductions (full train set) | – |

Exploratory (configs in `configs/exploratory/`): X3_mean and X3_max, i.e. protected PatchCore + `patchcore_hr` +
local ST + DINO reconstruction + `composition_hr`, with mean or max fusion (`docs/X3.md`).

## Outputs
New runs write the files below to `runs_rerun/<experiment>/<dataset>/seed<k>/` (exploratory: `runs_rerun/exploratory/...`):

* `predictions.csv`: one row per test image, with raw and standardized branch scores, the fused score, labels,
  per-branch status and the calibration binding hash.
* `category_metrics.csv`: per-category AUROC with bootstrap CI, LOCO logical/structural AUROC, per-branch AUROC and
  support counts.
* `aggregate.json`: macro and pooled AUROC, per-branch macro AUROC and partial-run flags.
* `figures/roc_curves.png`: ROC curves of the fused score and every branch, with their AUROCs.
* `environment.json`, `provenance.json`, `config_used.json`.
* `<category>/seed<k>/calibration.json`: normalizers, fusion rule, active branches (and OR offsets), bound together.

Large artifacts go under `work_dir/`: `models/`, `scores/`, `cache/` and `loco_maps/`.

## Compute (measured on an RTX 5070 Laptop, 8 GB)
Current settings: DINO variants use TF32 matmuls plus cached frozen-encoder outputs (approved by the project owner).

| Step (per category) | Time |
|---|---|
| PatchCore fit + scoring | 1–2 min |
| Composition fit + scoring | 2–3 min |
| Local ST, 70,000 steps | ~30 min |
| DINO reconstruction, 5,000 iterations, batch 16 | ~57 min (1.5 it/s) |
| **Reference branches, one category, end to end** | **~87 min** |

The original faithful-fp32 DINO setting measured about 108 min per category; it's still available as `matmul_precision: fp32`.

**Run rules learned the hard way:**
* **One GPU training job at a time.** DINO training nearly fills the 8 GB card. Running two jobs spilled memory into
  system RAM and made everything about 100× slower. CPU-only work is fine alongside.
* **On Windows, run DINO training under a CUDA memory cap** (`scripts/run_capped.py --fraction 0.8 -- <cli args>`).
  Without it, PyTorch's cache can grow into system memory during the hard-mining warm-up until it runs out
  (`docs/METHODS.md`, "Run incidents").
* **Keep the machine plugged in, with sleep disabled.** Standby freezes training.
* Every stage is resumable: finished categories are skipped, and interrupted training resumes from checkpoints.
