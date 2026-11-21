# PatchCore + DINOv2 anomaly detection

This project detects defects in images from the MVTec AD and MVTec LOCO AD benchmarks. Its full model combines PatchCore, a local student–teacher model, DINOv2 reconstruction, and composition features.

## Start here: set up from GitHub on Windows

Install Python 3.12, use an NVIDIA GPU with a suitable driver, and allow **at least 40 GB free** for datasets, models, dependencies, and temporary downloads. Read [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md) before using the research datasets and weights.

Extract this repository into a **new, short folder**, such as `C:\pcf`, so that `setup_online.ps1` sits directly inside it. The repository name is long, and Windows' default extraction nests it inside a second folder of the same name. In a deep location such as `Downloads`, some project and Python package paths then exceed Windows' 260-character path limit, which can make the extraction or the package installation fail. Enabling Windows long-path support also avoids this.

Open PowerShell in that folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_online.ps1
```

This installs the pinned environment, downloads and verifies the archives, restores datasets under `data/`, models and scores under `work/`, and historical results under `runs/` and `runs_rerun/`. It fetches Imagenette and two pretrained weight files from their publishers. It does not train a model. The initial downloads are over 20 GB plus Python packages, so setup is **not instant**. A normal GitHub **Download ZIP contains source only**; this setup command obtains the large assets separately.

Preview the download list with `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_online.ps1 -Plan`. For selected downloads after environment setup, use `.\.venv\Scripts\python.exe scripts/download_artifacts.py --groups models scores`. Available groups: `datasets models scores outputs weights imagenette`.

Completed files are preserved. If setup detects a conflicting existing file, it stops; use a fresh extraction rather than overwriting earlier experiments. A retry may download an archive again. Archives are removed from the temporary download cache after successful extraction to save disk space.

The [model and research artifact release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/model-artifacts-2026-11) contains all original saved model files, DINOv2 weights, historical outputs, logs and extra source files. See [the coverage and verification record](docs/UPLOAD_VERIFICATION.md) for exclusions and test limits. After setup, follow section B to prepare and evaluate. You can read the historical results immediately in section A without installing anything.

An [optional archive of the original Windows environment](docs/ENVIRONMENT_ARCHIVE.md) is available for backup/recovery. It contains all original `.venv` files, but is machine-specific and is not used by normal setup.

## A. See the existing results

The repository includes the reported summary tables:

| Dataset | Model | Result | Status |
|---|---|---:|---|
| MVTec AD, 15 categories | G_full | macro image AUROC 0.994 | Confirmatory historical result |
| MVTec LOCO AD, 5 categories | G_full | macro image AUROC 0.860 | Confirmatory historical result |
| MVTec LOCO AD, 5 categories | X3_mean | macro image AUROC 0.885 | Exploratory historical result |

Read [the detailed results and caveats](docs/RESULTS.md). To download the optional precomputed score cache, use the [Saved anomaly-score cache release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/score-cache-2026-11). Extract `patchcore-score-cache.zip` at the repository root so its files land under `work/scores/`. The archive contains 188 numerical score files (about 82 MiB); it contains no images, models, weights, or Python environment. It skips work only for matching experiment settings and dataset manifests. You still need the correct datasets and dependencies. Once the full archive or a compatible score cache is available, you can regenerate a report from saved outputs:

```powershell
.\hybridad.cmd report --out .\logs\RESULTS.generated.md
```

Open `logs\RESULTS.generated.md` to see the report. This reads saved scores; it does not train a model or evaluate new images. Using `--out` keeps the checked-in `docs/RESULTS.md` intact.

## B. Run a new experiment

### Requirements and inputs

- Windows and Python 3.12 are the documented environment. The original tested machine used Python 3.12.10, PyTorch with CUDA 12.8, and an NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM.
- `setup_online.ps1` already places the datasets, pretrained weights, Imagenette, trained models and saved scores where the pipeline expects them. Without it, download these inputs from their publishers into the locations listed in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md). Review the dataset terms before using or redistributing the files.

To rebuild the pinned CUDA environment on Windows if needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
```

If the existing `.venv` is broken or the checkout was moved to another PC, rebuild it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1 -Rebuild
```

Rebuilding downloads the packages listed in `requirements.lock.txt` and replaces only the checkout's `.venv` folder.

### Datasets stored somewhere else

Only needed if your datasets are not under `data/`; the online setup already puts them there. Copy `configs/local.example.yaml` to `configs/local.yaml` and set the two dataset paths to your folders. Remove the `work_dir` line unless you want models and caches elsewhere: with it, the pipeline looks for models and scores in that folder instead of the restored `work/`, and would train and score everything again. `configs/local.yaml` is ignored by Git and stays on your PC.

### Prepare and evaluate one category

Open PowerShell in the repository folder. These commands run the full G_full configuration for one MVTec AD category, `bottle`. The prepare command uses the Windows CUDA memory cap needed for DINO training:

```powershell
.\hybridad.cmd manifests --dataset mvtec_ad --category bottle
.\.venv\Scripts\python.exe .\scripts\run_capped.py --fraction 0.8 -- prepare --experiment configs/experiments/G_full.yaml --dataset mvtec_ad --category bottle
.\hybridad.cmd evaluate --experiment configs/experiments/G_full.yaml --dataset mvtec_ad --category bottle --exploratory
```

`prepare` fits any missing model branches, scores the calibration and test images, and calibrates the score fusion without reading test labels. `evaluate` reads the test labels and writes the metrics. New evaluations from this source release are labelled **EXPLORATORY** because this is an AUROC-only derivative of the original preregistered code. Keep the `--exploratory` flag; it makes that status explicit.

To process every category in one dataset, omit `--category bottle`, for example:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_capped.py --fraction 0.8 -- prepare --experiment configs/experiments/G_full.yaml --dataset mvtec_ad
.\hybridad.cmd evaluate --experiment configs/experiments/G_full.yaml --dataset mvtec_ad --exploratory
```

Use `--dataset mvtec_loco` for LOCO. Use `--dataset all` to run both datasets. Training and scoring every category can take many hours.

### What takes time, and how to avoid unnecessary work

The calibration calculation itself is small: it fits score normalizers and a fusion record from already saved normal-image scores. The longer step is `prepare`, which may need to train model branches and score images to create those calibration scores. Calibration-data preparation can take **about 20 minutes or longer per category** when compatible score files are not already available. DINO reconstruction training alone took about **20–57 minutes per category** in the archived runs; the full reference branches took about **87 minutes per category** on the original RTX 5070 Laptop. Timing varies with hardware, disk speed, and which outputs can be reused.

The pipeline is resumable and reuses compatible saved models and score files. Keep the same checkout, configuration, dataset manifests, and `work` folder between runs; rerunning `prepare` then skips branches whose required scores are already present. Avoid deleting `work` or `runs_rerun` if you want to resume. Run one GPU training job at a time and keep the PC plugged in and awake. The `scripts/run_capped.py --fraction 0.8` wrapper limits PyTorch CUDA cache growth on Windows and avoids WDDM spilling allocations into system RAM, which caused severe slowdowns in the original runs. Use it for DINO training as shown above.

Do not lower the number of training steps, shrink images, change the calibration split, or change the normalizer quantile just to make a run faster: those changes can alter results and make them incomparable to the documented experiment. The default calibration procedure in `src/hybridad/calibration.py` does not need a speed shortcut; reusing its valid cached scores is the safe time saver.

### Where outputs go

- New scores, predictions, metrics, and reports go under `runs_rerun` (or the configured run folder).
- Large trained models, score caches, and feature caches go under `work`.
- Calibration records are saved as `calibration.json` for each experiment/category/seed.
- Evaluation writes `predictions.csv`, `category_metrics.csv`, `aggregate.json`, provenance, and figures.
- The CLI prints the aggregate AUROC when evaluation finishes. Synthetic smoke results are only code checks; they are **not** benchmark results.

For pixel-level localization after evaluation, run:

```powershell
.\hybridad.cmd localize --experiment configs/experiments/G_full.yaml --dataset mvtec_ad --category bottle
```

For command options, run `.\hybridad.cmd <command> --help`, such as `.\hybridad.cmd prepare --help`.

## C. Run on another PC

Use `setup_online.ps1` as described at the top of this README. You do not need the original Windows folder. Its default dataset locations match `configs/base.yaml`, so no machine-specific config is required in a fresh checkout. The manual input sources and licenses remain listed in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md).

## Project notes

- The original benchmark results and methodological limitations are in [docs/RESULTS.md](docs/RESULTS.md) and [docs/METHODS.md](docs/METHODS.md).
- X3 was designed after viewing MVTec LOCO test results; its results are exploratory. See [docs/X3.md](docs/X3.md).
- The code in this repository is released under the [MIT License](LICENSE). Third-party code under `third_party/` keeps its own licenses, and the datasets keep their CC BY-NC-SA 4.0 terms; sources and licenses are listed in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md).
- Notes for restoring the project on the original research PC from its local archive are in [docs/ORIGINAL_PC.md](docs/ORIGINAL_PC.md).
- See [architecture.html](architecture.html) for diagrams and [visualiser.html](visualiser.html) for the interactive historical results viewer.
