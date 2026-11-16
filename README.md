# PatchCore + DINOv2 anomaly detection

This project detects defects in images from the MVTec AD and MVTec LOCO AD benchmarks. Its full model combines PatchCore, a local student–teacher model, DINOv2 reconstruction, and composition features.

This README is the run guide. Start with **A. See the existing results** if you only want to inspect the completed experiment, or **B. Run the saved project on the original PC** to restore the full research archive. A GitHub clone alone does not contain the datasets, model files, or Python environment. An optional saved-score cache is available from the release linked below.

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

## B. Run the saved project on the original Windows PC

This is the quickest way to run the archived project. It uses the original research folder already on the PC and restores it into a separate checkout.

**On the original PC, this restore has already been done:** use the separate checkout in `Documents\Codex\patchcore-run`. Do not copy the archive again unless you are setting up a different checkout.

1. Download the repository ZIP from GitHub and extract it to a new folder. Do not extract it over the original `kutty dinosaur` folder.
2. Make sure the original folder is at `%USERPROFILE%\OneDrive\Desktop\kutty dinosaur` and that the C: drive has about **29 GB free** (the copy is about 27 GB, plus 2 GB spare).
3. Double-click `setup_local_archive.cmd` in the extracted repository and let it finish. It copies the datasets, pretrained weights, trained models, saved scores, results, and Python environment, then runs the quick check. The copy can take several minutes.
4. When it reports `ALL CHECKS PASSED`, open PowerShell in the extracted repository folder and run:

   ```powershell
   .\hybridad.cmd --help
   .\hybridad.cmd report --out .\logs\RESULTS.generated.md
   ```

The original folder is not changed. The setup stops if it finds a different file where it needs to copy one. It also stops if `runs_check\` or `logs\self_check.json` already exists, so it cannot overwrite a previous check. For a fresh attempt, use a fresh repository extraction. If the archive is stored elsewhere, open PowerShell in the extracted folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_local_archive.ps1 -Archive "D:\path\to\kutty dinosaur"
```

The full archive is about 27 GB. GitHub contains the source and documentation, and the release provides the optional numerical score cache. GitHub does not host the datasets, pretrained weights, trained checkpoints, or Python environment.

## C. Run a new experiment

### Requirements and inputs

- Windows and Python 3.12 are the documented environment. The original tested machine used Python 3.12.10, PyTorch with CUDA 12.8, and an NVIDIA RTX 5070 Laptop GPU with 8 GB VRAM.
- You need the original MVTec AD and/or MVTec LOCO AD images, pretrained model weights, and Imagenette for the local student–teacher penalty branch. Download these from their publishers and put them in the locations in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md). Review the dataset terms before using or redistributing the files.
- If you restored the original archive using section B, these inputs and the original `.venv` are already in the checkout.

To rebuild the pinned CUDA environment on Windows if needed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1
```

If the existing `.venv` is broken or the checkout was moved to another PC, rebuild it with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1 -Rebuild
```

Rebuilding downloads the packages listed in \`requirements.lock.txt\` and replaces only the checkout's \`.venv\` folder.

### Point the download at your dataset folder

A GitHub source download does not include the benchmark images. On this PC, the datasets are in \`%USERPROFILE%\\OneDrive\\Desktop\\datasets\`, with \`MVTecAD\` and \`MVTec_LOCO\` inside. Create the ignored, machine-local config once from PowerShell in the repository folder:

\`\`\`powershell
if (Test-Path .\\configs\\local.yaml) { throw "configs/local.yaml already exists; preserve it and check its dataset paths before continuing." }
$datasetRoot = (Join-Path $env:USERPROFILE 'OneDrive\\Desktop\\datasets').Replace('\\', '/')
if (-not (Test-Path (Join-Path $env:USERPROFILE 'OneDrive\\Desktop\\datasets\\MVTecAD')) -or
    -not (Test-Path (Join-Path $env:USERPROFILE 'OneDrive\\Desktop\\datasets\\MVTec_LOCO'))) {
    throw "Expected MVTecAD and MVTec_LOCO folders under $env:USERPROFILE\\OneDrive\\Desktop\\datasets"
}
@"
paths:
  datasets:
    mvtec_ad: "$datasetRoot/MVTecAD"
    mvtec_loco: "$datasetRoot/MVTec_LOCO"
"@ | Set-Content -LiteralPath .\\configs\\local.yaml -Encoding utf8
\`\`\`

This writes \`configs/local.yaml\`, which is ignored by Git and stays on your PC. It does not copy or upload the images. The dataset paths are used by \`hybridad\` commands, including \`prepare\`.

### Prepare and evaluate one category

Open PowerShell in the repository folder. These commands run the full G_full configuration for one MVTec AD category, `bottle`. The prepare command uses the Windows CUDA memory cap needed for DINO training:

```powershell
.\hybridad.cmd manifests --dataset mvtec_ad --category bottle
.\.venv\Scripts\python.exe .\scripts\run_capped.py --fraction 0.8 -- prepare --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle
.\hybridad.cmd evaluate --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle --exploratory
```

`prepare` fits any missing model branches, scores the calibration and test images, and calibrates the score fusion without reading test labels. `evaluate` reads the test labels and writes the metrics. New evaluations from this source release are labelled **EXPLORATORY** because this is an AUROC-only derivative of the original preregistered code. Keep the `--exploratory` flag; it makes that status explicit.

To process every category in one dataset, omit `--category bottle`, for example:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_capped.py --fraction 0.8 -- prepare --experiment experiments/G_full.yaml --dataset mvtec_ad
.\hybridad.cmd evaluate --experiment experiments/G_full.yaml --dataset mvtec_ad --exploratory
```

Use `--dataset mvtec_loco` for LOCO. Use `--dataset all` to run both datasets. Training and scoring every category can take many hours.

### What takes time, and how to avoid unnecessary work

The calibration calculation itself is small: it fits score normalizers and a fusion record from already saved normal-image scores. The longer step is `prepare`, which may need to train model branches and score images to create those calibration scores. Calibration-data preparation can take **about 20 minutes or longer per category** when compatible score files are not already available. DINO reconstruction training alone took about **20–57 minutes per category** in the archived runs; the full reference branches took about **87 minutes per category** on the original RTX 5070 Laptop. Timing varies with hardware, disk speed, and which outputs can be reused.

The pipeline is resumable and reuses compatible saved models and score files. Keep the same checkout, configuration, dataset manifests, and `work\` folder between runs; rerunning `prepare` then skips branches whose required scores are already present. Avoid deleting `work\` or `runs_rerun\` if you want to resume. Run one GPU training job at a time and keep the PC plugged in and awake. The `scripts/run_capped.py --fraction 0.8` wrapper limits PyTorch CUDA cache growth on Windows and avoids WDDM spilling allocations into system RAM, which caused severe slowdowns in the original runs. Use it for DINO training as shown above.

Do not lower the number of training steps, shrink images, change the calibration split, or change the normalizer quantile just to make a run faster: those changes can alter results and make them incomparable to the documented experiment. The default calibration procedure in `src/hybridad/calibration.py` does not need a speed shortcut; reusing its valid cached scores is the safe time saver.

### Where outputs go

- New scores, predictions, metrics, and reports go under `runs_rerun\` (or the configured run folder).
- Large trained models, score caches, and feature caches go under `work\`.
- Calibration records are saved as `calibration.json` for each experiment/category/seed.
- Evaluation writes `predictions.csv`, `category_metrics.csv`, `aggregate.json`, provenance, and figures.
- The CLI prints the aggregate AUROC when evaluation finishes. Synthetic smoke results are only code checks; they are **not** benchmark results.

For pixel-level localization after evaluation, run:

```powershell
.\hybridad.cmd localize --experiment experiments/G_full.yaml --dataset mvtec_ad --category bottle
```

For command options, run `.\hybridad.cmd <command> --help`, such as `.\hybridad.cmd prepare --help`.

## D. Run on a different PC without the original archive

1. Clone or download this repository.
2. In PowerShell, run `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_env.ps1` to install the pinned environment. This requires Python 3.12 and internet access.
3. Download the datasets and weights from their publishers as described in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md). Keep the expected folder names and file names. MVTec AD and LOCO AD are separately licensed datasets; Imagenette follows ImageNet access terms.
4. Verify the tracked MVTec image files:

   ```powershell
   python .\scripts\verify_datasets.py --ad-root "$env:USERPROFILE\OneDrive\Desktop\datasets\MVTecAD" --loco-root "$env:USERPROFILE\OneDrive\Desktop\datasets\MVTec_LOCO"
   ```

5. Follow section C to prepare, evaluate, and inspect outputs. Trained models are not included in a GitHub-only checkout and must be trained or separately restored. Historical numerical scores can be downloaded from the [Saved anomaly-score cache release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/score-cache-2026-11) and extracted at the repository root; check the release notes and confirm compatibility before relying on them.

## Project notes

- The original benchmark results and methodological limitations are in [docs/RESULTS.md](docs/RESULTS.md) and [docs/METHODS.md](docs/METHODS.md).
- X3 was designed after viewing MVTec LOCO test results; its results are exploratory. See [docs/X3.md](docs/X3.md).
- Dataset, weight, and third-party code sources and licenses are listed in [DATA_AND_LICENSES.md](DATA_AND_LICENSES.md).
- See [architecture.html](architecture.html) for diagrams and [visualiser.html](visualiser.html) for the interactive historical results viewer.
