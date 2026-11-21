# Notes for the original research PC

These steps apply only to the PC where the original `kutty dinosaur` research folder lives. On any other PC, use `setup_online.ps1` as described at the top of the [README](../README.md).

## Restore from the original research folder

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

The original folder is not changed. The setup stops if it finds a different file where it needs to copy one. It also stops if `runs_check` or `logs\self_check.json` already exists, so it cannot overwrite a previous check. For a fresh attempt, use a fresh repository extraction. If the archive is stored elsewhere, open PowerShell in the extracted folder and run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_local_archive.ps1 -Archive "D:\path\to\kutty dinosaur"
```

The full original archive is about 27 GB. The public releases now host MVTec datasets, saved models, scores, DINOv2 weights and research outputs. The online setup in the [README](../README.md) obtains the remaining publisher inputs and rebuilds the Python environment.

After the restore, prepare and evaluate as described in section B of the [README](../README.md).

## Point the pipeline at the original dataset folder

A GitHub source download does not include the benchmark images. On this PC, the datasets are in `%USERPROFILE%\OneDrive\Desktop\datasets`, with `MVTecAD` and `MVTec_LOCO` inside. Create the ignored, machine-local config once from PowerShell in the repository folder:

```powershell
if (Test-Path .\configs\local.yaml) { throw "configs/local.yaml already exists; preserve it and check its dataset paths before continuing." }
$datasetRoot = (Join-Path $env:USERPROFILE 'OneDrive\Desktop\datasets').Replace('\', '/')
if (-not (Test-Path (Join-Path $env:USERPROFILE 'OneDrive\Desktop\datasets\MVTecAD')) -or
    -not (Test-Path (Join-Path $env:USERPROFILE 'OneDrive\Desktop\datasets\MVTec_LOCO'))) {
    throw "Expected MVTecAD and MVTec_LOCO folders under $env:USERPROFILE\OneDrive\Desktop\datasets"
}
@"
paths:
  datasets:
    mvtec_ad: "$datasetRoot/MVTecAD"
    mvtec_loco: "$datasetRoot/MVTec_LOCO"
"@ | Set-Content -LiteralPath .\configs\local.yaml -Encoding utf8
```

This writes `configs/local.yaml`, which is ignored by Git and stays on your PC. It does not copy or upload the images. The dataset paths are used by `hybridad` commands, including `prepare`.
