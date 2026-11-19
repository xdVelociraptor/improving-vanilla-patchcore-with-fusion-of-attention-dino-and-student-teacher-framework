# Upload coverage and verification

Checked on 2026-11-19. The source ZIP plus the release assets and publisher downloads form the restore path; the source ZIP alone is not the full archive.

## 1. Original-file coverage

The inventory accounts for all 61,518 files in the original research folder:

| Disposition | Original files |
|---|---:|
| Source tree or model/score/output releases | 2,102 |
| Original PDF, renamed to `docs/REPORT.pdf` | 1 |
| MVTec dataset release, including masks and notices | 11,558 |
| Imagenette train split, obtained from Fast.ai | 9,469 |
| WideResNet and EfficientAD teacher, obtained from publishers | 2 |
| Machine-specific `.venv`, optional environment release; normal setup rebuilds it | 38,386 |

No original file is unaccounted for. This is a path/coverage audit, not a claim that every source file is unchanged: the repository contains documented code and privacy edits. All 259 original saved-model files are included across 21 ZIPs. The output supplement has 1,452 entries, including its added notice. Machine-specific home paths were sanitized in 85 text outputs. See [coverage counts](artifact-coverage.json) and [supplement file hashes](artifact-files.json).

## 2. Integrity

All 23 new ZIPs passed full ZIP CRC checks. Each of the 259 model files was also compared by SHA-256 with its original file. All three restored pretrained weight files match the originals. All 9,469 original Imagenette train files match the publisher restore. The supplement has no conflicting files in the fresh source checkout.

The MVTec release was previously checked against every original dataset path and size, including masks and notices. A fresh manifest-image check in this audit found 5,354 AD images and 3,651 LOCO images matching their recorded hashes, with zero failures. That image check does not independently cover every mask; the archive/path checks cover the remaining dataset files.

## 3. Public delivery

All three releases are published. Anonymous GitHub API requests verified all 45 GitHub-hosted downloads referenced in [artifact-downloads.json](../artifact-downloads.json): uploaded state, exact URL and SHA-256 digest. See [release verification](release-verification.json).

The restore script then downloaded and extracted the outputs, DINOv2 weights, saved scores, WideResNet weights, EfficientAD teacher archive and Imagenette archive without GitHub login, validating the pinned archive hashes. See [publisher comparison](publisher-verification.json). The full MVTec/model payload was not downloaded again in this final pass; published digests were checked against the already-verified archives.

## 4. Setup and runtime

- Restore tests: path traversal rejection, existing-file preservation, required archive member checking, and ZIP extraction all passed.
- Applicable test suite: **32 passed, 5 skipped, 24 deselected**. Weight/data/reference/slow cases were not run in this suite. Native FAISS failed inside the sandbox but imported correctly outside it; the successful suite ran outside the sandbox using the existing Python environment.
- PowerShell setup scripts parsed without errors; the CLI help command succeeded.
- Downloaded scores successfully recalibrated and predicted all **20 MVTec AD/LOCO categories** using G_full. The calibration/prediction loop took about 2 seconds on this machine with dependencies already loaded. This timing excludes installation, downloads, image scoring and training.
- One exploratory bottle evaluation produced image AUROC **1.0** from downloaded scores. Historical report generation also succeeded. These checks reuse archived measurements, not a new independent benchmark run. See [runtime record](cached-runtime-verification.json).

## Limits and non-mirrored inputs

A completely fresh Windows installation, fresh GPU training, and new-image inference have not been validated in this audit. The full one-command installation was not repeated from an empty environment; runtime checks used the existing pinned environment. Python 3.12, a suitable NVIDIA driver, internet and sufficient disk space remain required. New or changed experiments can still require training and scoring; calibration-data preparation can take 20 minutes or more per category.

Imagenette, WideResNet and the EfficientAD teacher are fetched from publishers rather than mirrored because redistribution terms are not established for this repository. The `.venv` is now also preserved in an optional [environment archive](ENVIRONMENT_ARCHIVE.md); normal setup rebuilds it because it contains machine-specific paths and installed dependencies. These are explicit distribution choices, not silently missing inputs. See [licenses and sources](../DATA_AND_LICENSES.md).

## Environment archive addition, 2026-11-20

Four optional ZIPs preserve all 38,386 original environment files. Every archived file was compared by SHA-256 with its original, and the archive path inventory has no omissions or duplicates. Release upload digests were verified before publication. This adds archival coverage; it does not turn the original environment into a portable installation. Redistribution permission has not been established for Imagenette or the two remaining pretrained weights, so those remain verified publisher downloads.
