# Model metadata and research artifacts

This directory includes the original model metadata JSON files: settings, model and input fingerprints, and diagnostics for the saved experiments.

The original `work/models/` also contains trained checkpoints and memory banks (about 8.56 GB); these models are not included in this source repository and remain in the original research folder. A separate [GitHub release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/score-cache-2026-11) provides 188 numerical score files (about 82 MiB). Download `patchcore-score-cache.zip` and extract it at the repository root to restore `work/scores/`. The scores can save scoring/calibration preparation only when the experiment and dataset manifests match. They do not replace datasets, trained models, pretrained weights, or the Python environment. See the release notes and [run guide](../README.md).

The saved model metadata and input hashes can help identify the required artifacts. Dataset images, pretrained weights, logs and the Python environment are distributed or generated separately; see [the data and license guide](../DATA_AND_LICENSES.md) and [README](../README.md).
