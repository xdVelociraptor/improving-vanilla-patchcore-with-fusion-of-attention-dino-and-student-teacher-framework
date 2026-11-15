# Model metadata and research artifacts

This directory includes the original model metadata JSON files: settings, model and input fingerprints, and diagnostics for the saved experiments.

The original `work/models/` also contains trained checkpoints and memory banks (about 8.56 GB); `work/scores/` contains score files (about 86 MB). These binary artifacts are not included in this source repository. They remain in the original research folder. Without them, reproducing the saved-score workflow requires restoring the original artifacts or training and scoring again.

The saved model metadata and input hashes can help identify the required artifacts. Dataset images, pretrained weights, logs and the Python environment are distributed or generated separately; see [the data and license guide](../DATA_AND_LICENSES.md) and [README](../README.md).
