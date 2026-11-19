# Data, weights and third-party code

This source snapshot contains code, split metadata and historical summary results. The repository source tree does not contain dataset images. MVTec AD and MVTec LOCO AD are also available as optional category archives in the [MVTec dataset release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/mvtec-datasets-2026-11); retain the included notices and follow their license terms. The original publisher pages remain the authoritative sources. Weights, trained artifacts, Imagenette, and a Python environment are separate inputs.

| Input | Obtain from | Where to place it |
|---|---|---|
| Original MVTec AD (15 categories; not AD 2) | [MVTec AD download and license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | `data/MVTecAD/`, with `bottle/`, `cable/`, etc. directly below it |
| MVTec LOCO AD (5 categories) | [MVTec LOCO AD download and license](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | `data/MVTec_LOCO/`, with `breakfast_box/`, `juice_bottle/`, etc. directly below it |
| Full-size Imagenette v2 train split | [Fast.ai Imagenette and archive links](https://github.com/fastai/imagenette) | `external_data/imagenette2/train/` |
| WideResNet-50-2 weights | [Torchvision weight file](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth) | `weights/wide_resnet50_2-95faca4d.pth` |
| DINOv2 ViT-B/14 with registers weights | [Meta weight file](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) | `weights/dinov2_vitb14_reg4_pretrain.pth` |
| EfficientAD PDN-S teacher weights | [Anomalib release and asset](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights), a third-party release rather than the EfficientAD authors' weights | `weights/efficientad/efficientad_pretrained_weights/pretrained_teacher_small.pth` |

MVTec publishes both benchmark datasets under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Preserve the dataset authors' attribution and observe the noncommercial and share-alike terms. The separate [MVTec dataset release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/mvtec-datasets-2026-11) is an unofficial, unchanged-file archive of the supplied MVTec AD and LOCO AD folders; it includes the source notices and `SHA256SUMS.txt`. Imagenette is a subset of ImageNet; see [ImageNet access terms](https://image-net.org/accessagreement).

The original archive's SHA-256 fingerprints for the weights and Imagenette archive are in [the provenance registry](src/hybridad/provenance.py). The code verifies weights before loading them. Verify the original MVTec image files against the saved split manifests with `python scripts/verify_datasets.py`. The manifests contain relative image and mask identifiers, labels, roles and image checksums; they do not contain image bytes. The verifier checks the images listed in the manifests; it does not verify masks or files outside the manifests. MVTec AD masks are under category `ground_truth/`; preserve the downloaded annotations for localization.

Included third-party source:

- [Meta DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8): only the base model and layer source used by the pipeline, with the [Apache-2.0 license](third_party/dinov2/LICENSE). Separate Cell DINO, X-ray, and text model material is outside this snapshot.
- [Amazon Science PatchCore](https://github.com/amazon-science/patchcore-inspection/tree/fcaa92f124fb1ad74a7acf56726decd4b27cbcad): reference Python source for tests, with [Apache-2.0 license](third_party/patchcore-inspection/LICENSE) and [NOTICE](third_party/patchcore-inspection/NOTICE).
- MVTec LOCO AD evaluator v2.0: [BSD 3-Clause terms](third_party/mvtec_loco_ad_evaluation/LICENSE.txt) accompany the local copy.

When citing the benchmarks, use the citation instructions on the MVTec pages: Bergmann et al., *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (2021), and Bergmann et al., *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization* (2022).

## Online restore and archived additions

`setup_online.ps1` uses the pinned URLs and SHA-256 values in `artifact-downloads.json`. Large files reside in GitHub Releases; they are not included in the source ZIP. The [model artifact release](https://github.com/xdVelociraptor/improving-vanilla-patchcore-with-fusion-of-attention-dino-and-student-teacher-framework/releases/tag/model-artifacts-2026-11) adds all original saved models, logs and research outputs, the missing composition figure, and unmodified DINOv2 base weights with their Apache-2.0 license. The optional extra third-party source includes additional DINO variants with their own supplied licenses; the base Apache license must not be assumed to govern every variant.

WideResNet weights and the EfficientAD teacher are obtained directly from their publisher URLs, rather than mirrored here. Torchvision [notes that pretrained models can have separate terms](https://docs.pytorch.org/vision/stable/models.html); the EfficientAD release has an [unresolved upstream license clarification](https://github.com/open-edge-platform/anomalib/issues/3710). Imagenette is also downloaded from Fast.ai rather than mirrored. Review the relevant publisher terms for your use. The downloader verifies the pinned original archive hashes before extraction.

The original `.venv` is available as an [optional archival release](docs/ENVIRONMENT_ARCHIVE.md), retaining its installed-package notices. Normal setup still rebuilds it from `requirements.lock.txt` because the archived environment is machine-specific. Private machine paths in 85 archived text outputs were replaced by `<USER_HOME>`; the originals remain untouched. Historical logs, predictions and intermediate results do not constitute a new independent reproduction of the published metrics.
