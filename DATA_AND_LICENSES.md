# Data, weights and third-party code

This source snapshot contains code, split metadata and historical summary results. Download separately distributed inputs from their publishers, checking the relevant terms first. The original research archive included images, weights, trained artifacts and a virtual environment; those files are outside this repository.

| Input | Obtain from | Where to place it |
|---|---|---|
| Original MVTec AD (15 categories; not AD 2) | [MVTec AD download and license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | `data/MVTecAD/`, with `bottle/`, `cable/`, etc. directly below it |
| MVTec LOCO AD (5 categories) | [MVTec LOCO AD download and license](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad) | `data/MVTec_LOCO/`, with `breakfast_box/`, `juice_bottle/`, etc. directly below it |
| Full-size Imagenette v2 train split | [Fast.ai Imagenette and archive links](https://github.com/fastai/imagenette) | `external_data/imagenette2/train/` |
| WideResNet-50-2 weights | [Torchvision weight file](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth) | `weights/wide_resnet50_2-95faca4d.pth` |
| DINOv2 ViT-B/14 with registers weights | [Meta weight file](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) | `weights/dinov2_vitb14_reg4_pretrain.pth` |
| EfficientAD PDN-S teacher weights | [Anomalib release and asset](https://github.com/open-edge-platform/anomalib/releases/tag/efficientad_pretrained_weights), a third-party release rather than the EfficientAD authors' weights | `weights/efficientad/efficientad_pretrained_weights/pretrained_teacher_small.pth` |

MVTec publishes both benchmark datasets under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Preserve the dataset authors' attribution and review the noncommercial and share-alike terms. Imagenette is a subset of ImageNet; see [ImageNet access terms](https://image-net.org/accessagreement). Availability of an archive does not establish redistribution rights for its individual images. This project links to publisher downloads and does not mirror them.

The original archive's SHA-256 fingerprints for the weights and Imagenette archive are in [the provenance registry](src/hybridad/provenance.py). The code verifies weights before loading them. Verify the original MVTec image files against the saved split manifests with `python scripts/verify_datasets.py`. The manifests contain relative image and mask identifiers, labels, roles and image checksums; they do not contain image bytes. The verifier checks the images listed in the manifests; it does not verify masks or files outside the manifests. MVTec AD masks are under category `ground_truth/`; preserve the downloaded annotations for localization.

Included third-party source:

- [Meta DINOv2](https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8): only the base model and layer source used by the pipeline, with the [Apache-2.0 license](third_party/dinov2/LICENSE). Separate Cell DINO, X-ray, and text model material is outside this snapshot.
- [Amazon Science PatchCore](https://github.com/amazon-science/patchcore-inspection/tree/fcaa92f124fb1ad74a7acf56726decd4b27cbcad): reference Python source for tests, with [Apache-2.0 license](third_party/patchcore-inspection/LICENSE) and [NOTICE](third_party/patchcore-inspection/NOTICE).
- MVTec LOCO AD evaluator v2.0: [BSD 3-Clause terms](third_party/mvtec_loco_ad_evaluation/LICENSE.txt) accompany the local copy.

When citing the benchmarks, use the citation instructions on the MVTec pages: Bergmann et al., *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (2021), and Bergmann et al., *Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization* (2022).
