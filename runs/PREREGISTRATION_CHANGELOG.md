# Pre-registration changelog

## v1 -> v2 (2026-09-26, 01:55 local)
v1 (`PREREGISTRATION_v1_superseded.json`, code hash 135b7716b151d059) was superseded BEFORE any evaluation of an
experiment that it and v2 treat differently. No test labels had been joined for any experiment when v2 was written.
Reason: the user asked for the full suite within ~10 hours and approved speed-ups:
* DINO reconstruction variants: `matmul_precision: tf32` (documented numeric deviation from Dinomaly's fp32);
  this changes every DINO variant ID, so all DINO models are trained under v2 only.
* Exact speed-ups (bitwise-equality tests in tests/test_speedups_exact.py): cached frozen-encoder outputs for DINO
  training; preloaded penalty-image crops for the local student-teacher. These changed source code (hence the new code
  hash) but not any model, score or configuration of PatchCore, local ST or composition.
Everything else (experiments, fusion rules, calibration, protocol, acceptance criteria) is unchanged.

## v2 -> v3 (2026-09-26, 14:29 local)
Bug fix only, no configuration change: resuming an interrupted DINO-reconstruction training failed
("RNG state must be a torch.ByteTensor") because RNG states were restored from a checkpoint mapped to the GPU.
Fixed in src/hybridad/branches/dino_recon.py; regression test tests/test_resume.py. No evaluated model used this
path (all DINO models so far trained without interruption). Experiment specs are unchanged; only the code hash
changes. Results evaluated under v2 (MVTec LOCO A-H and variants; MVTec AD A and H) keep the confirmatory status
recorded at their evaluation time.
