# Software validation record

Validation date: 2026-10-23.

Environment: Windows, Python 3.12, PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, NumPy 2.2.6, SciPy 1.16.3, pytest 8.4.2. CPU threads were limited to two for small-image checks.

## Automated tests

`python -m pytest -q`: **39 passed**.

Checks cover zero-reliability equivalence, gate bounds and motion gradients, arbitrary image sizes and finite gradients, shifted-window wrap masking, frozen-classifier gradient transmission, identity reconstruction losses, ablation shapes, invalid reliability rejection, track confirmation and gap expiry, covariance PSD and descriptor normalization, innovation gating, isolated tracker state, letterboxing, unchanged pixels outside ROI crops, split leakage rejection and pipeline metric matching.

Maintenance regressions cover best-PSNR checkpoint persistence and legacy log recovery, source/group/hash split rejection, readable errors for blurred-only training manifests, removal of inactive ablation parameters, configuration JSON round trips, attention MAC counting, both Adam betas for the detector optimizer, fast-motion identity continuity through five missing frames, nine-frame sequence generation and changed-source rejection before output reuse.

## End-to-end execution

`python tools/smoke_test.py --output runs/smoke` completed successfully:

1. Generated train/validation/test toy images (8/4/4 geometric shapes).
2. Trained a two-class MobileNetV3-Small for one epoch.
3. Ran two restoration initialization steps and saved checkpoints.
4. Loaded initialization and frozen classifier checkpoints, ran two recognition-supervised steps, saved checkpoints.
5. Evaluated four held-out toy images and wrote metrics JSON.
6. Restored an image and confirmed its output dimensions.

Default network forward/backward also passed at 128x128. The parameter count measured by `benchmark.py` was 848,613. Timing values depend on the machine and are produced by the benchmark command.

At batch one and 128x128 the counter reports 1,783,719,424 MACs for executed convolution, linear and attention matrix products. FFT and elementwise work are excluded; this is not a full-operation count. The full 1,620-frame published archive was rechecked for hashes, dimensions, temporal metadata, source exclusion and source/group split isolation. Existing release files and their checksums are unchanged.

Toy fixtures are generated solely to test software execution. The optional external YOLO/video integration and dataset converters can be run with user-supplied data and checkpoints using the documented commands.

The detector optimizer adjustment follows the [Ultralytics trainer extension API](https://docs.ultralytics.com/guides/custom-trainer/) and [optimizer construction interface](https://docs.ultralytics.com/reference/engine/trainer/). Its Adam parameter update is unit tested. Training a traffic-sign detector and timing the complete video pipeline require user-supplied annotated data and trained checkpoints; those experiments were not run.
