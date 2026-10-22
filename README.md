# KG-TSRDeblur

PyTorch reference implementation of **KG-TSRDeblur: Kalman-Guided Lightweight Transformer for Traffic Sign Motion Deblurring and Recognition in Autonomous Driving**.

**Release status:** this code was newly implemented from the author-supplied manuscript. It is not recovered original experimental code. Missing implementation settings were selected for this release and are documented in [IMPLEMENTATION.md](docs/IMPLEMENTATION.md). No original datasets, experimental logs, detector/classifier checkpoints or trained restoration weights were supplied. The manuscript's accuracy, runtime and model size have **not** been reproduced by this repository. The default restorer has **848,613 parameters (0.849M)**, whereas the manuscript reports 9.1M; the unpublished stage depths prevent establishing architectural equivalence. Do not cite software smoke tests as evidence for the manuscript's numerical results.

[Implementation and limitations](docs/IMPLEMENTATION.md) · [Data format](docs/DATA.md) · [Validation record](docs/VALIDATION.md)

## Included

- Eight-state constant-velocity Kalman filter, IoU/Hungarian association, innovation gating, three-hit confirmation and five-frame propagation.
- Size-normalized state/covariance descriptors and reliability scores, equations (1)--(6).
- NAF-style encoder/decoder, four alternating regular/shifted window attention blocks, reliability-bounded motion adapter and spatial/frequency refinement, equations (7)--(9).
- MobileNetV3-Small sharp-ROI pretraining, reconstruction initialization and frozen-classifier recognition supervision, equations (10)--(11).
- Paired ROI evaluation, class-correct pipeline precision/recall, image/video inference, TT100K conversion, causal sequence caching, synthetic software fixtures and tests.
- Optional Ultralytics YOLOv8n training and inference integration. The detector requires traffic-sign training data and weights.

## Install

Python 3.10+ is required. CPU validation used Python 3.12, PyTorch 2.7.1 and torchvision 0.22.1. A CUDA GPU is recommended for research training.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

For CUDA, use the matching wheel command from [PyTorch](https://pytorch.org/get-started/locally/). The tested direct dependency versions are in `requirements-tested.txt`; this is not a cross-platform transitive lockfile. Video/detector extras: `python -m pip install -e ".[video]"`. Ultralytics has its own license; the MIT license here covers the newly written repository code, not third-party packages, datasets or checkpoints.

## Runnable software smoke test

```bash
python tools/smoke_test.py --output runs/smoke
```

This generates geometric toy images, trains a two-class classifier for one epoch, runs two restoration initialization steps and two recognition-supervised steps, evaluates, restores one image and checks the outputs. These are **software tests on artificial shapes**, not traffic-sign research experiments. The resulting toy checkpoints are not pretrained research models.

## Train with real data

Obtain data separately and prepare train/validation/test JSONL manifests as described in [DATA.md](docs/DATA.md). All examples below run from the repository root. The classifier and restoration commands reject shared sharp paths or sequence IDs across training and validation manifests.

```bash
# 1. Classifier on sharp training ROIs: 100 epochs, Adam, batch 64.
python -m kg_tsrdeblur.train_classifier --manifest data/train.jsonl --val-manifest data/val.jsonl --classes 20 --epochs 100 --batch-size 64 --device cuda --output runs/classifier

# 2. Reconstruction initialization: 100,000 iterations, lr=2e-4.
python -m kg_tsrdeblur.train --manifest data/train.jsonl --val-manifest data/val.jsonl --config configs/paper_reference.json --stage init --device cuda --output runs/init

# 3. Recognition-oriented optimization: 50,000 iterations, lr=1e-4.
python -m kg_tsrdeblur.train --manifest data/train.jsonl --val-manifest data/val.jsonl --stage joint --init-checkpoint runs/init/best_psnr.pt --classifier runs/classifier/best.pt --device cuda --output runs/joint

# 4. Test once after selecting checkpoints using validation only.
python -m kg_tsrdeblur.evaluate --manifest data/test.jsonl --checkpoint runs/joint/best_psnr.pt --classifier runs/classifier/best.pt --classes 20 --device cuda --output runs/test_metrics.json
```

Repeat research experiments with seeds **42, 43 and 44** (`--seed`) and preserve split files, configuration files, class mappings, environment and logs. The code saves actual validation/test metrics; no manuscript result is embedded as a measured output. `best_psnr.pt` is selected by validation PSNR. For recognition-focused model selection, evaluate a predeclared checkpoint selection protocol on validation data before testing; this release does not claim to recover the paper's undisclosed selection protocol.

Resume a stage with `--resume runs/init/latest.pt --stage init`, the same data and total `--steps`. Optimizer and schedule are restored; data-loader/RNG position is not restored, so resumption is not bitwise identical to an uninterrupted run.

## Inference

Supply a checkpoint produced by training. Single images use zero motion reliability.

```bash
python -m kg_tsrdeblur.infer --input sign.png --checkpoint runs/joint/best_psnr.pt --output runs/restored.png
python -m kg_tsrdeblur.pipeline --input driving.mp4 --detector sign_yolov8n.pt --restorer runs/joint/best_psnr.pt --classifier runs/classifier/best.pt --tracker-config configs/tracker_reference.json --output runs/restored_video.mp4
python -m kg_tsrdeblur.evaluate_pipeline --predictions runs/restored_video.jsonl --ground-truth data/video_gt.jsonl --output runs/pipeline_metrics.json
python -m kg_tsrdeblur.benchmark --checkpoint runs/joint/best_psnr.pt --device cuda --output runs/roi_benchmark.json
```

Video inference writes restored frames and per-frame candidate JSONL. Pixels outside expanded ROI crops remain unchanged before video encoding. At most eight ROIs are restored per frame. Pipeline metrics require both category correctness and IoU >= 0.5. Benchmark timings cover the restoration network only, and must not be reported as complete-pipeline FPS. FFT/attention MACs are not currently counted.

## Ablations

Create a copy of the model JSON and set `image_only`, `spatial_only` or `conv_bottleneck` to `true`. For fixed reliability, use `--fixed-reliability` in video inference and a consistently modified training manifest assigning 1 to confirmed tracks only. For the no-recognition-loss variant, pass `--no-recognition-loss` in the joint stage. Retrain every architecture/loss variant with the same budget and seeds. Use `evaluate --blurred-baseline` for unprocessed ROIs. Ablation switches provide experimental controls; no ablation results are included.

## Citation and provenance

See [CITATION.cff](CITATION.cff) for the manuscript title/authors. Publication venue and DOI were not supplied. The PDF and private/raw driving data are not redistributed. Module docstrings and [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) map code to the manuscript's equations and distinguish reported settings from reference choices. [PyTorch FFT](https://docs.pytorch.org/docs/stable/fft.html), [scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), and [torchvision MobileNetV3](https://docs.pytorch.org/vision/stable/models/mobilenetv3.html) supply the underlying APIs.
