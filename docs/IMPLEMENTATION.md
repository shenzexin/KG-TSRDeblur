# Method mapping and implementation decisions

This document maps the method equations to code and records the settings used by this implementation.

## Mapping

| Manuscript | Code |
| --- | --- |
| Eq. (1)--(3): state, prediction, normalized innovation, association | `tracking.py`: `KalmanTrack`, `MultiObjectTracker` |
| Eq. (4)--(5): ROI-scaled covariance and reliability | `KalmanTrack.prior` |
| Eq. (6): 16D descriptor -> 32D learned motion embedding | `prior` and `model.py:MotionAdapter.encoder` |
| Eq. (7): r-bounded spatial motion adapter | `MotionAdapter.forward` |
| Section 2.2: NAF-style encoder/decoder, alternating shifted windows | `NAFBlock`, `WindowBlock`, `KGTSRDeblur` |
| Eq. (8)--(9): bounded spectral gains and residual fusion | `SpatialFrequencyRefinement`, `KGTSRDeblur.forward` |
| Eq. (10)--(11): recognition, L1, SSIM and complex Fourier losses | `losses.py`, `train.py`, `classifier.py` |

## Settings explicitly reported in the manuscript

- YOLOv8n, detector input 640; MobileNetV3-Small; 20 classes.
- 128x128 ROI, 10% expansion, aspect-ratio-preserving padding, up to eight candidates per frame.
- Base width 32; two downsampling stages; four bottleneck blocks; four heads; 8x8 windows; motion dimension 32.
- Track confirmation after three observations and at most five missing-frame propagations.
- Classifier/detector training: 100 epochs, initial learning rate 1e-3, batch sizes 64/16.
- Restoration: 100,000 initialization iterations at 2e-4, then 50,000 recognition-oriented iterations at 1e-4; batch size 16; Adam betas (0.9,0.9), no weight decay, cosine floor 1e-6.
- Loss weights: SSIM 0.1, FFT 0.01, recognition 0.1. Classifier frozen during restoration optimization; no gradients through tracking/detection.
- Seeds 42,43,44; TT100K official train/test and 10% train-to-validation holdout; private videos split by sequence.

## Supplementary choices made for this implementation

| Unspecified detail | Reference choice and reason |
| --- | --- |
| Encoder/decoder stage depths | (2,2)/(2,2), channels 32/64/128; a minimal compact implementation |
| Transformer details | Pre-channel-LN; QKV dot-product attention; no learned positional bias; 2x feed-forward expansion with depthwise 3x3 and GELU; shift 4; explicit cyclic-boundary and padding masks |
| Skip fusion and upsampling | Concatenation + 1x1 convolution, pixel shuffle; 2x2 stride-2 convolution downsampling |
| NAF-style block details | Multiplicative gate, simplified channel attention, zero-initialized residual scales, 2x expansion |
| Motion MLP and adapters | 16->32->32 GELU MLP, linear channel projection, 1x1 gate, two-layer GELU correction |
| Frequency refinement width | 16 compressed channels; real gain from two 1x1 layers with GELU; FFT float32 with matching ortho normalization |
| Kalman noise | dt in seconds; white acceleration std=300 px/s^2; measurement std=3 px; initial velocity std=max(100 px/s, one corresponding box dimension/dt); initial geometry variance=9 |
| Assignment gates | IoU>=0.01; normalized innovation<=9.4877/4 (chosen numerical gate); no category-based rejection |
| Reliability penalties | alpha=1, beta=0.1, gamma=0.5; retained last accepted confidence/innovation across gaps |
| Confirmation | Three accumulated accepted observations including initialization, separated by frame calls; tentative tracks dropped on first miss |
| Box used for ROI | Current detection when matched; predicted box during gaps; covariance normalization uses selected unexpanded box |
| Candidate filtering | Confidence>=0.25, class-agnostic NMS at IoU 0.5; current detections before predicted candidates, confidence then track ID |
| ROI expansion | Total width/height increased 10% (5% per side), clipped to image; bicubic resize, black padding |
| SSIM convention | RGB mean; data range 1; Gaussian 11x11 sigma1.5; valid convolution support; full padded ROI scored |
| Frequency loss | Full orthonormal fft2; mean complex magnitude difference (not separate real/imaginary L1) |
| Classifier preprocessing | Input 128x128; ImageNet RGB mean/std in graph; weights=None, train on sharp labeled ROIs |
| Sharp/blur augmentation | No stochastic flips/crops, avoiding inconsistent geometric motion descriptors |
| Synthetic blur | Uniform length 3..25 px, angle [0,180), normalized rasterized line kernel; reflection boundaries; not an exposure simulator |
| 20 TT100K categories | If a list is absent, top 20 by training-only frequency, deterministic tie break; not recovered paper categories |
| Checkpoint selection | Best validation PSNR; classifier best validation accuracy; every 1000 restoration steps plus final step |

## Parameter counting and evaluation

With `configs/paper_reference.json`, the restorer contains 848,613 parameters including the motion encoder. Stage depths and channel widths can be adjusted in the configuration. `benchmark.py` measures the configured network's parameter count and per-ROI latency. The operation counter includes executed convolution/linear layers and attention QK/AV products. FFT input sizes are recorded separately; FFT, normalization, nonlinearities and elementwise operations are excluded from counted MACs, so this is not a total-operation estimate.

Pipeline ROI caching uses detector observations for tracking, and aligned sharp frames/annotations only for training targets and label assignment. A teacher detection or ground-truth trajectory must not silently replace noisy detector observations when claiming full-pipeline results. The cache drops unmatched training ROIs with no label; pipeline evaluation counts missed and misclassified signs.

Resume restores optimizer/scheduler/model, but not exact shuffle or RNG position. GPU nondeterminism may remain. Benchmarks exclude detector, classifier, tracking, disk I/O and encoding.

## Third-party scope

Neural blocks and tracking here are newly written. PyTorch/torchvision and SciPy supply primitives; optional Ultralytics supplies the detector. Their software/data/model licenses remain separate. Dataset download and private-video redistribution are not included. No manuscript PDF, correspondence email or private user file is included in the public package.

## Correctness maintenance

- Image-only and spatial-only architectures omit their inactive motion and spectral modules. Retrain ablations independently; legacy model weights discard only those known inactive parameter prefixes on load. Optimizer state from an older, unpruned ablation is not compatible with the pruned architecture; start a new stage with `--init-checkpoint`.
- Checkpoints store the best validation PSNR before saving both latest and best files. Older checkpoints can recover that value from their adjacent validation log; otherwise resume fails explicitly rather than overwriting a better checkpoint.
- Split validation compares sequence IDs, source IDs, scene groups, source hashes and image paths. The restoration loader rejects blurred-only sequence manifests with an explicit schema error.
- Initial velocity uncertainty scales with target size and frame interval. Both IoU and normalized-innovation gates remain required. The more permissive overlap gate supports large inter-frame shifts while still rejecting nonoverlapping boxes; it may increase ambiguous assignments in crowded scenes and must be calibrated on held-out annotated videos. The acceleration setting and gates are supplementary settings, not recovered manuscript values.
- The published five-frame dataset remains unchanged. It is too short for three-hit confirmation followed by five missing observations and reacquisition. Build a separate dataset with `--frames 9` or more for that protocol; do not relabel generated sequences as captured videos.
- Dataset generation checks source inventory identity before reusing files and verifies each private source hash after inventory. Use a new output folder after changing sources or generation settings.
- Video inference accepts `--timing-report runs/pipeline_timing.json --timing-warmup 10`. The report times detector inference, tracking, ROI preparation, restoration, classification and compositing, with device synchronization. It excludes video decoding/encoding and log writes. It requires actual trained checkpoints and is distinct from the restorer-only benchmark.

The default full model still has 848,613 parameters. No architectural evidence supplied with the manuscript determines a unique 9.1-million-parameter reconstruction. Changing widths merely to match that count would not establish equivalent experiments. The released sequences have no aligned sharp targets or traffic-sign annotations, so they support sequence loading and qualitative inspection, not the manuscript's quantitative restoration/recognition results.
