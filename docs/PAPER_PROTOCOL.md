# Manuscript experiment correspondence

The public implementation follows the manuscript's causal restoration method. Its release assets and reference configuration describe the public package.

| Component | Implementation |
| --- | --- |
| Eight-dimensional constant-velocity Kalman state, association and reliability, Eqs. (1)-(6) | tracking.py |
| Reliability-bounded residual adapter before Transformer, Eq. (7) | model.py / MotionAdapter |
| Spatial and spectral refinement and residual RGB prediction, Eqs. (8)-(9) | model.py / SpatialFrequencyRefinement and KGTSRDeblur |
| Reconstruction initialization, frozen-classifier optimization and frequency supervision, Eqs. (10)-(11) | train.py / losses.py |
| Table 3 pipeline recall | evaluate_pipeline.py, class-correct one-to-one matching at IoU >= 0.5 |
| Figure 2(c) observation-gap scoring | evaluate_gaps.py, class-independent one-to-one matching on unexpanded boxes, followed by class scoring |

## Data and checkpoints

- The manuscript driving-video dataset contains 120 captured sequences and 36,000 annotated frames: 84 sequences and 25,200 frames for training, 12 and 3,600 for validation, and 24 and 7,200 for testing. Figure 2 uses 18-frame windows from the captured sequences. See [dataset details](DATA.md#manuscript-driving-video-dataset).
- The public release consists of 324 five-frame synthetic sequences (1,620 blurred frames) based on 27 source photographs. It lacks sharp targets, class labels and tracking boxes.
- The manuscript also uses 12-frame TT100K-derived sequences created with continuous translation and scaling and exposure integration. The exposure midpoint supplies the sharp target and transformed annotation. The 12-frame converter is documented in [Data preparation](DATA.md#tt100k-derived-12-frame-sequences). The sequence-cache command accepts externally prepared paired sequences and detector observations. Reproducing the reported experiment requires the original synthesis settings, split and 20-class mapping.
- The compact reference config has 848,613 parameters and about 1.784 G counted MACs per 128 x 128 ROI. The default configuration and Table 2 reporting precision are described below. Experiment metrics are associated with the architecture, checkpoint, and data used for each run.
- The manuscript's NAFNet baseline is a separate model. The public image_only switch ablates motion conditioning within KG-TSRDeblur. Evaluate NAFNet with its own configuration and checkpoint.

## Figure 2(c) observation-gap scoring

Withhold detector updates after confirmation for one, three or five frames and run the three methods on shared Kalman predicted-box crops. Supply one JSONL row per withheld frame with sequence_id, frame_index, gap_length and annotations. Each annotation has an unexpanded xyxy box and an integer label. Predictions use sequence_id, frame_index and candidates with box and label. Box values must be the unexpanded coordinates, with candidates in the same order across methods. Give independent clips or repeated gap trials unique sequence IDs.

Run `pipeline.py` through `python -m kg_tsrdeblur.pipeline` for each trial with `--sequence-id TRIAL --withheld-frames data/withheld.jsonl`. Frame indices are zero-based, and each trial defines one contiguous gap. Select windows with confirmed tracks before the gap. The pipeline retains these frames in the stream and supplies empty detections to the tracker. It exports `box` (unexpanded), `crop_bounds` (expanded and clipped), and the predicted class per candidate. Use the same video, tracker configuration, detector and gap schedule for full and fixed runs, adding `--fixed-reliability` to the fixed run. Concatenate per-trial JSONL outputs for each method. For NAFNet, classify its outputs on the same video crops using `crop_bounds`, and retain the original `box`, trial ID, frame index and candidate order when exporting its labels.

    python -m kg_tsrdeblur.evaluate_gaps --ground-truth data/withheld.jsonl --nafnet runs/nafnet.jsonl --fixed runs/fixed.jsonl --full runs/full.jsonl --output runs/gap_metrics.json

The evaluator maximizes valid one-to-one match count at IoU >= 0.5, then maximizes total IoU. Each ground-truth instance without a valid match counts as incorrect. It reports ground-truth, matched and correctly classified counts for each method and gap. Score exported predictions after the inference runs. Include an explicit row with an empty candidates list for frames with no valid prediction. The evaluator rejects missing log rows so an incomplete export cannot be interpreted as a missed detection.

## Reporting

For Tables 1-3, calculate each metric independently on both datasets for each of seeds 42, 43 and 44. Average the dataset scores with equal weight for each seed, then average the three seed-level scores. Table 1 uses the sample standard deviation across those three seed-level scores. Table 3 uses the 12-frame synthetic sequences for TT100K pipeline recall. Count model parameters and MACs for each evaluated configuration independently of the dataset aggregation. For Table 4, normalize exposure-time mean box-corner path length by the midpoint box's shorter side and freeze the validation-set 33rd and 67th percentile thresholds for testing. Compute accuracy on each dataset separately before averaging and keep the same ROIs for the full model and its zero-reliability control.

## Default model configuration

`configs/parameter_matched_9p1m.json` specifies the default training and benchmark architecture. It uses a 468-channel Transformer bottleneck and a two-layer motion encoder with an 18,994-unit hidden layer and a 32-dimensional output. The motion encoder runs once per ROI before spatial broadcast. Base width 32, two downsampling stages, four attention blocks, four heads and 8 x 8 windows are retained.

The layer shapes give 9,099,991 parameters and 9,429,537,760 counted MACs per 128 x 128 RGB ROI. At the precision of Table 2 these are 9.1 M and 9.4 G. The analytical MAC count follows the benchmark's convolution, linear and QK/AV attention convention. The configuration specifies architectural complexity; recognition and restoration scores depend on the trained checkpoint and evaluation data.

Use `python -m kg_tsrdeblur.benchmark --config configs/parameter_matched_9p1m.json` for device-level profiling. `configs/paper_reference.json` selects the original compact architecture. Loading a checkpoint reconstructs the architecture from its saved configuration; an omitted bottleneck or motion hidden width retains its original compact default for that checkpoint.

## Experiment records

For Table 2, record each model configuration and checkpoint identifier, input size, tested device and precision, MAC counting scope, and the metrics and latency logs for each dataset and seed. For Table 4, record the source validation and test splits, the two numerical thresholds, and per-dataset bin counts and predictions for each method. Store the full and zero-reliability runs under the same checkpoint identifier. These records define the inputs to the reported averages.
