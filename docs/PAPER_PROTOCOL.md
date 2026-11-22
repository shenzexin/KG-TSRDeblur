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

- Manuscript driving-video experiments use 120 annotated captured sequences and 36,000 frames. Figure 2 uses 18-frame windows from that dataset.
- The public release consists of 324 five-frame synthetic sequences (1,620 blurred frames) based on 27 source photographs. It lacks sharp targets, class labels and tracking boxes.
- The manuscript also uses 12-frame TT100K-derived sequences created with continuous translation and scaling and exposure integration. The exposure midpoint supplies the sharp target and transformed annotation. The 12-frame converter is documented in [Data preparation](DATA.md#tt100k-derived-12-frame-sequences). The sequence-cache command accepts externally prepared paired sequences and detector observations. Reproducing the reported experiment requires the original synthesis settings, split and 20-class mapping.
- The public reference restorer has 848,613 parameters and about 1.784 G counted MACs at batch size one and 128 x 128 input. The manuscript reports 9.1 M and 9.4 G for its experiment. Its original architecture configuration and checkpoint are required to reproduce that run. The reference MAC counter excludes FFT, normalization and elementwise operations.
- The manuscript's NAFNet baseline is a separate model. The public image_only switch ablates motion conditioning within KG-TSRDeblur. Evaluate NAFNet with its own configuration and checkpoint.

## Figure 2(c) observation-gap scoring

Withhold detector updates after confirmation for one, three or five frames and run the three methods on shared Kalman predicted-box crops. Supply one JSONL row per withheld frame with sequence_id, frame_index, gap_length and annotations. Each annotation has an unexpanded xyxy box and an integer label. Predictions use sequence_id, frame_index and candidates with box and label. Box values must be the unexpanded coordinates, with candidates in the same order across methods. Give independent clips or repeated gap trials unique sequence IDs.

    python -m kg_tsrdeblur.evaluate_gaps --ground-truth data/withheld.jsonl --nafnet runs/nafnet.jsonl --fixed runs/fixed.jsonl --full runs/full.jsonl --output runs/gap_metrics.json

The evaluator maximizes valid one-to-one match count at IoU >= 0.5, then maximizes total IoU. Each ground-truth instance without a valid match counts as incorrect. It reports ground-truth, matched and correctly classified counts for each method and gap. Score exported predictions after the inference runs. Include an explicit row with an empty candidates list for frames with no valid prediction. The evaluator rejects missing log rows so an incomplete export cannot be interpreted as a missed detection.

## Reporting

For Tables 1-3, calculate each metric independently on both datasets for each of seeds 42, 43 and 44. Average the dataset scores with equal weight for each seed, then average the three seed-level scores. Table 1 uses the sample standard deviation across those three seed-level scores. Table 3 uses the 12-frame synthetic sequences for TT100K pipeline recall. Count model parameters and MACs for each evaluated configuration independently of the dataset aggregation. For Table 4, normalize exposure-time mean box-corner path length by the midpoint box's shorter side and freeze the validation-set 33rd and 67th percentile thresholds for testing. Compute accuracy on each dataset separately before averaging and keep the same ROIs for the full model and its zero-reliability control.

## Parameter-aligned configuration

`configs/parameter_matched_9p1m.json` uses a 496-channel Transformer bottleneck and keeps the reference model's base width of 32, two downsampling stages, four attention blocks, four heads, 8 x 8 windows, and 32-dimensional motion embedding. Its layer shapes give 9,134,501 parameters and 10,427,503,616 counted MACs for one 128 x 128 ROI under the repository's convolution, linear and attention counting convention. These are analytical counts for the new configuration. The 9.1 M rounded parameter count aligns with the manuscript; the approximately 10.43 G counted MACs differ from the manuscript's 9.4 G.

Run `python -m kg_tsrdeblur.benchmark --config configs/parameter_matched_9p1m.json` to profile the configuration. For training, pass the same config to the initialization stage and use the resulting checkpoint for the joint stage. Reference checkpoint tensor shapes are compatible with the default reference configuration. The wider configuration uses its own weights.
