# Data preparation

## Paired ROI manifest

UTF-8 JSONL, one sample per line. Paths are relative to the manifest file. Sharp and blur source crops must be aligned and the same size; they are letterboxed to 128x128 at load time. Labels are integer indices of one fixed shared class mapping. Example (paths illustrate the schema; images are not bundled):

```json
{"blur":"train/0001_blur.png","sharp":"train/0001_sharp.png","label":0,"reliability":0}
```

For sequential samples add `sequence_id`, `frame_index`, `track_id`, a 16-number `descriptor`, and `reliability` in [0,1]. Descriptor layout:

1. cx/W, cy/H, w/W, h/H (Kalman posterior geometry, full-frame dimensions).
2. vx*dt/roi_w, vy*dt/roi_h, vw*dt/roi_w, vh*dt/roi_h.
3. Eight values log(1+diag(A P A^T)); A=diag(1/roi_w,1/roi_h,1/roi_w,1/roi_h,dt/roi_w,dt/roi_h,dt/roi_w,dt/roi_h).

Use the unexpanded selected ROI dimensions for normalization. Nonsequential TT100K images use r=0. Missing descriptors also default to zeros only when r=0. Descriptor consistency across coordinate transforms is the caller's responsibility; this release avoids geometric training augmentation.

## TT100K

Download the original dataset separately under its applicable terms. Supply its annotations and official train/test image-ID lists:

```bash
python -m kg_tsrdeblur.prepare tt100k --root data/tt100k --annotations data/tt100k/annotations.json --train-ids data/tt100k/train/ids.txt --test-ids data/tt100k/test/ids.txt --output data/tt100k_pairs --seed 42
```

Converter expects official-style `{"imgs":{"image_id":{"path":"train/image_id.jpg","objects":[{"category":"...","bbox":{"xmin":0,"ymin":0,"xmax":10,"ymax":10}}]}}}`. It holds out 10% of training **images**, then crops objects, preventing different crops from the same image crossing splits. A JSON array of exactly 20 category strings may be passed with `--classes`. Without it, top20 training-frequency categories are selected and recorded in `classes.json`. Source image IDs, blur lengths and angles are saved. These are explicit replacement choices; original manuscript classes and splits were not provided.

Blur length is sampled from 3..25 pixels and angle uniformly over [0,180). This is a reference degradation generator, not the original experimental protocol. The 20 categories shared with private videos cannot be established without the private annotation mapping.

## Causal video priors

Process each source sequence independently and chronologically, including empty detection frames. Input JSONL contains full-frame paths, noisy detector observations and training-only GT annotations:

```json
{"sequence_id":"seq001","frame_index":0,"blur":"blur/000000.png","sharp":"sharp/000000.png","detections":[[20,30,70,80,0.91]],"annotations":[{"box":[20,30,70,80],"label":0}]}
```

```bash
python -m kg_tsrdeblur.prepare sequence --frames data/seq001/frames.jsonl --fps 30 --output data/cached/seq001
```

`detections` format is x1,y1,x2,y2,confidence, in full-frame pixels. A fresh tracker is used for every invocation; no future observations enter prior construction. GT annotations assign labels to IoU>=0.5 candidates only; the tracker never consumes them. Sharp and blur full frames must be geometrically aligned. Naturally blurred frames without sharp references belong to recognition/pipeline evaluation, not paired PSNR/SSIM evaluation.

Split source sequences into train/validation/test **before** creating crops or synthetic blur. To combine sequence manifests, rewrite image paths relative to the combined manifest, or preserve sequence subdirectories. Keep all versions of a source clip in one partition. Store the class mapping and sequence split files with each research run. The manuscript's 84/12/24 sequence split cannot be instantiated without its source videos and IDs.

## Detector data

The optional `train_detector` wrapper expects an Ultralytics dataset YAML with full-frame images, labels in normalized YOLO xywh format and traffic-sign classes. Export this from your ground-truth annotations using the same source partitions. Do not train on ROI-only manifests or use generic COCO weights as a trained traffic-sign detector.

```bash
python -m kg_tsrdeblur.train_detector --data data/signs.yaml --epochs 100 --batch-size 16 --device 0
```

## Pipeline ground truth

`evaluate_pipeline` expects one annotated frame per line:

```json
{"frame_index":0,"annotations":[{"box":[20,30,70,80],"label":0}]}
```

Evaluate one sequence per invocation, then aggregate TP/prediction/GT counts across sequences before computing precision/recall. Predictions without a correct class-and-IoU match count as false positives; GT signs in frames with no output count as misses. Macro-F1 for paired ROI evaluation averages over all configured classes (undefined classes contribute zero). Include the metric conventions when comparing with other implementations.
