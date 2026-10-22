"""Causal region pipeline, with a detector-independent API and optional YOLO video CLI."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .data import crop_roi, image_tensor
from .tracking import MultiObjectTracker, TrackerConfig
from .utils import load_restorer
from .classifier import load_classifier


class PerceptionPipeline:
    def __init__(self, restorer, classifier=None, tracker_config=None, max_rois=8, fixed_reliability=False):
        self.restorer, self.classifier = restorer.eval(), classifier
        if classifier is not None:
            classifier.eval()
        self.device = next(restorer.parameters()).device
        self.tracker = MultiObjectTracker(tracker_config)
        self.max_rois, self.fixed_reliability = max_rois, fixed_reliability

    @torch.no_grad()
    def step(self, frame, detections, dt=1/30):
        frame = frame.convert('RGB')
        candidates = self.tracker.step(detections, frame.size, dt)[:self.max_rois]
        rendered = frame.copy()
        if not candidates:
            return rendered, []
        images, descriptors, reliabilities, crops = [], [], [], []
        for c in candidates:
            roi, bounds, padding = crop_roi(frame, c['box'])
            images.append(image_tensor(roi))
            descriptors.append(c['descriptor'])
            r = float(c['confirmed']) if self.fixed_reliability else c['reliability']
            reliabilities.append(r)
            c['effective_reliability'] = r
            crops.append((bounds, padding))
        batch = torch.stack(images).to(self.device)
        desc = torch.from_numpy(np.stack(descriptors)).to(self.device)
        reliability = torch.tensor(reliabilities, device=self.device)
        restored, gates = self.restorer(batch, desc, reliability, return_gate=True)
        restored = restored.clamp(0, 1)
        probabilities = self.classifier(restored).softmax(1).cpu() if self.classifier else None
        records = []
        for i, (c, (bounds, (left, top, nw, nh))) in enumerate(zip(candidates, crops)):
            pixels = (restored[i].permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
            roi = Image.fromarray(pixels).crop((left, top, left+nw, top+nh))
            x1, y1, x2, y2 = bounds
            rendered.paste(roi.resize((x2-x1, y2-y1), Image.Resampling.BICUBIC), (x1, y1))
            record = {k: v for k, v in c.items() if k != 'descriptor'}
            record.update(mean_gate=gates[i].mean().item(), crop_bounds=list(bounds))
            if probabilities is not None:
                record.update(label=int(probabilities[i].argmax()), class_confidence=float(probabilities[i].max()))
            records.append(record)
        return rendered, records


def main():
    p = argparse.ArgumentParser(description='YOLOv8n -> Kalman -> ROI restoration -> MobileNetV3-Small')
    p.add_argument('--input', required=True)
    p.add_argument('--detector', required=True, help='Traffic-sign fine-tuned YOLOv8n checkpoint (not generic COCO weights)')
    p.add_argument('--restorer', required=True)
    p.add_argument('--classifier', required=True)
    p.add_argument('--output', default='runs/video.mp4')
    p.add_argument('--tracker-config')
    p.add_argument('--fixed-reliability', action='store_true')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    import cv2
    from ultralytics import YOLO
    cfg = TrackerConfig(**json.loads(Path(a.tracker_config).read_text())) if a.tracker_config else None
    pipe = PerceptionPipeline(load_restorer(a.restorer, a.device)[0], load_classifier(a.classifier, a.device),
                              cfg, fixed_reliability=a.fixed_reliability)
    detector = YOLO(a.detector)
    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {a.input}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Video encoder unavailable')
    frame_index = 0
    try:
        with output.with_suffix('.jsonl').open('w', encoding='utf-8') as log:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                prediction = detector.predict(bgr, imgsz=640, conf=0.25, iou=0.5, agnostic_nms=True,
                                              device=a.device, verbose=False)[0]
                boxes = prediction.boxes
                detections = np.c_[boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()]
                frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                restored, records = pipe.step(frame, detections, dt=1/fps)
                writer.write(cv2.cvtColor(np.asarray(restored), cv2.COLOR_RGB2BGR))
                log.write(json.dumps(dict(frame_index=frame_index, candidates=records))+'\n')
                frame_index += 1
    finally:
        cap.release()
        writer.release()


if __name__ == '__main__':
    main()
