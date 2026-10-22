"""Manifest loader. All motion values must be produced causally in original frame coordinates."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def image_tensor(image):
    return torch.from_numpy(np.asarray(image.convert('RGB'), dtype=np.float32).copy()).permute(2, 0, 1)/255


def letterbox(image, size=128):
    image = image.convert('RGB')
    w, h = image.size
    scale = size/max(w, h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
    result = Image.new('RGB', (size, size), (0, 0, 0))
    left, top = (size-nw)//2, (size-nh)//2
    result.paste(resized, (left, top))
    return result, (left, top, nw, nh)


def crop_roi(frame, box, expansion=0.1, size=128):
    """Expansion=0.1 increases total width/height by 10%, i.e. 5% on each edge."""
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*expansion/2, (y2-y1)*expansion/2
    w, h = frame.size
    bounds = (max(0, int(np.floor(x1-dx))), max(0, int(np.floor(y1-dy))),
              min(w, int(np.ceil(x2+dx))), min(h, int(np.ceil(y2+dy))))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Empty ROI")
    roi, padding = letterbox(frame.crop(bounds), size)
    return roi, bounds, padding


class ROIDataset(Dataset):
    def __init__(self, manifest, size=128, sharp_only=False):
        path = Path(manifest)
        self.root, self.size, self.sharp_only = path.resolve().parent, size, sharp_only
        self.records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.records:
            raise ValueError("Manifest is empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        row = self.records[i]
        def load(key):
            with Image.open(self.root / row[key]) as im:
                return im.convert('RGB')
        sharp = load('sharp')
        blur = sharp.copy() if self.sharp_only else load('blur')
        if sharp.size != blur.size:
            raise ValueError("Blur/sharp inputs must be geometrically aligned with equal sizes")
        sharp = image_tensor(letterbox(sharp, self.size)[0])
        blur = image_tensor(letterbox(blur, self.size)[0])
        descriptor = np.asarray(row.get('descriptor', [0]*16), dtype=np.float32)
        r = float(row.get('reliability', 0))
        if descriptor.shape != (16,) or not np.isfinite(descriptor).all() or not np.isfinite(r) or not 0 <= r <= 1:
            raise ValueError("Invalid causal prior")
        if r and ('descriptor' not in row or not row.get('sequence_id')):
            raise ValueError("Nonzero motion requires a sequence_id and an explicit descriptor")
        return dict(blur=blur, sharp=sharp, descriptor=torch.from_numpy(descriptor),
                    reliability=torch.tensor(r, dtype=torch.float32), label=int(row.get('label', -1)))
