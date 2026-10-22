import argparse
from pathlib import Path
import torch
from PIL import Image
from .data import image_tensor, letterbox
from .utils import load_restorer


def main():
    p = argparse.ArgumentParser(description='Restore a cropped traffic-sign image with zero motion reliability')
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    model, _ = load_restorer(a.checkpoint, a.device)
    model.eval()
    with Image.open(a.input) as source:
        original_size = source.size
        roi, (left, top, nw, nh) = letterbox(source)
    with torch.no_grad():
        result = model(image_tensor(roi)[None].to(a.device)).clamp(0, 1)[0]
    pixels = (result.permute(1, 2, 0).cpu().numpy()*255).round().astype('uint8')
    restored = Image.fromarray(pixels).crop((left, top, left+nw, top+nh))
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    restored.resize(original_size, Image.Resampling.BICUBIC).save(output)


if __name__ == '__main__':
    main()
