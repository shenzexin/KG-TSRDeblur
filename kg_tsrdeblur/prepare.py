"""Portable data utilities: toy fixtures, TT100K conversion, causal video ROI caching."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve
from .data import crop_roi
from .tracking import MultiObjectTracker


def motion_blur(image, length, angle):
    """Normalized line kernel with reflected boundaries; reference synthetic protocol."""
    length = int(length)
    if length < 1:
        raise ValueError('Blur length must be positive')
    radius = int(np.ceil(length/2))
    size = 2*radius+1
    kernel = np.zeros((size, size), dtype=np.float32)
    theta = np.deg2rad(angle)
    t = np.linspace(-(length-1)/2, (length-1)/2, max(2, length*8))
    xx, yy = np.rint(radius+t*np.cos(theta)).astype(int), np.rint(radius+t*np.sin(theta)).astype(int)
    np.add.at(kernel, (yy, xx), 1)
    kernel /= kernel.sum()
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    blurred = np.stack([convolve(pixels[:, :, c], kernel, mode='reflect') for c in range(3)], axis=-1)
    return Image.fromarray(blurred.round().clip(0, 255).astype('uint8'))


def write_manifest(path, rows):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


def toy(output, seed=42):
    """Generated shapes are for software validation only, never research evaluation."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, count in [('train', 8), ('val', 4), ('test', 4)]:
        folder = out/split
        folder.mkdir(exist_ok=True)
        rows = []
        for i in range(count):
            pixels = rng.integers(10, 80, (128, 128, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            draw = ImageDraw.Draw(image)
            label = i % 2
            if label == 0:
                draw.ellipse((22, 22, 106, 106), fill='white', outline='red', width=8)
            else:
                draw.polygon([(64, 18), (110, 106), (18, 106)], fill='yellow', outline='red', width=8)
            draw.text((53, 55), str(i*10), fill='black')
            sharp, blur = f'{split}/{i}_sharp.png', f'{split}/{i}_blur.png'
            image.save(out/sharp)
            motion_blur(image, int(rng.integers(3, 14)), float(rng.uniform(0, 180))).save(out/blur)
            rows.append(dict(sharp=sharp, blur=blur, label=label, reliability=0))
        write_manifest(out/f'{split}.jsonl', rows)
    (out/'classes.json').write_text(json.dumps(['toy_circle', 'toy_triangle']), encoding='utf-8')
    print(f'Toy software-test fixtures written to {out}; NOT TT100K or paper results.')


def tt100k(a):
    root, out = Path(a.root), Path(a.output)
    annotations = json.loads(Path(a.annotations).read_text(encoding='utf-8'))['imgs']
    def ids(path):
        return [s.strip().removesuffix('.jpg').removesuffix('.png') for s in Path(path).read_text().splitlines() if s.strip()]
    training, testing = ids(a.train_ids), ids(a.test_ids)
    if set(training) & set(testing):
        raise ValueError('Official train/test ID lists overlap')
    counts = Counter(o['category'] for key in training for o in annotations[key].get('objects', []))
    if a.classes:
        classes = json.loads(Path(a.classes).read_text(encoding='utf-8'))
    else:
        classes = [name for name, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:20]]
    if len(classes) != 20 or len(set(classes)) != 20:
        raise ValueError('Provide exactly 20 distinct categories (paper list was not supplied)')
    rng = np.random.default_rng(a.seed)
    shuffled = rng.permutation(training).tolist()
    nval = max(1, round(len(shuffled)*0.1))
    split = dict(train=shuffled[nval:], val=shuffled[:nval], test=testing)
    out.mkdir(parents=True, exist_ok=True)
    (out/'classes.json').write_text(json.dumps(classes, indent=2), encoding='utf-8')
    (out/'source_splits.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    for split_name, keys in split.items():
        (out/split_name).mkdir(exist_ok=True)
        rows = []
        for key in keys:
            info = annotations[key]
            with Image.open(root/info['path']) as im:
                frame = im.convert('RGB')
            for j, obj in enumerate(info.get('objects', [])):
                category = obj['category']
                if category not in classes:
                    continue
                bbox = obj['bbox']
                box = [bbox[k] for k in ['xmin', 'ymin', 'xmax', 'ymax']]
                roi, _, _ = crop_roi(frame, box)
                sharp = f'{split_name}/{key}_{j}_sharp.png'
                blur = f'{split_name}/{key}_{j}_blur.png'
                roi.save(out/sharp)
                length, angle = int(rng.integers(3, 26)), float(rng.uniform(0, 180))
                motion_blur(roi, length, angle).save(out/blur)
                rows.append(dict(sharp=sharp, blur=blur, label=classes.index(category), reliability=0,
                                 source_image=key, blur_length=length, blur_angle=angle))
        write_manifest(out/f'{split_name}.jsonl', rows)
    print('Saved reference TT100K split and blur metadata; category/blur choices are not the original experiment.')


def cache_sequence(a):
    """Input one ordered sequence of full-frame detector outputs and paired sharp images."""
    source, out = Path(a.frames), Path(a.output)
    rows = [json.loads(x) for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    if len({r['sequence_id'] for r in rows}) != 1:
        raise ValueError('One sequence per cache invocation; tracker must reset between sequences')
    indices = [r['frame_index'] for r in rows]
    if any(j <= i for i, j in zip(indices, indices[1:])):
        raise ValueError('Frames must be strictly ordered')
    out.mkdir(parents=True, exist_ok=True)
    tracker = MultiObjectTracker()
    result = []
    for i, row in enumerate(rows):
        if i and row['frame_index'] != rows[i-1]['frame_index']+1:
            raise ValueError('Include every frame (empty detections allowed); no index jumps')
        with Image.open(source.parent/row['blur']) as im:
            blur_frame = im.convert('RGB')
        with Image.open(source.parent/row['sharp']) as im:
            sharp_frame = im.convert('RGB')
        if blur_frame.size != sharp_frame.size:
            raise ValueError('Sharp and blur frames must be geometrically aligned')
        candidates = tracker.step(row.get('detections', []), blur_frame.size, 1/a.fps)[:8]
        # GT is consulted only for labels, never for prior estimation or candidate boxes.
        from .tracking import box_iou
        for c in candidates:
            annotations = row.get('annotations', [])
            overlaps = [box_iou(np.array(c['box']), np.array(g['box'])) for g in annotations]
            if not overlaps or max(overlaps) < 0.5:
                continue
            label = annotations[int(np.argmax(overlaps))]['label']
            stem = f"{row['frame_index']:06d}_{c['track_id']}"
            blur_roi, _, _ = crop_roi(blur_frame, c['box'])
            sharp_roi, _, _ = crop_roi(sharp_frame, c['box'])
            blur_roi.save(out/f'{stem}_blur.png')
            sharp_roi.save(out/f'{stem}_sharp.png')
            result.append(dict(blur=f'{stem}_blur.png', sharp=f'{stem}_sharp.png', label=label,
                sequence_id=row['sequence_id'], frame_index=row['frame_index'], track_id=c['track_id'],
                descriptor=c['descriptor'].tolist(), reliability=c['reliability']))
    write_manifest(out/'manifest.jsonl', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('toy')
    t.add_argument('--output', default='data/toy')
    t.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('tt100k')
    for key in ['root', 'annotations', 'train-ids', 'test-ids', 'output']:
        t.add_argument('--'+key, required=True)
    t.add_argument('--classes', help='JSON list of 20 category strings; otherwise select top20 on training only')
    t.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('sequence')
    t.add_argument('--frames', required=True)
    t.add_argument('--output', required=True)
    t.add_argument('--fps', type=float, default=30)
    a = p.parse_args()
    if a.command == 'toy':
        toy(a.output, a.seed)
    elif a.command == 'tt100k':
        tt100k(a)
    elif a.fps <= 0:
        p.error('fps must be positive')
    else:
        cache_sequence(a)


if __name__ == '__main__':
    main()
