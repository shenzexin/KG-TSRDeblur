"""Generate aligned 12-frame TT100K sequences from explicit motion settings."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def generate(root, annotation_file, splits, classes, output, settings):
    vx, vy, zoom = (settings[key] for key in ('vx', 'vy', 'log_zoom_per_frame'))
    exposure, samples = settings['exposure_frames'], settings['exposure_samples']
    if not all(math.isfinite(value) for value in (vx, vy, zoom, exposure)):
        raise ValueError('Motion and exposure settings must be finite')
    if not 0 < exposure <= 1 or not isinstance(samples, int) or samples < 2:
        raise ValueError('Use an exposure in (0, 1] frames and at least two samples')
    if len(classes) != 20 or len(set(classes)) != 20:
        raise ValueError('Provide exactly 20 distinct TT100K category codes')
    ids = [key for split in ('train', 'val', 'test') for key in splits[split]]
    if len(set(ids)) != len(ids):
        raise ValueError('Each source image must belong to exactly one split')
    annotations = json.loads(annotation_file.read_text(encoding='utf-8'))['imgs']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'classes.json').write_text(json.dumps(classes, indent=2), encoding='utf-8')
    (output / 'generation.json').write_text(json.dumps(
        dict(settings=settings, frames=12, splits=splits, boundary='black',
             exposure_domain='linear RGB'), indent=2), encoding='utf-8')
    offsets = exposure * ((np.arange(samples) + 0.5) / samples - 0.5)
    lookup = np.arange(256, dtype=np.float32) / 255
    lookup = np.where(lookup <= .04045, lookup / 12.92, ((lookup + .055) / 1.055) ** 2.4)
    labels = {category: i for i, category in enumerate(classes)}

    for split in ('train', 'val', 'test'):
        for index, source in enumerate(splits[split]):
            info = annotations[source]
            folder = output / split / str(index)
            folder.mkdir(parents=True)
            with Image.open(root / info['path']) as image:
                rgb = image.convert('RGB')
            width, height = rgb.size
            center = np.asarray([width / 2, height / 2])
            objects = []
            for track, obj in enumerate(info.get('objects', [])):
                if obj['category'] in labels:
                    b = obj['bbox']
                    objects.append((track, labels[obj['category']],
                                    np.array([[b['xmin'], b['ymin']], [b['xmax'], b['ymin']],
                                              [b['xmin'], b['ymax']], [b['xmax'], b['ymax']]])))

            def geometry(t):
                return math.exp(zoom * t), center + np.asarray([vx * t, vy * t])

            def warp(t):
                scale, anchor = geometry(t)
                dx, dy = center - anchor / scale
                return rgb.transform(rgb.size, Image.Transform.AFFINE,
                                     (1 / scale, 0, dx, 0, 1 / scale, dy),
                                     resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))

            def corners_at(corners, t):
                scale, anchor = geometry(t)
                return scale * (corners - center) + anchor

            rows = []
            for frame in range(12):
                times = frame + offsets
                linear = np.zeros((height, width, 3), dtype=np.float32)
                for t in times:
                    linear += lookup[np.asarray(warp(t))]
                linear /= samples
                srgb = np.where(linear <= .0031308, 12.92 * linear,
                                1.055 * linear ** (1 / 2.4) - .055)
                blur_path, sharp_path = f'{frame:02}_blur.png', f'{frame:02}_sharp.png'
                Image.fromarray(np.uint8(np.rint(np.clip(srgb, 0, 1) * 255))).save(folder / blur_path)
                warp(frame).save(folder / sharp_path)
                targets = []
                for track, label, corners in objects:
                    midpoint = corners_at(corners, frame)
                    lower, upper = midpoint.min(0), midpoint.max(0)
                    short_side = float(min(upper - lower))
                    # Fully visible blur trajectories keep the geometry and target aligned.
                    trajectory = np.stack([corners_at(corners, t) for t in
                                           np.r_[frame - exposure / 2, times, frame + exposure / 2]])
                    if (short_side <= 0 or (trajectory < 0).any()
                            or (trajectory > [width, height]).any()):
                        continue
                    path = np.linalg.norm(np.diff(trajectory, axis=0), axis=2).sum(0).mean()
                    targets.append(dict(box=[*lower.tolist(), *upper.tolist()], label=label,
                                        track_id=track, severity=float(path / short_side)))
                rows.append(dict(sequence_id=f'tt100k:{source}', source_image=source,
                                 frame_index=frame, blur=blur_path, sharp=sharp_path,
                                 annotations=targets))
            (folder / 'frames.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('root', 'annotations', 'splits', 'classes', 'output', 'settings'):
        parser.add_argument('--' + field, required=True, type=Path)
    args = parser.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    generate(args.root, args.annotations, read(args.splits), read(args.classes),
             args.output, read(args.settings))


if __name__ == '__main__':
    main()
