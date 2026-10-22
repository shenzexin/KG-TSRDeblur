"""Build blurred-only synthetic motion sequences from privately held still images."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps


VERSION = '1.0.0'
FRAME_COUNT = 5
FPS = 30


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resize_rgb(path, side):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        scale = min(1., side / max(im.size))
        target = (max(1, round(im.width*scale)), max(1, round(im.height*scale)))
        if im.size != target:
            im = im.resize(target, Image.Resampling.LANCZOS)
        return np.asarray(im).copy()


def perceptual_hash(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coeff = cv2.dct(gray)[:8, :8].reshape(-1)[1:]
    return sum(int(v > np.median(coeff)) << i for i, v in enumerate(coeff))


def stable_seed(text, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{text}'.encode()).digest()[:8], 'little')


def inventory(source, side=1024, seed=42, priority_only=False):
    candidates = [p for p in Path(source).rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}]
    if priority_only:
        candidates = [p for p in candidates if p.stem.isdigit() and 1 <= int(p.stem) <= 28]
    candidates.sort(key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**15, p.name))
    unique, by_hash, aliases = [], {}, []
    for path in candidates:
        digest = sha256_file(path)
        if digest in by_hash:
            aliases.append(dict(source_name=path.name, canonical_source_id=by_hash[digest]['source_id'],
                                reason='byte_identical_source'))
            continue
        rgb = resize_rgb(path, side)
        priority = path.stem.isdigit() and 1 <= int(path.stem) <= 28
        gray = cv2.cvtColor(cv2.resize(rgb, (256, 256)), cv2.COLOR_RGB2GRAY)
        row = dict(source_id=f'source_{len(unique)+1:06d}', source_name=path.name, source_sha256=digest,
                   width=rgb.shape[1], height=rgb.shape[0], priority=priority,
                   phash=perceptual_hash(rgb), laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                   private_path=str(path))
        unique.append(row)
        by_hash[digest] = row
    # Group duplicate/near-duplicate content BEFORE assigning any splits.
    parents = list(range(len(unique)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    # Conservative visual-scene grouping of the prioritized source photographs.
    related_priority_stems = [{1, 2, 3, 4, 6}, {5, 7, 8}, {9}, {10, 11}, {12}, {13},
                              {14, 15}, {16, 17}, {18, 19, 20, 21, 22}, {23, 24, 25, 26}, {27, 28}]
    for related in related_priority_stems:
        members = [i for i, r in enumerate(unique) if Path(r['source_name']).stem.isdigit()
                   and int(Path(r['source_name']).stem) in related]
        for i in members[1:]:
            union(members[0], i)
    for i, a in enumerate(unique):
        for j in range(i):
            b = unique[j]
            if (a['phash'] ^ b['phash']).bit_count() <= 8:
                union(i, j)
    groups = {}
    for i, row in enumerate(unique):
        groups.setdefault(find(i), []).append(row)
    ordered_groups = sorted(groups.values(), key=lambda group: stable_seed(group[0]['source_id'], seed))
    ntest = max(1, round(len(ordered_groups)*.2)) if len(ordered_groups) >= 3 else 0
    nval = max(1, round(len(ordered_groups)*.1)) if len(ordered_groups) >= 3 else 0
    for index, group in enumerate(ordered_groups):
        group_id = 'group_'+group[0]['source_id'].removeprefix('source_')
        split = 'test' if index < ntest else ('val' if index < ntest+nval else 'train')
        for row in group:
            row.update(group_id=group_id, split=split)
    return unique, aliases, len(candidates)


def parameters(source_row, variant, seed):
    sequence_id = f"{source_row['source_id']}_v{variant:02d}"
    key = stable_seed(sequence_id, seed)
    rng = np.random.default_rng(key)
    severity = variant % 3 if source_row['priority'] else int(rng.integers(0, 3))
    speeds = [(7., 11.), (14., 21.), (26., 36.)]
    exposure_ranges = [(.65, .85), (.8, 1.05), (.95, 1.25)]
    speed = float(rng.uniform(*speeds[severity]))
    angle = float(rng.uniform(0, 2*np.pi))
    velocity = [speed*np.cos(angle), speed*np.sin(angle)]
    acceleration = rng.uniform(-.35, .35, 2).tolist()
    return dict(sequence_id=sequence_id, seed=key, severity=['mild', 'moderate', 'severe'][severity],
                velocity_px_per_frame=velocity, acceleration_px_per_frame2=acceleration,
                exposure_frames=float(rng.uniform(*exposure_ranges[severity])), exposure_samples=129)


def position(t, params):
    v = np.asarray(params['velocity_px_per_frame'])
    a = np.asarray(params['acceleration_px_per_frame2'])
    return v*t + .5*a*t*t


def exposure_kernel(t, params):
    """Subpixel exposure trajectory, rasterized with bilinear mass assignment."""
    length = params['exposure_frames']
    offsets_t = (np.arange(params['exposure_samples'])+.5)/params['exposure_samples']*length-length/2
    offsets = np.stack([position(t+tau, params)-position(t, params) for tau in offsets_t])
    # OpenCV filter2D computes correlation, so negate scene translation offsets.
    locations = -offsets
    radius = int(np.ceil(np.abs(locations).max()))+2
    kernel = np.zeros((2*radius+1, 2*radius+1), np.float32)
    for x, y in locations:
        xx, yy = x+radius, y+radius
        ix, iy = int(np.floor(xx)), int(np.floor(yy))
        fx, fy = xx-ix, yy-iy
        kernel[iy, ix] += (1-fx)*(1-fy)
        kernel[iy, ix+1] += fx*(1-fy)
        kernel[iy+1, ix] += (1-fx)*fy
        kernel[iy+1, ix+1] += fx*fy
    kernel /= kernel.sum()
    return kernel, offsets


def linearize(rgb):
    x = rgb.astype(np.float32)/255.
    return np.where(x <= .04045, x/12.92, ((x+.055)/1.055)**2.4)


def encode_srgb(linear):
    x = np.clip(linear, 0, 1)
    x = np.where(x <= .0031308, 12.92*x, 1.055*np.maximum(x, .0031308)**(1/2.4)-.055)
    return np.rint(x*255).clip(0, 255).astype(np.uint8)


def render_frame(linear, t, params):
    h, w = linear.shape[:2]
    displacement = position(t, params)
    kernel, offsets = exposure_kernel(t, params)
    margin = int(np.ceil(np.abs(displacement).max()+np.abs(offsets).max()))+8
    padded = cv2.copyMakeBorder(linear, margin, margin, margin, margin, cv2.BORDER_REFLECT_101)
    integrated = cv2.filter2D(padded, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    shifted = cv2.warpAffine(integrated, np.array([[1., 0., displacement[0]], [0., 1., displacement[1]]]),
                             (w+2*margin, h+2*margin), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
    return encode_srgb(shifted[margin:margin+h, margin:margin+w]), displacement, offsets


def generate_source(task):
    row, args = task
    cv2.setNumThreads(1)
    rgb = resize_rgb(row['private_path'], args['side'])
    linear = linearize(rgb)
    sequences, frames = [], []
    variants = args['priority_variants'] if row['priority'] else 1
    for variant in range(variants):
        params = parameters(row, variant, args['seed'])
        folder = Path(args['output'])/'sequences'/row['split']/params['sequence_id']
        folder.mkdir(parents=True, exist_ok=True)
        sequence_frames = []
        for index in range(FRAME_COUNT):
            path = folder/f'frame_{index:03d}.jpg'
            t = index-2
            if not path.exists():
                pixels, displacement, offsets = render_frame(linear, t, params)
                ok, compressed = cv2.imencode('.jpg', cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR),
                                              [cv2.IMWRITE_JPEG_QUALITY, args['jpeg_quality']])
                if not ok:
                    raise RuntimeError(f'Cannot encode {path.name}')
                temp = path.with_suffix('.partial')
                temp.write_bytes(compressed.tobytes())
                temp.replace(path)
            else:
                displacement = position(t, params)
            frame = dict(sequence_id=params['sequence_id'], source_id=row['source_id'], group_id=row['group_id'],
                split=row['split'], frame_index=index, timestamp_seconds=index/FPS, relative_time_frames=t,
                blur_path=path.relative_to(args['output']).as_posix(), width=row['width'], height=row['height'],
                displacement_px=[float(v) for v in displacement], synthetic=True, sha256=sha256_file(path), bytes=path.stat().st_size)
            sequence_frames.append(frame['blur_path'])
            frames.append(frame)
        sequences.append(dict(**params, source_id=row['source_id'], group_id=row['group_id'], split=row['split'],
            fps=FPS, frame_count=FRAME_COUNT, center_frame_index=2, synthetic=True,
            motion_model='global_image_plane_translation_with_constant_acceleration',
            frames=sequence_frames, center_triplet=sequence_frames[1:4]))
    return sequences, frames


def write_jsonl(path, records):
    Path(path).write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in records), encoding='utf-8')


def build(args):
    started = time.perf_counter()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_config = dict(side=args.side, priority_variants=args.priority_variants, seed=args.seed,
                      jpeg_quality=args.jpeg_quality, frame_count=FRAME_COUNT, version=VERSION,
                      priority_only=args.priority_only)
    config_path = out/'generation_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError('Existing output uses a different configuration; choose another directory')
    config_path.write_text(json.dumps(run_config, indent=2), encoding='utf-8')
    rows, aliases, raw_count = inventory(args.source, args.side, args.seed, args.priority_only)
    if args.limit:
        rows = rows[:args.limit]
    public_sources = [{k: v for k, v in r.items() if k not in {'private_path', 'phash'}} for r in rows]
    write_jsonl(out/'sources.jsonl', public_sources)
    (out/'source_aliases.json').write_text(json.dumps(aliases, indent=2), encoding='utf-8')
    print(json.dumps(dict(event='inventory_complete', raw_images=raw_count, unique_selected=len(rows),
          priority_sources=sum(r['priority'] for r in rows), source_groups=len({r['group_id'] for r in rows}))), flush=True)
    settings = dict(output=str(out), side=args.side, priority_variants=args.priority_variants,
                    seed=args.seed, jpeg_quality=args.jpeg_quality)
    sequences, frames = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(generate_source, (r, settings)) for r in rows]
        for i, future in enumerate(as_completed(pending), 1):
            seq, frm = future.result()
            sequences.extend(seq)
            frames.extend(frm)
            if i % 25 == 0 or i == len(rows):
                print(json.dumps(dict(event='progress', completed_sources=i, total_sources=len(rows),
                      sequences=len(sequences), frames=len(frames), elapsed_seconds=round(time.perf_counter()-started, 1))), flush=True)
    sequences.sort(key=lambda r: r['sequence_id'])
    frames.sort(key=lambda r: (r['sequence_id'], r['frame_index']))
    write_jsonl(out/'sequences.jsonl', sequences)
    write_jsonl(out/'frames.jsonl', frames)
    for split in ['train', 'val', 'test']:
        write_jsonl(out/f'{split}.jsonl', [r for r in sequences if r['split'] == split])
    stats = dict(name='KGTSR Synthetic Motion Sequences', version=VERSION, synthetic=True,
        source_origin='Author-provided still photographs; capture origin reported by the contributor',
        source_images_received=raw_count, unique_source_images_used=len(rows), source_images_released=0,
        sharp_frames_released=0, priority_source_count=sum(r['priority'] for r in rows),
        priority_variants=args.priority_variants, total_sequences=len(sequences), total_blurred_frames=len(frames),
        frames_per_sequence=FRAME_COUNT, fps=FPS, maximum_long_side=args.side, jpeg_quality=args.jpeg_quality,
        source_groups=len({r['group_id'] for r in rows}),
        sequences_by_split=dict(Counter(r['split'] for r in sequences)),
        frames_by_split=dict(Counter(r['split'] for r in frames)),
        sequences_by_severity=dict(Counter(r['severity'] for r in sequences)),
        total_image_bytes=sum(r['bytes'] for r in frames), duration_seconds=round(time.perf_counter()-started, 1),
        task_annotations='No traffic-sign boxes, identities, class labels, detector outputs, or clear targets are included',
        border_policy='reflection padding', color_policy='linear-light exposure integration; sRGB JPEG encoding; metadata stripped',
        source_blur_policy='Existing source blur is retained; no source is asserted to be a sharp ground truth',
        split_policy='conservative visual-scene and perceptual-content groups, seeded group allocation approximately 70/10/20',
        subset='priority_source_stems_1_to_28' if args.priority_only else 'all_source_images')
    (out/'dataset.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(dict(event='complete', **stats)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Private source folder; source files are never copied to output')
    p.add_argument('--output', required=True)
    p.add_argument('--side', type=int, default=1024)
    p.add_argument('--priority-variants', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--jpeg-quality', type=int, default=94)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Limit unique sources for a pilot build')
    p.add_argument('--priority-only', action='store_true', help='Use source filename stems 1..28 only')
    a = p.parse_args()
    if a.side < 128 or a.priority_variants < 1 or not 1 <= a.jpeg_quality <= 100 or a.workers < 1:
        p.error('Invalid dimensions, variants, JPEG quality or worker count')
    build(a)


if __name__ == '__main__':
    main()
