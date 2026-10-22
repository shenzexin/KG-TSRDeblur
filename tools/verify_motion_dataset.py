"""Validate manifests, image integrity, split isolation and exclusion of source files."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    read = lambda name: [json.loads(x) for x in (root/name).read_text(encoding='utf-8').splitlines() if x.strip()]
    summary = json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    sequences, frames, sources = read('sequences.jsonl'), read('frames.jsonl'), read('sources.jsonl')
    assert summary['synthetic'] is True
    assert summary['source_images_released'] == summary['sharp_frames_released'] == 0
    assert len(sequences) == summary['total_sequences']
    assert len(frames) == summary['total_blurred_frames']
    assert len(sources) == summary['unique_source_images_used']
    source_hashes = {s['source_sha256'] for s in sources}
    ids = {s['sequence_id'] for s in sequences}
    assert len(ids) == len(sequences)
    source_splits, group_splits = defaultdict(set), defaultdict(set)
    frame_by_path = {f['blur_path']: f for f in frames}
    assert len(frame_by_path) == len(frames)
    for s in sequences:
        assert s['synthetic'] is True and len(s['frames']) == 5
        assert s['center_frame_index'] == 2 and s['center_triplet'] == s['frames'][1:4]
        source_splits[s['source_id']].add(s['split'])
        group_splits[s['group_id']].add(s['split'])
        sequence_hashes = []
        for i, p in enumerate(s['frames']):
            f = frame_by_path[p]
            assert f['sequence_id'] == s['sequence_id'] and f['frame_index'] == i
            assert f['split'] == s['split'] and f['synthetic'] is True
            path = (root/p).resolve()
            assert path.is_relative_to(root) and path.suffix == '.jpg'
            digest = sha256(path)
            assert digest == f['sha256'] and digest not in source_hashes
            assert path.stat().st_size == f['bytes']
            with Image.open(path) as im:
                assert im.size == (f['width'], f['height'])
                assert max(im.size) <= summary['maximum_long_side']
                assert not im.getexif()
                im.verify()
            sequence_hashes.append(digest)
        assert len(set(sequence_hashes)) == 5
    assert all(len(v) == 1 for v in source_splits.values())
    assert all(len(v) == 1 for v in group_splits.values())
    actual_images = {p.relative_to(root).as_posix() for p in (root/'sequences').rglob('*.jpg')}
    assert actual_images == set(frame_by_path)
    for split in ['train', 'val', 'test']:
        assert {r['sequence_id'] for r in read(split+'.jsonl')} == {r['sequence_id'] for r in sequences if r['split'] == split}
    return dict(status='passed', sequences=len(sequences), blurred_frames=len(frames), unique_sources=len(sources),
                source_images_released=0, sharp_frames_released=0, source_group_overlap=0,
                images_by_split=dict(Counter(f['split'] for f in frames)),
                checked=['image decoding', 'dimensions', 'SHA-256', 'stripped EXIF', 'temporal ordering',
                         'distinct adjacent frames', 'source/group split isolation', 'source-file exclusion'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--report')
    a = p.parse_args()
    result = verify(a.root)
    if a.report:
        Path(a.report).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
