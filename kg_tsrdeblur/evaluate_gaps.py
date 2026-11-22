"""Score Figure 2 gaps from shared predicted-box logs and withheld-frame annotations."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracking import box_iou


def score_gap(predictions, annotations, threshold=0.5):
    """Associate unexpanded boxes before using class predictions."""
    costs = np.full((len(predictions), len(annotations)),
                    min(len(predictions), len(annotations)) + 1.0)
    overlaps = np.zeros_like(costs)
    for i, pred in enumerate(predictions):
        for j, target in enumerate(annotations):
            overlap = box_iou(np.asarray(pred['box']), np.asarray(target['box']))
            overlaps[i, j] = overlap
            if overlap >= threshold:
                costs[i, j] = 1 - overlap
    rows, columns = linear_sum_assignment(costs)
    matches = [(i, j) for i, j in zip(rows, columns) if overlaps[i, j] >= threshold]
    correct = sum(predictions[i]['label'] == annotations[j]['label'] for i, j in matches)
    return len(annotations), len(matches), correct


def read_frames(path):
    frames = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row.get('sequence_id'), row['frame_index'])
        if key in frames:
            raise ValueError('Duplicate sequence/frame key: {}'.format(key))
        frames[key] = row
    return frames


def evaluate(ground_truth, runs):
    totals = {}
    for key, frame in ground_truth.items():
        gap = frame['gap_length']
        if gap not in (1, 3, 5):
            raise ValueError('Only withheld frames with gap_length 1, 3 or 5 belong in this manifest')
        full = runs['full'].get(key, {}).get('candidates', [])
        for name, run in runs.items():
            candidates = run.get(key, {}).get('candidates', [])
            if [c['box'] for c in candidates] != [c['box'] for c in full]:
                raise ValueError('Predicted boxes or candidate order differ at {} for {}'.format(key, name))
            counts = totals.setdefault(gap, {}).setdefault(name, [0, 0, 0])
            for i, count in enumerate(score_gap(candidates, frame['annotations'])):
                counts[i] += count
    if set(totals) != {1, 3, 5}:
        raise ValueError('Evaluation requires withheld-frame annotations for all three gap lengths')
    return {
        gap: {
            method: dict(ground_truth=n, matched=m, correct=c,
                         top1_pct=100 * c / n if n else None)
            for method, (n, m, c) in methods.items()
        }
        for gap, methods in sorted(totals.items())
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', required=True)
    for name in ('nafnet', 'fixed', 'full'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runs = {name: read_frames(getattr(args, name)) for name in ('nafnet', 'fixed', 'full')}
    result = evaluate(read_frames(args.ground_truth), runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
