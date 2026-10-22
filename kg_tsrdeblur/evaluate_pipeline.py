"""Complete-pipeline class-correct precision/recall at IoU >= 0.5."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .tracking import box_iou


def match_counts(predictions, ground_truth, threshold=0.5):
    # Maximum-cardinality class-correct matching. Each sign counted at most once.
    costs = np.full((len(predictions), len(ground_truth)), 1e6)
    for i, pred in enumerate(predictions):
        for j, gt in enumerate(ground_truth):
            overlap = box_iou(np.array(pred['box']), np.array(gt['box']))
            if pred.get('label') == gt['label'] and overlap >= threshold:
                costs[i, j] = 1-overlap
    true_positive = 0
    if costs.size:
        ii, jj = linear_sum_assignment(costs)
        true_positive = int(sum(costs[i, j] < 1e6 for i, j in zip(ii, jj)))
    return true_positive, len(predictions), len(ground_truth)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True, help='pipeline.py output JSONL for a single sequence')
    p.add_argument('--ground-truth', required=True, help='One JSON per frame: frame_index, annotations=[{box,label}]')
    p.add_argument('--output', default='runs/pipeline_metrics.json')
    a = p.parse_args()
    def load(path):
        rows = [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
        keyed = {r['frame_index']: r for r in rows}
        if len(rows) != len(keyed):
            raise ValueError('Duplicate frame indices; evaluate sequences separately')
        return keyed
    pred, gt = load(a.predictions), load(a.ground_truth)
    if set(pred)-set(gt):
        raise ValueError('Predicted frames lack ground-truth annotations')
    tp = npred = ngt = 0
    for i, row in gt.items():
        t, n, g = match_counts(pred.get(i, {}).get('candidates', []), row['annotations'])
        tp, npred, ngt = tp+t, npred+n, ngt+g
    result = dict(true_positives=tp, predictions=npred, ground_truth=ngt,
        precision=tp/npred if npred else 0., recall=tp/ngt if ngt else 0.,
        policy='one-to-one class-correct matching, IoU>=0.5; absent predicted frames count as misses')
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
