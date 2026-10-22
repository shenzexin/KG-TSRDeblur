import json
import numpy as np
import pytest
import torch
from PIL import Image
from kg_tsrdeblur.model import KGTSRDeblur, ModelConfig
from kg_tsrdeblur.pipeline import PerceptionPipeline
from kg_tsrdeblur.data import ROIDataset, letterbox
from kg_tsrdeblur.utils import ensure_split_disjoint
from kg_tsrdeblur.evaluate import recognition_metrics


def test_letterbox_and_outside_roi_unchanged():
    frame = Image.fromarray(np.full((100, 200, 3), 90, np.uint8))
    roi, (_, _, w, h) = letterbox(frame)
    assert roi.size == (128, 128) and (w, h) == (128, 64)
    model = KGTSRDeblur(ModelConfig(width=8, transformer_blocks=2, refinement_channels=4))
    pipe = PerceptionPipeline(model)
    result, records = pipe.step(frame, [[50, 30, 80, 60, .9]])
    assert len(records) == 1 and records[0]['reliability'] == 0
    before, after = np.asarray(frame), np.asarray(result)
    assert np.array_equal(before[:20], after[:20])
    assert np.array_equal(before[:, 100:], after[:, 100:])


def test_leakage_guard(tmp_path):
    a, b = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    a.write_text(json.dumps(dict(sharp='a.png', sequence_id='seq1')))
    b.write_text(json.dumps(dict(sharp='b.png', sequence_id='seq1')))
    with pytest.raises(ValueError, match='leakage'):
        ensure_split_disjoint(a, b)


def test_recognition_metric_reference():
    result = recognition_metrics(torch.tensor([[2, 1], [0, 1]]))
    assert result['top1'] == .75
    assert result['macro_f1'] == pytest.approx((.8+2/3)/2)
