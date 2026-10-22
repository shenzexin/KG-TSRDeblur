import argparse
import json
from dataclasses import asdict
import numpy as np
import pytest
import torch
from kg_tsrdeblur.model import KGTSRDeblur, ModelConfig
from kg_tsrdeblur.tracking import MultiObjectTracker
from kg_tsrdeblur.data import ROIDataset
from kg_tsrdeblur.utils import ensure_split_disjoint, save_restorer, load_restorer, resume_best_psnr
from kg_tsrdeblur.benchmark import count_operations
from kg_tsrdeblur.train_detector import configure_paper_adam


def test_detector_adam_both_betas():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(2))], betas=(.9,.999))
    assert configure_paper_adam(optimizer) is optimizer
    assert optimizer.defaults['betas'] == (.9,.9)
    assert all(g['betas'] == (.9,.9) for g in optimizer.param_groups)
    with pytest.raises(TypeError):
        configure_paper_adam(torch.optim.SGD([torch.nn.Parameter(torch.ones(2))],lr=.01))


@pytest.mark.parametrize('key', ['source_id', 'group_id', 'source_sha256'])
def test_shared_source_variants_rejected(tmp_path, key):
    paths = [tmp_path/'train.jsonl', tmp_path/'val.jsonl']
    for i, path in enumerate(paths):
        path.write_text(json.dumps({'sharp': f'{i}.png', 'sequence_id': f'seq{i}', key: 'shared'}))
    with pytest.raises(ValueError, match='leakage'):
        ensure_split_disjoint(*paths)


def test_sequence_split_guard_and_training_schema(tmp_path):
    paths = [tmp_path/'train.jsonl', tmp_path/'val.jsonl']
    for i, path in enumerate(paths):
        path.write_text(json.dumps(dict(sequence_id=f'seq{i}', frames=[f'{i}.jpg'])))
    ensure_split_disjoint(*paths)
    with pytest.raises(ValueError, match='aligned sharp training targets'):
        ROIDataset(paths[0])


def test_resume_preserves_best_selection(tmp_path):
    model = KGTSRDeblur(ModelConfig(width=8, transformer_blocks=1, refinement_channels=4))
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    path = tmp_path/'latest.pt'
    save_restorer(path, model, opt, sched, 4, argparse.Namespace(stage='init', steps=10), best_psnr=32.)
    restored, checkpoint = load_restorer(path)
    best = resume_best_psnr(checkpoint, path)
    assert best == 32. and not 31. > best
    assert restored.config == ModelConfig(**json.loads(json.dumps(asdict(model.config))))
    checkpoint.pop('best_psnr')
    (tmp_path/'validation.jsonl').write_text('\n'.join(json.dumps(dict(step=s, psnr=p)) for s, p in [(2,32), (4,31), (6,40)]))
    assert resume_best_psnr(checkpoint, path) == 32.


def test_ablation_removes_parameters_and_matches_zero_motion():
    cfg = dict(width=8, transformer_blocks=1, refinement_channels=4)
    full = KGTSRDeblur(ModelConfig(**cfg)).eval()
    image = KGTSRDeblur(ModelConfig(**cfg, image_only=True)).eval()
    image.load_state_dict({k:v for k,v in full.state_dict().items() if not k.startswith('motion.')})
    x = torch.rand(1,3,32,32)
    with torch.no_grad():
        assert torch.equal(full(x), image(x))
    assert image.motion is None
    assert sum(p.numel() for p in image.parameters()) < sum(p.numel() for p in full.parameters())
    spatial = KGTSRDeblur(ModelConfig(**cfg, spatial_only=True))
    assert spatial.refine.spectral is None and spatial.refine.fusion is None
    counts = count_operations(full, x, torch.zeros(1,16), torch.ones(1))
    assert counts['attention_macs'] == 2*64**2*32
    assert counts['counted_macs'] > counts['convolution_macs'] > 0


@pytest.mark.parametrize('velocity', [(36.,0.), (25.,25.), (0.,-36.)])
def test_fast_track_confirmation_and_five_frame_reacquisition(velocity):
    tracker = MultiObjectTracker()
    ids, reliability = [], []
    for i in range(9):
        center = np.array([350.,350.])+np.array(velocity)*i
        detection = [[*(center-20), *(center+20), .99]] if i < 3 or i == 8 else []
        candidates = tracker.step(detection, (1024,1024))
        assert len(candidates) == 1
        ids.append(candidates[0]['track_id'])
        reliability.append(candidates[0]['reliability'])
    assert len(set(ids)) == 1 and all(r > 0 for r in reliability[2:])
    assert all(reliability[i+1] < reliability[i] for i in range(2,7))
    assert tracker.tracks[0].missing == 0


def test_separate_fast_targets_keep_ids():
    tracker = MultiObjectTracker()
    for i in range(5):
        a, b = 100+30*i, 800-25*i
        candidates = tracker.step([[a,100,a+40,140,.99], [b,600,b+40,640,.98]], (1024,768))
        assert [c['track_id'] for c in candidates] == [1,2]
    assert all(c['confirmed'] for c in candidates)
