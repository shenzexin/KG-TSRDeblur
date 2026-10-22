import json
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import ModelConfig, KGTSRDeblur


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    return ModelConfig(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else ModelConfig()


def load_restorer(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get('kind') != 'restorer':
        raise ValueError('Not a KG-TSRDeblur restorer checkpoint')
    model = KGTSRDeblur(ModelConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['model'])
    return model, ckpt


def save_restorer(path, model, optimizer, scheduler, step, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind='restorer', model=model.state_dict(), model_config=asdict(model.config),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
        args=vars(args), torch_version=str(torch.__version__), provenance='paper-based-reference'), path)


def ensure_split_disjoint(train_manifest, val_manifest):
    """Reject shared sequence IDs and shared sharp image paths across train/val."""
    def identifiers(path):
        p = Path(path).resolve()
        rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        return ({r['sequence_id'] for r in rows if r.get('sequence_id')},
                {str((p.parent/r['sharp']).resolve()) for r in rows})
    ts, tp = identifiers(train_manifest)
    vs, vp = identifiers(val_manifest)
    if ts & vs or tp & vp:
        raise ValueError('Train/validation leakage: shared sequences or sharp source paths')
