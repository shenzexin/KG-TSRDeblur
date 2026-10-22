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
    state = ckpt['model']
    # Older ablation checkpoints retained these inactive modules.
    inactive = []
    if model.config.image_only:
        inactive.append('motion.')
    if model.config.spatial_only:
        inactive.extend(['refine.spectral.', 'refine.fusion.'])
    state = {k: v for k, v in state.items() if not any(k.startswith(p) for p in inactive)}
    model.load_state_dict(state)
    return model, ckpt


def save_restorer(path, model, optimizer, scheduler, step, args, best_psnr=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind='restorer', model=model.state_dict(), model_config=asdict(model.config),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step,
        args=vars(args), best_psnr=best_psnr,
        torch_version=str(torch.__version__), provenance='paper-based-reference'), path)


def resume_best_psnr(checkpoint, checkpoint_path):
    """Recover selection state, including validation logs from older checkpoints."""
    score = checkpoint.get('best_psnr')
    if score is not None:
        return float(score)
    log = Path(checkpoint_path).parent/'validation.jsonl'
    if log.exists():
        scores = [float(r['psnr']) for line in log.read_text(encoding='utf-8').splitlines()
                  if line.strip() and (r := json.loads(line))['step'] <= checkpoint['step']]
        if scores:
            return max(scores)
    raise ValueError('Checkpoint has no best-PSNR state or validation log. '
                     'Use --init-checkpoint to start a new run without resuming selection state.')


def ensure_split_disjoint(train_manifest, val_manifest):
    """Reject overlap in sequence, source, scene group, content hash or image paths."""
    def identifiers(path):
        p = Path(path).resolve()
        rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        ids = {(key, str(r[key])) for r in rows
               for key in ('sequence_id', 'source_id', 'group_id', 'source_sha256') if r.get(key)}
        paths = {str((p.parent/r[key]).resolve()).casefold() for r in rows
                 for key in ('sharp', 'blur', 'blur_path') if isinstance(r.get(key), str)}
        paths.update(str((p.parent/value).resolve()).casefold() for r in rows
                     for value in r.get('frames', []) if isinstance(value, str))
        return ids, paths
    ts, tp = identifiers(train_manifest)
    vs, vp = identifiers(val_manifest)
    if ts & vs or tp & vp:
        raise ValueError('Train/validation leakage: shared sequence, source, group, hash or image path')
