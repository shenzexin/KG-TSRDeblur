"""Paired ROI metrics, optional recognition metrics; never inserts manuscript scores."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import ROIDataset
from .classifier import load_classifier
from .losses import psnr_per_image, ssim_per_image
from .utils import load_restorer


def recognition_metrics(confusion):
    cm = confusion.double()
    tp = cm.diag()
    denom = cm.sum(0)+cm.sum(1)
    f1 = torch.where(denom > 0, 2*tp/denom.clamp_min(1), 0)
    return dict(top1=(tp.sum()/cm.sum().clamp_min(1)).item(), macro_f1=f1.mean().item(),
                macro_f1_policy='all configured classes, zero for undefined classes', confusion=confusion.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--blurred-baseline', action='store_true')
    p.add_argument('--classifier')
    p.add_argument('--classes', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='runs/metrics.json')
    a = p.parse_args()
    if not a.blurred_baseline and not a.checkpoint:
        p.error('Supply --checkpoint or --blurred-baseline')
    model = load_restorer(a.checkpoint, a.device)[0].eval() if not a.blurred_baseline else None
    classifier = load_classifier(a.classifier, a.device) if a.classifier else None
    cm = torch.zeros(a.classes, a.classes, dtype=torch.long)
    n, psnr, ssim = 0, 0., 0.
    with torch.no_grad():
        for batch in DataLoader(ROIDataset(a.manifest), batch_size=a.batch_size):
            x, gt = batch['blur'].to(a.device), batch['sharp'].to(a.device)
            y = model(x, batch['descriptor'].to(a.device), batch['reliability'].to(a.device)) if model else x
            y = y.clamp(0, 1)
            psnr += psnr_per_image(y, gt).sum().item()
            ssim += ssim_per_image(y, gt).sum().item()
            n += len(x)
            if classifier:
                labels = batch['label']
                logits = classifier(y)
                if logits.shape[1] != a.classes or (labels < 0).any() or (labels >= a.classes).any():
                    raise ValueError('Classifier class count or evaluation labels mismatch')
                predicted = logits.argmax(1).cpu()
                cm += torch.bincount(labels*a.classes+predicted, minlength=a.classes*a.classes).reshape(a.classes, a.classes)
    result = dict(samples=n, psnr_db=psnr/n, ssim=ssim/n, image_policy='RGB [0,1], clipped output; full padded ROI',
                  ssim_policy='11x11 Gaussian sigma=1.5, valid support', checkpoint=a.checkpoint,
                  blurred_baseline=a.blurred_baseline)
    if classifier:
        result.update(recognition_metrics(cm))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
