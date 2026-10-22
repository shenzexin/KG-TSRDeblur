"""Restoration initialization and recognition-oriented optimization (Sec. 3.1)."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import ROIDataset
from .model import KGTSRDeblur
from .classifier import load_classifier
from .losses import restoration_loss, psnr_per_image, ssim_per_image
from .utils import seed_everything, load_config, load_restorer, save_restorer, ensure_split_disjoint, resume_best_psnr


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    psnr, ssim, n = 0., 0., 0
    for batch in loader:
        x = batch['blur'].to(device)
        gt = batch['sharp'].to(device)
        y = model(x, batch['descriptor'].to(device), batch['reliability'].to(device)).clamp(0, 1)
        psnr += psnr_per_image(y, gt).sum().item()
        ssim += ssim_per_image(y, gt).sum().item()
        n += len(x)
    model.train()
    return dict(psnr=psnr/n, ssim=ssim/n, samples=n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--val-manifest', required=True)
    p.add_argument('--config')
    p.add_argument('--stage', choices=['init', 'joint'], default='init')
    p.add_argument('--init-checkpoint')
    p.add_argument('--resume', help='Resume same stage; includes optimizer and schedule')
    p.add_argument('--classifier')
    p.add_argument('--steps', type=int)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='runs/restoration')
    p.add_argument('--validate-every', type=int, default=1000)
    p.add_argument('--no-recognition-loss', action='store_true', help='Joint-stage ablation only')
    args = p.parse_args()
    if args.resume and args.init_checkpoint:
        p.error('Use --resume or --init-checkpoint, not both')
    if args.stage == 'joint' and not (args.init_checkpoint or args.resume):
        p.error('Joint stage requires --init-checkpoint or --resume')
    if args.stage == 'joint' and not args.no_recognition_loss and not args.classifier:
        p.error('Joint stage requires a pretrained --classifier')
    args.steps = args.steps if args.steps is not None else (100000 if args.stage == 'init' else 50000)
    if args.steps < 1 or args.batch_size < 1 or args.validate_every < 1:
        p.error('Steps, batch-size and validation interval must be positive')
    seed_everything(args.seed)
    ensure_split_disjoint(args.manifest, args.val_manifest)
    device = torch.device(args.device)
    ckpt = None
    if args.resume or args.init_checkpoint:
        model, ckpt = load_restorer(args.resume or args.init_checkpoint, device)
        if args.config and load_config(args.config) != model.config:
            p.error('Config differs from checkpoint architecture')
    else:
        model = KGTSRDeblur(load_config(args.config)).to(device)
    lc = 0.1 if args.stage == 'joint' and not args.no_recognition_loss else 0.
    classifier = load_classifier(args.classifier, device) if lc else None
    train_set = ROIDataset(args.manifest)
    val_loader = DataLoader(ROIDataset(args.val_manifest), batch_size=args.batch_size, num_workers=args.workers)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, generator=generator)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4 if args.stage == 'init' else 1e-4,
                           betas=(0.9, 0.9), weight_decay=0.)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-6)
    start = 0
    if args.resume:
        if ckpt['args']['stage'] != args.stage or ckpt['args']['steps'] != args.steps:
            p.error('Resume requires same stage and total steps')
        opt.load_state_dict(ckpt['optimizer'])
        schedule.load_state_dict(ckpt['scheduler'])
        start = ckpt['step']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'arguments.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    model.train()
    iterator = iter(loader)
    best = resume_best_psnr(ckpt, args.resume) if args.resume else -float('inf')
    for step in range(start+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch['blur'], batch['descriptor'], batch['reliability'])
        loss, terms = restoration_loss(pred, batch['sharp'], classifier, batch['label'], lambda_c=lc)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        schedule.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps(dict(step=step, loss=loss.item(), lr=opt.param_groups[0]['lr'],
                **{k: v.item() for k, v in terms.items()})), flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            metrics = validate(model, val_loader, device)
            with (output/'validation.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(dict(step=step, **metrics))+'\n')
            print(json.dumps(dict(step=step, validation=metrics)), flush=True)
            improved = metrics['psnr'] > best
            if improved:
                best = metrics['psnr']
                save_restorer(output/'best_psnr.pt', model, opt, schedule, step, args, best)
            save_restorer(output/'latest.pt', model, opt, schedule, step, args, best)


if __name__ == '__main__':
    main()
