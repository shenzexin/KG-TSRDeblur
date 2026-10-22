import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import ROIDataset
from .classifier import SignClassifier
from .utils import seed_everything, ensure_split_disjoint


def main():
    p = argparse.ArgumentParser(description='Pretrain MobileNetV3-Small on sharp training ROIs')
    p.add_argument('--manifest', required=True)
    p.add_argument('--val-manifest', required=True)
    p.add_argument('--classes', type=int, default=20)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', default='runs/classifier')
    a = p.parse_args()
    if a.classes < 2 or a.epochs < 1 or a.batch_size < 2:
        p.error('Require classes>=2, epochs>=1, batch-size>=2')
    seed_everything(a.seed)
    ensure_split_disjoint(a.manifest, a.val_manifest)
    model = SignClassifier(a.classes).to(a.device)
    train = ROIDataset(a.manifest, sharp_only=True)
    val = ROIDataset(a.val_manifest, sharp_only=True)
    if len(train) < 2:
        p.error('Classifier training needs at least two samples for BatchNorm')
    loader = DataLoader(train, batch_size=min(a.batch_size, len(train)), shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=a.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.9), weight_decay=0)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.
    for epoch in range(1, a.epochs+1):
        model.train()
        for batch in loader:
            x, labels = batch['sharp'].to(a.device), batch['label'].to(a.device)
            if (labels < 0).any() or (labels >= a.classes).any():
                raise ValueError('All classifier training labels must be in [0,classes)')
            loss = F.cross_entropy(model(x), labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        schedule.step()
        model.eval()
        correct, count = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch['label'].to(a.device)
                if (labels < 0).any() or (labels >= a.classes).any():
                    raise ValueError('Invalid validation label')
                correct += (model(batch['sharp'].to(a.device)).argmax(1) == labels).sum().item()
                count += len(labels)
        accuracy = correct/count
        row = dict(epoch=epoch, sharp_val_top1=accuracy)
        print(json.dumps(row), flush=True)
        with (output/'validation.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        ckpt = dict(kind='classifier', model=model.state_dict(), num_classes=a.classes,
                    epoch=epoch, args=vars(a), sharp_val_top1=accuracy)
        torch.save(ckpt, output/'latest.pt')
        if accuracy > best:
            best = accuracy
            torch.save(ckpt, output/'best.pt')


if __name__ == '__main__':
    main()
