"""Measure actual implementation parameters and ROI timing on the current device."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .model import KGTSRDeblur
from .utils import load_config, load_restorer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--output', default='runs/benchmark.json')
    a = p.parse_args()
    if min(a.iterations, a.batch_size) <= 0 or a.warmup < 0:
        p.error('Invalid benchmark counts')
    model = load_restorer(a.checkpoint, a.device)[0] if a.checkpoint else KGTSRDeblur(load_config(a.config)).to(a.device)
    model.eval()
    x = torch.rand(a.batch_size, 3, 128, 128, device=a.device)
    d, r = torch.zeros(a.batch_size, 16, device=a.device), torch.ones(a.batch_size, device=a.device)
    cuda = x.is_cuda
    def sync():
        if cuda:
            torch.cuda.synchronize(x.device)
    with torch.no_grad():
        for _ in range(a.warmup):
            model(x, d, r)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(x.device)
        times = []
        for _ in range(a.iterations):
            sync()
            start = time.perf_counter()
            model(x, d, r)
            sync()
            times.append((time.perf_counter()-start)*1000)
    result = dict(parameters=sum(p.numel() for p in model.parameters()),
        batch_size=a.batch_size, mean_batch_ms=sum(times)/len(times),
        amortized_roi_ms=sum(times)/len(times)/a.batch_size, device=a.device,
        device_name=torch.cuda.get_device_name(x.device) if cuda else 'CPU', torch=str(torch.__version__),
        threads=torch.get_num_threads(), warmup=a.warmup, iterations=a.iterations,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(x.device) if cuda else None,
        macs=None, macs_note='Not counted: convolution-only counters omit attention and FFT costs',
        model_config=asdict(model.config), scope='restorer only; not complete-pipeline FPS', trained_checkpoint=bool(a.checkpoint))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
