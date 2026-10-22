"""Measure actual implementation parameters and ROI timing on the current device."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
import torch
from .model import KGTSRDeblur, WindowBlock, SpatialFrequencyRefinement
from .utils import load_config, load_restorer


@torch.no_grad()
def count_operations(model, image, descriptor, reliability):
    """Count executed Conv/Linear and QK/AV MACs; FFT and elementwise work are separate."""
    counts = dict(convolution_macs=0, linear_macs=0, attention_macs=0, fft_real_input_elements=0)
    handles = []
    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            counts['convolution_macs'] += output.numel() * (module.in_channels//module.groups) * module.kernel_size[0]*module.kernel_size[1]
        elif isinstance(module, torch.nn.Linear):
            counts['linear_macs'] += output.numel()*module.in_features
        elif isinstance(module, WindowBlock):
            b, c, h, w = inputs[0].shape
            s = module.window
            windows = b*((h+s-1)//s)*((w+s-1)//s)
            counts['attention_macs'] += 2*windows*(s*s)**2*c
        elif isinstance(module, SpatialFrequencyRefinement) and not module.spatial_only:
            x = inputs[0]
            counts['fft_real_input_elements'] += x.shape[0]*module.reduce.out_channels*x.shape[-2]*x.shape[-1]
    try:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, WindowBlock, SpatialFrequencyRefinement)):
                handles.append(module.register_forward_hook(hook))
        model(image, descriptor, reliability)
    finally:
        for handle in handles:
            handle.remove()
    counts['counted_macs'] = sum(counts[k] for k in ('convolution_macs', 'linear_macs', 'attention_macs'))
    counts['scope'] = 'Batch count; multiply-add=one MAC. FFT pair, normalization, activations, softmax and elementwise operations excluded.'
    return counts


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
        operations=count_operations(model, x, d, r),
        model_config=asdict(model.config), scope='restorer only; not complete-pipeline FPS', trained_checkpoint=bool(a.checkpoint))
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
