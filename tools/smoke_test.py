"""End-to-end software validation using generated shapes; no scientific reproduction claim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='runs/smoke')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '2'
    def run(module, *args):
        subprocess.run([sys.executable, '-m', 'kg_tsrdeblur.'+module, *map(str, args)],
                       cwd=root, env=env, check=True)
    data = out/'toy'
    run('prepare', 'toy', '--output', data)
    common = ['--manifest', data/'train.jsonl', '--val-manifest', data/'val.jsonl', '--device', 'cpu']
    run('train_classifier', *common, '--classes', 2, '--epochs', 1, '--batch-size', 4, '--output', out/'classifier')
    train_args = [*common, '--steps', 2, '--batch-size', 2, '--validate-every', 2]
    run('train', *train_args, '--stage', 'init', '--output', out/'init')
    run('train', *train_args, '--stage', 'joint', '--init-checkpoint', out/'init/latest.pt',
        '--classifier', out/'classifier/best.pt', '--output', out/'joint')
    run('evaluate', '--manifest', data/'test.jsonl', '--checkpoint', out/'joint/latest.pt',
        '--classifier', out/'classifier/best.pt', '--classes', 2, '--device', 'cpu', '--output', out/'metrics.json')
    run('infer', '--input', data/'test/0_blur.png', '--checkpoint', out/'joint/latest.pt',
        '--output', out/'restored.png', '--device', 'cpu')
    from PIL import Image
    with Image.open(out/'restored.png') as im:
        assert im.size == (128, 128)
    result = json.loads((out/'metrics.json').read_text())
    assert result['samples'] == 4 and 0 <= result['top1'] <= 1
    print('PASS: toy classifier -> initialization -> joint supervision -> evaluation -> image inference')
    print('These outputs validate software execution only, not manuscript results.')


if __name__ == '__main__':
    main()
