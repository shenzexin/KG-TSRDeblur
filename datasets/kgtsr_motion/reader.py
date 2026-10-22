"""Read five-frame blurred sequences without requiring unreleased source images."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


class BlurredSequences:
    def __init__(self, root, split=None):
        self.root = Path(root)
        file = self.root/(split+'.jsonl' if split else 'sequences.jsonl')
        self.rows = [json.loads(x) for x in file.read_text(encoding='utf-8').splitlines() if x.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = []
        for relative in row['frames']:
            with Image.open(self.root/relative) as im:
                frames.append(np.asarray(im.convert('RGB'), dtype=np.float32)/255.)
        # [time,channels,height,width], normalized RGB; no sharp or class-label field.
        return dict(frames=np.stack(frames).transpose(0, 3, 1, 2), metadata=row)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'])
    a = p.parse_args()
    data = BlurredSequences(a.root, a.split)
    print('Sequences:', len(data))
    if len(data):
        item = data[0]
        print('First sequence:', item['metadata']['sequence_id'])
        print('Tensor shape:', item['frames'].shape)


if __name__ == '__main__':
    main()
