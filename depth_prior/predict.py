#!/usr/bin/env python3
"""Generate the archived DAv2 prior without changing its input convention."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from Data.dataset import Dataset_test
from depth_anything_v2_yuan.dpt import DepthAnythingV2


class FineTunedDAV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DepthAnythingV2(encoder='vitb', features=128,
                                    out_channels=[96, 192, 384, 768])

    def forward(self, image):
        return F.interpolate(self.model(image), size=(475, 475),
                             mode='bilinear', align_corners=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'checkpoints/biophysgs_dav2_vitb.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--test-file', type=Path, help='Exclude held-out names from depth generation.')
    args = parser.parse_args()
    excluded = set(args.test_file.read_text().splitlines()) if args.test_file else set()
    paths = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in {'.png', '.jpg', '.jpeg'} and p.name not in excluded)
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit('No input RGB images found.')
    model = FineTunedDAV2().to(args.device).eval()
    model.load_state_dict(torch.load(args.weights, map_location='cpu')['model_state_dict'], strict=True)
    loader = DataLoader(Dataset_test([str(p) for p in paths]), batch_size=1,
                        shuffle=False, num_workers=args.workers)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index, image in enumerate(loader):
            depth = (model(image.to(args.device)) / 20.0).squeeze().cpu().numpy().astype(np.float16)
            if not np.isfinite(depth).all():
                raise RuntimeError(f'Non-finite depth for {paths[index].name}')
            np.save(args.output / (paths[index].stem + '.npy'), depth)
            print(f'{index + 1}/{len(paths)}: {paths[index].name}', flush=True)


if __name__ == '__main__':
    main()
