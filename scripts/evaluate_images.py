#!/usr/bin/env python3
"""Score full RGB frames under an explicit, named LPIPS convention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lpips
import numpy as np
from PIL import Image
import torch
from utils.loss_utils import ssim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--protocol', choices=['standardized', 'legacy'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    paths = sorted(p for p in args.renders.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not paths:
        raise SystemExit('No predicted images found.')
    pred_stems = {p.stem for p in paths}
    truths = {p.stem: p for p in args.gt.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    if len(pred_stems) != len(paths) or pred_stems != set(truths):
        raise SystemExit('Prediction and ground-truth image sets must match exactly, without duplicate stems.')
    if args.protocol == 'legacy':
        from lpipsPyTorch.modules.lpips import LPIPS
        network = LPIPS(net_type='vgg').to(args.device).eval()
    else:
        network = lpips.LPIPS(net='vgg', version='0.1').to(args.device).eval()
    rows = []
    with torch.no_grad():
        for path in paths:
            truth = truths[path.stem]
            pred = np.array(Image.open(path).convert('RGB'))
            gt = np.array(Image.open(truth).convert('RGB'))
            if pred.shape != gt.shape:
                raise ValueError(f'Image dimensions differ for {path.name}')
            x = torch.from_numpy(pred.copy()).permute(2, 0, 1)[None].to(args.device).float() / 255
            y = torch.from_numpy(gt.copy()).permute(2, 0, 1)[None].to(args.device).float() / 255
            mse = float(np.mean(((pred.astype(np.float64) - gt) / 255) ** 2))
            value = network(x, y, normalize=True) if args.protocol == 'standardized' else network(x, y)
            rows.append({'image': path.name, 'PSNR': float(-10 * np.log10(max(mse, 1e-30))),
                         'SSIM': float(ssim(x, y)), 'LPIPS': float(value),
                         'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'gt_sha256': hashlib.sha256(truth.read_bytes()).hexdigest()})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'per_view.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = {'views': len(rows), 'protocol': args.protocol, 'lpips_backbone': 'vgg',
               'lpips_input_normalized': args.protocol == 'standardized',
               **{key: float(np.mean([r[key] for r in rows])) for key in ['PSNR', 'SSIM', 'LPIPS']}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
