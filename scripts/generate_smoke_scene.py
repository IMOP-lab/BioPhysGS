#!/usr/bin/env python3
"""Make a procedural COLMAP-format test fixture; no patient images are used."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=16)
    args = parser.parse_args()
    if args.frames < 8:
        parser.error('At least eight frames are needed for a held-out view.')
    if args.output.exists():
        raise SystemExit('Choose a new output directory for the generated test fixture.')
    images = args.output / 'images'
    depths = args.output / 'depths'
    sparse = args.output / 'sparse/0'
    for p in [images, depths, sparse]:
        p.mkdir(parents=True)
    width, height, focal = 96, 64, 82.0
    cy, cx = height / 2, width / 2
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    extrinsics = []
    names = []
    for i in range(args.frames):
        center = 0.15 * np.sin(i * 2 * np.pi / args.frames)
        depth = np.full((height, width), 2.0)
        for _ in range(12):
            px = (xx - cx) / focal * depth + center
            py = (yy - cy) / focal * depth
            depth = 2.0 + 0.15 * np.sin(px * 2.5) * np.cos(py * 3.0)
        px = (xx - cx) / focal * depth + center
        py = (yy - cy) / focal * depth
        rgb = np.stack([0.4 + 0.2 * np.sin(px * 8), 0.4 + 0.2 * np.cos(py * 9),
                        0.5 + 0.15 * np.sin((px + py) * 7)], axis=-1)
        name = f'frame_{i:04d}.png'
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(images / name)
        np.save(depths / f'frame_{i:04d}.npy', depth.astype(np.float16))
        names.append(name)
        extrinsics.append(f'{i + 1} 1 0 0 0 {-center:.9f} 0 0 1 {name}\n\n')
    (sparse / 'cameras.txt').write_text(f'1 PINHOLE {width} {height} {focal} {focal} {cx} {cy}\n')
    (sparse / 'images.txt').write_text(''.join(extrinsics))
    points = []
    index = 0
    for y in np.linspace(-0.9, 0.9, 32):
        for x in np.linspace(-1.35, 1.35, 40):
            index += 1
            rgb = np.round(np.array([0.4 + 0.2 * np.sin(x * 8), 0.4 + 0.2 * np.cos(y * 9),
                                     0.5 + 0.15 * np.sin((x + y) * 7)]) * 255).astype(int)
            z = 2.0 + 0.15 * np.sin(x * 2.5) * np.cos(y * 3.0)
            points.append(f'{index} {x:.8f} {y:.8f} {z:.8f} {rgb[0]} {rgb[1]} {rgb[2]} 0 1 0\n')
    (sparse / 'points3D.txt').write_text(''.join(points))
    (args.output / 'test.txt').write_text('\n'.join(names[1::8]) + '\n')
    (args.output / 'fixture.json').write_text(json.dumps({
        'purpose': 'Software smoke test, not a paper experiment',
        'frames': args.frames, 'test_frames': names[1::8],
        'surface': 'z = 2 + 0.15 sin(2.5x) cos(3y)'}, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
