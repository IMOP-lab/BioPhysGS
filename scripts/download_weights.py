#!/usr/bin/env python3
"""Fetch the exact inference weights from the versioned GitHub release."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--only', choices=['depth', 'initializer', 'all'], default='all')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'configs/weights.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, record in manifest.items():
        if args.only != 'all' and record['component'] != args.only:
            continue
        target = args.output / name
        if target.exists():
            if target.stat().st_size == record['bytes'] and sha256(target) == record['sha256']:
                print('Verified existing', name)
                continue
            raise SystemExit(f'{target} exists but has a different hash; move it aside before downloading.')
        temporary = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(record['url'], timeout=60) as response, temporary.open('wb') as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        if temporary.stat().st_size != record['bytes'] or sha256(temporary) != record['sha256']:
            raise SystemExit(f'Incomplete or mismatched download: {temporary}')
        temporary.replace(target)
        print('Downloaded and verified', name)


if __name__ == '__main__':
    main()
