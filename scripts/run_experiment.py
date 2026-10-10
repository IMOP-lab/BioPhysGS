#!/usr/bin/env python3
"""Run an explicit paper profile with portable data and output paths."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RENDER_FLAGS = {'disable_bio_optical_field', 'disable_dual_layer',
                'disable_spatial_attenuation', 'disable_hemoglobin', 'disable_fresnel',
                'disable_surface_specular', 'disable_image_sss', 'generic_capacity_control'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='full')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--depths', default='depths')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--skip-metrics', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    profile_path = ROOT / 'configs/profiles' / (args.profile + '.json')
    profile = json.loads(profile_path.read_text())
    working = ROOT / profile['code_directory']
    iterations = args.iterations or profile['iterations']
    data, output = args.data.resolve(), args.output.resolve()
    train = [sys.executable, str(working / 'train.py'), '-s', str(data), '-m', str(output),
             '--eval', '--disable_viewer', '--debug_from', '-1',
             '--iterations', str(iterations), '--test_iterations', str(iterations),
             '--save_iterations', str(iterations), '--resolution', str(args.resolution)]
    if profile.get('fixed_seed') is not None:
        if args.seed != profile['fixed_seed']:
            parser.error(f"{args.profile} preserves the recorded fixed seed {profile['fixed_seed']}.")
    else:
        train.extend(['--seed', str(args.seed)])
    if args.depths:
        train.extend(['-d', args.depths])
    train.extend(profile['flags'])
    render = [sys.executable, str(working / 'render.py'), '-m', str(output),
              '--iteration', str(iterations), '--skip_train']
    render.extend(flag for flag in profile['flags'] if flag.removeprefix('--') in RENDER_FLAGS)
    evaluation = [sys.executable, str(ROOT / 'scripts/evaluate_images.py'),
                  '--renders', str(output / 'test' / f'ours_{iterations}' / 'renders'),
                  '--gt', str(output / 'test' / f'ours_{iterations}' / 'gt'),
                  '--protocol', profile['evaluation_protocol'], '--output', str(output / 'evaluation')]
    commands = [('train', train), ('render', render)]
    if not args.skip_metrics:
        commands.append(('evaluate', evaluation))
    for name, command in commands:
        print(name + ': ' + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise SystemExit('Output already exists. Choose a fresh directory to preserve earlier runs.')
    if not (data / 'images').is_dir() or not (data / 'sparse/0').is_dir():
        raise SystemExit('Expected images/ and sparse/0/ in --data.')
    if args.depths and not (data / args.depths).is_dir():
        raise SystemExit('Depth-prior directory is missing; use --depths "" explicitly for a no-prior run.')
    output.mkdir(parents=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    record = {'profile': args.profile, 'iterations': iterations, 'seed': args.seed,
              'paper_budget': profile['iterations'], 'is_budget_override': iterations != profile['iterations'],
              'commands': dict(commands), 'completed_stages': []}
    record_path = output / 'release_run.json'
    for name, command in commands:
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=working, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            record.update(status='failed', failed_stage=name, exit_code=result.returncode)
            record_path.write_text(json.dumps(record, indent=2) + '\n')
            raise SystemExit(f'{name} failed; inspect {output / (name + ".log")}')
        record['completed_stages'].append(name)
    record['status'] = 'completed'
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    print('Completed:', record_path)


if __name__ == '__main__':
    main()
