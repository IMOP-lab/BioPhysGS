#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${BIOPHYSGS_PYTHON:-python}" "$repo_dir/scripts/run_experiment.py" "$@"
