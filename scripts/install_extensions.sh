#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BIOPHYSGS_PYTHON:-python}"
if [[ -z "${CUDA_HOME:-}" ]]; then
    echo 'Set CUDA_HOME to a CUDA 11.8 toolkit containing bin/nvcc.' >&2
    exit 1
fi
"$python_bin" -c 'import torch; assert torch.version.cuda == "11.8", "Use PyTorch 2.1.0+cu118 for the archived reconstruction environment."'
export MAX_JOBS="${MAX_JOBS:-4}"
for package in diff-gaussian-rasterization simple-knn fused-ssim; do
    "$python_bin" -m pip install --no-build-isolation --no-deps "$repo_dir/submodules/$package"
done
"$python_bin" -m pip install --no-build-isolation --no-deps \
    'git+https://github.com/NVlabs/tiny-cuda-nn.git@3f6ef958edd4c5ab4d894b6e4813839cc809457c#subdirectory=bindings/torch'
