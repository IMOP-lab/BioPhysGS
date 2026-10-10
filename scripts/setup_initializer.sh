#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BIOPHYSGS_INIT_PYTHON:-python}"
fetch_source() {
    local directory="$1" url="$2" revision="$3"
    if [[ ! -d "$directory/.git" ]]; then
        git clone "$url" "$directory"
    fi
    git -C "$directory" checkout --detach "$revision"
    git -C "$directory" submodule update --init --recursive
}
fetch_source "$repo_dir/initialization/thirdparty/lietorch" \
    https://github.com/princeton-vl/lietorch.git 0fa9ce8ffca86d985eca9e189a99690d6f3d4df6
fetch_source "$repo_dir/initialization/thirdparty/eigen" \
    https://gitlab.com/libeigen/eigen.git 3d4ba855e014987cad86d62a8dff533492255695
cd "$repo_dir/initialization"
"$python_bin" setup.py build_ext --inplace
"$python_bin" setup.py install
