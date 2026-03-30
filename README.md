# BioPhysGS

BioPhysGS is a physics-guided Gaussian splatting codebase for scene reconstruction, rendering, and evaluation.

## Repository Layout

```text
.
|-- arguments/              # argument definitions
|-- gaussian_renderer/      # differentiable renderer
|-- lpipsPyTorch/           # local LPIPS implementation
|-- scene/                  # scene representation and loading
|-- utils/                  # utility helpers
|-- train.py                # training entry point
|-- render.py               # rendering entry point
|-- render_video.py         # video rendering
|-- metrics.py              # evaluation script
|-- convert.py              # COLMAP conversion helper
|-- runAll.sh               # batch training template
|-- environment.yml         # conda environment
|-- requirements.txt        # lightweight Python requirements
```

## Requirements

- Python 3.10
- PyTorch 2.1 with CUDA 11.8
- A working CUDA toolchain for custom extensions
- COLMAP-format input data for training

## Setup

Recommended:

```bash
conda env create -f environment.yml
conda activate biophysgs
```

Manual setup:

```bash
conda create -n biophysgs python=3.10
conda activate biophysgs
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit-dev=11.8 -c conda-forge
pip install -r requirements.txt
```

If you use custom CUDA rasterization or nearest-neighbor extensions, make sure their local packages are installed in your environment before training.

## Data Layout

```text
<scene_root>
|-- images/
|-- depths/                  # optional depth priors
`-- sparse/
    `-- 0/
        |-- cameras.bin
        |-- images.bin
        |-- points3D.bin
        `-- depth_params.json
```

## Usage

Training:

```bash
python train.py -s <scene_root>
```

Training with evaluation:

```bash
python train.py -s <scene_root> --eval
```

Training with depth priors:

```bash
python train.py -s <scene_root> -d <depth_dir>
```

Rendering:

```bash
python render.py -m <model_dir>
```

Metric evaluation:

```bash
python metrics.py -m <model_dir>
```

Batch script template:

```bash
bash runAll.sh
```

## Notes

- `runAll.sh` is only a template and should be edited for your own dataset paths and GPU ids.
- `metrics.py` and `render.py` require a full runtime environment with PyTorch installed.
- If you use depth supervision, make sure the depth files and `depth_params.json` are consistent with your scene layout.
