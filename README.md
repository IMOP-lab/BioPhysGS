# BioPhysGS

[中文说明](README_zh.md)

Research implementation of **BioPhysGS: Physics Guided 3D Gaussian Reconstruction
with Bio Optical Priors for Endoscopy**.

BioPhysGS reconstructs static or near-static local scenes from monocular RGB.
It combines dense-flow initialization, optical conditioning of Gaussian
appearance, and depth-conditioned image composition. Scene optimization is
offline; a trained scene supports subsequent interactive rendering.

This release includes training and rendering source, the custom CUDA rasterizer,
depth-prior inference and fine-tuning source, the dense-flow/COLMAP exporter,
experiment profiles, evaluation scripts, and recorded quantitative summaries.
Inference weights are downloadable from the
[v0.1.0 release](https://github.com/IMOP-lab/BioPhysGS/releases/tag/v0.1.0).

## Installation

Use Linux, an NVIDIA GPU, Python 3.10, and the CUDA 11.8 toolkit. The archived
reconstruction environment uses PyTorch 2.1.0, torchvision 0.16.0, and NumPy
1.26.4. A toolkit with `nvcc` is required in addition to PyTorch CUDA wheels.
The extension installer pins the tested tiny-cuda-nn 2.0 source commit.

```bash
git clone https://github.com/IMOP-lab/BioPhysGS.git
cd BioPhysGS
conda create -n biophysgs python=3.10 -y
conda activate biophysgs
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
export CUDA_HOME=/path/to/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/install_extensions.sh
python scripts/download_weights.py
```

The rasterizer is bundled because BioPhysGS uses its depth outputs and
four-component screen-space gradients for densification. Replacing it with a
stock rasterizer can change behavior or cause a shape mismatch.
Initialization uses a separate environment; see [data preparation](docs/DATA.md).

## Quick software check

Create a procedural fixture and exercise optimization, model saving,
model loading, rendering and evaluation:

```bash
python scripts/generate_smoke_scene.py --output data/smoke
python scripts/run_experiment.py --profile full --data data/smoke \
  --depths depths --output outputs/smoke --iterations 600 --gpu 0
```

The 600-step run activates depth, smoothness and TV terms after step 500.
It is a software check; paper experiments use 30,000 updates.

## Input and training

Prepared inputs contain undistorted RGB, fixed estimated poses, initial points
in COLMAP format, and training-view depth-prior arrays:

```text
scene/
  images/                  # RGB images
  depths/                  # <image_stem>.npy, fine-tuned DAv2 outputs / 20
  sparse/0/
    cameras.bin            # or cameras.txt; PINHOLE/SIMPLE_PINHOLE
    images.bin             # or images.txt
    points3D.bin           # or points3D.txt/points3D.ply
  test.txt                 # explicit held-out image names
```

Generate depth priors with the released ViT-B weights:

```bash
CUDA_VISIBLE_DEVICES=0 python depth_prior/predict.py \
  --images /path/to/scene/images --output /path/to/scene/depths \
  --test-file /path/to/scene/test.txt
```

Run a formal full-model/control profile:

```bash
python scripts/run_experiment.py --profile full --data /path/to/scene \
  --depths depths --output outputs/full_seed0 --seed 0 --gpu 0
python scripts/run_experiment.py --profile no_depth --data /path/to/scene \
  --depths depths --output outputs/no_depth_seed0 --seed 0 --gpu 0
```

The `hamlyn` profile selects the calibrated full-model source used for the revised
Hamlyn runs. It retains exported principal points and complete directional
conditioning. It also supports a new prepared scene with off-center intrinsics.
The formal profile preserves conventions used in archived SyntheticColon controls.

```bash
python scripts/run_experiment.py --profile hamlyn --data /path/to/hamlyn/scene \
  --depths depth_depany --output outputs/hamlyn_Z1 --gpu 0
```

Runs write commands/configurations and stage logs to their output directories.
Use `--dry-run` to inspect commands. Existing output directories are protected.
See [experiment protocols](docs/REPRODUCTION.md) for data mappings and metrics.

## Source layout

| Path | Purpose |
| --- | --- |
| `train.py`, `render.py`, `metrics.py` | Formal training, rendering and historical evaluation |
| `scene/endoscope_model.py` | Primitive optical conditioning and image composition |
| `gaussian_renderer/` | Differentiable rendering and appearance composition |
| `submodules/` | Exact rasterizer, nearest-neighbor and SSIM extension source |
| `variants/hamlyn/` | Calibrated full-model source for revised Hamlyn results |
| `initialization/` | Separate DroidCalib/DROID-based pose and point exporter |
| `depth_prior/` | Fine-tuned DAv2 model, inference and training source |
| `configs/` | Profiles and checked weight-download manifest |
| `scripts/` | Installation, smoke inputs, run and evaluation entry points |
| `results/` | Archived numerical comparison summaries |
| `provenance/` | Source hashes and release validation records |

## Data and weights

Obtain datasets from their providers under their access conditions. Raw clinical
videos and reconstructed patient scenes are not distributed here. The software
smoke fixture is generated procedurally.

The released DAv2 checkpoint contains the exact inference tensors of the
fine-tuned experimental checkpoint, with optimizer state removed.
`configs/weights.json` specifies SHA256 values and download URLs. Gaussian scenes
are optimized per sequence; their model files are generated by training.

## Attribution and licenses

The implementation builds on
[3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting),
[3D-UIR](https://github.com/bilityniu/3D-UIR),
[DroidCalib](https://github.com/boschresearch/DroidCalib),
[DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM), and
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
Original notices and component licenses are retained in [LICENSE.md](LICENSE.md),
[LICENSE.txt](LICENSE.txt), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and
component directories. Reconstruction code retains upstream research/non-commercial
restrictions; the separate initializer retains AGPL-3.0. ViT-B model weights
retain CC-BY-NC-4.0 terms.

Please cite the BioPhysGS manuscript and relevant upstream works. Bibliographic
information is in [CITATION.cff](CITATION.cff). The manuscript is under revision;
this repository does not assert publication or clinical validation.
