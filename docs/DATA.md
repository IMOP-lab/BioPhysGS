# Preparing data

## Already prepared COLMAP scenes

Reconstruction accepts pinhole images and COLMAP poses/points. Keep poses,
points and depth priors in their recorded coordinate/scale convention.
Absolute physical scale is not inferred from monocular RGB.
The recorded Fourier encoding requires a point cloud with nonzero spread on
each coordinate axis; an exactly axis-aligned planar fixture is degenerate.

Create `test.txt` with explicit held-out image names before training. Otherwise,
loaders use the archived every-eighth-frame rule. Held-out RGB and depth are
excluded from reconstruction losses. The estimated pose pipeline may process the
complete sequence, as recorded separately from the photometric train/test split.

## Dense-flow initialization

`initialization/estimate_and_export.py` preserves the recorded DroidCalib/DROID
settings and COLMAP export code. It is a separate program under AGPL-3.0.
Install dependencies in an independent environment: PyTorch, a matching CUDA
toolkit, OpenCV, SciPy, Open3D, matplotlib, tqdm and torch-scatter. The recorded
inference environment uses Python 3.10 and PyTorch 2.5.1/CUDA 12.1.

```bash
conda create -n biophysgs-init python=3.10 -y
conda activate biophysgs-init
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r initialization/requirements.txt
python -m pip install torch-scatter==2.1.2 \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
export CUDA_HOME=/path/to/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/setup_initializer.sh
python scripts/download_weights.py --only initializer
CUDA_VISIBLE_DEVICES=0 python initialization/estimate_and_export.py \
  --imagedir /path/to/input/images --weights checkpoints/droidcalib.pth \
  --reconstruction_path /path/to/export --num_images 200 --buffer 256 \
  --image_size_target 320 240
```

The exporter writes camera/pose/point files and trajectory diagnostics.
Place camera/point files in `scene/sparse/0/` and separately prepared RGB in
`scene/images/`. Resize/rectify RGB to the width and height in `cameras.txt`,
retaining its original filenames. Exported calibration describes the initializer's
working grid; it must match the RGB dimensions used for reconstruction.
Raw distorted images require dataset-specific rectification. `--calib` accepts
supplied intrinsics; `--no-opt_intr` fixes them.
Defaults are recorded settings, not universally optimal endoscope parameters.

## Fine-tuned depth prior

`depth_prior/predict.py` preserves the archived ViT-B architecture, historical
channel convention in `Data/dataset.py`, 518-pixel input resizing, 475-by-475
output interpolation, and output divisor 20. Arrays are saved as float16;
reconstruction resizes with OpenCV and converts them to float32. These are
relative priors in the recorded convention, not calibrated tissue depths.

```bash
python scripts/download_weights.py --only depth
CUDA_VISIBLE_DEVICES=0 python depth_prior/predict.py \
  --images /path/to/scene/images --output /path/to/scene/depths \
  --test-file /path/to/scene/test.txt
```

Fine-tuning retains the SyntheticColon I/II training and validation folders and
excludes listed test videos, including O1--O3. It requires raw SimCol3D and upstream
ViT-B weights:

```bash
cd depth_prior
mkdir -p 'Trained models'
python train_depthany.py --data-root /path/to/raw/SimCol3D \
  --pretrained-weights /path/to/depth_anything_v2_vitb.pth --continue false
```

The published checkpoint can be used directly. Release validation does not
repeat full depth-model fine-tuning.

## Dataset references

- SimCol3D: [official challenge repository](https://github.com/lalithjets/SimCol3D_challenge_2022).
- Hamlyn: [Hamlyn Centre datasets](https://hamlyn.doc.ic.ac.uk/vision/).
- EndoMapper: [official dataset repository](https://www.nature.com/articles/s41597-023-02564-7).
- C3VDv2: [dataset website](https://durrlab.github.io/C3VDv2/).

Dataset access and permitted uses are governed by their providers. This release
contains no raw clinical image/video dataset.
