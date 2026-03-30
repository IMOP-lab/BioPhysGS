# BioPhysGS

Codebase for our physics-guided Gaussian splatting pipeline.

## Structure

```text
.
|-- arguments/
|-- gaussian_renderer/
|-- lpipsPyTorch/
|-- scene/
|-- utils/
|-- train.py
|-- render.py
|-- render_video.py
|-- metrics.py
|-- convert.py
|-- runAll.sh
|-- environment.yml
|-- requirements.txt
```

## Setup

```bash
conda env create -f environment.yml
conda activate biophysgs
```

Or install manually:

```bash
conda create -n biophysgs python=3.10
conda activate biophysgs
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
conda install cudatoolkit-dev=11.8 -c conda-forge
pip install -r requirements.txt
```

## Data Layout

```text
<scene_root>
|-- images/
|-- depths/
`-- sparse/
    `-- 0/
        |-- cameras.bin
        |-- images.bin
        |-- points3D.bin
        `-- depth_params.json
```

## Usage

Train:

```bash
python train.py -s <scene_root>
```

Train with evaluation:

```bash
python train.py -s <scene_root> --eval
```

Train with depth priors:

```bash
python train.py -s <scene_root> -d <depth_dir>
```

Render:

```bash
python render.py -m <model_dir>
```

Evaluate:

```bash
python metrics.py -m <model_dir>
```

Batch script:

```bash
bash runAll.sh
```
