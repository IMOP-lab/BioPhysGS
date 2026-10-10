# Recorded experiment profiles

This release preserves two snapshots instead of merging historical camera and
metric conventions. `provenance/source_snapshot.json` records original file hashes;
`provenance/release_validation.json` records validation of distributed files.
Portability edits concern paths and release entry points. Core training/model
source is copied from the recorded experiments.

| Profile | Source | Updates | Evaluation |
| --- | --- | --- | --- |
| `full` | Formal source at repository root | 30,000 | Archived legacy VGG-LPIPS |
| `no_depth` | Depth alignment disabled | 30,000 | Legacy |
| `no_smooth`, `no_tv`, `no_scale` | Named regularizer disabled | 30,000 | Legacy |
| `all_losses_off` | Depth alignment, smoothness, TV, scale disabled | 30,000 | Legacy |
| `no_field`, `no_layers`, `no_field_and_layers` | Optical branch controls | 30,000 | Legacy |
| `generic_capacity` | Matched optical-branch parameter count | 30,000 | Legacy |
| `hamlyn` | `variants/hamlyn`, calibrated full-model rerun | 30,000 | Legacy |
| `endomapper` | Full source for new prepared EndoMapper runs | 30,000 | Standardized VGG-LPIPS |

The EndoMapper profile is a new runnable full-model profile; old nine-method rows
are archived records, not newly reproduced release runs. External baselines use
separate implementations, input poses and budgets. One command does not rerun
every historical baseline.

## SyntheticColon controls

Use recorded O1--O3 prepared inputs (`s1`, `s2`, `s3`) and seeds 0, 1 and 2.
Retain the same poses, points, prior maps and seeds in paired comparisons.
`full` and `no_depth` isolate depth supervision under shared initialization.
The 3DGS+DAv2 reference uses its separately adapted upstream implementation.
Optimizer/densification defaults are in `arguments/__init__.py`. Depth alignment,
smoothness and TV start after step 500; scale regularization starts immediately.
Default effective depth weight is one. The wrapper passes pipeline ablation
flags to training and rendering.

## Hamlyn

Z1/Z2/Z3 correspond to `capture-20110222T143408Z2`,
`capture-20110222T170751Z1` and `capture-20110222T171001Z2`, with 34/67/37 original
test views. Keep their frame selection and split. The calibrated `hamlyn` source
retains principal points and the 70-dimensional appearance-regressor input.
It preserves the recorded fixed seed 0; the wrapper rejects other seeds for this
snapshot rather than silently ignoring them.
The complete directional input is part of the full model, not DROID pose retuning.

Historical Hamlyn scoring used JPEG ground truths and quality-75 rendered JPEG
exports with `[0,1]` VGG-LPIPS inputs. Its rendering source saves predictions as
PNG for lossless archival. To reproduce the JPEG table, export quality-75 copies
before calling `scripts/evaluate_images.py --protocol legacy`. Wrapper PNG
scores are lossless software evaluations and must not be relabelled as that table.

## EndoMapper and efficiency

EndoMapper uses five fixed 40-frame local windows, 35/5 train/test views,
320-by-240 RGB and 30,000 updates. Test indices are 1/9/17/25/33. External methods
use COLMAP/DAv2; BioPhysGS uses dense-flow poses. Scores compare recorded
pipelines rather than isolating optical-model attribution.
`results/endomapper_12_methods.csv` retains original nine means and three added
endoscopy-specific baseline means; it is not produced by the smoke fixture.

`results/rtx3090_12_methods.csv` records an independent synthetic timing scene:
32 frames, 28/4 split, 256-by-256 rendering, 30,000 updates, 20 warm-up renders
and five synchronized passes. Added runs use the same physical RTX 3090 as
the original records. Clinical quality and synthetic timing are distinct.
FPS concerns subsequent rendering and excludes pose/depth preprocessing.

## Metrics and validation

`scripts/evaluate_images.py` checks matching image sets and scores full RGB
without crops/masks. PSNR uses RGB MSE; SSIM uses an 11-by-11 Gaussian window.
`standardized` uses LPIPS 0.1/VGG with `normalize=True`; `legacy` uses the
retained local wrapper on `[0,1]` tensors. Do not pool these conventions.
C3VDv2 reference scoring uses separate LPIPS-AlexNet and differing pose settings.

Release validation verifies source/weight hashes, an independent CUDA build,
short train/save/load/render/metric cycles, a depth-inference pass and the
initialization entry point. It does not repeat every 30k paper experiment or
validate clinical geometry. Library/GPU versions can affect exact outcomes.
