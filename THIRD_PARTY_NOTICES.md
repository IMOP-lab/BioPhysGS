# Upstream components

Original copyright headers and component licenses are retained. This summary
does not replace their full license texts.

| Component | Source | License location |
| --- | --- | --- |
| Gaussian scene, loaders, optimizer, renderer interface | graphdeco-inria/gaussian-splatting | `LICENSE.md` |
| Appearance/medium implementation lineage | bilityniu/3D-UIR | `LICENSE.txt`, `licenses/3D-UIR.txt` |
| Custom differentiable rasterizer | Retained experimental Gaussian Splatting extension | `submodules/diff-gaussian-rasterization/LICENSE.md` |
| Nearest-neighbor extension | simple-knn | `submodules/simple-knn/LICENSE.md` |
| Fused SSIM extension | fused-ssim | `submodules/fused-ssim/LICENSE` |
| GLM headers | g-truc/glm | `submodules/diff-gaussian-rasterization/third_party/glm/copying.txt` |
| Tiny CUDA networks (installation dependency) | NVlabs/tiny-cuda-nn | `licenses/tiny-cuda-nn.txt` |
| Local LPIPS wrapper | S-aiueo32/lpips-pytorch | `licenses/lpips-pytorch.txt` |
| LPIPS calibration weights | richzhang/PerceptualSimilarity | `licenses/LPIPS.txt` |
| Depth source and DINOv2-derived backbone | DepthAnything/Depth-Anything-V2 | `licenses/Depth-Anything-V2.txt`, source headers |
| Depth training/data helpers | Retained local fine-tuning source | `depth_prior/LICENSE` |
| Separate dense-flow initializer | boschresearch/DroidCalib, derived from DROID-SLAM | `initialization/LICENSE`, `initialization/3rd-party-licenses.txt` |

Reconstruction code retains Gaussian Splatting/Pi-Lab research and non-commercial
restrictions. The independent initializer retains AGPL-3.0. Depth Anything V2
**ViT-B weights**, including the fine-tuned release, retain CC-BY-NC-4.0 terms;
see `licenses/CC-BY-NC-4.0.txt`. There is no single permissive license for all parts.

Portability edits to initialization/depth-training helpers replace local-machine
defaults with explicit paths; numerical preprocessing and model/solver settings
are preserved. Release scripts add installation, weight retrieval, procedural
fixtures and execution/evaluation entry points.
