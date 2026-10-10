# BioPhysGS

BioPhysGS 的研究代码，包含正式实验和 Hamlyn 校准重跑所用的训练、渲染与评估实现。
方法面向静态或近静态的内窥镜局部场景，先进行离线重建，再交互浏览重建结果。

## 包含内容

- 原始实验版本的完整训练和模型代码，以及保留主点和方向输入的 Hamlyn 校准版本。
- 实验使用的 CUDA 光栅器、最近邻与 SSIM 扩展源码。
- DroidCalib/DROID 位姿与点云导出程序、微调 DAv2 深度先验的推理及训练代码。
- 消融配置、明确的训练/留出规则、两种历史指标约定及实验数值记录。
- 版本化权重下载、SHA256 核验和不含临床图像的程序化运行样例。

## 安装与快速验证

Linux、Python 3.10、NVIDIA GPU 和 CUDA 11.8 开发工具包：

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
python scripts/generate_smoke_scene.py --output data/smoke
python scripts/run_experiment.py --profile full --data data/smoke \
  --depths depths --output outputs/smoke --iterations 600 --gpu 0
```

600 步样例用于验证训练、保存、加载、渲染和指标计算，并覆盖第 500 步之后启用的损失；
论文实验采用 30,000 次优化更新。必须使用本仓库的自定义光栅器，其深度输出与四分量
屏幕空间梯度和默认 3DGS 扩展有区别。

## 数据与复现

准备 `images/`、`sparse/0/`、训练帧 `.npy` 深度先验和明确的 `test.txt`。
深度先验可使用发布的微调 ViT-B 权重生成：

```bash
CUDA_VISIBLE_DEVICES=0 python depth_prior/predict.py \
  --images /path/to/scene/images --output /path/to/scene/depths \
  --test-file /path/to/scene/test.txt
python scripts/run_experiment.py --profile hamlyn --data /path/to/scene \
  --depths depths --output outputs/new_scene --gpu 0
```

`full` 等配置保留正式 SyntheticColon 受控实验版本；`hamlyn` 保留修订稿中校准后的完整
版本与主点。保留两个快照便于核对历史实验，不将它们混成一个未经验证的新配置。
Hamlyn 历史表采用 JPEG 导出与 `[0,1]` VGG-LPIPS，EndoMapper 采用无损图像及标准化
LPIPS；运行时应遵循各自协议，不能混合统计。

详细说明见 [数据准备](docs/DATA.md) 和 [实验协议](docs/REPRODUCTION.md)。
微调深度权重和初始化权重见 [v0.1.0](https://github.com/IMOP-lab/BioPhysGS/releases/tag/v0.1.0)，
下载脚本会核对完整 SHA256。深度发布权重仅移除优化器状态，推理张量保持不变。
原始临床数据需从相应提供方取得，本仓库不分发患者视频或患者重建场景。

## 来源与许可

本实现基于 3DGS、3D-UIR、DroidCalib/DROID-SLAM 和 Depth Anything V2，并保留
相应版权和许可证。重建代码受上游研究/非商业使用条件约束，独立初始化组件采用
AGPL-3.0，ViT-B 模型权重沿用 CC-BY-NC-4.0。详见
[第三方来源](THIRD_PARTY_NOTICES.md) 和各组件许可证。
