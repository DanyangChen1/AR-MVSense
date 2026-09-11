# Lightweight rPPG Approximate Reproduction

这是论文 **Lightweight rPPG Framework for Real-Time Physiological Monitoring on AR Devices** 的可运行近似复现包。仓库包含模型实现、训练/评估程序、公开数据集预处理代码、选定权重和实测结果；不包含 UBFC-rPPG 或 MMPD 原始数据。

本项目不是作者官方源码。论文未披露的网络与训练细节采用了可追溯的工程补全，完整边界见 [docs/REPRODUCTION_BOUNDARIES.md](docs/REPRODUCTION_BOUNDARIES.md)。

## 已附权重与 UBFC-rPPG 结果

| 模型 | 权重 | MAE | RMSE | MAPE (%) | Pearson |
|---|---|---:|---:|---:|---:|
| Light-rPPGViT | `weights/light_rppgvit/ubfc_best.pt` | 0.293 | 0.655 | 0.324 | 0.9981 |
| rPPGViT | `weights/rppgvit/ubfc_best.pt` | 0.098 | 0.293 | 0.128 | 0.9998 |
| TSCAN（近似基线） | `weights/tscan/ubfc_epoch16.pth` | 3.027 | 5.150 | 3.167 | 0.9000 |

前两项达到或接近论文量级。TSCAN 为官方 rPPG-Toolbox 的提前停止近似结果。所有数字来自受试者级 6:2:2 划分，测试集 9 名受试者；主表没有主动引入数据泄漏。

## 安装

建议 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

先校验发布包和权重：

```bash
python scripts/verify_release.py
```

## UBFC-rPPG：复现已有权重结果

按数据集许可自行下载 Dataset 2，并整理成：

```text
data/raw/UBFC-rPPG/DATASET_2/
├── subject1/vid.avi
├── subject1/ground_truth.txt
└── ...
```

也可运行带完整性检查的下载器：

```bash
python scripts/download_datasets.py ubfc \
  --output data/raw/UBFC-rPPG --source kaggle --workers 2
```

预处理一次：

```bash
python scripts/preprocess.py ubfc \
  --input data/raw/UBFC-rPPG/DATASET_2 \
  --output data/cache/UBFC-rPPG-128 --size 128
```

评估附带权重：

```bash
python scripts/evaluate.py \
  --config configs/ubfc_light_rppgvit_physical_batch4.yaml \
  --checkpoint weights/light_rppgvit/ubfc_best.pt

python scripts/evaluate.py \
  --config configs/ubfc_rppgvit_window64_cosine_physical_batch4.yaml \
  --checkpoint weights/rppgvit/ubfc_best.pt
```

重新训练：

```bash
python scripts/train.py --config configs/ubfc_light_rppgvit_physical_batch4.yaml
python scripts/train.py --config configs/ubfc_rppgvit_window64_cosine_physical_batch4.yaml
```

## TSCAN 基线

仓库内附精简后的 rPPG-Toolbox 源码及其原许可证。先用本工程预处理/配置训练，或在已存在 Toolbox 缓存时直接评估：

```bash
python third_party/rPPG-Toolbox/main.py \
  --config_file configs/toolbox_baselines/ubfc/tscan_release_test.yaml
```

TSCAN 配置使用 `data/cache/rppg-toolbox/ubfc/tscan/`。如只想复现论文主模型，不需要运行此可选基线。

## MMPD / mini-MMPD

代码支持 MMPD `.mat` 数据：

```bash
python scripts/preprocess.py mmpd \
  --input data/raw/MMPD-mini \
  --output data/cache/MMPD-mini-128 --size 128

python scripts/train.py --config configs/mmpd_light_rppgvit.yaml
```

MMPD 需要用户本人接受数据发布条款。本包不绕过门禁、不上传数据，也不附未验证的 MMPD 指标。取得数据后可直接使用上述入口训练。

## 结果与可追溯性

- `weights/manifest.json`：权重 SHA-256、配置和指标来源；
- `results/summary.json`：推荐模型的机器可读结果；
- `results/paper_vs_local.csv`：论文表格与本地结果逐项对照；
- `results/*/split_subjects.json`：训练/验证/测试受试者列表；
- `docs/REPRODUCTION_BOUNDARIES.md`：近似复现、数据泄漏及平台测量边界。

若另做训练/测试重叠实验，请把目录和表格明确命名为 `leaky_demo`；此类数据只能展示拟合能力，不能替代独立测试结果。

## 说明

- 权重单文件均小于 GitHub 的 100 MB 限制；也可以使用 Git LFS 管理 `*.pt`、`*.pth`。
- 原始数据和预处理缓存被 `.gitignore` 排除。
- Snapdragon XR2 功耗和延迟必须在对应设备上实测；仓库中的 Apple M2 数据只是本地仿真。
- 发布前请由仓库所有者选择顶层代码许可证；第三方 rPPG-Toolbox 保留其原始 LICENSE。
