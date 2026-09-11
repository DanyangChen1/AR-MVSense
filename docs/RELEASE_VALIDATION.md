# GitHub 发布包验证记录

- 构建日期：2026-09-11
- 发布目录：`github_release/lightweight-rppg-reproduction`
- 包大小：约 66 MB
- 自动化测试：18/18 通过
- 权重检查：Light-rPPGViT、rPPGViT、TSCAN 三个权重的 SHA-256 均匹配，且均能严格加载到对应网络结构
- 数据清洁检查：未包含 `.avi`、`.mat` 或 `.npy` 原始数据/缓存；未发现超过 90 MB 的单文件
- 路径检查：发布配置未残留 `/Volumes/PHILIPS/OJID` 或 `/Users/cdy` 本机绝对路径
- 端到端检查：从发布目录运行 `scripts/evaluate.py`，Light-rPPGViT 在现有 UBFC-rPPG 缓存上成功完成 34 个测试批次

端到端复核指标：

| MAE | RMSE | MAPE (%) | Pearson | 测试视频数 |
|---:|---:|---:|---:|---:|
| 0.29296875 | 0.6550980403 | 0.3243707546 | 0.9981010542 | 9 |

该记录证明发布包内代码、配置和权重能够协同运行；原始公开数据仍须使用者按各数据集条款自行获取。
