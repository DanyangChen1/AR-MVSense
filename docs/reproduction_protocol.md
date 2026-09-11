# 实验复现协议与证据边界

## 1. 原文明确给出的设置（reported）

| 项目 | Lightweight rPPG 论文 | 实现 |
|---|---|---|
| 数据集 | UBFC-rPPG、MMPD | 两者均支持 |
| 采样率 | 30 fps | `fps: 30` |
| ROI | Haar 首帧检测，后续固定同一框 | 固定首个可用帧的最大人脸框；检测失败时记录中心裁剪回退 |
| 输入预处理 | 帧差后除以序列标准差 | `(F[t+1]-F[t]) / std` |
| 片段长度 | 128 帧 | `clip_length: 128`，实际读取 129 帧形成 128 个差分 |
| 划分 | 6:2:2 | 按数值排序的受试者 ID 划分，避免受试者泄漏 |
| 优化器 | Adam | Adam |
| 初始学习率 | `1e-2` | `lr: 0.01` |
| batch size | 4 | `batch_size: 4` |
| 损失 | Negative Pearson Correlation Loss | `NegativePearsonLoss` |
| 指标 | MAE、RMSE、MAPE、Pearson rho | 逐视频 FFT-HR 后汇总 |
| 标准模型结构 | 卷积编码器 + 3x CCA + 4x SWTA + 波形头 | 对应实现 |
| 轻量模型结构 | 轻量卷积编码器 + 3x 重参数 CCA + 4x TASA + 波形头 | 对应实现 |
| AR 测试输入 | 128x128、batch 1、1000 次 | benchmark 默认一致 |

## 2. 由参考论文补全的设置（inferred）

| 缺失项 | 采用值 | 补全依据/理由 |
|---|---:|---|
| 标准模型 attention heads | 4 | PhysMViT 明确给出 `h=4` |
| 标准模型 epoch | 30 | PhysMViT 明确给出 30 epochs |
| 标准模型 dropout | 0.3 | PhysMViT 明确给出 0.3 |
| 轻量模型 epoch | 50 | RePhys 明确给出 50 iterations（按 epoch 解释） |
| 轻量模型 dropout | 0.2 | RePhys 明确给出 0.2 |
| PConv 比例 | 1/4 通道 | RePhys 明确给出 first 1/4 channels |
| CCA/SWTA hidden width | 124 | 以论文约 1.12M 参数为约束选择；部署参数约 1.11M |
| Light hidden width | 137 | 以论文约 0.70M 参数和 6.93G FLOPs 双重约束选择；训练态 0.700M/6.901G，部署态约 0.697M/6.780G |
| 编码器时间降采样/波形上采样 | 标准版 2x；轻量版保持 T；末端线性上采样 | 标准版图中明确有 Upsample但倍率未给；轻量版保持 T 与 RePhys Fig. 1 各 stage 的尺寸标注一致 |
| 编码器空间降采样 | 标准版 4x；轻量版 8x | 原文未给逐层 stride；以论文 20.12G/6.93G FLOPs 比例为约束，并保留 16x16 的轻量版空间特征图 |
| SWTA 窗口/移位 | 16 / 8 个潜在时间步 | 原文只给符号 `Wt`、`s`，未给数值；标准版 2x 时间降采样后对应原视频约 1.07 s / 0.53 s |
| FFT HR 范围 | 0.75-2.5 Hz | rPPG-Toolbox 的复现推荐值（45-150 BPM） |
| 随机种子 | 42 | 原文未报告，固定以支持复跑 |
| 学习率调度 | 无 | 原文仅报告初始学习率，避免引入未披露策略 |
| 默认受试者划分 | 数值 ID 排序后 6:2:2 | 原文未给 ID 清单；另支持 `seeded_random` 作为单变量复现迭代 |
| 默认差分对齐 | 读取 129 帧产生 128 个差分 | 原文未说明尾部；另支持读取 128 帧并补零的 `zero_tail` 口径 |

当前 Apple M2 主机仅有 8 GB 统一内存，PyTorch 构建未启用 MPS；直接使用物理 batch 4 会产生严重换页长尾。UBFC 正式训练因此保持论文的有效 `batch_size: 4`，但以 `micro_batch_size: 1` 累积 4 次梯度后执行一次 Adam 更新。除 BatchNorm 的批维统计外，其梯度平均与一次四样本更新一致；该硬件执行差异会写入 resolved config 和实验报告，不冒充作者的物理 batch 4 环境。

## 3. 无法从论文唯一复原的部分

1. CCA 的精确通道表、patch/stride、Q/K/V 投影尺寸与级联连接细节。
2. SWTA 窗口长度、移位步长、位置编码与 attention mask 细节。
3. TASA gating CNN 的层数、核大小、门控初始化和边界处理。
4. 结构化剪枝比例、重要性准则、QAT 位宽/observer/backend 与微调轮数。
5. UBFC 与 MMPD 的确切受试者 ID 划分、随机种子、是否使用重叠片段。
6. HR 后处理窗口、频带、FFT 点数、是否使用标签文件中的直接 HR 行。
7. 表中 FLOPs 的计数规则，以及 latency 的预热、线程、精度与统计量。
8. MMPD 是完整 320x240 版本还是 80x60 mini 版本；当前实现不会把两者结果混称。

因此，目标是复现方法趋势、训练链路和可部署性证据；若要宣称重现论文表 1-3 的数值，需要作者源码、原始 split 清单、checkpoint 和 XR2 部署工程。

特别地，主论文在“Comprehensive Lightweight Optimization”中只用一句话声明采用 QAT 和结构化剪枝，没有报告任何可执行设置。RePhys 参考论文仅包含 LGTSM、PConv 和 MobileOne 式结构重参数化，也没有 QAT 或剪枝流程，因而不能作为这两项缺失参数的来源。表 3 中 Light-rPPGViT 的 `0.70 M / 6.93 G / 34 ms` 又与 RePhys 表 V 的对应数值完全相同；本工程将其保留为论文目标，而不会把当前通过宽度反推得到的近似复杂度冒充为“QAT/剪枝已复现”。最终完成门禁要求另行生成 `outputs/optimization/light_rppgvit.json`，其中必须记录并实测重参数等价性、QAT 设置与结构化剪枝设置及其验证集/测试集影响。

受控优化入口为 `scripts/optimize_light.py`。默认推断设置是 PyTorch `x86` 8-bit QAT（激活 per-tensor、权重 per-output-channel）以及逐卷积/线性层 20% 的 L1 输出通道掩码，各自从同一浮点最佳检查点独立微调 5 个 epoch。由于掩码没有进行物理通道压缩，脚本只报告结构化稀疏率与精度，不将稀疏模型的桌面耗时解释为部署加速。所有这些设置均在结果中标记为 `inferred`。

## 4. 数据与划分

### UBFC-rPPG

使用作者公开 Drive 中的 `DATASET_2`。文件夹实际含 42 个受试者目录。官方 Drive 配额受限时，可使用 Kaggle version 1 镜像，但必须先验证其 84 个文件严格对应相同 42 个受试者、服务端文件大小完整，并保留来源记录；已验证 subject1 标签与官方副本 SHA-256 相同。按受试者数字 ID 排序后：前 60% 训练、中间 20% 验证、最后 20% 测试。此做法比依赖文件系统 `glob` 顺序更可复核。subject27 允许共享数据但禁止在 PPT、网站、报告或论文中展示其人脸图像，因此所有可视化导出必须排除该受试者。

### MMPD

官网要求签署 release agreement，并由教师用机构邮箱申请。完整集约 370 GB，mini 版约 48 GB。本机剩余空间不足以安全容纳完整集和缓存，默认配置面向 mini 版并将缓存尺寸设为 80；训练时再升采样到 128。mini 结果必须单独标注，不能与论文可能使用的完整集直接对等。

## 5. 指标口径

模型输出与标签均是差分波形。评估时先累加、线性去趋势并做 0.75-2.5 Hz 一阶 Butterworth 带通；每个原始视频的所有非重叠 clip 按起点拼接，然后以 periodogram/FFT 取主频转 BPM。最终在“每视频 HR”上计算：

- `MAE = mean(abs(pred_hr - gt_hr))`
- `RMSE = sqrt(mean((pred_hr - gt_hr)^2))`
- `MAPE = 100 * mean(abs(pred_hr - gt_hr) / gt_hr)`
- `Pearson = corr(pred_hr, gt_hr)`

同时输出波形 Pearson，作为信号重建诊断，但不拿它替代论文的 HR 指标。

## 6. 部署验证边界

本机可验证参数量、近似 FLOPs、TorchScript 导出、训练态/部署态重参数等价性和 Mac CPU/MPS 延迟。Snapdragon XR2 的 34 ms 与增量 0.16 W 只能通过 Android systrace 和 Qualcomm Trepn Profiler 在真实硬件上复测；桌面计时或估算不能替代该证据。

### 对比基线口径

TSCAN、PhysNet、DeepPhys、PhysFormer 和 EfficientPhys 直接使用官方 rPPG-Toolbox 实现。配置生成器保留各模型官方模板中的预处理、输入排列、片段长度、训练轮数和学习率；只统一为论文的 batch size 4、同数据集受试者级 6:2:2、验证集选最佳 epoch 和整段 FFT 指标。为避免文件系统顺序改变划分，UBFC 和 MMPD loader 均按数值受试者 ID 排序。MMPD mini 使用全部已标注条件，不沿用某个跨数据集示例配置中的条件筛选。

工具箱局部补丁仅缩小可选依赖导入、开放 dataloader worker 数配置并固定目录排序，不修改五个模型、损失函数、预处理数学或指标算法。补丁保存在 `patches/rppg-toolbox-minimal-imports.patch`。

## 7. 论文报告值（仅作为复现目标）

| 数据集 | 模型 | MAE | RMSE | MAPE (%) | Pearson |
|---|---|---:|---:|---:|---:|
| UBFC-rPPG | rPPGViT | 0.27 | 0.30 | 0.28 | 0.99 |
| UBFC-rPPG | Light-rPPGViT | 0.46 | 0.88 | 0.48 | 0.98 |
| MMPD | rPPGViT | 0.65 | 1.52 | 0.81 | 0.99 |
| MMPD | Light-rPPGViT | 0.68 | 1.55 | 1.31 | 0.98 |

| 模型 | 参数量 | FLOPs | XR2 latency |
|---|---:|---:|---:|
| rPPGViT | 1.12 M | 20.12 G | 40 ms |
| Light-rPPGViT | 0.70 M | 6.93 G | 34 ms |

这些数值来自论文，未标作本工程实测；本机仿真结果见 `docs/local_simulation.md`。
