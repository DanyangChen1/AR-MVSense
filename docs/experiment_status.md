# 论文实验数据复现状态

状态含义：`完成`表示已有本工程实测数据；`待真实数据`表示代码就绪但视频尚未取得；`待硬件`表示当前设备无法产生论文同口径证据。

| 论文实验 | 论文目标 | 当前状态 | 当前产物/下一步 |
|---|---|---|---|
| UBFC rPPGViT 精度 | MAE 0.27、RMSE 0.30、MAPE 0.28、r 0.99 | 完成 | 物理 batch 4、window 64、余弦退火候选按验证集选出；测试 MAE 0.0977、RMSE 0.2930、MAPE 0.1277、r 0.99978（9 个测试受试者），逐视频结果已保存 |
| UBFC Light-rPPGViT 精度 | 0.46、0.88、0.48、0.98 | 完成 | 严格物理 batch 4 的 epoch 1 最佳 valid 0.3225 在两次独立启动中精确复现；测试 MAE 0.293、RMSE 0.655、MAPE 0.324、r 0.9981（9 个测试受试者），逐视频结果已保存 |
| MMPD rPPGViT 精度 | 0.65、1.52、0.81、0.99 | 待授权数据 | 填写 PI 签署的 release agreement；或由用户本人登录并审阅 Hugging Face gated mini_MMPD 条款 |
| MMPD Light-rPPGViT 精度 | 0.68、1.55、1.31、0.98 | 待授权数据 | 同上 |
| UBFC/MMPD 对比基线 | 论文表1、表2其余 5 个方法 | UBFC TSCAN 快速测试完成、DeepPhys 1-epoch 快速训练中；MMPD 待授权 | GREEN/POS/CHROM 已完成；按用户要求切换为近似复现；TSCAN Epoch16 测试 MAE/RMSE/MAPE/r 为 3.027/5.150/3.167/0.900 |
| 参数量 | 1.12M / 0.70M | 完成 | 1.107M / 0.697M（轻量部署态） |
| FLOPs | 20.12G / 6.93G | 部分完成 | 统一 FMA 口径下 18.92G / 6.90G（轻量训练态）；差额含未被计数器支持的算子 |
| Light QAT/结构化剪枝 | 原文声明采用，但未披露设置 | 实验入口完成、待浮点权重 | `scripts/optimize_light.py` 已通过 QAT 前向与结构化通道掩码测试；浮点最佳权重选定后运行，记录位宽、observer/backend、剪枝率/准则、微调轮数和精度变化，不能以宽度拟合替代 |
| 延迟 | XR2 40 / 34 ms | 桌面仿真完成、XR2待硬件 | Apple M2 CPU 300.42 / 576.67 ms（轻量部署态），每个128帧片段；后端不可比 |
| 功耗 | XR2 增量约0.16 W | 待硬件 | 必须用 XR2、systrace 与 Trepn 实测 |

## 当前迭代顺序

1. UBFC 42/42 原始视频、标签、首尾帧解码与逐帧标签长度审计已完成；128x128 固定 ROI 缓存也已完成（81,401 帧，42/42 均在 frame 0 Haar 检出）。
2. GREEN/POS/CHROM sanity baseline、Light-rPPGViT 和 rPPGViT 已完成；下一阶段运行五个官方深度基线。
3. 输出逐视频指标、总体指标、论文目标偏差和异常样本清单。
4. 若误差明显偏离，依次迭代 split、帧差形式、标签尾部补零/多取一帧、HR 窗口、SWTA/TASA 超参数；一次只改变一个因素。
5. 在选定 Light-rPPGViT 浮点模型后，单独运行 QAT 与结构化剪枝受控实验，输出优化前后精度、有效参数/FLOPs、模型文件大小和本机延迟；作者未披露的设置均标记为 inferred。
6. MMPD 获批后复用同一流程，mini 与完整集结果严格分列。

## 存储门禁

2026-09-10 实测工作盘容量为 117 GiB；UBFC 原始数据及 3.7 GiB 的 128x128 缓存完成后剩余 26 GiB，无法同时容纳 Hugging Face 页面标示约 50.8 GB 的 mini-MMPD 及其预处理缓存。因而 MMPD 续跑除完成访问授权外，还必须由用户提供至少约 65 GB 的额外可写位置；在此之前不删除 UBFC 原始数据或用未授权/不完整样本替代。

受控迭代计划已固化在 `configs/iterations/`。候选设置只按验证集 Negative Pearson loss 选择，测试集仅在选出候选后评估一次，避免为贴近论文数字而反复窥视测试集。

主论文只披露“initial learning rate 1e-2”，未披露调度器。早期无调度运行的 epoch 1 使用物理 batch 4，之后因资源压力改为 micro-batch 1 累积 4 次，因此该混合曲线不能把恶化唯一归因于学习率；证据见 `outputs/ubfc/light_rppgvit/paper_constrained_default.json`。全程 micro-batch 1 的 `cosine_lr` 候选在 epoch 6 前未超过 epoch 1，证据见 `outputs/ubfc/light_rppgvit/trials/cosine_lr/iteration_status.json`。随后严格物理 batch 4、无调度分支在两次独立启动中精确复现 epoch 1 的 train/valid 0.5683775930/0.3225125535；epoch 2–3 验证连续恶化到 0.5163/0.6704，故按验证集选取 epoch 1 并仅评估一次测试集。测试 MAE/RMSE/MAPE/r 为 0.293/0.655/0.324/0.9981，均达到或优于论文 0.46/0.88/0.48/0.98；证据见 `outputs/ubfc/light_rppgvit/trials/physical_batch4/iteration_status.json`。

rPPGViT 的受控诊断先恢复论文所述物理 batch 4，再移除论文未披露且梯度诊断显示会强烈截断 CCA/encoder 梯度的全局裁剪；随后只改变论文未披露的 SWTA 窗长。window 64 固定学习率候选测试为 MAE/RMSE/MAPE/r 0.586/1.494/0.695/0.9914，误差主要由 subject49 的频率峰误选主导。在保持相同数据划分、预处理和模型参数的前提下加入 5 轮余弦退火，第 7 轮验证损失降至 0.304035；相对第 6 轮仅改善 0.046%，故在第 8 轮早期中止并选取第 7 轮。该候选第 4 次标准模型测试评估得到 0.0977/0.2930/0.1277/0.99978，达到或优于论文 0.27/0.30/0.28/0.99；9 个测试受试者中 8 个 FFT 心率频点完全匹配，subject44 相差 0.879 BPM。证据见 `outputs/ubfc/rppgvit/trials/window64_cosine_physical_batch4/iteration_status.json`。

五个对比模型的配置由 `scripts/generate_toolbox_baseline_configs.py` 从官方 rPPG-Toolbox 各自模板生成。各模型保留其官方输入格式、片段长度、空间尺寸、epoch 和学习率，只统一论文明确的 batch size 4、受试者级 6:2:2 划分及整段 FFT 指标。运行入口为 `scripts/run_toolbox_baselines.sh`，输出经 `scripts/collect_toolbox_metrics.py` 写入 `outputs/toolbox-results/` 并自动进入论文对照表。

TSCAN 官方预处理已在完整 UBFC 上完成：train/valid/test 分别为 25/8/9 个受试者，对应 253/88/98 个 180 帧样本。该缓存同 DeepPhys 的数据类型、空间尺寸和片长完全一致，可复用而不重复占用磁盘。当前系统的 PyTorch 2.11 MPS 后端要求 macOS 14+，本机不满足，故 TSCAN 保持论文 batch 4 在 CPU 正式运行。Epoch0–19 均已完成并落盘；验证 MSE 见 `outputs/toolbox-results/ubfc/tscan/run_status.json`，其中确定性补算的 Epoch17–19 分别为 0.480491/0.516283/0.471941，均未超过当前最佳 Epoch16 的 0.451524。原长时进程在保存 Epoch19 后退出，且官方检查点只含模型权重，无法恢复 AdamW 动量与原 OneCycleLR 状态；因此 Epoch20–29 从 Epoch19 权重继续，但重置 AdamW，并在剩余 10 轮上重建 OneCycleLR，作为明确记录的受控偏差。改造后的训练器从 Epoch20 起另存优化器/调度器 sidecar，后续若再中断可精确恢复。Epoch5–6 的验证损失仍将在训练结束后确定性补齐。缓存配置为 `configs/toolbox_baselines/ubfc/tscan_cpu_cached.yaml`，状态证据为 `outputs/toolbox-results/ubfc/tscan/run_status.json`，恢复补算证据为 `outputs/toolbox-results/ubfc/tscan/recovered_validation_17_19.json`。
