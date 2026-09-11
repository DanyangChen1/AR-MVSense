# 网络结构逐图审计

本审计把主论文 Fig. 2-3、PhysMViT Fig. 1-2 和 RePhys Fig. 1-4 与当前实现逐项核对。目标是区分原文明示结构、参考论文补全和仅为匹配成本而作的推断，避免把“能运行”误写成“作者源码复现”。

| 结构证据 | 原文明示 | 当前实现 | 证据状态 |
|---|---|---|---|
| rPPGViT 总体 | convolutional encoder -> 3x CCA -> 4x SWTA -> upsample/head | 同序及同深度 | matched |
| CCA | 高低分辨率 Q/K/V cross-scale attention，learnable residual gamma，Add&Norm + S-T FFN | pooled adjacent scales、MHA、gamma、FFN/residual | matched-with-inferred tokenization |
| SWTA | 局部窗口 MHA，相邻层 shift，Add&Norm + S-T FFN | 交替 0/半窗 shift、masked window MHA、双残差 | matched-with-inferred window size |
| Light 总体 | light encoder -> 3x re-parameterized CCA -> 4x TASA -> upsample/head | 同序及同深度 | matched |
| re-parameterized CCA | 3x3/1x1/identity 训练分支可合并；PConv、SE、Add&Norm、S-T FFN | 可验证等价的 RepConv、1/4 PConv、SE、双残差归一化、时序聚合 FFN | matched-with-inferred channel layout |
| TASA | Gating CNN 产生自适应权重；前后帧 temporal shift；3x3 Conv、SE、Add&Norm、S-T FFN | 输入驱动 depthwise Conv1d gate、公式 `X + alpha*(X[t-1]-X[t+1])`、3x3 depthwise Conv、SE、双残差归一化 | matched-with-inferred gate CNN |
| PConv 比例 | 主论文未给；RePhys 实验设置明确 first 1/4 | `pconv_ratio=0.25` | inferred-from-reference |
| QAT/结构化剪枝 | 主论文宣称使用，但未给位宽、observer、后端、比例或微调方案；RePhys 也未包含二者 | 当前宽度拟合不算复现；已加入独立实验产物与完成门禁 | pending-controlled-implementation |

参考论文与主论文存在不能混用的设置：RePhys 明确使用 72x72 输入和 MSE loss，而主论文使用 Negative Pearson loss，并在 AR 成本实验明确给出 128x128 输入。因此当前主实验遵循主论文的 128x128/Negative Pearson，只从 RePhys 继承其明确披露而主论文未给出的 3 stages、PConv 前 1/4、dropout 0.2 和 50 iterations/epochs 候选。PhysMViT 的相对帧差公式也只作为敏感性实验；主论文明确采用简单帧差除以序列标准差。

成本交叉验证采用由五个官方基线确认的“一次 FMA 计一次”口径。当前轻量训练态为 699,715 参数、6.901G 已识别 FMA，分别相对论文 0.70M/6.93G 偏差 -0.04%/-0.41%；部署态重参数合并后的数值为 696,838/6.780G。逐元素、归一化、激活和线性上采样未被 FVCore 完整计入，因此 FLOPs 仍标记为已识别下界。

仍无法从三篇论文确定：具体卷积通道表、CCA token 尺寸、SWTA window/shift、Gating CNN 核与激活、SE reduction、FFN 的精确空间作用域、剪枝率和量化配置。真实数据阶段将只用验证集对这些候选做受控选择，测试集在选定配置后评估一次。
