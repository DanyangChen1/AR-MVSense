# 近似复现边界说明

本目录交付的是论文 *Lightweight rPPG Framework for Real-Time Physiological Monitoring on AR Devices* 的近似复现与工程仿真，不是作者原始代码、原始权重或逐位一致的官方结果。

## 可以如何使用

- 用于验证论文流程是否可运行、指标量级是否接近、模型是否能在本机训练与推理。
- 可缩短训练轮数、采用验证集提前停止、使用作者未披露设置的合理默认值。
- 为快速演示可以建立另行标记为 `leaky_demo` 的泄漏实验，但该结果只能证明拟合能力，不能表述为独立测试集泛化性能。
- 当前主结果仍采用受试者级 6:2:2 无交叉划分；没有为了贴近论文数字而主动引入数据泄漏。

## 不能据此声称

- 不能声称已获得作者源码、官方权重或完全复现全部论文数字。
- 不能把含训练/验证/测试受试者重叠的结果写成无泄漏测试结果。
- 不能把 Apple M2 的延迟或功耗换算成 Snapdragon XR2 实机测量。
- 不能在未接受 MMPD 数据条款时绕过访问门禁；不能展示 UBFC subject27 的人脸图像。

## 当前近似项

1. 论文未完整披露网络细节，`rPPGViT` 与 `Light-rPPGViT` 是结合论文正文、PhysMViT 和 RePhys 的透明补全实现。
2. TSCAN 使用官方 rPPG-Toolbox 和完整 UBFC-rPPG。原 30 轮训练在 Epoch19 后中断，快速交付采用验证集最优 Epoch16；Epoch20 的未完成批次不计入模型。
3. TSCAN 的测试只使用此前未参与训练和选模的 9 名受试者；该项没有数据泄漏。
4. MMPD/mini-MMPD 仍受数据授权和磁盘容量限制；工程保留下载检查、配置与续跑入口，但不会伪造其结果。
5. XR2 延迟和功耗没有对应硬件，故只保留本机仿真值并显式标记为平台不匹配。

## 交付物位置

- 主程序：`src/`、`scripts/`、`configs/`
- 参考基线程序：`third_party/rPPG-Toolbox/`
- 模型权重：`outputs/ubfc/**/best.pt` 与 `outputs/toolbox/ubfc/**/PreTrainedModels/*.pth`
- 机器可读指标：`outputs/**/metrics.json`、`outputs/comparison/`
- 运行说明：`README.md`
- 详细实验状态：`docs/experiment_status.md`

所有对外引用应同时保留本文件，并明确使用“近似复现”“本地仿真”或“受控偏差”字样。
