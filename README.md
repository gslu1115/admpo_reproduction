# ADMPO：Figure 2 与 Figure 4 复现

本项目复现论文 *Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning* 的两个主文实验：Figure 2 的长期动力学预测误差，以及 Figure 4 的模型误差—不确定性相关性。

> 当前状态：代码与实验协议已实现，环境门、smoke、pilot 和五种子正式结果将按顺序写入本仓库。正式结果生成前，README 不预填论文数值，也不把 smoke 输出当作复现结果。

## 复现范围

- Figure 2：在 `hopper-medium-v2`、`hopper-medium-replay-v2`、`walker2d-medium-v2`、`walker2d-medium-replay-v2` 上比较 ADM、ensemble dynamics 和 bootstrapping RNN 的 100 步累计预测误差。
- Figure 4：在两个 medium-replay 任务上比较 ADM 与 ensemble 的模型误差和不确定性相关性，采样策略包括随机动作、ADMPO-OFF 学习策略和 BC 行为策略代理。
- 固定种子：`0, 1, 2, 3, 4`。
- 目标：复现主要趋势和结论，不要求与论文像素级或数值级一致。

本项目不复现论文的在线学习实验、D4RL/NeoRL 完整性能表、附录消融实验或其他基线。

## 环境与硬件

- WSL：`Ubuntu-22.04-Recovery`
- Conda 环境：`/home/lul/miniconda3/envs/admpo`
- Python 3.9
- PyTorch 2.0.1+cu118
- Gym 0.23.1、D4RL 1.1、mujoco-py 2.1.2.14
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8 GB

本项目不会修改 `.bashrc`。每次进入新的 shell 后执行：

```bash
conda activate admpo
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/home/lul/.mujoco/mujoco210/bin:/usr/lib/wsl/lib"
export MUJOCO_GL=egl
export D4RL_SUPPRESS_IMPORT_ERROR=1
pip install -e . --no-deps
```

克隆仓库时还需初始化固定版本的官方代码：

```bash
git submodule update --init --recursive
```

## 数据集

数据集由 D4RL 放置在 `~/.d4rl/`，不会进入 Git。四个数据文件的路径、字节数和 SHA256 会记录在 `results/manifests/figure2-full.json`。

## 运行方法

先检查 GPU、数据和 MuJoCo oracle：

```bash
python -m admpo_repro check --phase full
```

低成本端到端检查：

```bash
python -m admpo_repro run figure2 --phase smoke --seeds 0 --resume
python -m admpo_repro run figure4 --phase smoke --seeds 0 --resume
```

完整 pilot：

```bash
python -m admpo_repro run figure2 --phase pilot --seeds 0 --resume
python -m admpo_repro run figure4 --phase pilot --seeds 0 --resume
```

五种子正式实验：

```bash
python -m admpo_repro run figure2 --phase full --seeds 0 1 2 3 4 --resume
python -m admpo_repro run figure4 --phase full --seeds 0 1 2 3 4 --resume
```

从已有 CSV 重绘图片：

```bash
python -m admpo_repro plot figure2
python -m admpo_repro plot figure4
```

## 实验口径

### Figure 2

- ADM 与 bootstrapping RNN 使用 3 层 GRU、隐藏维度 200 和 4 个残差块；最大历史长度为 5。
- Ensemble 使用 7 个成员和验证误差最小的 5 个 elite，主干为 4×200 Swish MLP。
- BC 使用 2×256 ReLU MLP、动作 MSE、学习率 `3e-4`、batch size 256。
- 每个模型从自己的预测状态调用同一个 BC 策略，并把该动作同时施加于配对的 MuJoCo oracle。
- 状态误差是状态维度 MSE；正式图报告五种子均值与 SEM，纵轴采用对数尺度。

### Figure 4

- ADMPO-OFF：`m=5`、`H=5`、`β=0.1`、real ratio `0.5`、5000×1000 次 SAC 更新。
- Hopper 的 `alpha=0.05`，Walker2d 的 `alpha=0.1`，均关闭自动温度调节。
- 主不确定性指标为不同回溯预测或 elite 预测均值之间的标准差：`sqrt(mean_state(var_hypothesis(mean)))`。
- `total_uncertainty` 额外加入模型输出的 aleatoric 方差，只作为诊断列，不用于主图标题中的 Pearson 相关系数。
- 每种策略、每个模型、每个种子收集 500 个有效点；另保存直接来自 D4RL 的状态—动作诊断，不混入主图。

## 复现结果

正式运行完成后，本节将自动/人工据实补充以下内容：

- `results/figure2/figure2.png` 与 `figure2.pdf`
- `results/figure2/summary.csv`
- `results/figure4/figure4.png` 与 `figure4.pdf`
- `results/figure4/correlations.csv`
- 每个任务、模型、种子的结论对照及运行耗时

在五种子结果完整前，不对“复现成功”作结论。

## 目录与产物策略

- `configs/`：正式参数与 smoke 覆盖参数。
- `src/admpo_repro/`：唯一保留的训练、评估和绘图实现。
- `tests/`：数据边界、模型形状、指标公式和绘图测试。
- `results/`：可提交的逐种子汇总、统计表、PNG/PDF 和 manifest。
- `runs/`：Git 忽略的检查点、训练记录和本地中间产物。
- `vendor/ADMPO/`：固定提交的官方 Git submodule。

## 偏差与已知限制

- 原论文仓库没有公开 Figure 2/4 的完整训练和绘图脚本，因此 bootstrapping RNN、ensemble 和评估器参考同作者后续 ADM-v2 实现重建。
- D4RL 没有提供 medium/replay 数据集实际行为策略的可调用模型，因此主实验使用 BC 作为行为策略代理，并保留 dataset-direct 诊断。
- Hopper/Walker 的 Gym observation 会把速度裁剪到 `[-10, 10]`；被裁剪的 observation 不再包含可精确恢复的完整 MuJoCo 状态。Oracle 环境门只在速度未触及裁剪边界的连续转移上验证，并在 manifest 记录排除数量。
- 论文不公开原始种子，本项目预先固定为 `0–4`。
- 8 GB 显存下只串行运行一个训练进程；完整五种子实验预计需要约 2–3 周，最终以 pilot 实测时间为准。

第三方版本与用途见 [THIRD_PARTY.md](THIRD_PARTY.md)。
