# ADMPO：Figure 2 与 Figure 4 部分复现

本项目复现论文 *Any-step Dynamics Model Improves Future Predictions for Online and Offline Reinforcement Learning* 的 Figure 2（长期动力学预测误差）与 Figure 4（模型误差—不确定性相关性）。目标是复现主要趋势，不要求与论文数值或图像像素级一致。

> 当前状态：Figure 2 已按重新核对后的离线轨迹评估口径完成代码修正和 smoke 验证，正式 RTX 5090 训练尚未开始。此前使用 BC 与 MuJoCo oracle 的 Figure 2 输出属于旧协议，不会进入新的统计结果。

## 当前复现范围

### Figure 2

- 数据集：`hopper-medium-replay-v2`、`walker2d-medium-replay-v2`。
- 模型：ADM、Bootstrapping RNN、7成员/5 elite Ensemble。
- 模型种子：`0, 1, 2`。
- 按完整轨迹划分训练/验证/测试集，并使三部分转移数量尽量接近 `80%/10%/10%`。
- 三种模型和三个模型种子共用同一固定划分（`split seed=202405`）。
- 使用训练集统计量进行状态和动作归一化。
- 最终图报告三种子均值与 SEM；由于计算预算限制，这属于三种子部分复现。

### Figure 4

Figure 4 仍保留此前的限时实现，最终清理与结果说明将在 Figure 2 正式训练完成后处理。本轮代码修改没有把 Figure 2 的旧评估方式继续用于正式结果。

本项目不复现在线学习实验、D4RL/NeoRL 完整性能表、附录消融实验或其他基线。

## Figure 2 实验口径

- ADM 与 Bootstrapping RNN 使用3层GRU、隐藏维度200、4个残差块，最大历史长度 `m=5`。
- ADM学习从起始状态和动作子序列预测任意步之后的状态增量；RNN使用逐状态—动作历史预测下一状态。
- 两个序列模型每个epoch获得相同数量的优化器更新，batch size均为256。
- Ensemble使用7个独立成员、验证误差最低的5个elite，主干为4×200 Swish MLP，并使用独立bootstrap采样。
- 10%验证轨迹只用于早停和elite选择；10%测试轨迹不参与训练或模型选择。
- 每次评估从测试轨迹固定抽取100个窗口，滚动100步。
- 每一步动作直接采用对应测试轨迹中记录的真实动作。
- 模型状态递归使用自身预测均值，真实目标为同一轨迹中对应的未来状态。
- Ensemble预测取5个elite预测均值的平均；不采样成员，也不加入aleatoric噪声。
- 不训练BC，不调用MuJoCo oracle，不根据模型预测的终止状态删除轨迹。
- 指标为原始状态维度MSE，异常或溢出误差截断到float32最大值。

## 环境

### 本地验证环境

- WSL：`Ubuntu-22.04-Recovery`
- Python 3.9
- PyTorch 2.0.1+cu118
- GPU：RTX 4060 Laptop 8GB

### RTX 5090 正式环境

- Ubuntu 22.04
- Python 3.10
- PyTorch 2.8.0+cu128（租赁平台镜像）
- RTX 5090 32GB

Figure 2直接读取缓存的D4RL HDF5，不导入Gym或MuJoCo。PyTorch版本变化和完整包版本会记录在运行manifest中。

安装项目：

```bash
python -m pip install -e . --no-deps
```

固定官方子模块：

```bash
git submodule update --init --recursive
```

## 数据集

数据文件默认位于 `~/.d4rl/datasets/`，不会进入Git。运行manifest记录文件路径、字节数和SHA256。

正式运行前检查缓存、GPU和轨迹划分：

```bash
python -m admpo_repro check --phase full
```

## 运行命令

低成本全链路检查：

```bash
python -m admpo_repro run figure2 --phase smoke --seeds 0 --workers 1
```

正式pilot（结果可被full复用）：

```bash
python -m admpo_repro run figure2 --phase pilot --seeds 0 --workers 1 --resume
```

RTX 5090三种子正式实验：

```bash
python -m admpo_repro run figure2 --phase full --seeds 0 1 2 --workers 0 --resume
```

`--workers 0`表示按显存自动选择。32GB GPU默认并行6个ADM/RNN任务或3个Ensemble任务；可以根据pilot实测利用率显式调整。并行只改变作业调度，不改变batch size、随机种子或模型超参数。

训练中的每个模型都会更新 `*.progress.json`，其中包含当前epoch、早停计数和验证误差；断开VS Code或SSH不会影响持久会话中的训练。

从CSV重新绘图：

```bash
python -m admpo_repro plot figure2 --phase full
```

## 结果与检查点位置

- 正式检查点：`runs/figure2/trajectory_80_10_10_3seeds/formal/`（Git忽略）。
- 正式逐种子CSV：`results/figure2/trajectory_80_10_10_3seeds/full/per_seed/`。
- 正式图片与汇总：`results/figure2/trajectory_80_10_10_3seeds/full/`。
- 运行manifest：`results/manifests/figure2-trajectory_80_10_10_3seeds-full.json`。
- smoke与pilot结果分别放入独立目录；旧版结果保留在旧路径，不会被新绘图器读取。

正式运行完成后，本节将据实补充PNG/PDF、逐任务结果表、step 100对比、实测耗时以及与论文结论是否一致。若趋势不成立，不筛选种子或调整图像掩盖结果。

## 已知限制

- 论文及官方仓库未公开Figure 2的完整训练与绘图脚本，RNN、Ensemble和部分训练细节参考同作者后续ADM-v2及OfflineRL-Kit重建。
- 论文没有完整披露数据轨迹划分、测试窗口和随机种子；本项目固定并在manifest中记录这些选择。
- 三个种子的误差估计弱于原计划的五种子，因此同时报告标准差和SEM，并明确标注为部分复现。
- PyTorch 2.8.0是RTX 5090兼容性要求带来的环境偏差；本地smoke仍使用PyTorch 2.0.1验证接口兼容性。

第三方版本和借鉴关系见 [THIRD_PARTY.md](THIRD_PARTY.md)。
