# 第三方来源与固定版本

本项目的复现代码是独立组织的实验实现。以下项目用于核对网络结构、训练口径与评估公式。

## ADMPO

- 仓库：https://github.com/HxLyn3/ADMPO
- 固定提交：`68c28b9bbf6b95a801ebb30ea47294ad2d8d9cb3`
- 用途：ADM、ADMPO-OFF、SAC、任务终止函数和论文超参数的主要依据。
- 形式：`vendor/ADMPO` Git submodule；其 MIT 许可证保留在 submodule 内。

## ADM-v2

- 仓库：https://github.com/LAMDA-RL/adm2
- 参考提交：`5690f37df2175615754190d0172e380c5a31e139`
- 用途：原论文仓库缺失的 bootstrapping RNN、ensemble、MuJoCo oracle、rollout MSE 与均值分歧不确定性实现口径。
- 形式：仅作实现参考，不作为 vendor 或运行依赖。

## OfflineRL-Kit

- 仓库：https://github.com/yihaosun1124/OfflineRL-Kit
- 参考提交：`951302eed019f047c490cfe05b23beb2cf29f714`
- 用途：7-model/5-elite ensemble 配置和 BC 训练设置。
- 形式：仅作实现参考，不作为 vendor 或运行依赖。

## D4RL 与 MuJoCo

- D4RL：https://github.com/Farama-Foundation/d4rl
- mujoco-py：https://github.com/openai/mujoco-py
- 数据集与 MuJoCo 二进制不进入本仓库；数据文件哈希写入运行 manifest。
