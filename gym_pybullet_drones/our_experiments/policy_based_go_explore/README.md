# Policy-Based Go-Explore（目标条件策略版）

基于 **Policy-Based Go-Explore** 算法的单智能体无人机目标覆盖探索实验。

## 算法概述

与原始 Go-Explore 的核心区别：**用目标条件策略替代动作回放**完成 return 阶段。

每轮迭代执行：
1. **选择**：从 Archive 中按分数比例采样一个目标 Cell
2. **Return 阶段**：策略网络接收目标 Cell 的 XY 坐标，学习导航到目标位置
3. **Explore 阶段**：到达目标 Cell 后，策略继续探索，发现新 Cell
4. **PPO 更新**：用 GAE 计算优势函数，更新策略和价值网络
5. **Archive 更新**：将新发现的 Cell 存入 Archive

### 对随机环境的鲁棒性

| 机制 | 说明 |
|---|---|
| **目标条件策略** | 策略看到目标坐标，学习导航能力而非记忆固定路径 |
| **子目标轨迹跟踪** | 长距离路径分解为 1m 间隔的子目标，提供密集引导奖励 |
| **Phase 标识** | 策略知道当前处于 return/explore，可采用不同行为模式 |

## Cell 设计

Cell Key = `(grid_x, grid_y, n_captured_targets)`

- XY 位置离散化为 0.5m × 0.5m 网格
- **加入领域知识**：已捕获目标数编码进 Cell Key
- 同一位置、不同覆盖进度被视为不同的 Cell

## 环境

- 使用 `OurSingleRLAviary`（单智能体，Gymnasium API）
- 动作空间：VEL（速度控制，2维）
- 环境包装器 `GoExploreEnvWrapper` 注入 goal/phase 信息

## 训练命令

```powershell
# 默认参数
python -m gym_pybullet_drones.our_experiments.policy_based_go_explore.train

# 自定义参数
python -m gym_pybullet_drones.our_experiments.policy_based_go_explore.train \
    --total_iterations 5000 \
    --n_envs 4 \
    --return_max_steps 200 \
    --explore_max_steps 300
```

## 主要参数（`config.py`）

### 环境参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `arena_size` | 10.0 | 场地边长（米） |
| `target_count` | 18 | 目标数量 |
| `obstacle_count` | 6 | 障碍物数量 |
| `ctrl_freq` | 60 | 控制频率（Hz） |

### Go-Explore 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `return_max_steps` | 200 | Return 阶段最大步数 |
| `explore_max_steps` | 300 | Explore 阶段最大步数 |
| `cell_size` | 0.5 | 网格单元边长（米） |
| `max_cells` | 10000 | Archive 最大容量 |
| `n_envs` | 4 | 并行环境数 |

### 子目标跟踪参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `sub_goal_spacing` | 1.0 | 子目标间距（米） |
| `sub_goal_reach_thresh` | 0.5 | 子目标到达阈值（米） |
| `sub_goal_reward` | 1.0 | 到达子目标的额外奖励 |
| `potential_reward_scale` | 0.1 | 势场引导奖励系数 |

### 网络与 PPO 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `obs_embed_dim` | 128 | 观测嵌入维度 |
| `gru_hidden` | 128 | GRU 隐藏层大小 |
| `goal_embed_dim` | 16 | 目标嵌入维度 |
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `clip_eps` | 0.2 | PPO 裁剪系数 |
| `n_epochs` | 4 | 每次更新的 epoch 数 |
| `batch_size` | 256 | 小批量大小 |
| `total_iterations` | 5000 | 总训练轮数 |
| `device` | cpu | 训练设备（cpu/cuda） |

## 输出文件

```
results/go_explore/
├── archive.json          # 发现的 Cell 存档
├── model_iter{N}.pt      # 定期保存的模型
└── model_final.pt        # 最终模型
```

## 文件结构

| 文件 | 说明 |
|---|---|
| `train.py` | 主训练循环 |
| `config.py` | 超参数配置 |
| `archive.py` | Cell Archive 实现 |
| `goal_conditioned_env.py` | 环境包装器（注入 goal/phase） |
| `networks.py` | 目标条件 Actor-Critic 网络 |
| `ppo.py` | PPO 训练器（带 valid-mask） |
| `rollout_buffer.py` | GAE Rollout Buffer（带 goal/phase） |
| `trajectory_tracker.py` | 子目标轨迹跟踪器 |

## 与原始 Go-Explore 的区别

| 特性 | 原始 Go-Explore | Policy-Based 版 |
|---|---|---|
| Return 方式 | 动作回放 | 目标条件策略 |
| 训练阶段 | Phase 1 探索 + Phase 2 鲁棒化 | 一步完成 |
| 抗随机性 | 依赖确定性环境 | 策略可适应随机性 |
| 网络输入 | 仅观测 | 观测 + 目标 + 阶段标识 |
| 奖励设计 | 环境原始奖励 | Return 阶段用子目标引导奖励 |
