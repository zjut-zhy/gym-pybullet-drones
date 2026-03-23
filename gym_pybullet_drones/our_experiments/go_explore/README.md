# Original Go-Explore（动作回放版）

基于 **Go-Explore** 算法的单智能体无人机目标覆盖探索实验。

## 算法概述

Go-Explore 分为两个阶段：

### Phase 1：探索（`train.py`）

每轮迭代执行：
1. **选择**：从 Archive 中按分数比例采样一个目标 Cell
2. **回放**：`reset()` 后按存储的动作序列精确回放，到达目标 Cell
3. **探索**：在目标 Cell 附近执行随机动作，发现新 Cell
4. **更新**：将新发现的 Cell 存入 Archive

### Phase 2：鲁棒化（`robustify.py`）

使用 **反向课程学习** 训练 PPO 策略：
1. 从 Archive 中取出最优轨迹
2. 初始：回放轨迹前缀，策略只负责最后几步
3. 逐步缩短回放前缀，策略负责越来越多的步骤
4. 最终：策略自主完成整个任务

## Cell 设计

Cell Key = `(grid_x, grid_y, n_captured_targets)`

- XY 位置离散化为 0.5m × 0.5m 网格
- **加入领域知识**：已捕获目标数编码进 Cell Key
- 同一位置、不同覆盖进度被视为不同的 Cell

## 环境

- 使用 `OurSingleRLAviary`（单智能体，Gymnasium API）
- 动作空间：VEL（速度控制，2维）
- 确定性环境：固定种子 → 动作回放精确复现

## 训练命令

### Phase 1：探索

```powershell
# 默认参数
python -m gym_pybullet_drones.our_experiments.go_explore.train

# 自定义参数
python -m gym_pybullet_drones.our_experiments.go_explore.train \
    --total_iterations 5000 \
    --n_envs 4 \
    --explore_steps 300 \
    --cell_size 0.5
```

### Phase 2：鲁棒化

```powershell
# 需要先完成 Phase 1 生成 archive.json
python -m gym_pybullet_drones.our_experiments.go_explore.robustify \
    --archive_path results/go_explore/archive.json \
    --total_iterations 3000 \
    --n_envs 4
```

## 主要参数

### Phase 1 参数（`config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_iterations` | 5000 | 总探索轮数 |
| `n_envs` | 4 | 并行环境数 |
| `explore_steps` | 300 | 每轮随机探索步数 |
| `cell_size` | 0.5 | 网格单元边长（米） |
| `max_cells` | 10000 | Archive 最大容量 |
| `arena_size` | 10.0 | 场地边长（米） |
| `target_count` | 18 | 目标数量 |
| `obstacle_count` | 6 | 障碍物数量 |

### Phase 2 参数（`robustify.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_iterations` | 3000 | PPO 训练轮数 |
| `policy_steps` | 300 | 每轮策略交互步数 |
| `curriculum_step` | 5 | 每次缩短的回放步数 |
| `curriculum_interval` | 50 | 每隔多少轮缩短一次 |
| `lr` | 3e-4 | 学习率 |
| `device` | cpu | 训练设备 |

## 输出文件

```
results/go_explore/
├── archive.json          # Phase 1 发现的 Cell 存档
results/go_explore_phase2/
├── model_iter{N}.pt      # Phase 2 定期保存的模型
└── model_final.pt        # Phase 2 最终模型
```

## 文件结构

| 文件 | 说明 |
|---|---|
| `train.py` | Phase 1 主训练循环 |
| `robustify.py` | Phase 2 鲁棒化训练 |
| `config.py` | Phase 1 超参数 |
| `archive.py` | Cell Archive 实现 |
| `networks.py` | Actor-Critic 网络（无目标条件） |
| `ppo.py` | PPO 训练器 |
| `rollout_buffer.py` | GAE Rollout Buffer |
