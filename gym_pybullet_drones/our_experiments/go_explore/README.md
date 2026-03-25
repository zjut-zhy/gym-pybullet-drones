# Go-Explore（树状归档 + 后向鲁棒化）

基于 **Go-Explore** 算法的单智能体无人机目标覆盖实验。

## 算法概述

完整流水线分为三个阶段：

### Phase 1：确定性探索与树状归档（`train.py`）

在**完全确定性**的环境中（所有环境实例共享相同种子），系统性地探索整个状态空间：

1. **选择**：从 Archive 中按分数比例采样一个 Cell
2. **传送**：`env.restore_snapshot(cell.snapshot)` 直接传送到目标 Cell
3. **探索**：执行随机动作，发现新 Cell
4. **归档**：将新 Cell 存入 Archive（树结构），记录 `parent_key` + `actions_from_parent`

每个 Cell 存储一个指向父节点的指针，形成一棵以初始状态为根的**探索树**。

### Bridge Layer：Demo 提取（`gen_demo.py`）

从 Archive 中找到**覆盖目标最多**（其次累计奖励最高）的黄金 Cell：

1. **树溯源**：沿 `parent_key` 指针链从叶节点回溯到根节点
2. **翻转拼接**：将所有动作片段首尾拼接，得到从初始状态到最优 Cell 的完整动作序列
3. **确定性回放**：在相同种子的环境中重新执行这条动作序列，收集每一步的 snapshot 作为 waypoint
4. **保存 Demo**：打包 `(obs, action, reward, snapshot, return)` 存为 `.demo.pkl`

### Phase 2：后向课程 + PPO + SIL 鲁棒化（`robustify.py`）

在**带随机性**的真实环境中训练神经网络策略：

1. **后向课程（Backward Algorithm）**
   - 初始：将智能体传送到 Demo 轨迹**接近终点**的 waypoint
   - 策略用 PPO 在线学习从该 waypoint 完成剩余任务
   - 当成功率 ≥ 80%（覆盖 18 个目标），起点**向前推移**
   - 重复直到策略能从初始状态鲁棒通关

2. **自我模仿学习（SIL）**
   - Demo 轨迹预加载到 SIL Buffer
   - 在线高回报轨迹也加入 Buffer
   - SIL Loss = `max(0, R_demo - V(s)) · log π(a_demo|s)`
   - 与 PPO Loss 联合优化

## Cell 设计

Cell Key = `(grid_x, grid_y, n_captured_targets)`

- XY 位置离散化为 0.5m × 0.5m 网格
- **领域知识**：已覆盖目标数编码进 Cell Key
- **Pareto 更新**：仅当新轨迹在步数和奖励上都不差时才覆盖

## 一键训练

```powershell
# 双击运行或命令行执行
run_all.bat
```

该脚本依次执行：Phase 1 → Bridge → Phase 2 → GUI Demo

## 分步命令

```powershell
# Phase 1: 确定性探索
python -m gym_pybullet_drones.our_experiments.go_explore.train \
    --total_iterations 5000 --n_envs 4

# Bridge: 树溯源 + 回放生成 Demo
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \
    --archive_path results/go_explore/archive.json

# Phase 2: 后向课程 + PPO + SIL
python -m gym_pybullet_drones.our_experiments.go_explore.robustify \
    --demo_path results/go_explore/best_demo.demo.pkl \
    --total_iterations 3000 --n_envs 4

# Demo: 加载模型演示
python -m gym_pybullet_drones.our_experiments.go_explore.demo \
    --model_path results/go_explore_phase2/model_final.pt \
    --n_episodes 3
```

## 主要参数

### Phase 1（`config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_iterations` | 5000 | 总探索轮数 |
| `n_envs` | 4 | 并行环境数 |
| `explore_steps` | 300 | 每轮随机探索步数 |
| `cell_size` | 0.5 | 网格单元边长（米） |
| `max_cells` | 10000 | Archive 最大容量 |
| `seed` | 42 | 确定性种子 |

### Phase 2（`robustify.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_iterations` | 3000 | PPO 训练轮数 |
| `backward_step_size` | 50 | 每次后退的 waypoint 步数 |
| `success_threshold` | 0.8 | 触发后退的成功率 |
| `success_captures` | 18 | "成功"所需的目标覆盖数 |
| `sil_coef` | 0.1 | SIL Loss 权重 |
| `max_backward_iters` | 500 | 单个 level 最大迭代数 |

## 输出文件

```
results/go_explore/
├── archive.json              # Phase 1 Archive 元数据
├── cell_data/                # 每个 Cell 的 snapshot + actions_from_parent
│   ├── cell_0.pkl
│   └── ...
└── best_demo.demo.pkl        # Bridge 生成的 Demo 数据

results/go_explore_phase2/
├── model_iter{N}.pt          # Phase 2 定期保存的模型
└── model_final.pt            # Phase 2 最终模型
```

## 文件结构

| 文件 | 说明 |
|---|---|
| `train.py` | Phase 1：确定性探索主循环 |
| `gen_demo.py` | Bridge：树溯源 + 确定性回放 Demo 生成 |
| `robustify.py` | Phase 2：后向课程 + PPO + SIL 训练 |
| `demo.py` | 加载训练好的模型进行 GUI 演示 |
| `run_all.bat` | 一键训练脚本（Phase 1 → Bridge → Phase 2 → Demo） |
| `config.py` | Phase 1 超参数 |
| `archive.py` | 树状 Cell Archive 实现 |
| `networks.py` | GRU Actor-Critic 网络 |
| `ppo.py` | PPO + SIL 训练器 |
| `sil_buffer.py` | 自我模仿学习回放缓冲区 |
| `rollout_buffer.py` | GAE Rollout Buffer |
