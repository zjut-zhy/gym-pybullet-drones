# Go-Explore（Per-Cell 完整动作序列 + 后向鲁棒化）

基于 **Go-Explore** 算法的单智能体无人机目标覆盖实验。

## 算法概述

完整流水线分为三个阶段：

### Phase 1：确定性探索与归档（`train.py`）

在**完全确定性**的环境中（所有环境实例共享 `environment_seed`，每次 `reset` 传入固定种子），系统性地探索整个状态空间：

1. **选择**：从 Archive 中按 `score` 比例采样一个 Cell（前 `warmup_iterations` 轮始终从 reset 开始）
2. **传送**：通过 **动作回放** 精确恢复——`env.reset(seed)` + 回放该 Cell 的 `full_action_sequence`，保证物理引擎状态 bit-identical
3. **探索**：执行 `explore_steps` 步随机动作，每个动作保持 30 步（~1s at 30Hz），模拟持续方向飞行，避免动作剧变导致姿态越限
4. **归档**：将新 Cell 存入 Archive，每个 Cell 独立保存从 `env.reset()` 到该 Cell 的 **完整动作序列**

**环境配置**：Phase 1 关闭目标接近引导奖励（`enable_target_attraction=False`），避免 shaping reward 干扰 Archive 的轨迹质量评估。累积奖励仅反映捕获奖励 + 安全惩罚。

**终止处理**：探索中若环境 `done`，仅当 `n_captured >= target_count`（全覆盖成功）时才将最后一步存入 Archive。其他终止状态（姿态越限、出界、碰撞、超时）不存入，否则回放到该 Cell 后环境处于 `done` 状态，无法继续探索。

**Pareto 更新**：已有 Cell 发现更优路径时（cost 更低 **或** reward 更高，且另一指标不变差），更新其 `full_action_sequence`。因 Phase 1 关闭了 target_attraction，累积奖励仅含捕获奖励和障碍/边界惩罚，Pareto 比较有意义。采用**懒构建**优化：仅在确认需要创建新 Cell 或 Pareto 更优时才构建完整动作序列。

**Done Cell 过滤**：`select()` 排除已捕获所有目标的 Done Cell（`n_captured >= target_count`），避免回放到已终止状态浪费迭代。Done Cell 保留在 Archive 中供 demo 导出。

> **为什么不用快照（snapshot）恢复？**
>
> PyBullet 的 `resetBasePositionAndOrientation`、`saveState/restoreState`、`saveBullet` 均无法保存物理约束求解器的 warm-start 状态（拉格朗日乘子等内部数据）。即使用完全相同的值调用 `resetBase*`（理论上的 no-op），`stepSimulation()` 的结果仍会发生偏差（~1.22e-04）。因此本实现使用动作回放作为唯一的确定性状态恢复手段。

### Bridge Layer：Demo 提取（`gen_demo.py`）

从 Archive 中提取**所有成功轨迹**（`n_captured >= target_count`），按累计奖励降序排列：

1. **筛选**：调用 `archive.get_successful_cells(target_count)` 获取所有完成全覆盖的 Cell；若无成功 Cell 则 fallback 到 `get_best_cell()`
2. **确定性回放**：在相同种子的环境中逐条重新执行动作序列，收集每一步的 `(obs, action, reward, n_captured)`
3. **MC Returns**：计算 γ=0.99 折扣回报
4. **保存**：
   - `demos.pkl`：包含所有成功轨迹的合集（`combined["demos"]` 列表）
   - `demos_best.demo.pkl`：最优单条轨迹（向后兼容）

**多轨迹优势**：多条不同路径的成功经验提升 Phase 2 行为克隆/RL 微调的泛化性和鲁棒性。

### Phase 2：后向课程 + PPO + SIL 鲁棒化（`robustify.py`）

使用**单环境**训练 GRU Actor-Critic 策略，交替使用 backward waypoint 和 random reset，训练参数与 `sb3rl/train.py` 对齐以确保公平比较：

1. **交替 Reset**
   - 奇数 episode：通过 action replay 恢复到 Demo 的 backward waypoint（课程学习）
   - 偶数 episode：使用随机种子 `env.reset(seed=random)`（泛化训练）
   - 两种模式交替进行，兼顾课程收敛和环境泛化

2. **后向课程（Backward Algorithm）**
   - 初始：从 Demo 接近终点的 waypoint 开始（`start_idx = demo_n_steps - backward_step_size`）
   - 当最近 `eval_window` 个 episode 成功率 ≥ `success_threshold`，起点后退 `backward_step_size` 步
   - 重复直到 `start_idx=0` 且成功率达标

2. **自我模仿学习（SIL）**
   - Demo 轨迹预加载到 SIL Buffer（循环缓冲区，容量 `sil_capacity`）
   - 在线高回报轨迹（`ep_reward > sil_online_threshold`）也加入 Buffer
   - SIL Loss = `-E[ max(0, R_demo - V(s)) · log π(a_demo|s) ]` + 0.5 · value regression
   - 与 PPO Loss 联合优化（`total_loss = ppo_loss + sil_coef × sil_loss`）

3. **PPO 细节**（与 sb3rl 一致）
   - Clipped Surrogate Objective，ratio clamp 至 `[0, 10]`
   - GAE(λ) 计算 advantage，标准化后训练
   - NaN/Inf 安全防护，跳过无效 batch

4. **TensorBoard 日志对齐**（与 sb3rl 完全可比）
   - `rollout/ep_rew_mean`、`rollout/ep_len_mean`：最近 100 个 episode 的滑动平均（`deque(maxlen=100)`，与 SB3 `stats_window_size` 默认值一致）
   - 记录频率：每次 PPO update（即每 `n_steps` 环境步）记录一次，与 SB3 PPO `log_interval=1` 对齐
   - `eval/mean_reward`、`eval/mean_ep_length`：指标名与 SB3 EvalCallback 一致
   - 每次训练自动创建编号子目录（`run_1`、`run_2`…），与 SB3 的 `PPO_1`、`PPO_2` 机制类似，避免新旧数据混叠

5. **评估机制**（与 sb3rl EvalCallback 对齐）
   - 每 `eval_freq`（默认 10000）个环境步触发一次评估
   - 创建独立的评估环境，使用随机种子运行 `n_eval_episodes`（默认 3）个 episode
   - 评估时使用**确定性策略**（不含探索噪声），与 SB3 的 `deterministic=True` 一致
   - 记录 `eval/mean_reward`、`eval/mean_ep_length`、`eval/mean_captures`
   - 若当前评估 reward 超过历史最佳，自动保存 `best_model.pt`

## Cell 设计

Cell Key = `(grid_x, grid_y, n_captured_targets)`

- XY 位置离散化为 `cell_size` × `cell_size` 网格（默认 0.5m × 0.5m）
- **领域知识**：已覆盖目标数编码进 Cell Key
- **Pareto 更新**：已有 Cell 发现 `(cost ≤, reward ≥)` 且严格更优的新路径时，更新 full_action_sequence

Score 计算: `score = 1/(1 + visit_count) + (n_captured / trajectory_cost) * (max_steps / target_count)`

- **新颖性** (`1/(1+visit_count)`)：访问越少分数越高，鼓励探索不常去的区域
- **捕获效率**：衡量单位步数的目标捕获效率，归一化到 ~[0,1] 区间与新颖性量级匹配。高捕获数的 Cell 不再因步数多而被惩罚
- **Done Cell 过滤**：已完成全覆盖的 Cell 不参与 `select()` 采样

每个 Cell 存储：
- `full_action_sequence`：从 `env.reset()` 到此 Cell 的完整动作列表（独立副本），用于动作回放确定性恢复
- `trajectory_cost`：从 reset 到此 Cell 的总步数
- `cumulative_reward`：累积奖励
- `visit_count` / `score`：用于选择的权重

## 网络结构

**ObsEncoder**（Per-Key MLP）：
- `self_state` → MLP(6 → 64 → 64)
- `target_state` → MLP(54 → 64 → 64)
- `obstacle_state` → MLP(24 → 64 → 32)
- 拼接后投影至 `obs_embed_dim`（128）

**ActorCritic**（GRU-based）：
- GRU（input=128, hidden=128, layers=1）
- Actor: `Linear(128 → 2)` + learnable log_std, tanh squashing
- Critic: `Linear(128 → 1)`
- log_prob 含 tanh correction: `log π(a) = log π(z) - Σ log(1 - tanh²(z))`

## 一键训练

```powershell
# 双击运行或命令行执行
run_all.bat
```

该脚本依次执行：Phase 1（20000 iters）→ Bridge → Phase 2（1M timesteps）→ GUI Demo

## 分步命令

```powershell
# Phase 1: 确定性探索
python -m gym_pybullet_drones.our_experiments.go_explore.train --total_iterations 10000

# Bridge: 回放所有成功 cell 动作序列生成多条 Demo
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo --archive_path results/go_explore/archive.json

# Phase 2: 后向课程 + PPO + SIL
python -m gym_pybullet_drones.our_experiments.go_explore.robustify --demo_path results/go_explore/demos_best.demo.pkl --total_timesteps 1000000

# Demo: 加载模型演示
python -m gym_pybullet_drones.our_experiments.go_explore.demo --model_path results/go_explore_phase2/model_final.pt --n_episodes 3

# Archive Demo: 直接回放 Archive 中成功 Cell 的动作序列
python -m gym_pybullet_drones.our_experiments.go_explore.demo_archive --archive_path results/go_explore/archive.json --n_demos 5 --playback_speed 2
```

## 主要参数

### Phase 1（`config.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_iterations` | 5000 | 总探索轮数 |
| `n_envs` | 1 | 环境数（固定种子下多环境无意义） |
| `explore_steps` | 300 | 每轮随机探索步数 |
| `warmup_iterations` | 100 | 前 N 轮从 reset 开始，不使用 cell 选择 |
| `cell_size` | 0.5 | 网格单元边长（米） |
| `max_cells` | 10000 | Archive 最大容量 |
| `arena_size` | 10.0 | 竞技场边长（米） |
| `target_count` | 18 | 目标个数 |
| `obstacle_count` | 6 | 障碍物个数 |
| `log_interval` | 10 | 日志打印间隔 |
| `save_interval` | 100 | Archive 保存间隔 |
| `seed` | 42 | 确定性种子 |

### Phase 2（`robustify.py` → `RobustifyConfig`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `total_timesteps` | 1000000 | 总环境交互步数（与 sb3rl 一致） |
| `n_steps` | 2048 | 每轮 PPO rollout 步数（与 sb3rl 一致） |
| `eval_freq` | 10000 | 每 N 步评估一次（与 sb3rl 一致） |
| `n_eval_episodes` | 3 | 每次评估的 episode 数 |
| `backward_step_size` | 50 | 每次后退的 waypoint 步数 |
| `success_threshold` | 0.8 | 触发后退的成功率 |
| `eval_window` | 20 | 成功率滑动窗口大小（episodes） |
| `max_backward_iters` | 200 | 单个 level 最大 PPO 更新数 |
| `success_captures` | 18 | "成功"所需的目标覆盖数 |
| `sil_coef` | 0.1 | SIL Loss 权重 |
| `sil_capacity` | 50000 | SIL Buffer 容量 |
| `lr` | 3e-4 | Adam 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ |
| `clip_eps` | 0.2 | PPO clip ε |
| `entropy_coef` | 0.01 | 熵正则系数 |
| `n_epochs` | 5 | PPO mini-epoch 数 |
| `batch_size` | 256 | PPO mini-batch 大小 |
| `obs_embed_dim` | 128 | Obs encoder 输出维度 |
| `gru_hidden` | 128 | GRU 隐层维度 |
| `log_interval` | 100 | 控制台打印间隔（PPO 更新次数） |

## 输出文件

```
results/go_explore/
├── archive.json              # Phase 1 Archive 元数据
├── action_sequences/         # 每个 Cell 的完整动作序列
│   ├── cell_0_actions.pkl
│   └── ...
├── demos.pkl                 # 所有成功轨迹合集（多条 Demo）
└── demos_best.demo.pkl       # 最优单条 Demo（向后兼容）

results/go_explore_phase2/
├── best_model.pt             # Phase 2 最佳评估模型
└── model_final.pt            # Phase 2 最终模型

runs/go_explore_phase2/       # TensorBoard 日志（自动编号子目录）
├── run_1/                    # 第 1 次训练
├── run_2/                    # 第 2 次训练
└── ...
```

## 文件结构

| 文件 | 说明 |
|---|---|
| `train.py` | Phase 1：确定性探索主循环（动作回放 + Pareto 更新 + sticky action） |
| `gen_demo.py` | Bridge：回放所有成功 cell 动作序列生成多条 Demo（含 MC returns） |
| `robustify.py` | Phase 2：后向课程 + PPO + SIL + 交替 reset + TensorBoard |
| `demo.py` | 加载 Phase 2 模型进行 GUI 演示 |
| `demo_archive.py` | 直接回放 Archive 中成功 Cell 的动作序列（支持变速播放） |
| `run_all.bat` | 一键训练脚本（Phase 1 → Bridge → Phase 2 → Demo） |
| `config.py` | Phase 1 超参数（`GoExploreConfig` dataclass） |
| `archive.py` | Per-Cell Archive 实现（Pareto 更新 + Done Cell 过滤 + 持久化） |
| `networks.py` | Per-Key MLP ObsEncoder + GRU ActorCritic（tanh squashing） |
| `ppo.py` | PPO + SIL 联合训练器（含 NaN 安全防护） |
| `sil_buffer.py` | 循环 SIL 回放缓冲区（demo 预加载 + 在线高回报轨迹） |
| `rollout_buffer.py` | GAE Rollout Buffer（标准 PPO 数据收集） |
