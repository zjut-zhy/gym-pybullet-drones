# SGE-MambaRL

**SAC-Guided Go-Explore + Mamba3-based RL**

## 框架概述

```
Phase 1 (本目录)                  Phase 2 (本目录)
┌─────────────────────┐           ┌──────────────────────────────┐
│ train.py (SGE)      │           │ robustify.py                 │
│   SAC-guided探索    │           │   SB3 PPO                    │
│   ↓                 │           │   + MambaActorCriticPolicy   │
│ gen_demo.py (复用)  │──demo──→  │   + BackwardCurriculum       │
│   ↓                 │           │   + SIL                      │
│ archive.json        │           │                              │
└─────────────────────┘           └──────────────────────────────┘
```

- **Phase 1 (SGE)**：SAC-guided Go-Explore 探索稀疏奖励环境（`train.py`），产出 archive 和 demo 轨迹
  - 与原版 Go-Explore 的区别：用 **SAC 策略**（最大熵RL）代替随机动作做探索，探索更高效
- **Phase 2 (MambaRL)**：使用 Mamba3 作为 RL 策略网络的时序编码器（`robustify.py`），从头训练，用 demo 做 SIL + 后向课程引导

## Mamba3 策略架构

```
obs = {self_state, target_state, obstacle_state}
    ↓
ObsEncoder (per-key MLP) → d_model 维度嵌入
    ↓
[Mamba3 Block + LayerNorm + Residual] × n_layers
    ↓
Actor head → π(a|s)    Critic head → V(s)
```

**Mamba3 的优势**：选择性状态空间模型（Selective SSM）通过输入依赖的门控机制，
能自动"记住"关键避障事件、"遗忘"无关信息，天然适合动态避障的长程决策。

## 安装 Mamba3

```bash
# 需要 Linux + CUDA + PyTorch
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

安装后在 `mamba_policy.py` 中设置 `USE_REAL_MAMBA = True`。

未安装时自动使用纯 PyTorch 的 Fallback SSM（仅用于代码调试，不用于正式训练）。

## 使用方法

```bash
# Phase 1: SAC-Guided Go-Explore
python -m gym_pybullet_drones.our_experiments.sge_mambarl.train --total_iterations 20000

# Bridge: 生成 Demo（复用 go_explore 的 gen_demo）
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \
    --archive_path results/sge_mambarl_phase1/archive.json \
    --output_path results/sge_mambarl_phase1/demos_best.demo.pkl

# Phase 2: Mamba RL 训练
python -m gym_pybullet_drones.our_experiments.sge_mambarl.robustify \
    --demo_path results/sge_mambarl_phase1/demos_best.demo.pkl \
    --total_timesteps 1000000

# Demo: GUI 可视化
python -m gym_pybullet_drones.our_experiments.sge_mambarl.demo \
    --model_path results/sge_mambarl/best_model.zip --n_episodes 3
```

或使用一键脚本：`run_all.bat`

## 超参数

### Mamba 策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mamba_d_model` | 128 | Mamba 模型维度 |
| `mamba_d_state` | 64 | SSM 状态维度 |
| `mamba_n_layers` | 2 | Mamba 层数 |
| `mamba_headdim` | 32 | SSM head 维度 |

### SAC 探索器（Phase 1）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sac_lr` | 3e-4 | SAC 学习率 |
| `sac_buffer_size` | 100000 | SAC replay buffer 容量 |
| `sac_batch_size` | 256 | SAC mini-batch 大小 |
| `sac_learning_starts` | 1000 | SAC 训练前的随机探索步数 |
| `sac_gamma` | 0.99 | SAC 折扣因子 |
| `sac_tau` | 0.005 | 软目标更新系数 |

### 后向课程

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `backward_step_size` | 50 | 每次后退的 waypoint 步数 |
| `success_threshold` | 0.8 | 触发后退的成功率 |
| `eval_window` | 20 | 成功率滑动窗口大小 |
| `max_backward_iters` | 30 | 单个 level 最大更新数 |
| `curriculum_ratio` | 0.5 | backward reset 的概率 |

### SIL

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sil_coef` | 0.1 | SIL Loss 权重 |
| `sil_capacity` | 50000 | SIL Buffer 容量 |
| `sil_updates_per_rollout` | 5 | 每次 rollout 后 SIL 更新次数 |

### PPO（与 sb3rl 一致）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 3e-4 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `n_steps` | 2048 | PPO rollout 长度 |
| `n_epochs` | 5 | PPO mini-epoch 数 |
| `batch_size` | 256 | mini-batch 大小 |

## 输出文件

```
results/sge_mambarl/
├── best_model.zip        # 最佳评估模型（SB3 格式）
├── final_model.zip       # 最终模型
└── evaluations.npz       # 评估记录

runs/sge_mambarl/         # TensorBoard 日志
└── PPO_1/
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | **Phase 1**: SAC-Guided Go-Explore（SAC策略替代随机动作） |
| `config.py` | Phase 1 超参数（SGEConfig） |
| `mamba_policy.py` | **Phase 2 核心**: Mamba3 Actor-Critic 策略 |
| `robustify.py` | **Phase 2**: SB3 PPO + Mamba 策略 + 后向课程 + SIL |
| `sil_buffer.py` | 循环 SIL 回放缓冲区 |
| `demo.py` | 加载模型 GUI 演示 |
| `run_all.bat` | 一键训练脚本（Phase 1 + Bridge + Phase 2 + Demo） |
