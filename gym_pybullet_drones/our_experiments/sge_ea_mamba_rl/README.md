# SEAM-RL: SGE + EA-Mamba RL

**SAC-Guided Go-Explore + Entity-Attention-Mamba RL**

## 框架概述

```
Phase 1 (本目录)                  Phase 2 (本目录)
┌─────────────────────┐           ┌──────────────────────────────┐
│ train.py (SGE)      │           │ robustify.py                 │
│   SAC-guided探索    │           │   SB3 PPO                    │
│   ↓                 │           │   + EAMambaActorCriticPolicy │
│ gen_demo.py (复用)  │──demo──→  │   + BackwardCurriculum       │
│   ↓                 │           │   + SIL                      │
│ archive.json        │           │                              │
└─────────────────────┘           └──────────────────────────────┘
```

- **Phase 1 (SGE)**：SAC-guided Go-Explore 探索稀疏奖励环境（`train.py`），产出 archive 和 demo 轨迹
- **Phase 2 (EA-Mamba RL)**：使用 Entity Attention + Mamba 作为策略网络（`robustify.py`），结合 SIL + 后向课程训练

## EA-Mamba 策略架构

```
obs = {self_state, target_state, obstacle_state}
              ↓
┌──────────────────────────────────────────────┐
│  EntityAttentionEncoder (空间维度)            │
│                                              │
│  self_state (6d) → SelfEncoder → query (64d) │
│                                              │
│  target_state (54d)                          │
│    → reshape 18×3 → TokenEncoder → 18×d_k   │
│    → CrossAttention(query, keys) → 64d       │
│                                              │
│  obstacle_state (24d)                        │
│    → reshape 6×4 → TokenEncoder → 6×d_k     │
│    → CrossAttention(query, keys) → 32d       │
│                                              │
│  Concat(64+64+32) → Projection → d_model    │
└──────────────────────────────────────────────┘
              ↓
┌──────────────────────────────────────────────┐
│  Mamba SSM Stack (时间维度)                   │
│  [MambaBlock + LayerNorm + Residual] × 2     │
└──────────────────────────────────────────────┘
              ↓
Actor head → π(a|s)    Critic head → V(s)
```

**与 sge_mamba_rl 的区别**：
- 旧版 `ObsEncoder`：将 `target_state(54d)` 视为一个平坦向量，通过 MLP 隐式学习
- 新版 `EntityAttentionEncoder`：将其 reshape 为 18×3 的逐实体 token，通过 **交叉注意力** 显式建模各实体的决策相关性

## 安装 Mamba

```bash
# 需要 Linux + CUDA + PyTorch
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

安装后在 `ea_mamba_policy.py` 中设置 `USE_REAL_MAMBA = True`。

未安装时自动使用纯 PyTorch 的 Fallback SSM（仅用于代码调试）。

## 使用方法

```bash
# Phase 1: SAC-Guided Go-Explore
python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.train --total_iterations 20000

# Bridge: 生成 Demo
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \
    --archive_path results/sge_ea_mamba_rl_phase1/archive.json \
    --output_path results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl

# Phase 2: EA-Mamba RL 训练
python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.robustify \
    --demo_path results/sge_ea_mamba_rl_phase1/demos_best.demo.pkl \
    --total_timesteps 1000000

# Demo: GUI 可视化
python -m gym_pybullet_drones.our_experiments.sge_ea_mamba_rl.demo \
    --model_path results/sge_ea_mamba_rl/best_model.zip --n_episodes 3
```

或使用一键脚本：`run_all.bat`

## 超参数

### EA-Mamba 策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mamba_d_model` | 128 | 模型维度 |
| `mamba_d_state` | 64 | SSM 状态维度 |
| `mamba_n_layers` | 2 | Mamba 层数 |
| `mamba_headdim` | 32 | SSM head 维度 |
| `ea_d_k` | 32 | Entity token 编码维度 |
| `ea_d_attn_target` | 64 | Target 注意力输出维度 |
| `ea_d_attn_obstacle` | 32 | Obstacle 注意力输出维度 |
| `ea_n_heads` | 2 | 注意力头数 |

### 后向课程 / SIL / PPO

同 `sge_mamba_rl`，详见 `robustify.py` 中的 `RobustifyConfig`。

## 输出文件

```
results/sge_ea_mamba_rl/
├── best_model.zip        # 最佳评估模型（SB3 格式）
├── final_model.zip       # 最终模型
└── evaluations.npz       # 评估记录

runs/sge_ea_mamba_rl/     # TensorBoard 日志
└── PPO_1/
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `ea_mamba_policy.py` | **核心**: EntityAttention + Mamba Actor-Critic 策略 |
| `train.py` | **Phase 1**: SAC-Guided Go-Explore |
| `config.py` | Phase 1 超参数（SGEConfig） |
| `robustify.py` | **Phase 2**: SB3 PPO + EA-Mamba + 后向课程 + SIL |
| `sil_buffer.py` | 循环 SIL 回放缓冲区 |
| `demo.py` | 加载模型 GUI 演示 |
| `run_all.bat` | 一键训练脚本 |
