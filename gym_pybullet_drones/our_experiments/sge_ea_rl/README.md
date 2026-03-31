# SGE-EA (消融实验)

**SAC-Guided Go-Explore + Entity-Attention RL（无 Mamba）**

## 概述

消融实验：仅使用 Entity Attention 的空间感知，**去除 Mamba 时序建模**。  
用于验证 Mamba 时序编码对导航性能的贡献。

与完整版 `sge_ea_mamba_rl` 的区别：

| | sge_ea_mamba_rl | sge_ea_rl（本模块） |
|---|---|---|
| 空间感知 | ✅ Entity Attention | ✅ Entity Attention |
| 时序建模 | ✅ Mamba SSM | ❌ 无（每步独立） |
| 推理特点 | 上下文感知（利用历史） | 仅当前观测 |

## EA-only 策略架构

```
obs = {self_state, target_state, obstacle_state}
              ↓
┌──────────────────────────────────────────────┐
│  EntityAttentionEncoder (空间维度)            │
│                                              │
│  self_state → query                          │
│  target_state → 18×3 tokens → CrossAttn      │
│  obstacle_state → 6×4 tokens → CrossAttn     │
│  Concat → Projection → d_model              │
└──────────────────────────────────────────────┘
              ↓  (无 Mamba，直接输出)
Actor head → π(a|s)    Critic head → V(s)
```

## 使用方法

```bash
# Phase 1: SAC-Guided Go-Explore
python -m gym_pybullet_drones.our_experiments.sge_ea_rl.train --total_iterations 20000

# Bridge: 生成 Demo
python -m gym_pybullet_drones.our_experiments.go_explore.gen_demo \
    --archive_path results/sge_ea_rl_phase1/archive.json \
    --output_path results/sge_ea_rl_phase1/demos_best.demo.pkl

# Phase 2: EA RL 训练
python -m gym_pybullet_drones.our_experiments.sge_ea_rl.robustify \
    --demo_path results/sge_ea_rl_phase1/demos_best.demo.pkl \
    --total_timesteps 1000000

# Demo: GUI 可视化
python -m gym_pybullet_drones.our_experiments.sge_ea_rl.demo \
    --model_path results/sge_ea_rl/best_model.zip --n_episodes 3
```

或使用一键脚本：`run_all.bat`

## 超参数

### EA 策略

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ea_d_model` | 128 | 模型维度 |
| `ea_d_k` | 32 | Entity token 编码维度 |
| `ea_d_attn_target` | 64 | Target 注意力输出维度 |
| `ea_d_attn_obstacle` | 32 | Obstacle 注意力输出维度 |
| `ea_n_heads` | 2 | 注意力头数 |

### 后向课程 / SIL / PPO

同 `sge_ea_mamba_rl`，详见 `robustify.py` 中的 `RobustifyConfig`。

## 输出文件

```
results/sge_ea_rl/
├── best_model.zip        # 最佳评估模型
├── final_model.zip       # 最终模型
└── evaluations.npz       # 评估记录

runs/sge_ea_rl/           # TensorBoard 日志
└── PPO_1/
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `ea_policy.py` | **核心**: EntityAttention Actor-Critic 策略（无 Mamba） |
| `train.py` | **Phase 1**: SAC-Guided Go-Explore |
| `config.py` | Phase 1 超参数 |
| `robustify.py` | **Phase 2**: SB3 PPO + EA + 后向课程 + SIL |
| `sil_buffer.py` | 循环 SIL 回放缓冲区 |
| `demo.py` | 加载模型 GUI 演示 |
| `run_all.bat` | 一键训练脚本 |
