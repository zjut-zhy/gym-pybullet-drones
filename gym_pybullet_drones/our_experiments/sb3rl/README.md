# SB3 RL Training — Single-Agent Target Coverage

使用 Stable-Baselines3 的 4 种算法训练 `OurSingleRLAviary`。

## 支持算法

| 算法 | 类型 | 说明 |
|------|------|------|
| PPO  | On-policy  | 近端策略优化，稳定性好 |
| SAC  | Off-policy | 软演员-评论家，带自动熵调节 |
| TD3  | Off-policy | 双延迟 DDPG，减少 Q 值过估计 |
| DDPG | Off-policy | 确定性策略梯度 |

## 单算法训练

```bash
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ppo
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo sac
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo td3
python -m gym_pybullet_drones.our_experiments.sb3rl.train --algo ddpg
```

## 全部运行

从项目根目录 (`gym-pybullet-drones/`) 执行：

```powershell
.\gym_pybullet_drones\our_experiments\sb3rl\run_all.bat
.\gym_pybullet_drones\our_experiments\sb3rl\run_all.bat --total_timesteps 500000
```

## 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algo` | ppo | 算法选择 |
| `--total_timesteps` | 1000000 | 训练总步数 |
| `--eval_freq` | 10000 | 评估频率 |
| `--n_eval_episodes` | 3 | 每次评估的 episode 数 |
| `--gui` | True | 训练后是否显示 GUI 演示 |
| `--plot` | True | 是否显示学习曲线 |
| `--output_folder` | results | 输出目录 |

## 输出文件

```
results/
  ppo-03.21.2026_17.06.54/
    best_model.zip          # 评估最优模型
    final_model.zip         # 最终模型
    evaluations.npz         # 评估数据
    learning_curve.png      # 学习曲线图
```
