# MARL 训练脚本使用文档

## 文件说明

| 文件 | 算法 | 类型 | 特点 |
|------|------|------|------|
| `mappo.py` | MAPPO | On-policy | PPO + GAE，适合合作任务 |
| `masac.py` | MASAC | Off-policy | 自动熵调节，探索性强 |
| `maddpg.py` | MADDPG | Off-policy | 确定性策略 + 高斯探索噪声 |
| `matd3.py` | MATD3 | Off-policy | MADDPG 改进版：双Q + 延迟策略更新 |

所有脚本采用**参数共享**：所有无人机使用同一个神经网络。

---

## 快速开始

```bash
# 激活环境
conda activate drones

# 进入目录
cd d:\Users\ZHY\Documents\GitHub\gym-pybullet-drones

# 最简运行（默认 2 架无人机，50 万步）
python gym_pybullet_drones/our_experiments/marl/mappo.py

# 查看所有可用参数
python gym_pybullet_drones/our_experiments/marl/mappo.py --help
```

---

## 通用参数

所有 4 个脚本共享以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-drones` | 2 | 无人机数量（最大 4） |
| `--total-timesteps` | 500000 | 总训练步数 |
| `--seed` | 1 | 随机种子 |
| `--cuda` / `--no-cuda` | True | 是否使用 GPU |
| `--save-model` / `--no-save-model` | True | 是否保存模型 |
| `--gamma` | 0.99 | 折扣因子 |
| `--track` | False | 启用 Weights & Biases 记录 |

---

## 各算法专有参数

### MAPPO (`mappo.py`)

```bash
python mappo.py \
  --num-drones 4 \
  --total-timesteps 1000000 \
  --learning-rate 3e-4 \
  --num-steps 2048 \
  --num-minibatches 32 \
  --update-epochs 10 \
  --clip-coef 0.2 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --gae-lambda 0.95 \
  --max-grad-norm 0.5 \
  --target-kl 0.03
```

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `--learning-rate` | 3e-4 | 学习率 | 训练不稳定可降到 1e-4 |
| `--num-steps` | 2048 | 每次 rollout 收集的 per-drone 步数 | 增大可提高稳定性，但更慢 |
| `--num-minibatches` | 32 | mini-batch 数量 | batch_size = num_steps / num_minibatches |
| `--update-epochs` | 10 | 每次 rollout 后更新的 epoch 数 | 太大会过拟合当前 batch |
| `--clip-coef` | 0.2 | PPO clip 系数 | 一般不用改 |
| `--ent-coef` | 0.01 | 熵正则系数 | 增大鼓励探索 |
| `--vf-coef` | 0.5 | value loss 系数 | 一般不用改 |
| `--gae-lambda` | 0.95 | GAE λ | 越大 variance 越高但 bias 越低 |
| `--target-kl` | None | KL 散度阈值（提前停止） | 设 0.01~0.05 可防止策略突变 |

### MASAC (`masac.py`)

```bash
python masac.py \
  --num-drones 4 \
  --total-timesteps 1000000 \
  --policy-lr 3e-4 \
  --q-lr 1e-3 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --learning-starts 5000 \
  --tau 0.005 \
  --autotune \
  --alpha 0.2
```

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `--policy-lr` | 3e-4 | Actor 学习率 | — |
| `--q-lr` | 1e-3 | Q 网络学习率 | — |
| `--buffer-size` | 1000000 | 经验回放大小 | 内存不够可降到 100000 |
| `--batch-size` | 256 | 训练 batch 大小 | — |
| `--learning-starts` | 5000 | 开始训练前的随机探索步数 | — |
| `--tau` | 0.005 | target 网络软更新系数 | — |
| `--autotune` / `--no-autotune` | True | 自动调节熵系数 α | 推荐开启 |
| `--alpha` | 0.2 | 固定熵系数（autotune 关闭时使用） | — |
| `--policy-frequency` | 2 | 每训练 N 步 Q 才更新一次 actor | — |

### MADDPG (`maddpg.py`)

```bash
python maddpg.py \
  --num-drones 4 \
  --total-timesteps 1000000 \
  --learning-rate 3e-4 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --learning-starts 25000 \
  --exploration-noise 0.1 \
  --tau 0.005
```

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `--learning-rate` | 3e-4 | Actor 和 Q 共用学习率 | — |
| `--exploration-noise` | 0.1 | 高斯探索噪声标准差 | 增大提高探索 |
| `--learning-starts` | 25000 | 随机探索步数 | 太少会导致 Q 不稳定 |
| `--policy-frequency` | 2 | 延迟 actor 更新频率 | — |

### MATD3 (`matd3.py`)

```bash
python matd3.py \
  --num-drones 4 \
  --total-timesteps 1000000 \
  --learning-rate 3e-4 \
  --buffer-size 1000000 \
  --batch-size 256 \
  --learning-starts 25000 \
  --exploration-noise 0.1 \
  --policy-noise 0.2 \
  --noise-clip 0.5 \
  --tau 0.005
```

| 参数 | 默认值 | 说明 | 调参建议 |
|------|--------|------|----------|
| `--policy-noise` | 0.2 | target policy smoothing 噪声 | TD3 专有 |
| `--noise-clip` | 0.5 | target 噪声截断范围 | TD3 专有 |
| 其余同 MADDPG | | | |

---

## 推荐运行配置

### 快速测试（验证能跑通）

```bash
python mappo.py --num-drones 2 --total-timesteps 10000 --num-steps 512
python masac.py --num-drones 2 --total-timesteps 10000 --learning-starts 1000
python maddpg.py --num-drones 2 --total-timesteps 10000 --learning-starts 1000
python matd3.py --num-drones 2 --total-timesteps 10000 --learning-starts 1000
```

### 正式训练

```bash
# MAPPO（推荐先试的算法）
python mappo.py --num-drones 4 --total-timesteps 2000000 --num-steps 4096

# MASAC（探索性最强）
python masac.py --num-drones 4 --total-timesteps 2000000 --learning-starts 10000

# MATD3（通常比 MADDPG 稳定）
python matd3.py --num-drones 4 --total-timesteps 2000000 --learning-starts 25000
```

---

## 训练输出

### 模型保存位置

```
runs/
  mappo__1__1710812345/
    mappo.cleanrl_model          # 模型权重
    events.out.tfevents.xxx      # TensorBoard 日志
```

### 查看训练曲线

```bash
tensorboard --logdir runs

# 先激活环境
conda activate drones
# 用 python -m 方式启动
python -m tensorboard.main --logdir runs
```

然后浏览器打开 `http://localhost:6006`。

### 关键指标

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `charts/episodic_return` | 每个 episode 的总奖励 | 上升 ↑ |
| `charts/SPS` | 每秒训练步数 | 稳定 |
| `losses/value_loss` | V/Q 网络损失 | 先升后降 |
| `losses/policy_loss` | 策略损失 | 波动但整体下降 |
| `losses/entropy` | 策略熵 (MAPPO) | 缓慢下降 |
| `losses/alpha` | 自动熵系数 (MASAC) | 自动调节 |

---

## 状态空间说明

每架无人机的观测经过展平后为 **108 维**向量，拼接顺序为：

```
[self_state(6) | teammate_state(24) | target_state(54) | obstacle_state(24)]
```

| 组件 | 维度 | 内容 |
|------|------|------|
| `self_state` | 6 | 自身位置(3) + 速度(3) |
| `teammate_state` | 24 | 4 slots × 6 维（相对位置 + 相对速度） |
| `target_state` | 54 | 18 slots × 3 维（相对位置） |
| `obstacle_state` | 24 | 6 slots × 4 维（相对位置 + 半径） |

动作空间为 **2 维**连续值 `[-1, 1]`：水平 XY 速度指令。

---

## 常见问题

**Q: 哪个算法推荐先试？**
A: MAPPO。On-policy 算法在合作任务上通常收敛更好。

**Q: 训练卡在 `learning_starts` 阶段没输出？**
A: 正常，off-policy 算法（MASAC/MADDPG/MATD3）需要先收集随机数据填充 buffer，此期间无训练日志输出。

**Q: 内存不够怎么办？**
A: 减小 `--buffer-size`（如 100000），或减少 `--num-drones`。

**Q: 怎么用 CPU 训练？**
A: 加 `--no-cuda` 参数。

**Q: 怎么对比算法？**
A: 用不同 `--seed`（如 1, 2, 3）跑多次，然后在 TensorBoard 里对比 `episodic_return` 曲线。
