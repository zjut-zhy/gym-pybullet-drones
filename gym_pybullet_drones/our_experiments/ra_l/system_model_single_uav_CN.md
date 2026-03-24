# 系统模型与问题表述（单机版）

## A. 问题陈述 (Problem Statement)

我们考虑由一架无人机、$M$ 个移动目标区域 $\mathcal{G} = \{G_1, \dots, G_M\}$ 和 $K$ 个动态障碍区域 $\mathcal{T} = \{T_1, \dots, T_K\}$ 组成的三维环境中的目标覆盖问题，目标区域和障碍区域统称为实体 $\mathcal{E} = \{\mathcal{G}, \mathcal{T}\}$。每个任务开始时，所有实体的位置均在有界任务区域内随机初始化。无人机具有固定的观测半径 $R_{\mathrm{obs}}$（最大感知范围）和覆盖半径 $R_{\mathrm{cov}}$（覆盖目标所需的最小接近距离）。**系统目标是在有限的时间范围 $T_{\max}$ 内最大化目标区域覆盖，同时确保与所有障碍区域的避碰。**

---

## B. 系统模型 (System Model)

我们考虑一个有界三维任务区域 $\mathcal{W} = [-L/2, L/2]^{2} \times [0, H_{\mathrm{w}}]$，其中 $L$ 为正方形操作区域的边长，$H_{\mathrm{w}}$ 为围合刚性边界的高度。环境包含无人机、$M$ 个目标和 $K$ 个障碍区域。无人机在时间步 $t$ 的位置记为 $\mathbf{p}^{(t)} = (x^{(t)}, y^{(t)}, z^{(t)})^{\top}$（世界坐标系），速度为 $\mathbf{v}^{(t)} = (\dot{x}^{(t)}, \dot{y}^{(t)}, \dot{z}^{(t)})^{\top} \in \mathbb{R}^3$，速度上限为 $v_{\max}$，即 $\lVert\mathbf{v}^{(t)}\rVert \le v_{\max}$。

### 四旋翼动力学

无人机被建模为具有质量 $m$、转动惯量矩阵 $\mathbf{J} \in \mathbb{R}^{3 \times 3}$、机体半径 $r_{\mathrm{uav}}$ 和推力系数 $k_f$ 的四旋翼。姿态由欧拉角 $\boldsymbol{\Phi} = (\phi, \theta, \psi)^{\top}$ 描述，分别表示横滚角、俯仰角和偏航角。四个旋翼（$l = 1,\dots,4$）以角速度 $\omega_l$ 旋转，产生推力 $f_l = k_f\,\omega_l^2$。世界坐标系下的平动和转动动力学为：

$$m\,\ddot{\mathbf{p}} = \mathbf{R}(\boldsymbol{\Phi})\begin{pmatrix}0\\0\\\sum_{l=1}^{4} f_l\end{pmatrix} - mg\,\mathbf{e}_3$$

$$\mathbf{J}\,\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times \mathbf{J}\,\boldsymbol{\omega}$$

其中 $\mathbf{R}(\boldsymbol{\Phi}) \in SO(3)$ 为机体坐标系到世界坐标系的旋转矩阵，$g$ 为重力加速度，$\mathbf{e}_3 = (0,0,1)^{\top}$ 为世界 $z$ 轴方向的单位向量，$\boldsymbol{\omega} \in \mathbb{R}^{3}$ 为机体角速度，$\boldsymbol{\tau} = (\tau_\phi, \tau_\theta, \tau_\psi)^{\top}$ 为通过分配矩阵由旋翼推力导出的机体坐标系力矩向量。

### 级联 PID 控制器

级联 PID 控制器通过三个阶段将高层速度指令映射为旋翼转速：位置环、姿态环和电机混控。

**速度指令。** 给定策略输出的动作 $\mathbf{a} = (a^{x}, a^{y})^{\top} \in [-1,1]^{2}$，期望速度构造为：

$$\mathbf{v}^{\mathrm{cmd}} = v_{\max}\,\frac{\mathbf{a}}{\lVert\mathbf{a}\rVert}$$

其中 $v_{\max}$ 为最大飞行速度。仅命令 $xy$ 分量，高度保持在 $z_0 = H_{\mathrm{w}}/2$。参考位置在每个控制步更新为 $\mathbf{p}^{\mathrm{ref}} = \mathbf{p}^{(t)} + \mathbf{v}^{\mathrm{cmd}} \cdot \Delta t$，其中 $\Delta t$ 为控制周期。

**位置环。** PID 控制器根据位置和速度误差计算期望推力向量：

$$\mathbf{F}^{\mathrm{des}} = \mathbf{K}_p^{\mathrm{pos}}\,\mathbf{e}_{p} + \mathbf{K}_i^{\mathrm{pos}}\!\int\!\mathbf{e}_{p}\,\mathrm{d}t + \mathbf{K}_d^{\mathrm{pos}}\,\mathbf{e}_{v} + mg\,\mathbf{e}_3$$

其中 $\mathbf{e}_{p} = \mathbf{p}^{\mathrm{ref}} - \mathbf{p}$ 为位置误差，$\mathbf{e}_{v} = \mathbf{v}^{\mathrm{cmd}} - \mathbf{v}$ 为速度误差，$\mathbf{K}_p^{\mathrm{pos}}, \mathbf{K}_i^{\mathrm{pos}}, \mathbf{K}_d^{\mathrm{pos}} \in \mathbb{R}^{3 \times 3}$ 分别为位置环的比例、积分和微分增益矩阵。标量推力通过将 $\mathbf{F}^{\mathrm{des}}$ 投影到当前机体 $z$ 轴获得，期望姿态 $\boldsymbol{\Phi}^{\mathrm{des}}$ 从 $\mathbf{F}^{\mathrm{des}}$ 的方向提取。

**姿态环。** 第二级 PID 控制器产生期望机体坐标系力矩：

$$\boldsymbol{\tau}^{\mathrm{des}} = \mathbf{K}_p^{\mathrm{att}}\,\mathbf{e}_{\Phi} + \mathbf{K}_i^{\mathrm{att}}\!\int\!\mathbf{e}_{\Phi}\,\mathrm{d}t + \mathbf{K}_d^{\mathrm{att}}\,\mathbf{e}_{\omega}$$

其中 $\mathbf{e}_{\Phi}$ 为当前姿态 $\boldsymbol{\Phi}$ 与期望姿态 $\boldsymbol{\Phi}^{\mathrm{des}}$ 之间的旋转误差，$\mathbf{e}_{\omega}$ 为角速率误差，$\mathbf{K}_p^{\mathrm{att}}, \mathbf{K}_i^{\mathrm{att}}, \mathbf{K}_d^{\mathrm{att}} \in \mathbb{R}^{3 \times 3}$ 为姿态环的 PID 增益矩阵。

**电机混控。** 标量推力和期望力矩通过混控矩阵 $\mathbf{M} \in \mathbb{R}^{4\times3}$ 组合为四个电机指令：

$$\boldsymbol{\omega}^{\mathrm{rpm}} = \alpha_{\mathrm{pwm}}\,\mathrm{clip}\!\bigl(T_{\mathrm{base}}\,\mathbf{1}_4 + \mathbf{M}\,\boldsymbol{\tau}^{\mathrm{des}},\; \omega_{\min},\, \omega_{\max}\bigr) + \beta_{\mathrm{pwm}}$$

其中 $T_{\mathrm{base}}$ 为悬停推力指令，$\mathbf{1}_4 = (1,1,1,1)^{\top}$ 将其等分到四个电机，$\mathrm{clip}(\cdot,\, \omega_{\min},\, \omega_{\max})$ 将 PWM 信号钳位在硬件限幅 $[\omega_{\min},\, \omega_{\max}]$ 内，$\alpha_{\mathrm{pwm}}$ 和 $\beta_{\mathrm{pwm}}$ 为 PWM 到 RPM 转换的仿射系数。

### 观测与覆盖指示函数

无人机监测其感知范围内的实体。设 $\mathbf{p}_j^{(t)}$ 为实体 $E_j$ 在时间步 $t$ 的位置。观测指示函数定义为：

$$I_{\mathrm{obs}}(E_j, t) = \begin{cases} 1 & \text{若 } \lVert\mathbf{p}^{(t)} - \mathbf{p}_j^{(t)}\rVert \le R_{\mathrm{obs}} \\ 0 & \text{其他} \end{cases}$$

其中 $I_{\mathrm{obs}}(E_j, t) = 1$ 表示无人机在时间 $t$ 检测到实体 $E_j$。设 $\mathbf{p}_m^{(t)}$ 为目标 $G_m$ 的位置。覆盖指示函数为：

$$I_{\mathrm{cov}}(G_m, t) = \begin{cases} 1 & \text{若 } \lVert\mathbf{p}^{(t)} - \mathbf{p}_m^{(t)}\rVert \le R_{\mathrm{cov}} \\ 0 & \text{其他} \end{cases}$$

其中 $I_{\mathrm{cov}}(G_m, t) = 1$ 表示目标 $G_m$ 被成功覆盖。目标是最大化回合时域 $T$ 内的累积覆盖量：

$$J = \sum_{t=0}^{T} \sum_{m=1}^{M} I_{\mathrm{cov}}(G_m, t)$$

目标被覆盖后立即在随机无碰撞位置重生，以保持任务复杂度恒定。

### 动态实体

每个目标区域 $G_m$ 被建模为半径 $r_{\mathrm{tgt}}$ 的圆柱区域。目标区域在 $xy$ 平面上以恒定速度 $v_{\mathrm{tgt}}$ 移动，航向每 $T_{\mathrm{hold}}$ 个控制步从 $\mathcal{U}(0, 2\pi)$ 均匀重新采样。每个障碍区域 $T_k$ 为半径 $r_k \sim \mathcal{U}(r_{\min}^{\mathrm{obs}}, r_{\max}^{\mathrm{obs}})$ 的圆柱区域，其中 $r_{\min}^{\mathrm{obs}}$ 和 $r_{\max}^{\mathrm{obs}}$ 分别为允许的最小和最大障碍半径。障碍区域在 $xy$ 平面上以速度 $v_{\mathrm{obs}}$ 移动，遵循与目标区域相同的随机航向协议。边界反射使所有实体保持在任务区域内。

### 安全约束

为避免碰撞并保持在任务区域内，无人机必须保持安全距离。设 $\mathbf{p}_k^{(t)}$ 为障碍区域 $T_k$ 在时间 $t$ 的中心位置。设 $R_{\mathrm{safe}}^{1} = r_{\mathrm{uav}} + r_k$ 为无人机与障碍区域边界之间的最小安全距离，$R_{\mathrm{safe}}^{2} = r_{\mathrm{uav}}$ 为无人机与任务区域边界之间的最小安全距离。约束为：

$$\begin{cases} \lVert\mathbf{p}^{(t)} - \mathbf{p}_k^{(t)}\rVert \ge R_{\mathrm{safe}}^{1}, & \forall\, T_k \in \mathcal{T} \\ d_{\mathrm{b}}^{(t)} \ge R_{\mathrm{safe}}^{2} \end{cases}$$

其中 $d_{\mathrm{b}}^{(t)} = \min\!\bigl(\tfrac{L}{2} - |x^{(t)}|,\; \tfrac{L}{2} - |y^{(t)}|\bigr)$ 为无人机中心到任务区域边界的最小距离。



---

## C. 问题转化 (Problem Transformation)

目标覆盖任务被建模为**部分可观测马尔可夫决策过程**（POMDP），由元组 $\langle \mathcal{S}, \mathcal{O}, \mathcal{A}, \mathcal{P}, R, \gamma \rangle$ 定义，其中 $\mathcal{S}$ 为全局状态空间，$\mathcal{O}$ 为观测空间，$\mathcal{A}$ 为动作空间，$\mathcal{P}(\mathbf{s}' \mid \mathbf{s}, a)$ 为状态转移函数，$R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 为奖励函数，$\gamma \in (0,1]$ 为折扣因子。无人机遵循参数为 $\theta$ 的参数化策略 $\pi_{\theta}(a \mid o)$，旨在最大化期望折扣回报 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$，其中 $r_{t+k}$ 为时刻 $t+k$ 所获奖励。

### 观测空间 (Observation Space)

在每个时间步 $t$，无人机接收局部观测 $o^{(t)}$，由三个部分组成：

1. **自身状态** $[\mathbf{p}^{(t)},\, \mathbf{v}^{(t)}]$：无人机的位置和速度，分别按区域半边长 $L/2$ 和最大速度 $v_{\max}$ 归一化。
2. **目标状态** $[\mathbf{p}_m^{(t)} - \mathbf{p}^{(t)}]$，$m = 1,\dots,M$：每个目标相对于无人机的归一化相对位置。
3. **障碍区域状态** $[\mathbf{p}_k^{(t)} - \mathbf{p}^{(t)},\, r_k]$，$k = 1,\dots,K$：每个障碍区域的相对位置和半径，半径按 $R_{\mathrm{obs}}$ 归一化。

对于超出观测半径 $R_{\mathrm{obs}}$ 的实体，对应条目设为零向量，表示未观测到。

### 动作空间 (Action Space)

动作为 $xy$ 平面上的二维速度方向，$a^{(t)} = (a^{x}, a^{y}) \in [-1,1]^{2}$。策略 $\pi_{\theta}(a \mid o)$ 输出归一化动作，经 $v_{\max}$ 缩放后通过 B 节所述的级联 PID 控制器转换为旋翼转速。

### 奖励函数 (Reward Function)

奖励函数的设计借鉴了**人工势场**（APF）方法，将无人机接近危险区域时递增的排斥惩罚与引导其趋向目标的吸引奖励相结合。奖励分解为三个分量：障碍区域惩罚、目标奖励和边界惩罚。所有分量共享一个统一的势场核函数：

$$\varphi(d,\,d_0) = \alpha\,\left(\frac{1}{d - d_0 + 1} - \frac{1}{\sigma + 1}\right)^{\!2}$$

其中 $\alpha$ 为可调缩放系数，$d$ 为当前距离，$d_0$ 为发生硬碰撞的接触距离，$\sigma$ 为无人机的威胁区宽度。势函数 $\varphi$ 在 $d = d_0 + \sigma$ 处为零，并在 $d \to d_0$ 时单调递增。下文中下标 $xy$ 表示三维位置在 $xy$ 平面上的投影，如 $\mathbf{p}_{xy} = (x, y)^{\top}$。

**1) 障碍区域惩罚：** 对于障碍区域 $T_k$（半径 $r_k$），无人机与障碍区域中心的平面距离为 $d = \lVert\mathbf{p}_{xy} - \mathbf{p}_{k,xy}\rVert$，其中 $\mathbf{p}_{xy}$ 和 $\mathbf{p}_{k,xy}$ 分别为无人机和障碍区域位置的 $xy$ 投影。设接触距离 $D_k = r_{\mathrm{uav}} + r_k$。惩罚为：

$$r_{\mathrm{obs}} = \begin{cases} -20 & d \le D_k \\ -\varphi(d,\, D_k) & D_k < d < D_k + \sigma \\ 0 & \text{其他} \end{cases}$$

**2) 目标奖励：** 设 $d^{*} = \min_{m} \lVert\mathbf{p}_{xy} - \mathbf{p}_{m,xy}\rVert$ 为无人机到最近目标区域中心的平面距离，其中 $\mathbf{p}_{m,xy}$ 为目标区域 $G_m$ 位置的 $xy$ 投影。奖励为：

$$r_{\mathrm{tgt}} = \begin{cases} +10 & d^{*} \le R_{\mathrm{cov}} \\ \varphi(d^{*},\, R_{\mathrm{cov}}) & R_{\mathrm{cov}} < d^{*} < R_{\mathrm{cov}} + \sigma \\ 0 & \text{其他} \end{cases}$$

**3) 边界惩罚：** 沿用安全约束中定义的 $d_{\mathrm{b}}^{(t)}$，惩罚为：

$$r_{\mathrm{bnd}} = \begin{cases} -20 & d_{\mathrm{b}} \le r_{\mathrm{uav}} \\ -\varphi(d_{\mathrm{b}},\, r_{\mathrm{uav}}) & r_{\mathrm{uav}} < d_{\mathrm{b}} < r_{\mathrm{uav}} + \sigma \\ 0 & \text{其他} \end{cases}$$

总奖励为：

$$R = r_{\mathrm{obs}} + r_{\mathrm{tgt}} + r_{\mathrm{bnd}}$$

### 回合终止条件

回合在以下条件下终止：(i) 累计捕获数达到 $\lceil \eta \cdot M \rceil$，其中 $\eta \in (0,1]$ 为覆盖完成比率；(ii) 安全约束被违反；(iii) 回合时长超过最大时间范围 $T_{\max}$。

