# 基于泊松过程的 MARL 市值资产定价模型推导

## 1. 模型假设与定义

传统的 Black-Scholes 模型假设资产价格遵循几何布朗运动 (GBM)，即由连续的维纳过程 (Wiener Process) 驱动。然而，在多智能体强化学习 (MARL) 中，智能体的能力往往呈现**阶梯式**或**顿悟式**的增长（例如解开某个谜题、发现特定策略），而非连续的平滑改进。

为了更准确地描述这一现象，我们提出使用**几何泊松过程 (Geometric Poisson Process)** 替代原来的模型。

### 1.1 动态方程 (SDE)

假设智能体市值 $S_t$ 满足以下随机微分方程：

$$ dS_t = \mu S_t dt + \sigma S_t dP_t $$

其中：
- $\mu$：漂移率（Drift），表示除跳跃外的确定性增长趋势。
- $\sigma$：**跳跃幅度因子**（非波动率）。当事件发生时，股价变为原来的 $(1+\sigma)$ 倍。如果是下跌跳跃，则 $\sigma < 0$（但在本推导中主要考虑能力提升，故假设 $\sigma > 0$）。
- $P_t$：标准的**泊松过程 (Poisson Process)**，强度参数为 $\lambda$。
  - $dP_t = 1$ 的概率为 $\lambda dt$（发生跳跃）。
  - $dP_t = 0$ 的概率为 $1 - \lambda dt$（无跳跃）。
  - $\mathbb{E}[dP_t] = \lambda dt$。

---

## 2. 利用伊藤公式 (Ito's Formula) 求解 $S_t$

对于包含跳跃过程的半鞅 (Semimartingale) $X_t$，伊藤公式的微分形式为：

$$ df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX^c_t + \frac{1}{2}\frac{\partial^2 f}{\partial x^2} d\langle X^c \rangle_t + \left[ f(t, X_t) - f(t, X_{t-}) \right] $$

在我们的模型中，$S_t$ 没有连续扩散部分（即没有 $dW_t$ 项），因此 $S_t^c$ 仅包含漂移项，二次变差 $d\langle S^c \rangle_t = 0$。

### 2.1 猜测解的形式

我们猜测解的形式为指数增长与跳跃次数的乘积：

$$ S_t = S_0 e^{\alpha t} (1+\sigma)^{P_t} $$

我们需要确定参数 $\alpha$ 使得该解满足微分方程。

令 $f(t, P_t) = S_0 e^{\alpha t} (1+\sigma)^{P_t}$。应用伊藤公式：

1. **时间偏导项**：
   $$ \frac{\partial f}{\partial t} = \alpha S_0 e^{\alpha t} (1+\sigma)^{P_t} = \alpha S_t $$

2. **连续部分**：
   无连续扩散项，故为 0。

3. **跳跃部分**：
   当 $P_t$ 发生跳跃（$P_t \to P_t + 1$）时：
   $$ \Delta S_t = S_t(P_t+1) - S_t(P_t) = S_0 e^{\alpha t} (1+\sigma)^{P_t+1} - S_0 e^{\alpha t} (1+\sigma)^{P_t} $$
   $$ \Delta S_t = S_t \cdot (1+\sigma) - S_t = \sigma S_t $$
   
   因此，跳跃部分的微分形式为 $\sigma S_t dP_t$。

将几部分合并：
$$ dS_t = \alpha S_t dt + \sigma S_t dP_t $$

将其与原始方程 $dS_t = \mu S_t dt + \sigma S_t dP_t$ 对比，可得：
$$ \alpha = \mu $$

### 2.2 解析解

因此，随机微分方程的解析解为：

$$ \boxed{S_t = S_0 e^{\mu t} (1+\sigma)^{P_t}} $$

这表示在时间 $t$ 内，资产价格自然增长 $e^{\mu t}$ 倍，并经历了 $P_t$ 次幅度为 $\sigma$ 的跳跃乘数。

---

## 3. 风险中性定价 (Risk-Neutral Pricing)

为了对期权进行定价，我们需要在**风险中性测度 $\mathbb{Q}$** 下进行计算。根据无套利定价理论，风险中性世界中资产的期望回报率必须等于无风险利率 $r$。

### 3.1 确定风险中性漂移率

计算 $S_t$ 的期望增长：

$$ \mathbb{E}[dS_t] = \mu S_t dt + \sigma S_t \mathbb{E}[dP_t] $$

由于 $\mathbb{E}[dP_t] = \lambda dt$，故：

$$ \mathbb{E}[dS_t] = (\mu + \sigma \lambda) S_t dt $$

令期望回报率等于 $r$：
$$ \mu + \sigma \lambda = r \implies \mu = r - \sigma \lambda $$

因此，在风险中性测度下，$S_T$ 的表达式为：

$$ S_T = S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^{P_T} $$

其中 $P_T$ 服从参数为 $\lambda T$ 的泊松分布。

---

## 4. 欧式看涨期权定价公式

期权价值 $C$ 是其在到期日 $T$ 的折现期望收益：

$$ C = e^{-rT} \mathbb{E}^{\mathbb{Q}} \left[ \max(S_T - K, 0) \right] $$

利用全期望公式，我们可以对跳跃次数 $k$ 进行求和：

$$ \mathbb{E}[X] = \sum_{k=0}^{\infty} P(P_T = k) \cdot \mathbb{E}[X \mid P_T = k] $$

已知泊松分布概率质量函数：
$$ P(P_T = k) = \frac{e^{-\lambda T} (\lambda T)^k}{k!} $$

当跳跃次数固定为 $k$ 时，$S_T$ 是确定的值（不再随机）：
$$ S_T^{(k)} = S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k $$

因此：

$$ C = e^{-rT} \sum_{k=0}^{\infty} \frac{e^{-\lambda T} (\lambda T)^k}{k!} \max\left( S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k - K, \, 0 \right) $$

### 4.1 最终解析解形式

梳理上述公式：

$$ C(S_0, K, T) = \sum_{k=0}^{\infty} \frac{e^{-(r+\lambda)T} (\lambda T)^k}{k!} \max\left( S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k - K, \, 0 \right) $$

## 5. 基于求和交换与CCDF的最终解析解

为了便于计算，我们可以消去 $\max$ 函数，将原式转化为两个泊松累积概率的差。

### 5.1 寻找临界跳跃次数 $k^*$

期权行权的条件是 $S_T > K$。由于 $S_T(k)$ 随 $k$ 单调递增，存在一个最小整数 $k^*$ 使得当 $P_T \ge k^*$ 时，$S_T > K$。

解不等式：
$$ S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k > K $$

取对数求解 $k$：
$$ k \log(1+\sigma) > \log(K/S_0) - (r - \sigma \lambda)T $$

得到临界值：
$$ k^* = \left\lceil \frac{\log(K/S_0) - (r - \sigma \lambda)T}{\log(1+\sigma)} \right\rceil $$
且 $k^*$ 必须非负，故 $k^* \leftarrow \max(0, k^*)$。

### 5.2 拆分求和项

将原求和式写为从 $k^*$ 开始的求和：

$$ C = \sum_{k=k^*}^{\infty} \frac{e^{-(r+\lambda)T} (\lambda T)^k}{k!} \left[ S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k - K \right] $$

拆分为 $A$（资产部分）和 $B$（现金部分）：$C = A - B$。

#### B项（现金部分）：
$$ B = \sum_{k=k^*}^{\infty} e^{-rT} K \frac{e^{-\lambda T} (\lambda T)^k}{k!} = K e^{-rT} \sum_{k=k^*}^{\infty} \text{Poisson}(k; \lambda T) $$
$$ B = K e^{-rT} Q(k^*; \lambda T) $$
其中 $Q(k; \mu) = P(X \ge k | X \sim \text{Poi}(\mu))$ 是泊松分布的右尾概率（CCDF）。

#### A项（资产部分）：
$$ A = S_0 e^{-\sigma \lambda T} \sum_{k=k^*}^{\infty} \frac{e^{-\lambda T} (\lambda T)^k (1+\sigma)^k}{k!} $$
合并指数项与幂次项：
$$ A = S_0 \sum_{k=k^*}^{\infty} e^{-\lambda(1+\sigma)T} \frac{[\lambda T (1+\sigma)]^k}{k!} $$
*(注：前面系数的指数合并过程: $-\sigma \lambda T - \lambda T = -\lambda(1+\sigma)T$)*

这里我们发现求和项正是参数为 $\lambda' T = \lambda(1+\sigma)T$ 的泊松分布概率质量函数。
$$ A = S_0 Q(k^*; \lambda(1+\sigma)T) $$

### 5.3 最终的优雅公式（类 Black-Scholes 形式）

$$ \boxed{C = S_0 Q(k^*; \lambda(1+\sigma)T) - K e^{-rT} Q(k^*; \lambda T)} $$

**数学意义对比：**
| 术语 | Black-Scholes (高斯) | Poisson Model (跳跃) |
| :--- | :--- | :--- |
| **ITM 概率** | $\Phi(d_2)$ | $Q(k^*; \lambda T)$ |
| **Delta** | $\Phi(d_1)$ | $Q(k^*; \lambda(1+\sigma)T)$ |
| **波动源** | 连续扩散 $\sigma \sqrt{T}$ | 离散强度 $\lambda T$ 与幅度 $\sigma$ |

这个公式在计算上非常高效，因为它只依赖于正则化不完全伽马函数 $P(s, x)$（Regularized Incomplete Gamma Function），这在 `scipy.special` 或 `torch` 中都有高效实现。

---

## 6. 用于 MARL 的 TD-Error 定价整形（从稀疏奖励到可收敛信号）

本推导给出了在“固定跳幅 + 泊松到达”下的解析定价式。将其用于 MARL 时，可以把每个智能体看成一条“能力资本/市值”曲线：

1. **单轮 episode 聚合**：先把环境的团队回报聚合为单轮总收益 $G$（例如折扣和）。
2. **Shapley 奖励分配**：用合作博弈 $v(S)$ 的 Shapley Value 得到每个智能体的已实现贡献 $\phi_i$。
3. **Lévy/泊松定价（本金 + 预期）**：把 $\phi_i$ 驱动的市值过程视为 Lévy 过程的一个特例（几何泊松），计算“本金 $S$ + 预期增量 $C$”的定价值
   $$\Pi_i = S_i + C_i.$$
4. **替换/增强 TD error**：用 $\Pi_i$ 作为 critic 目标值或 TD 奖励整形项，从而在奖励稀疏时仍能获得稳定学习信号。

更完整的理论连接方式与可落地的 TD 形式见：
- [docs/levy_td_shaping.md](docs/levy_td_shaping.md)

### 6.1 关于“前期波动大、后期波动小”的说明

需要强调：若 $\lambda$ 和跳幅参数 $\sigma$ 都是常数（齐次泊松 + 固定跳幅），则过程增量平稳，单位时间的对数方差强度为常数，**不会自动**呈现“前期波动大、后期波动小”。

若你想体现 RL 的“前期探索/顿悟更频繁、后期收敛更平稳”，更自然的建模方式是令 $\lambda(t)$ 递减（非齐次泊松）或令 $\sigma(t)$ 递减（跳幅衰减），或使它们依赖于学习状态（如 TD-error、策略熵）。
