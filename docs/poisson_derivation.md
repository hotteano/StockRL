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

整理上述公式，得到纯泊松过程下的看涨期权定价公式：

$$ \boxed{C(S_0, K, T) = \sum_{k=0}^{\infty} \frac{e^{-(r+\lambda)T} (\lambda T)^k}{k!} \max\left( S_0 e^{(r - \sigma \lambda)T} (1+\sigma)^k - K, \, 0 \right)} $$

### 4.2 直观解释

1. **加权求和**：期权价格是无数种可能情境的加权和。每种情境对应在时间 $T$ 内发生了 $k$ 次能力突破。
2. **确定性收益**：一旦确定了发生 $k$ 次突破，最终股价就是确定的（因为我们移除了布朗运动噪声）。只要 $S_T^{(k)} > K$，该情境就有正收益。
3. **阈值效应**：对于虚值期权（Out-of-the-money），只有当突破次数 $k$ 达到一定数量时，期权才会被行权。这体现了模型对“高潜力、高爆发”智能体的估值偏好。
