# 核心理论推导：从离散到连续

本文档详细推导 Equity-MARL 的核心数学框架。

---

## 一、Shapley Value → 标的资产价格的映射

### 1.1 问题陈述

我们需要将智能体 $i$ 在每轮获得的 Shapley 值 $\phi_i(t)$ 映射为一个可用于期权定价的"标的资产价格"$S_i(t)$。

### 1.2 映射函数的设计原则

设计的映射 $f: \{\phi_i(\tau)\}_{\tau \leq t} \to S_i(t)$ 需要满足：

1. **非负性**：$S_i(t) \geq 0$（股价不能为负）
2. **单调性**：高贡献 → 高股价
3. **无记忆性**：当前价格只依赖历史，不依赖未来
4. **可微性**：便于梯度计算

### 1.3 推荐的映射函数

**方案A：对数收益率累积**

$$S_i(t) = S_i(0) \cdot \exp\left(\sum_{\tau=1}^{t} r_i(\tau)\right)$$

其中对数收益率定义为：

$$r_i(\tau) = \log\left(1 + \frac{\phi_i(\tau) - \bar{\phi}(\tau)}{\bar{\phi}(\tau) + \epsilon}\right)$$

- $\bar{\phi}(\tau)$：所有智能体在时刻 $\tau$ 的平均 Shapley 值
- $\epsilon$：防止除零的小常数

**性质验证**：
- ✅ 非负性：指数函数保证 $S_i(t) > 0$
- ✅ 单调性：$\phi_i > \bar{\phi}$ 时 $r_i > 0$，股价上涨
- ✅ 无记忆性：只用历史数据
- ✅ 可微性：组合函数可微

**方案B：指数移动平均 (EMA)**

$$S_i(t) = (1-\alpha) S_i(t-1) + \alpha \cdot \max(\phi_i(t), 0)$$

更简单，但可能不满足GBM假设。

---

## 二、离散时间框架：CRR 二叉树

### 2.1 模型设定

在 CRR 模型中，我们假设每个时间步 $\Delta t$，智能体的"股价"只有两种可能变化：

```
               S_i · u    (上涨)
              ↗
    S_i ─────
              ↘
               S_i · d    (下跌)
```

其中上涨因子和下跌因子为：

$$u = e^{\sigma_i \sqrt{\Delta t}}, \quad d = e^{-\sigma_i \sqrt{\Delta t}} = \frac{1}{u}$$

### 2.2 波动率估计

智能体 $i$ 的波动率 $\sigma_i$ 通过历史 Shapley 值计算：

$$\sigma_i(t) = \sqrt{\frac{1}{W-1} \sum_{\tau=t-W+1}^{t} \left(r_i(\tau) - \bar{r}_i\right)^2} \cdot \sqrt{\frac{1}{\Delta t}}$$

其中：
- $W$：滑动窗口大小
- $r_i(\tau)$：对数收益率
- $\bar{r}_i$：窗口内平均收益率

### 2.3 风险中性概率

在风险中性测度下，上涨概率为：

$$p = \frac{e^{r\Delta t} - d}{u - d}$$

**在 MARL 中 $r$ 的定义**：

$r$ 可以解释为"系统基准增长率"，有几种定义方式：

1. **常数设定**：$r = 0$（无基准）
2. **平均增长率**：$r = \frac{1}{n}\sum_i \mathbb{E}[r_i]$
3. **折扣率**：$r = -\log(\gamma)$，其中 $\gamma$ 是 RL 的折扣因子

### 2.4 期权价值的递归计算

设期权的执行价格为 $K_i$（智能体 $i$ 的"预期基准贡献"），可定义为：

$$K_i = \mathbb{E}[S_i(T)] \text{ 或 } K_i = S_i(0) \cdot e^{rT}$$

**后向递归**：

从终端时刻 $T$ 开始反向计算：

$$C_i(T, j) = \max(S_i(T, j) - K_i, 0)$$

其中 $S_i(T, j) = S_i(0) \cdot u^j \cdot d^{T-j}$ 是在 $T$ 时刻经历 $j$ 次上涨后的股价。

对于 $t < T$：

$$\boxed{C_i(t, j) = e^{-r\Delta t}\left[p \cdot C_i(t+1, j+1) + (1-p) \cdot C_i(t+1, j)\right]}$$

**时间复杂度**：$O(T^2)$ per agent

---

## 三、从离散到连续：极限定理

### 3.1 中心极限定理的应用

当 $\Delta t \to 0$（时间步无限细分），二叉树模型收敛到连续模型。

设总时间 $T$ 分为 $N$ 步，则 $\Delta t = T/N$。

定义对数股价：
$$X_i(t) = \log S_i(t)$$

在二叉树模型中：
$$X_i(t + \Delta t) - X_i(t) = \begin{cases} +\sigma_i\sqrt{\Delta t} & \text{概率 } p \\ -\sigma_i\sqrt{\Delta t} & \text{概率 } 1-p \end{cases}$$

### 3.2 期望和方差

$$\mathbb{E}[X_i(t+\Delta t) - X_i(t)] = \sigma_i\sqrt{\Delta t}(2p - 1)$$

将 $p = \frac{e^{r\Delta t} - d}{u - d}$ 展开到一阶：

$$p \approx \frac{1}{2} + \frac{r - \frac{\sigma_i^2}{2}}{2\sigma_i}\sqrt{\Delta t}$$

因此：

$$\mathbb{E}[\Delta X_i] \approx \left(r - \frac{\sigma_i^2}{2}\right)\Delta t$$

$$\text{Var}[\Delta X_i] \approx \sigma_i^2 \Delta t$$

### 3.3 Donsker 定理与收敛

当 $N \to \infty$，由 **Donsker 不变原理**（函数型中心极限定理）：

$$X_i(t) \xrightarrow{d} X_i(0) + \left(r - \frac{\sigma_i^2}{2}\right)t + \sigma_i W_t$$

其中 $W_t$ 是标准维纳过程（布朗运动）。

转换回股价：

$$\boxed{S_i(t) = S_i(0) \cdot \exp\left[\left(r - \frac{\sigma_i^2}{2}\right)t + \sigma_i W_t\right]}$$

这就是著名的**几何布朗运动 (GBM)** 的解。

---

## 四、Black-Scholes 偏微分方程

### 4.1 随机微分方程形式

GBM 的 SDE 形式为：

$$\boxed{dS_i = rS_i \, dt + \sigma_i S_i \, dW_t}$$

（在风险中性测度下，漂移率为 $r$）

### 4.2 Itô 引理

设期权价值 $C(S_i, t)$ 是 $(S_i, t)$ 的函数，由 **Itô 引理**：

$$dC = \frac{\partial C}{\partial t}dt + \frac{\partial C}{\partial S_i}dS_i + \frac{1}{2}\frac{\partial^2 C}{\partial S_i^2}(dS_i)^2$$

代入 $dS_i = rS_i dt + \sigma_i S_i dW_t$：

$$(dS_i)^2 = \sigma_i^2 S_i^2 dt + O(dt^{3/2})$$

因此：

$$dC = \left(\frac{\partial C}{\partial t} + rS_i\frac{\partial C}{\partial S_i} + \frac{1}{2}\sigma_i^2 S_i^2\frac{\partial^2 C}{\partial S_i^2}\right)dt + \sigma_i S_i\frac{\partial C}{\partial S_i}dW_t$$

### 4.3 无风险组合构造

构造一个组合 $\Pi$：
- 持有 1 份期权（价值 $C$）
- 卖空 $\Delta = \frac{\partial C}{\partial S_i}$ 份"股票"

组合价值变化：

$$d\Pi = dC - \Delta \cdot dS_i = \left(\frac{\partial C}{\partial t} + \frac{1}{2}\sigma_i^2 S_i^2\frac{\partial^2 C}{\partial S_i^2}\right)dt$$

这个组合是**无风险的**（没有 $dW_t$ 项），因此必须满足：

$$d\Pi = r\Pi \, dt = r(C - \Delta S_i)dt$$

### 4.4 Black-Scholes PDE

整理得：

$$\boxed{\frac{\partial C}{\partial t} + rS_i\frac{\partial C}{\partial S_i} + \frac{1}{2}\sigma_i^2 S_i^2\frac{\partial^2 C}{\partial S_i^2} = rC}$$

**边界条件**（欧式看涨期权）：
$$C(S_i, T) = \max(S_i - K_i, 0)$$

---

## 五、Black-Scholes 解析解

### 5.1 Feynman-Kac 定理

Black-Scholes PDE 的解可以表示为期望：

$$C(S_i, t) = e^{-r(T-t)} \mathbb{E}^{\mathbb{Q}}\left[\max(S_i(T) - K_i, 0) \,\big|\, S_i(t) = S_i\right]$$

其中 $\mathbb{Q}$ 是风险中性测度。

### 5.2 显式解

$$\boxed{C(S_i, t) = S_i \Phi(d_1) - K_i e^{-r(T-t)} \Phi(d_2)}$$

其中：

$$d_1 = \frac{\ln(S_i/K_i) + (r + \sigma_i^2/2)(T-t)}{\sigma_i\sqrt{T-t}}$$

$$d_2 = d_1 - \sigma_i\sqrt{T-t}$$

$\Phi(\cdot)$ 是标准正态分布的累积分布函数。

### 5.3 Greeks（希腊字母）

希腊字母描述期权价值对各参数的敏感性，在 MARL 中可用于动态调整策略：

| Greek | 公式 | MARL 解释 |
|-------|------|----------|
| **Delta** $\Delta$ | $\frac{\partial C}{\partial S_i} = \Phi(d_1)$ | 智能体贡献变化 1 单位时，期权价值的变化 |
| **Gamma** $\Gamma$ | $\frac{\partial^2 C}{\partial S_i^2} = \frac{\phi(d_1)}{S_i\sigma_i\sqrt{T-t}}$ | 二阶敏感性，衡量 Delta 的稳定性 |
| **Theta** $\Theta$ | $\frac{\partial C}{\partial t}$ | 时间衰减，探索→利用的转换速率 |
| **Vega** $\nu$ | $\frac{\partial C}{\partial \sigma_i} = S_i\phi(d_1)\sqrt{T-t}$ | 波动率变化的影响 |

---

## 六、与强化学习的统一

### 6.1 策略梯度与 Delta 的联系

在强化学习中，策略梯度为：

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q(s_t, a_t)\right]$$

在 Equity-MARL 中，我们可以将 Q 值替换为期权调整后的值：

$$Q^{\text{option}}_i(s, a) = w_i(t) \cdot C_i(t) \cdot \phi_i(t)$$

其中：
- $w_i(t)$：Meta-Investor 分配的仓位
- $C_i(t)$：期权价值（潜力估计）
- $\phi_i(t)$：即时 Shapley 值

### 6.2 Bellman 方程的扩展

传统 Bellman 方程：

$$Q(s, a) = r + \gamma \max_{a'} Q(s', a')$$

Equity-MARL 扩展：

$$Q_i^{\text{eq}}(s, a_i) = \phi_i + \gamma \cdot \Delta_i \cdot Q_i^{\text{eq}}(s', a_i')$$

其中 $\Delta_i = \Phi(d_1)$ 是 Delta 值，起到**动态折扣**的作用：
- 高 Delta（高确定性）：更相信未来价值
- 低 Delta（高不确定性）：更依赖即时奖励

---

## 七、实现检查清单

### 7.1 离散模型实现

- [ ] Shapley Value 计算（蒙特卡洛近似）
- [ ] 波动率估计（滑动窗口标准差）
- [ ] 二叉树期权定价（后向递归）
- [ ] Meta-Investor 仓位优化

### 7.2 连续模型实现

- [ ] GBM 参数估计（最大似然）
- [ ] Black-Scholes 解析解计算
- [ ] Greeks 计算（数值或解析）
- [ ] PDE 数值求解（可选，用于异常期权）

---

## 八、参考文献方向

1. **Shapley in ML**: Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions" (SHAP)
2. **MARL Credit Assignment**: Rashid et al. (2018). "QMIX: Monotonic Value Function Factorisation"
3. **Option Pricing**: Black & Scholes (1973). "The Pricing of Options and Corporate Liabilities"
4. **Continuous-time RL**: Wang et al. (2020). "Reinforcement Learning in Continuous Time and Space"
5. **Risk-sensitive RL**: Tamar et al. (2015). "Policy Gradients with Variance Related Risk Criteria"

---

*本推导证明了 Equity-MARL 框架的数学自洽性和可实现性。*
