# 高级模型：非平稳性与分布假设修正

## 一、问题诊断

### 1.1 Black-Scholes 的核心假设

标准 GBM 假设：
$$dS = \mu S \, dt + \sigma S \, dW_t$$

**隐含假设**：
1. **恒定波动率**：$\sigma$ 是常数
2. **对数正态分布**：$\ln(S_T/S_0) \sim \mathcal{N}((\mu - \sigma^2/2)T, \sigma^2 T)$
3. **连续路径**：没有跳跃

### 1.2 RL 场景的特殊性

| 假设 | 金融市场 | RL 训练过程 | 问题 |
|------|---------|------------|------|
| 恒定 σ | ⚠️ 近似成立 | ❌ 严重违反 | 前期高波动，后期低波动 |
| 正态分布 | ⚠️ 厚尾 | ❌ 偏态+厚尾 | 策略突变导致跳跃 |
| 连续路径 | ⚠️ 有跳跃 | ❌ 频繁跳跃 | 探索导致的不连续 |

---

## 二、非平稳波动率的解决方案

### 2.1 方案一：时变波动率 (Time-Varying Volatility)

最简单的修正——让 $\sigma$ 随时间衰减：

$$\sigma(t) = \sigma_0 \cdot e^{-\lambda t} + \sigma_{\infty}$$

其中：
- $\sigma_0$：初始波动率（高，对应探索阶段）
- $\sigma_{\infty}$：终态波动率（低，对应收敛阶段）
- $\lambda$：衰减速率

**优点**：简单，符合 RL 的"前肥后细"直觉
**缺点**：需要预设衰减形式

### 2.2 方案二：GARCH 模型 (波动率聚集)

让波动率自回归地依赖历史：

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

其中：
- $\omega$：基础方差
- $\alpha$：收益冲击的影响
- $\beta$：波动率持续性

**GARCH(1,1) 适合 RL 的原因**：
- 捕捉"波动率聚集"：高波动后跟着高波动
- 自适应：不需要预设衰减形式
- 可估计：用最大似然估计参数

### 2.3 方案三：随机波动率 (Heston 模型)

让波动率本身也是随机过程：

$$dS = \mu S \, dt + \sqrt{v} S \, dW_1$$
$$dv = \kappa(\theta - v) \, dt + \xi \sqrt{v} \, dW_2$$

其中：
- $v = \sigma^2$：瞬时方差
- $\kappa$：均值回复速度
- $\theta$：长期方差
- $\xi$：波动率的波动率 (vol of vol)
- $\rho = \text{Corr}(dW_1, dW_2)$：杠杆效应

**Heston 模型适合 RL 的原因**：
- 波动率均值回复：最终趋向 $\theta$（收敛）
- 允许波动率的波动：捕捉不确定性的不确定性

---

## 三、分布假设的修正

### 3.1 为什么泊松分布的直觉是对的？

你的直觉非常准确！RL 中的"突变"更像**离散跳跃事件**而非连续扩散：

- **探索动作**：随机尝试新策略 → 收益跳跃
- **策略更新**：梯度下降导致的突然改变
- **环境变化**：非平稳环境的突然切换

这些都是**稀疏、离散**的事件，更适合用**泊松过程**而非布朗运动描述。

### 3.2 方案一：Merton 跳跃扩散模型

在 GBM 基础上加入泊松跳跃：

$$dS = \mu S \, dt + \sigma S \, dW + S \, dJ$$

其中跳跃项：
$$dJ = (e^Y - 1) \, dN(\lambda)$$

- $N(\lambda)$：强度为 $\lambda$ 的泊松过程
- $Y \sim \mathcal{N}(\mu_J, \sigma_J^2)$：跳跃幅度（对数正态）

**泊松跳跃在 RL 中的解释**：
- $\lambda$：每单位时间"策略突变"的期望次数
- $\mu_J$：平均跳跃幅度
- $\sigma_J$：跳跃幅度的不确定性

### 3.3 方案二：混合泊松-正态 (Compound Poisson)

将收益分解为两部分：

$$r_t = \underbrace{\mu + \sigma \epsilon_t}_{\text{正常扩散}} + \underbrace{\sum_{i=1}^{N_t} J_i}_{\text{跳跃}}$$

其中：
- $\epsilon_t \sim \mathcal{N}(0, 1)$：正常波动
- $N_t \sim \text{Poisson}(\lambda t)$：跳跃次数
- $J_i$：第 $i$ 次跳跃的幅度

**这种模型完美契合 RL**：
- 大部分时间：小幅正常波动（策略微调）
- 偶尔：大幅跳跃（探索或策略突破）

### 3.4 方案三：状态切换模型 (Regime-Switching)

假设系统在多个"状态"之间切换：

$$r_t | S_t = s \sim \mathcal{N}(\mu_s, \sigma_s^2)$$
$$P(S_{t+1} = j | S_t = i) = p_{ij}$$

例如两状态模型：
- 状态 1（探索）：高 $\mu$，高 $\sigma$
- 状态 2（利用）：低 $\mu$，低 $\sigma$

**隐马尔可夫模型 (HMM)** 可以自动识别状态转换！

---

## 四、RL 专用的"前肥后细"模型

### 4.1 时间非齐次泊松过程

让跳跃强度随时间衰减：

$$\lambda(t) = \lambda_0 \cdot e^{-\gamma t}$$

**解释**：
- 训练初期：$\lambda(0) = \lambda_0$ 高（频繁探索跳跃）
- 训练后期：$\lambda(t) \to 0$（趋于稳定）

### 4.2 完整的 RL 适配模型

结合上述思想，提出 **RL-adapted Jump-Diffusion**：

$$dS = \mu(t) S \, dt + \sigma(t) S \, dW + S \, dJ(t)$$

其中所有参数都是时间相关的：

$$\mu(t) = \mu_{\infty} + (\mu_0 - \mu_{\infty}) e^{-\alpha t}$$
$$\sigma(t) = \sigma_{\infty} + (\sigma_0 - \sigma_{\infty}) e^{-\beta t}$$
$$\lambda(t) = \lambda_0 \cdot e^{-\gamma t}$$

**参数解释**：

| 参数 | 含义 | RL 对应 |
|------|------|---------|
| $\mu_0, \mu_{\infty}$ | 漂移率从高到低 | 学习速率衰减 |
| $\sigma_0, \sigma_{\infty}$ | 波动率从高到低 | 探索→利用 |
| $\lambda_0, \gamma$ | 跳跃强度衰减 | 策略突变减少 |

---

## 五、期权定价的修正

### 5.1 跳跃扩散下的期权定价

Merton (1976) 给出了跳跃扩散下的期权定价公式：

$$C = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!} C_{BS}(S, K, r_n, \sigma_n, T)$$

其中：
- $\lambda' = \lambda(1 + \kappa)$，$\kappa = E[e^Y - 1]$
- $r_n = r - \lambda\kappa + n\ln(1+\kappa)/T$
- $\sigma_n^2 = \sigma^2 + n\sigma_J^2/T$

**实现方式**：截断无穷级数，取前 $N$ 项（通常 $N=20$ 足够）

### 5.2 GARCH 期权定价

在 GARCH 模型下，期权价格需要数值方法：
- **蒙特卡洛模拟**：模拟波动率路径
- **树方法**：扩展二叉树包含波动率状态
- **特征函数法**：用 FFT 求解

### 5.3 建议的实现顺序

1. **第一阶段**：时变波动率 $\sigma(t)$（最简单）
2. **第二阶段**：GARCH(1,1)（自适应波动率）
3. **第三阶段**：跳跃扩散（完整模型）
4. **第四阶段**：状态切换（如果需要）

---

## 六、数学推导：时变跳跃扩散

### 6.1 模型定义

$$\frac{dS}{S} = (\mu(t) - \lambda(t)\kappa) dt + \sigma(t) dW + (e^Y - 1) dN(\lambda(t))$$

### 6.2 特征函数

对数股价的特征函数：

$$\phi(u; t, T) = \exp\left[\int_t^T \psi(u; s) ds\right]$$

其中：
$$\psi(u; s) = iu\mu(s) - \frac{1}{2}u^2\sigma^2(s) + \lambda(s)(e^{iu\mu_J - \frac{1}{2}u^2\sigma_J^2} - 1 - iu\kappa)$$

### 6.3 期权定价

用 Lewis (2001) 的 FFT 方法：

$$C(S, K, T) = S - \frac{\sqrt{SK}}{\pi} \int_0^{\infty} \text{Re}\left[\frac{e^{-iuk}\phi(u - i/2; 0, T)}{u^2 + 1/4}\right] du$$

其中 $k = \ln(K/S)$。

---

## 七、总结与建议

### 你的两个观察都非常准确：

1. **非平稳性** → 用 **时变参数** 或 **GARCH** 解决
2. **"前肥后细"** → 用 **跳跃扩散** + **衰减强度** 解决

### 推荐的模型演进路径：

```
Black-Scholes (baseline)
    ↓
时变波动率 σ(t) (简单修正)
    ↓
GARCH 波动率 (自适应)
    ↓
跳跃扩散 (完整模型)
    ↓
状态切换 + 跳跃 (最完整)
```

### 论文角度的创新点：

这种"RL 适配的跳跃扩散模型"是一个很好的**理论贡献**：
- 首次将 **时间非齐次泊松过程** 引入 MARL 信用分配
- 建立了 **探索-利用** 与 **波动率衰减** 的数学联系
- 提供了 **比 Black-Scholes 更准确** 的 Agent 估值

---

*下一步：我可以帮你实现这些高级模型的代码*
