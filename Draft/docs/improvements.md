# Equity-MARL 改进方案 (v2.0)

基于深度审查反馈，本文档记录对原始框架的三大核心改进。

---

## 一、股价定义的修正：引入衰减系数

### 1.1 问题分析

原始定义：
$$S_i(t) = S_i(0) \cdot \exp\left(\sum_{\tau=1}^{t} r_i(\tau)\right)$$

**问题**：当 $T \to \infty$，所有 $S_i(t) \to \infty$，导致：
1. 数值溢出
2. 梯度爆炸
3. 期权价值失去区分度

### 1.2 修正方案：Retained Earnings with Depreciation

引入衰减系数 $\lambda \in (0, 1)$（类似于通胀或折旧）：

$$\boxed{S_i(t) = (1-\lambda) \cdot S_i(t-1) \cdot \exp(r_i(t)) + \lambda \cdot S_{\text{base}}}$$

其中：
- $\lambda$：衰减率（建议 $\lambda \in [0.01, 0.1]$）
- $S_{\text{base}}$：基准股价（如初始值 1.0）

**性质**：
- 股价有界：$S_i(t) \in [S_{\text{base}} \cdot \lambda/(1-(1-\lambda)e^{r_{\max}}), S_{\text{base}} \cdot \lambda/(1-(1-\lambda)e^{-r_{\max}})]$
- 保持相对排序：高贡献 Agent 仍然有高股价
- 遗忘历史：早期表现的影响逐渐衰减

### 1.3 另一种方案：相对股价

$$S_i(t) = \frac{\phi_i(t)}{\sum_j \phi_j(t)} \cdot n$$

直接用相对 Shapley 值作为"股价"，归一化到平均值 1。

---

## 二、执行价格 K：从固定到浮动（亚式期权）

### 2.1 问题分析

原始定义：$K_i = S_i(0) \cdot e^{rT}$

**问题**：
1. 非平稳环境中，早期估计的 $r$ 可能不准确
2. 所有 Agent 可能都是 Deep ITM 或 Deep OTM
3. 期权价值失去"筛选"能力

### 2.2 解决方案：浮动执行价格（Asian Option Style）

定义执行价格为**滑动窗口平均**：

$$\boxed{K_i(t) = \frac{1}{W} \sum_{\tau=t-W+1}^{t} S_i(\tau)}$$

或者定义为**跨 Agent 平均**（衡量"跑赢大盘"的能力）：

$$\boxed{K(t) = \bar{S}(t) = \frac{1}{n}\sum_{j=1}^{n} S_j(t)}$$

### 2.3 新的期权价值解释

使用浮动执行价格后：

$$C_i(t) = \mathbb{E}[\max(S_i(T) - K(T), 0)]$$

**含义**：$C_i(t)$ 衡量的是 Agent $i$ **超越平均水平的潜力**。

- $C_i > 0$：预期跑赢大盘
- $C_i \approx 0$：预期持平
- $C_i$ 很高：高成长潜力股

### 2.4 对 Black-Scholes 的影响

对于亚式期权，没有简单的解析解，但有以下近似方法：

**方法1：几何平均近似**

用几何平均代替算术平均，得到近似解析解：

$$C_{\text{Asian}} \approx C_{\text{BS}}(S, K_{\text{adj}}, \sigma_{\text{adj}})$$

其中调整后的波动率：
$$\sigma_{\text{adj}} = \sigma \cdot \sqrt{\frac{1}{3}}$$

**方法2：蒙特卡洛模拟**

直接模拟路径并计算期望（计算成本较高但精确）。

**方法3：离散二叉树（推荐）**

在 MARL 的离散时间步中，直接使用二叉树后向递归，将 $K$ 设为动态更新的平均值。

---

## 三、Markowitz 投资组合理论：考虑相关性

### 3.1 问题分析

原始仓位分配：$w_i \propto C_i$

**问题**：没有考虑 Agent 之间的相关性。如果 Agent A 和 B 高度正相关，同时重仓它们风险很大。

### 3.2 解决方案：均值-方差优化

**协方差矩阵估计**：

$$\Sigma_{ij}(t) = \frac{1}{W-1} \sum_{\tau=t-W+1}^{t} (\phi_i(\tau) - \bar{\phi}_i)(\phi_j(\tau) - \bar{\phi}_j)$$

**优化目标**：

$$\max_{\mathbf{w}} \quad \mathbf{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^T \Sigma \mathbf{w}$$

其中：
- $\boldsymbol{\mu} = [C_1(t), ..., C_n(t)]^T$：期权价值向量（预期收益）
- $\Sigma$：Shapley 值的协方差矩阵（风险）
- $\lambda$：风险厌恶系数

**约束条件**：
$$\sum_i w_i = 1, \quad w_i \geq 0$$

### 3.3 解析解（无约束情况）

$$\mathbf{w}^* = \frac{1}{\lambda} \Sigma^{-1} \boldsymbol{\mu}$$

然后归一化：$\mathbf{w} = \mathbf{w}^* / \sum_i w_i^*$

### 3.4 带约束的数值解

使用二次规划（Quadratic Programming）：

```python
from scipy.optimize import minimize

def portfolio_optimize(mu, Sigma, lambda_risk):
    n = len(mu)
    
    def objective(w):
        return -(w @ mu - 0.5 * lambda_risk * w @ Sigma @ w)
    
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # 权重和为1
    ]
    bounds = [(0, 1) for _ in range(n)]  # 非负权重
    
    result = minimize(objective, x0=np.ones(n)/n, 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x
```

### 3.5 效果

1. **多样性增强**：如果两个高收益 Agent 高度相关，系统只选其中更强的一个
2. **风险对冲**：会分配部分仓位给负相关的 Agent
3. **鲁棒性提升**：避免过度依赖单一策略

---

## 四、隐含波动率 (IV) 调节探索率

### 4.1 概念映射

| 金融概念 | MARL 对应 |
|---------|----------|
| 历史波动率 (HV) | Agent 历史表现的方差 |
| 隐含波动率 (IV) | 市场（Meta-Investor）对 Agent 未来波动的预期 |
| IV > HV | 市场认为该 Agent 未来更不确定 → 应增加探索 |
| IV < HV | 市场认为该 Agent 已稳定 → 应减少探索 |

### 4.2 隐含波动率的计算

给定 Meta-Investor 分配的仓位 $w_i$ 和理论期权价值 $C_i^{\text{BS}}(\sigma_i^{\text{HV}})$：

定义"市场期权价格"为：
$$C_i^{\text{market}} = w_i \cdot \sum_j C_j^{\text{BS}}$$

然后反解 IV：
$$\sigma_i^{\text{IV}} = \text{BS}^{-1}(C_i^{\text{market}})$$

这需要数值求解（如 Newton-Raphson）。

### 4.3 探索率调节

$$\epsilon_i = \epsilon_{\text{base}} \cdot \left(\frac{\sigma_i^{\text{IV}}}{\sigma_i^{\text{HV}}}\right)^\alpha$$

其中 $\alpha > 0$ 是调节灵敏度的超参数。

**效果**：
- 当市场认为 Agent 未来波动大（潜力未开发）时，增加探索
- 当市场认为 Agent 已稳定时，减少探索，专注利用

---

## 五、泡沫破裂后的参数重组

### 5.1 泡沫检测

$$\text{Bubble}_i(t) = \frac{w_i(t)}{\text{relative } C_i(t)}$$

定义**泡沫破裂事件**：

$$\text{Crash}_i(t) = \mathbb{1}\left[\text{Bubble}_i(t-1) > \theta_{\text{bubble}} \land \frac{\phi_i(t)}{\phi_i(t-1)} < \theta_{\text{drop}}\right]$$

即：之前被高估（$\text{Bubble} > \theta_{\text{bubble}}$），且 Shapley 值暴跌（$< \theta_{\text{drop}}$）。

### 5.2 参数重组策略

当 $\text{Crash}_i(t) = 1$ 时，执行"破产重组"：

**策略1：Partial Reset**
```python
def partial_reset(agent):
    # 保留特征提取器（CNN/RNN 层）
    for name, param in agent.feature_extractor.named_parameters():
        param.requires_grad = False  # 冻结
    
    # 重置策略头（全连接层）
    for layer in agent.policy_head:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
```

**策略2：Knowledge Distillation**
```python
def distill_to_new_agent(old_agent, new_agent, replay_buffer):
    # 用旧 Agent 的特征提取器输出作为教师信号
    for batch in replay_buffer:
        with torch.no_grad():
            teacher_features = old_agent.feature_extractor(batch.obs)
        
        student_features = new_agent.feature_extractor(batch.obs)
        loss = F.mse_loss(student_features, teacher_features)
        loss.backward()
        optimizer.step()
```

**策略3：Soft Reset（推荐）**
```python
def soft_reset(agent, reset_ratio=0.5):
    for param in agent.policy_head.parameters():
        noise = torch.randn_like(param) * param.std()
        param.data = (1 - reset_ratio) * param.data + reset_ratio * noise
```

### 5.3 效果

1. **逃离局部最优**：被高估但实际表现差的 Agent 得到"重生"机会
2. **保留有用知识**：特征提取器的学习成果被保留
3. **加速收敛**：比完全随机初始化更快恢复

---

## 六、更新后的完整算法

```
Algorithm: Equity-MARL v2.0 (E-MARL v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For t = 1 to T:
  
  ┌─────────────────────────────────────┐
  │ 1. 执行层                           │
  └─────────────────────────────────────┘
  For each agent i:
    ε_i = ε_base · (σ_IV / σ_HV)^α      # IV 调节探索率
    a_i = π_i(s; ε_i)
  R = Env.step(a)

  ┌─────────────────────────────────────┐
  │ 2. 清算层                           │
  └─────────────────────────────────────┘
  φ = ShapleyValue(R)                    # 蒙特卡洛近似

  ┌─────────────────────────────────────┐
  │ 3. 估值层 (v2.0 改进)                │
  └─────────────────────────────────────┘
  For each agent i:
    # 带衰减的股价更新
    S_i ← (1-λ)·S_i·exp(r_i) + λ·S_base
    
    # 浮动执行价格（亚式期权）
    K_i ← MovingAverage(S_i, W)
    
    # 期权定价
    σ_i ← Std(φ_i history)
    C_i ← BlackScholes(S_i, K_i, σ_i, T-t)

  ┌─────────────────────────────────────┐
  │ 4. 操盘层 (Markowitz 优化)           │
  └─────────────────────────────────────┘
  μ = [C_1, ..., C_n]
  Σ = Cov(φ history)
  w* = argmax_w (w'μ - λ/2·w'Σw)         # 均值-方差优化

  ┌─────────────────────────────────────┐
  │ 5. 泡沫检测与重组                    │
  └─────────────────────────────────────┘
  For each agent i:
    If Bubble_i > θ AND φ_i dropped sharply:
      SoftReset(agent_i)

  ┌─────────────────────────────────────┐
  │ 6. 学习层                           │
  └─────────────────────────────────────┘
  For each agent i:
    r_i^w = w_i · Δ_i · φ_i              # Delta 动态折扣
    π_i ← PolicyGradient(r_i^w)
```

---

## 七、实验验证路线图

### Phase 1: 概念验证 (Toy Environment)

| 实验 | 环境 | 验证目标 |
|------|------|---------|
| 1.1 | Multi-armed Bandit | $S_i(t)$ 与最优臂的相关性 |
| 1.2 | Matrix Game | Shapley Value 的准确性 |
| 1.3 | Simple Gridworld | 二叉树 vs BS 解析解的一致性 |

### Phase 2: 单一改进验证 (MPE/SMAC)

| 实验 | 对比组 | 验证目标 |
|------|--------|---------|
| 2.1 | QMIX vs QMIX+Δ-discount | Greeks 动态折扣的效果 |
| 2.2 | 简单分配 vs Markowitz | 投资组合优化的多样性增强 |
| 2.3 | 无重组 vs Soft Reset | 泡沫重组的逃离局部最优能力 |

### Phase 3: 完整系统验证

| 实验 | Baseline | 指标 |
|------|----------|------|
| 3.1 | QMIX, MAPPO, SHAQ | 最终收益、收敛速度 |
| 3.2 | - | Agent 多样性（策略熵） |
| 3.3 | - | 对环境非平稳性的适应能力 |

---

*Updated: January 2026*
*Version: 2.0*
