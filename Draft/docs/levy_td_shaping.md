# Shapley → Lévy 定价 → TD Error Shaping：一个用于稀疏奖励的 MARL 理论模型

## 0. 目标与直觉

在团队稀疏奖励的 MARL 中，常见问题是：大多数时间步（甚至大多数 episode）团队回报为 0 或极少出现正反馈，导致信用分配与价值学习信号极弱，训练难以收敛。

本模型把“**贡献分配**”与“**未来潜力估值**”显式分离：

1. 用单轮 episode 总收益构造合作博弈，并用 **Shapley Value** 将“已实现收益”公平分摊给每个智能体；
2. 把每个智能体的能力演化视作 **Lévy 过程**（跳跃过程是其特例），并用一个“基准测度/风险中等假设”把“未来潜力”转化为 **期权溢价（premium）** 或更一般的**估值信号**；
3. 用该估值信号替换/增强原有的 TD 目标或 TD error，从而在奖励稀疏时仍有稳定的、可解释的学习信号。

> 注：这里的“风险中性/基准测度”是 RL 建模选择（归一化规则），不需要对应真实金融市场。

---

## 1. Episode 总收益与 Shapley 信用分配

设第 $e$ 个 episode 的团队总收益为

$$G^{(e)} = \sum_{t=0}^{T-1} \gamma^t\, r^{\text{team}}_t.$$

构造合作博弈 $(N,v)$：$N=\{1,\dots,n\}$ 为智能体集合，$v(S)$ 为仅允许联盟 $S\subseteq N$ 参与时的（期望）episode 总收益。

Shapley 分配为

$$\phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\,[v(S\cup\{i\})-v(S)].$$

在实现层面，$v(S)$ 可用“mask 后重放/重采样 episode”的方式估计，也可用近似价值函数（如 centralized critic）来估计。

效率公理保证：

$$\sum_i \phi_i = v(N) \approx G^{(e)}.$$

我们把 $\phi_i$ 视为该 episode 对智能体 $i$ 的“**已实现分红/现金流**”。

---

## 2. Lévy 过程建模：能力跃迁的跳跃结构

为表达“顿悟式提升/策略发现”，把智能体 $i$ 的隐含市值（或能力资本）过程记为 $S^i_t$，采用 Lévy 过程驱动的对数价格：

$$\log S^i_t = \log S^i_0 + \mu_i t + L^i_t,$$

其中 $L^i_t$ 是 Lévy 过程（具有独立、平稳增量），允许由跳跃主导。

一个与本文仓库推导一致、且最贴合“顿悟跃迁”的特例是 **复合泊松（compound Poisson）**：

$$L^i_t = \sum_{k=1}^{N^i_t} Y^i_k,$$

其中 $N^i_t\sim\text{Poi}(\lambda_i t)$ 表示“突破/顿悟次数”，$Y^i_k$ 为跳跃幅度（可取常数 $\log(1+\sigma_i)$，也可取分布）。

当 $Y^i_k\equiv \log(1+\sigma_i)$ 时，得到几何泊松过程：

$$S^i_t = S^i_0\,e^{\mu_i t}(1+\sigma_i)^{N^i_t}.$$

---

## 3. 风险中性/风险中等（基准）假设：把“潜力”变成可学习信号

我们选定一个基准测度 $\mathbb{Q}$（可称“定价测度”），并规定折现后的过程为鞅：

$$\mathbb{E}^{\mathbb{Q}}[S^i_T\mid\mathcal{F}_t] = S^i_t\,e^{r(T-t)}.$$

在几何泊松且跳幅为常数 $\sigma_i$、强度为 $\lambda_i$ 的情形，这对应漂移修正

$$\mu_i = r - \sigma_i\lambda_i.$$

### 3.1 “期权溢价（premium）”作为非平凡估值

在标准金融口径中：

- 标的资产的“价格”是 $S_t$；
- 期权定价公式给出的“价格”通常是 **期权溢价** $C_t$（premium）。

在 RL 里我们真正需要的是一个**非平凡、可密集学习**的信号来表达“未来潜力”。因此更自然的选择是直接使用期权溢价：

$$\text{Premium}_i(t)=C_i(t)=e^{-r\tau}\,\mathbb{E}^{\mathbb{Q}}\big[\max(S^i_{t+\tau}-K_i,0)\mid\mathcal{F}_t\big].$$

在你已有的泊松推导（固定跳幅）下，可进一步化为 CCDF 的闭式：

$$C = S\,Q(k^*;\lambda(1+\sigma)\tau) - K e^{-r\tau} Q(k^*;\lambda\tau).$$

> 解释：$S$ 是当前“资本/市值”，$K$ 是基准（例如跨智能体平均、或自身历史均值），$\tau$ 是“剩余训练周期/剩余 episode 比例”。

### 3.2 如果你确实想要“单个标量总估值”：它对应一个组合头寸价值

如果你希望一个“总估值”标量把“当前能力水平”和“未来潜力”都包含进去，可以定义**组合头寸价值**（而不是把它称为“定价公式的结果”）：

$$\Pi_i(t)=w_S S^i_t + w_C C_i(t).$$

当 $w_S=w_C=1$ 时，$\Pi_i=S+C$ 表示“持有 1 单位标的 + 1 份看涨期权”的组合价值。
这时 $\Pi_i-S$ 的确就是期权溢价 $C$，但它并不“平凡”，因为 **$C$ 本身就是你希望注入 TD 的非平凡信号**。

---

## 4. 用估值信号替换/增强 TD error

下面给出两种最常用、也最容易落地的连接方式。

### 4.1 作为 critic 的“目标值”（value target）

对于 episode 结束时刻（或每个 step 聚合后），把 $C_i$（或你选定的 $\Pi_i$）当作监督信号：

$$\mathcal{L}_V = \big(V_{\theta,i}(s) - \text{Signal}_i\big)^2,\quad \text{Signal}_i\in\{C_i,\Pi_i\}.$$

直觉：即使 $G^{(e)}$ 很稀疏，$C_i$ 仍可通过波动与剩余时间价值给出非零的“未来潜力”，从而稳定训练。

### 4.2 作为 TD error 的“替代奖励”（TD reward shaping）

对 TD(0) 形式：

$$\delta^i_t = \tilde r^i_t + \gamma V_{\theta,i}(s_{t+1}) - V_{\theta,i}(s_t).$$

用估值信号构造 $\tilde r^i_t$ 的常见方式：

- **末端注入**（最贴合 episode 总收益定义）：只在 terminal step 用
  $$\tilde r^i_{T-1}=\text{Signal}_i,\quad \tilde r^i_{t<T-1}=0.$$
- **均匀摊销**：
  $$\tilde r^i_t = \text{Signal}_i/T.$$
- **混合**：
  $$\tilde r^i_t=(1-\eta)\,r^{\text{shapley}}_t + \eta\,r^{\text{price}}_t,$$
  其中 $r^{\text{price}}_t$ 由 $\Pi_i$ 构造。

---

## 5. 稀疏奖励为何更容易收敛（机制解释）

- Shapley 把“偶发的团队成功”拆成可解释的个体贡献，避免纯均分/纯局部奖励带来的高方差。
- Lévy/泊松定价把“未来可能的跃迁次数/幅度”折算为 premium，使信号不只依赖一次性爆发。
- 定价引擎天然引入时间到期 $\tau$：前期 premium 大（探索价值高），后期 premium 衰减（收敛更稳）。

---

## 6. 与仓库代码的最小对接点

本仓库的 `EquityMARL.step()` 已能返回：

- `shapley_values`：$\phi_i$
- `prices`：$S^i_t$（本金/资本）
- `option_values`：$C_i$（预期增量/premium）
- `priced_values`：$S^i_t + C_i$（组合头寸价值：标的 + 期权）
- `pricing_premium`：$C_i$（期权溢价，本质“定价公式的结果”）

奖励稀疏场景下更推荐直接用 `pricing_premium`（即 $C_i$）做 shaping；若你需要把“当前能力水平”一并编码进信号，再使用 `priced_values`。
