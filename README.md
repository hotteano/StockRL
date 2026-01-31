# Equity-MARL: 金融动力学驱动的多智能体强化学习框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **将 MARL 中的信用分配问题转化为投资组合管理问题**

## 🎯 核心思想

Equity-MARL (E-MARL) 将多智能体强化学习（MARL）抽象为金融市场：

| 金融概念 | MARL 对应 |
|---------|----------|
| **个股** | 智能体（Agent） |
| **股息** | Shapley Value（贡献度） |
| **期权价值** | 动态估值（潜力评估） |
| **投资组合** | 注意力权重分配 |
| **泡沫回调** | 正则化机制 |

## ✨ 主要特性

### 1. 博弈论最优的信用分配
使用 **Shapley Value** 解决"大锅饭"问题，基于每个 Agent 的边际贡献公平分配奖励。

### 2. 风险感知的动态估值
通过 **Black-Scholes 期权定价**，不仅考虑期望收益，还考虑波动率（风险）。

### 3. 投资组合式的权重优化
**Markowitz 均值-方差优化** 考虑 Agent 之间的相关性，自动实现多样化。

### 4. 自适应探索-利用平衡
期权的 **时间价值衰减** 天然对应从探索到利用的转换。

### 5. 泡沫检测与自我纠错
当 Agent 被高估但表现不佳时，触发 **参数重组**，逃离局部最优。

## 📐 数学框架

### Shapley Value → 信用分配
$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$$

### Black-Scholes PDE → 动态估值
$$\frac{\partial C}{\partial t} + rS\frac{\partial C}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} = rC$$

### Markowitz 优化 → 权重分配
$$\max_{\mathbf{w}} \quad \mathbf{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^T \Sigma \mathbf{w}$$

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基础使用

```python
from emarl import EquityMARL

# 初始化
emarl = EquityMARL(n_agents=4, total_steps=10000)

# 在训练循环中
for step in range(total_steps):
    team_reward = env.step(actions)  # 获得团队奖励
    
    # E-MARL 处理
    result = emarl.step(
        team_reward=team_reward,
        value_function=your_value_function
    )
    
    # 使用加权奖励更新策略
    weighted_rewards = result['weighted_rewards']
    for i, agent in enumerate(agents):
        agent.update(weighted_rewards[i])
```

### 简化接口

```python
from emarl import EasyEquityMARL

emarl = EasyEquityMARL(n_agents=4)

# 简单两行搞定
rewards, weights = emarl.process_reward(team_reward)
```

## 📁 项目结构

```
StockRL/
├── emarl/
│   ├── __init__.py          # 包入口
│   ├── option_pricing.py    # 期权定价（Black-Scholes, 二叉树）
│   ├── shapley.py           # Shapley Value 计算
│   ├── valuation.py         # 估值引擎（股价追踪）
│   ├── meta_investor.py     # Meta-Investor（Markowitz优化）
│   ├── bubble_detector.py   # 泡沫检测与参数重组
│   └── emarl_framework.py   # 主框架（整合所有组件）
├── docs/
│   ├── mathematical_derivation.md  # 完整数学推导
│   ├── core_derivation.md          # 核心推导（二叉树→BS）
│   └── improvements.md             # 改进方案（v2.0）
├── examples/
│   └── demo.py               # 演示脚本
├── requirements.txt
└── README.md
```

## 🔬 与现有方法对比

| 方法 | 信用分配 | 风险敏感 | 动态权重 | 可解释性 |
|------|---------|---------|---------|---------|
| VDN | ❌ 简单加和 | ❌ | ❌ | ⚠️ |
| QMIX | ⚠️ 非负约束 | ❌ | ❌ | ⚠️ |
| SHAQ | ✅ Shapley | ❌ | ❌ | ✅ |
| **E-MARL** | ✅ Shapley | ✅ 波动率 | ✅ 期权仓位 | ✅ 金融语义 |

## 🧪 运行测试

```bash
# 测试各模块
python -m emarl.option_pricing
python -m emarl.shapley
python -m emarl.valuation
python -m emarl.meta_investor
python -m emarl.emarl_framework

# 运行演示
cd examples && python demo.py
```

## 📖 文档

详细的数学推导请参阅：
- [完整数学推导](docs/mathematical_derivation.md)
- [核心推导：从二叉树到 Black-Scholes](docs/core_derivation.md)
- [改进方案 v2.0](docs/improvements.md)

## 🎓 学术贡献

本框架首次将以下三个领域统一：
1. **合作博弈论**（Shapley Value）
2. **金融工程**（期权定价）
3. **强化学习**（策略梯度）

### 创新点
- **Greeks 的 RL 解释**：Delta 作为动态折扣因子，Theta 作为探索率衰减
- **泡沫机制**：内置正则化，防止过拟合单一 Agent
- **金融可解释性**：每个组件都有清晰的金融语义

## 📝 引用

如果您使用了本框架，请引用：

```bibtex
@software{emarl2026,
  title={Equity-MARL: A Financial Dynamics Framework for Multi-Agent Reinforcement Learning},
  author={StockRL Project},
  year={2026},
  url={https://github.com/your-repo/StockRL}
}
```

## 📄 License

MIT License

---

*Made with ❤️ for the MARL research community*
