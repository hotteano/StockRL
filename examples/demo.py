"""
Equity-MARL Demo Script

This script demonstrates the core features of the Equity-MARL framework.
Run with: python examples/demo.py
"""

import sys
sys.path.insert(0, '..')

import torch
import numpy as np
import matplotlib.pyplot as plt
from emarl import EquityMARL, BlackScholesLayer


def demo_black_scholes():
    """Demonstrate the Differentiable Black-Scholes Layer."""
    print("\n" + "=" * 60)
    print("Demo 1: Differentiable Black-Scholes Layer")
    print("=" * 60)
    
    bs = BlackScholesLayer()
    
    # Create parameters with gradients
    S = torch.tensor([100.0], requires_grad=True)
    K = torch.tensor([100.0])
    sigma = torch.tensor([0.2])
    T = torch.tensor([1.0])
    r = 0.05
    
    # Compute option price and Greeks
    C, greeks = bs(S, K, sigma, T, r, return_greeks=True)
    
    print(f"\n期权定价结果 (Option Pricing Results):")
    print(f"  标的价格 (Spot): {S.item():.2f}")
    print(f"  执行价格 (Strike): {K.item():.2f}")
    print(f"  波动率 (Volatility): {sigma.item():.2%}")
    print(f"  到期时间 (Time to Maturity): {T.item():.2f} years")
    print(f"  无风险利率 (Risk-free Rate): {r:.2%}")
    print(f"\n  期权价值 (Option Value): {C.item():.4f}")
    print(f"  Delta (Δ): {greeks['delta'].item():.4f}")
    print(f"  Gamma (Γ): {greeks['gamma'].item():.6f}")
    print(f"  Theta (Θ): {greeks['theta'].item():.4f}")
    print(f"  Vega (ν): {greeks['vega'].item():.4f}")
    
    # Demonstrate differentiability
    C.backward()
    print(f"\n  梯度 ∂C/∂S (Gradient): {S.grad.item():.4f}")
    print("  ✓ 完全可微分，支持端到端训练!")


def demo_equity_marl():
    """Demonstrate the full Equity-MARL framework."""
    print("\n" + "=" * 60)
    print("Demo 2: Equity-MARL Framework")
    print("=" * 60)
    
    n_agents = 4
    n_episodes = 50
    steps_per_episode = 20
    
    # Initialize framework
    emarl = EquityMARL(
        n_agents=n_agents,
        total_steps=n_episodes * steps_per_episode,
        risk_aversion=0.5,
        bubble_threshold=1.8,
        device='cpu'
    )
    
    # Agent performance profiles (simulated)
    # Agent 0: Strong and stable
    # Agent 1: Average
    # Agent 2: Weak but improving
    # Agent 3: High variance (risky)
    
    agent_profiles = {
        0: {'mean': 1.5, 'std': 0.1, 'trend': 0.0},
        1: {'mean': 1.0, 'std': 0.2, 'trend': 0.0},
        2: {'mean': 0.6, 'std': 0.15, 'trend': 0.01},
        3: {'mean': 1.2, 'std': 0.8, 'trend': 0.0},
    }
    
    # Track history
    weight_history = []
    shapley_history = []
    option_history = []
    
    print("\n开始训练模拟 (Starting Training Simulation)...")
    
    for episode in range(n_episodes):
        for step in range(steps_per_episode):
            t = episode * steps_per_episode + step
            
            # Generate team reward
            team_reward = 10.0 + np.random.randn() * 2
            
            # Define value function based on agent profiles
            def value_function(mask):
                value = 0
                for i in range(n_agents):
                    if mask[i] > 0.5:
                        profile = agent_profiles[i]
                        contribution = (
                            profile['mean'] + 
                            profile['trend'] * t / 100 + 
                            np.random.randn() * profile['std']
                        )
                        value += max(0, contribution)
                return value / n_agents * team_reward
            
            # E-MARL step
            result = emarl.step(
                team_reward=torch.tensor(team_reward),
                value_function=value_function
            )
            
            # Record history
            weight_history.append(result['weights'].numpy())
            shapley_history.append(result['shapley_values'].numpy())
            option_history.append(result['option_values'].detach().numpy())
    
    # Print final results
    print("\n最终结果 (Final Results):")
    stats = emarl.get_summary_statistics()
    
    print(f"\n  Agent | Shapley均值 | 最终权重 | 期权价值 | 崩盘次数")
    print(f"  " + "-" * 55)
    for i in range(n_agents):
        shapley_mean = stats['mean_shapley'][i]
        final_weight = stats['final_weights'][i]
        option_mean = stats['mean_option_value'][i]
        crashes = stats['total_crashes'][i]
        print(f"    {i}   |   {shapley_mean:6.3f}   |  {final_weight:5.1%}  |  {option_mean:6.3f}  |    {int(crashes)}")
    
    # Plot results
    plot_results(weight_history, shapley_history, option_history, n_agents)


def plot_results(weight_history, shapley_history, option_history, n_agents):
    """Plot training results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    weight_arr = np.array(weight_history)
    shapley_arr = np.array(shapley_history)
    option_arr = np.array(option_history)
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    labels = [f'Agent {i}' for i in range(n_agents)]
    
    # Plot 1: Portfolio weights over time
    ax1 = axes[0, 0]
    for i in range(n_agents):
        ax1.plot(weight_arr[:, i], color=colors[i], label=labels[i], alpha=0.8)
    ax1.set_title('Portfolio Weights (仓位权重)', fontsize=12)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Weight')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Shapley values over time
    ax2 = axes[0, 1]
    window = 20
    for i in range(n_agents):
        # Smoothed version
        smoothed = np.convolve(shapley_arr[:, i], np.ones(window)/window, mode='valid')
        ax2.plot(smoothed, color=colors[i], label=labels[i], alpha=0.8)
    ax2.set_title('Shapley Values (贡献度)', fontsize=12)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Shapley Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Option values
    ax3 = axes[1, 0]
    for i in range(n_agents):
        smoothed = np.convolve(option_arr[:, i], np.ones(window)/window, mode='valid')
        ax3.plot(smoothed, color=colors[i], label=labels[i], alpha=0.8)
    ax3.set_title('Option Values (期权估值)', fontsize=12)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Option Value')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final weight distribution
    ax4 = axes[1, 1]
    final_weights = weight_arr[-1]
    bars = ax4.bar(labels, final_weights, color=colors)
    ax4.set_title('Final Portfolio Distribution (最终仓位分布)', fontsize=12)
    ax4.set_ylabel('Weight')
    ax4.set_ylim(0, 1)
    for bar, w in zip(bars, final_weights):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{w:.1%}', ha='center', va='bottom', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('emarl_demo_results.png', dpi=150)
    print("\n  ✓ 结果图已保存至 emarl_demo_results.png")
    plt.show()


def demo_greeks_interpretation():
    """Demonstrate how Greeks are used in E-MARL."""
    print("\n" + "=" * 60)
    print("Demo 3: Greeks 在 MARL 中的解释")
    print("=" * 60)
    
    print("""
    在 Equity-MARL 中，期权的希腊字母有特殊含义：
    
    ┌─────────┬────────────────────────────────────────────┐
    │ Greek   │ MARL 解释                                  │
    ├─────────┼────────────────────────────────────────────┤
    │ Delta Δ │ 动态折扣因子                               │
    │         │ - Δ → 1: 表现好的 Agent，看重长期 (γ ≈ 1)  │
    │         │ - Δ → 0: 表现差的 Agent，聚焦即时 (γ ≈ 0)  │
    ├─────────┼────────────────────────────────────────────┤
    │ Gamma Γ │ Delta 的稳定性                             │
    │         │ - 高 Γ: Agent 处于临界状态，需要关注       │
    │         │ - 低 Γ: Agent 状态稳定                     │
    ├─────────┼────────────────────────────────────────────┤
    │ Theta Θ │ 时间价值衰减（探索→利用的转换）            │
    │         │ - 训练初期: 高时间价值 → 多探索            │
    │         │ - 训练后期: 低时间价值 → 多利用            │
    ├─────────┼────────────────────────────────────────────┤
    │ Vega ν  │ 波动率敏感性                               │
    │         │ - 高 ν: Agent 表现受环境不确定性影响大     │
    │         │ - 低 ν: Agent 表现稳定，不受波动影响       │
    └─────────┴────────────────────────────────────────────┘
    
    这种映射使得 MARL 的行为更加可解释！
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("   Equity-MARL: 金融动力学驱动的多智能体强化学习")
    print("=" * 60)
    
    demo_black_scholes()
    demo_greeks_interpretation()
    demo_equity_marl()
    
    print("\n" + "=" * 60)
    print("Demo 完成! ✓")
    print("=" * 60)
