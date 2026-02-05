"""
OptionRL vs Q-Learning: 学习曲线与策略可视化

生成:
1. 学习曲线对比图 (成功率 vs Episode)
2. Q值热力图 (展示策略演变)
3. 期权价格 C(s) 可视化
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, List
import os

# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# LongChain 环境
# =============================================================================

class LongChainEnv:
    """一维长链环境：从状态 0 走到状态 n-1，只有终点有奖励"""
    
    def __init__(self, n_states: int = 30, slip_prob: float = 0.1):
        self.n_states = n_states
        self.n_actions = 2  # 0: 左, 1: 右
        self.slip_prob = slip_prob
        self.goal_state = n_states - 1
        self.state = 0
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action: int):
        # 有一定概率滑动（随机动作）
        if random.random() < self.slip_prob:
            action = random.randint(0, 1)
        
        if action == 1:  # 右
            self.state = min(self.state + 1, self.n_states - 1)
        else:  # 左
            self.state = max(self.state - 1, 0)
        
        # 只有到达终点才有奖励
        if self.state == self.goal_state:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_matrix(self):
        """获取转移概率矩阵 P[s, a, s']"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # 实际动作
                if a == 1:  # 右
                    next_s = min(s + 1, self.n_states - 1)
                else:  # 左
                    next_s = max(s - 1, 0)
                
                # 滑动时的动作
                if a == 1:
                    slip_s = max(s - 1, 0)
                else:
                    slip_s = min(s + 1, self.n_states - 1)
                
                P[s, a, next_s] += 1 - self.slip_prob
                P[s, a, slip_s] += self.slip_prob
        return P


# =============================================================================
# GridMaze 环境
# =============================================================================

class GridMazeEnv:
    """网格迷宫：从左上角走到右下角"""
    
    def __init__(self, size: int = 10):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右
        self.goal_state = self.n_states - 1
        self.state = 0
        
    def reset(self):
        self.state = 0
        return self.state
    
    def _pos_to_state(self, row, col):
        return row * self.size + col
    
    def _state_to_pos(self, state):
        return state // self.size, state % self.size
    
    def step(self, action: int):
        row, col = self._state_to_pos(self.state)
        
        if action == 0:  # 上
            row = max(row - 1, 0)
        elif action == 1:  # 下
            row = min(row + 1, self.size - 1)
        elif action == 2:  # 左
            col = max(col - 1, 0)
        else:  # 右
            col = min(col + 1, self.size - 1)
        
        self.state = self._pos_to_state(row, col)
        
        if self.state == self.goal_state:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_matrix(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            row, col = self._state_to_pos(s)
            for a in range(self.n_actions):
                if a == 0:  # 上
                    next_row, next_col = max(row - 1, 0), col
                elif a == 1:  # 下
                    next_row, next_col = min(row + 1, self.size - 1), col
                elif a == 2:  # 左
                    next_row, next_col = row, max(col - 1, 0)
                else:  # 右
                    next_row, next_col = row, min(col + 1, self.size - 1)
                next_s = self._pos_to_state(next_row, next_col)
                P[s, a, next_s] = 1.0
        return P


# =============================================================================
# DP 计算期权价格
# =============================================================================

def compute_option_price_dp(P: np.ndarray, goal_state: int, gamma: float = 0.99, 
                            n_iterations: int = 100) -> np.ndarray:
    """用 DP 计算期权价格 C(s) = E[成功到达目标 | s]"""
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    V[goal_state] = 1.0
    
    for _ in range(n_iterations):
        V_new = np.zeros(n_states)
        V_new[goal_state] = 1.0
        for s in range(n_states):
            if s == goal_state:
                continue
            # 均匀随机策略
            total = 0.0
            for a in range(n_actions):
                for s_next in range(n_states):
                    total += P[s, a, s_next] * V[s_next] / n_actions
            V_new[s] = total
        V = V_new
    return V


def compute_option_price_with_policy(P: np.ndarray, Q: np.ndarray, goal_state: int,
                                     eps: float = 0.1, gamma: float = 0.99,
                                     n_iterations: int = 30) -> np.ndarray:
    """用当前策略计算期权价格（优化版）"""
    n_states, n_actions, _ = P.shape
    V = np.zeros(n_states)
    V[goal_state] = 1.0
    
    # 预计算策略概率
    greedy_actions = np.argmax(Q, axis=1)
    
    for _ in range(n_iterations):
        V_new = np.zeros(n_states)
        V_new[goal_state] = 1.0
        for s in range(n_states):
            if s == goal_state:
                continue
            greedy_a = greedy_actions[s]
            total = 0.0
            for a in range(n_actions):
                if a == greedy_a:
                    pi_a = 1 - eps + eps / n_actions
                else:
                    pi_a = eps / n_actions
                # 使用矩阵乘法加速
                total += pi_a * np.dot(P[s, a], V)
            V_new[s] = total
        V = V_new
    return V


# =============================================================================
# 训练函数（带历史记录）
# =============================================================================

def train_q_learning(env, n_episodes: int = 5000, alpha: float = 0.1, 
                     gamma: float = 0.99, eps: float = 0.1,
                     record_every: int = 100) -> Tuple[List[float], List[np.ndarray]]:
    """Q-Learning 训练，返回成功率曲线和 Q 表快照"""
    Q = np.zeros((env.n_states, env.n_actions))
    successes = []
    Q_history = []
    
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0
        max_steps = env.n_states * 3
        
        while not done and steps < max_steps:
            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))
            
            s_next, r, done = env.step(a)
            td_target = r + (0.0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
            steps += 1
        
        successes.append(1.0 if r > 0 else 0.0)
        
        if (ep + 1) % record_every == 0:
            Q_history.append(Q.copy())
    
    return successes, Q_history


def train_optionrl(env, P: np.ndarray, n_episodes: int = 5000, alpha: float = 0.1,
                   gamma: float = 0.99, r_rate: float = 0.05, eps: float = 0.1,
                   update_C_every: int = 50, blend_ratio: float = 0.5,
                   record_every: int = 100) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """OptionRL 训练，返回成功率曲线、Q 表快照和 C 值快照"""
    Q = np.zeros((env.n_states, env.n_actions))
    successes = []
    Q_history = []
    C_history = []
    
    disc = math.exp(-r_rate)
    C = compute_option_price_dp(P, env.goal_state)
    
    for ep in range(n_episodes):
        if ep > 0 and ep % update_C_every == 0:
            C = compute_option_price_with_policy(P, Q, env.goal_state, eps=eps)
        
        current_blend = max(0.1, blend_ratio * (1 - ep / n_episodes))
        
        s = env.reset()
        done = False
        steps = 0
        max_steps = env.n_states * 3
        
        while not done and steps < max_steps:
            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))
            
            s_next, r, done = env.step(a)
            
            remaining = max_steps - steps
            C_next = (disc ** max(remaining, 1)) * C[s_next]
            Q_bootstrap = 0.0 if done else gamma * np.max(Q[s_next])
            
            td_target = r + current_blend * C_next + (1 - current_blend) * Q_bootstrap
            Q[s, a] += alpha * (td_target - Q[s, a])
            
            s = s_next
            steps += 1
        
        successes.append(1.0 if r > 0 else 0.0)
        
        if (ep + 1) % record_every == 0:
            Q_history.append(Q.copy())
            C_history.append(C.copy())
    
    return successes, Q_history, C_history


# =============================================================================
# 可视化函数
# =============================================================================

def smooth_curve(data: List[float], window: int = 100) -> np.ndarray:
    """平滑曲线"""
    data = np.array(data)
    if len(data) < window:
        return np.cumsum(data) / (np.arange(len(data)) + 1)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_learning_curves(succ_q: List[float], succ_opt: List[float], 
                         title: str, save_path: str):
    """绘制学习曲线对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    smooth_q = smooth_curve(succ_q)
    smooth_opt = smooth_curve(succ_opt)
    
    ax.plot(smooth_q, label='Q-Learning', color='#e74c3c', linewidth=2, alpha=0.8)
    ax.plot(smooth_opt, label='OptionRL-DP', color='#3498db', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (smoothed)', fontsize=12)
    ax.set_title(f'{title}: Learning Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # 添加注释
    final_q = smooth_q[-1] if len(smooth_q) > 0 else 0
    final_opt = smooth_opt[-1] if len(smooth_opt) > 0 else 0
    ax.annotate(f'Final: {final_q:.1%}', xy=(len(smooth_q)-1, final_q),
                xytext=(-80, 20), textcoords='offset points',
                fontsize=10, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.5))
    ax.annotate(f'Final: {final_opt:.1%}', xy=(len(smooth_opt)-1, final_opt),
                xytext=(-80, -30), textcoords='offset points',
                fontsize=10, color='#3498db',
                arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 学习曲线已保存: {save_path}")


def plot_chain_policy_evolution(Q_history_q: List[np.ndarray], 
                                 Q_history_opt: List[np.ndarray],
                                 C_history: List[np.ndarray],
                                 n_states: int,
                                 title: str, save_path: str):
    """绘制 LongChain 的策略演变"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    n_snapshots = min(4, len(Q_history_opt))
    indices = [int(i * (len(Q_history_opt) - 1) / (n_snapshots - 1)) for i in range(n_snapshots)]
    
    # 上半部分：Q 值演变
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    states = np.arange(n_states)
    
    # Q-Learning Q值
    for idx in indices:
        Q = Q_history_q[idx] if idx < len(Q_history_q) else Q_history_q[-1]
        V = np.max(Q, axis=1)
        ep = (idx + 1) * 100
        ax1.plot(states, V, label=f'Ep {ep}', alpha=0.7)
    ax1.set_xlabel('State', fontsize=10)
    ax1.set_ylabel('max Q(s,a)', fontsize=10)
    ax1.set_title('Q-Learning: Value Evolution', fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # OptionRL Q值
    for idx in indices:
        Q = Q_history_opt[idx] if idx < len(Q_history_opt) else Q_history_opt[-1]
        V = np.max(Q, axis=1)
        ep = (idx + 1) * 100
        ax2.plot(states, V, label=f'Ep {ep}', alpha=0.7)
    ax2.set_xlabel('State', fontsize=10)
    ax2.set_ylabel('max Q(s,a)', fontsize=10)
    ax2.set_title('OptionRL: Value Evolution', fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 中间：期权价格 C(s) 演变
    ax3 = fig.add_subplot(gs[1, :])
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(indices)))
    for i, idx in enumerate(indices):
        C = C_history[idx] if idx < len(C_history) else C_history[-1]
        ep = (idx + 1) * 100
        ax3.plot(states, C, label=f'Ep {ep}', color=colors[i], linewidth=2)
    ax3.set_xlabel('State', fontsize=10)
    ax3.set_ylabel('Option Price C(s)', fontsize=10)
    ax3.set_title('OptionRL: Option Price C(s) Evolution', fontsize=11)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Goal')
    
    # 下半部分：最终策略对比
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    
    Q_final_q = Q_history_q[-1] if Q_history_q else np.zeros((n_states, 2))
    Q_final_opt = Q_history_opt[-1] if Q_history_opt else np.zeros((n_states, 2))
    
    policy_q = np.argmax(Q_final_q, axis=1)
    policy_opt = np.argmax(Q_final_opt, axis=1)
    
    # 策略可视化：箭头表示
    for s in range(n_states):
        color_q = '#e74c3c' if policy_q[s] == 1 else '#95a5a6'  # 右=红, 左=灰
        color_opt = '#3498db' if policy_opt[s] == 1 else '#95a5a6'
        
        ax4.arrow(s, 0.5, 0.3 if policy_q[s] == 1 else -0.3, 0, 
                  head_width=0.15, head_length=0.1, fc=color_q, ec=color_q)
        ax5.arrow(s, 0.5, 0.3 if policy_opt[s] == 1 else -0.3, 0,
                  head_width=0.15, head_length=0.1, fc=color_opt, ec=color_opt)
    
    ax4.set_xlim(-1, n_states)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel('State', fontsize=10)
    ax4.set_title('Q-Learning: Final Policy (→=Right, ←=Left)', fontsize=11)
    ax4.axvline(x=n_states-1, color='green', linestyle='--', linewidth=2, label='Goal')
    
    ax5.set_xlim(-1, n_states)
    ax5.set_ylim(0, 1)
    ax5.set_xlabel('State', fontsize=10)
    ax5.set_title('OptionRL: Final Policy (→=Right, ←=Left)', fontsize=11)
    ax5.axvline(x=n_states-1, color='green', linestyle='--', linewidth=2, label='Goal')
    
    fig.suptitle(f'{title}: Policy and Value Evolution', fontsize=14, fontweight='bold', y=1.02)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 策略演变图已保存: {save_path}")


def plot_grid_policy(Q: np.ndarray, C: np.ndarray, size: int, 
                     title: str, save_path: str):
    """绘制 GridMaze 的策略热力图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Q值热力图
    V = np.max(Q, axis=1).reshape(size, size)
    im1 = axes[0].imshow(V, cmap='RdYlGn', aspect='equal')
    axes[0].set_title('Q-Learning: max Q(s,a)', fontsize=11)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 期权价格热力图
    C_grid = C.reshape(size, size)
    im2 = axes[1].imshow(C_grid, cmap='Blues', aspect='equal')
    axes[1].set_title('OptionRL: Option Price C(s)', fontsize=11)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 策略箭头图
    policy = np.argmax(Q, axis=1).reshape(size, size)
    axes[2].imshow(np.ones((size, size)), cmap='gray', alpha=0.1, aspect='equal')
    
    # 箭头方向：0=上, 1=下, 2=左, 3=右
    dx = [0, 0, -0.3, 0.3]
    dy = [-0.3, 0.3, 0, 0]
    
    for i in range(size):
        for j in range(size):
            a = policy[i, j]
            axes[2].arrow(j, i, dx[a], dy[a], head_width=0.15, head_length=0.1,
                         fc='#3498db', ec='#3498db')
    
    axes[2].set_title('OptionRL: Final Policy', fontsize=11)
    axes[2].plot(size-1, size-1, 'g*', markersize=20, label='Goal')
    axes[2].plot(0, 0, 'ro', markersize=10, label='Start')
    axes[2].legend(loc='upper left')
    
    for ax in axes:
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  🗺️ 网格策略图已保存: {save_path}")


def plot_summary_comparison(results: dict, save_path: str):
    """绘制汇总对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    envs = list(results.keys())
    q_rates = [results[e]['q_final'] for e in envs]
    opt_rates = [results[e]['opt_final'] for e in envs]
    
    x = np.arange(len(envs))
    width = 0.35
    
    # 最终成功率对比
    bars1 = axes[0].bar(x - width/2, q_rates, width, label='Q-Learning', color='#e74c3c', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, opt_rates, width, label='OptionRL-DP', color='#3498db', alpha=0.8)
    
    axes[0].set_ylabel('Final Success Rate', fontsize=12)
    axes[0].set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(envs, rotation=15, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars1, q_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, opt_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    
    # 首次成功 Episode 对比
    q_first = [results[e]['q_first'] if results[e]['q_first'] is not None else 5000 for e in envs]
    opt_first = [results[e]['opt_first'] if results[e]['opt_first'] is not None else 5000 for e in envs]
    
    bars3 = axes[1].bar(x - width/2, q_first, width, label='Q-Learning', color='#e74c3c', alpha=0.8)
    bars4 = axes[1].bar(x + width/2, opt_first, width, label='OptionRL-DP', color='#3498db', alpha=0.8)
    
    axes[1].set_ylabel('First Success Episode', fontsize=12)
    axes[1].set_title('Sample Efficiency (First Success)', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(envs, rotation=15, ha='right')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # 标记"从未成功"
    for i, (q, o) in enumerate(zip(q_first, opt_first)):
        if q >= 5000:
            axes[1].text(x[i] - width/2, q + 50, 'Never', ha='center', fontsize=8, color='#e74c3c')
        if o >= 5000:
            axes[1].text(x[i] + width/2, o + 50, 'Never', ha='center', fontsize=8, color='#3498db')
    
    fig.suptitle('OptionRL vs Q-Learning: Extreme Sparse Reward Benchmark', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 汇总对比图已保存: {save_path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print(" 🎨 OptionRL vs Q-Learning: 可视化实验")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = "Draft/examples/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    n_episodes = 3000  # 减少 episode 数加快速度
    
    # =========================================================================
    # 实验 1: LongChain-30
    # =========================================================================
    print("\n🔗 实验 1: LongChain-30")
    env = LongChainEnv(n_states=30, slip_prob=0.1)
    P = env.get_transition_matrix()
    
    print("  训练 Q-Learning...")
    succ_q, Q_hist_q = train_q_learning(env, n_episodes=n_episodes)
    
    print("  训练 OptionRL...")
    env = LongChainEnv(n_states=30, slip_prob=0.1)  # 重置
    succ_opt, Q_hist_opt, C_hist = train_optionrl(env, P, n_episodes=n_episodes)
    
    # 记录结果
    first_q = next((i for i, s in enumerate(succ_q) if s > 0), None)
    first_opt = next((i for i, s in enumerate(succ_opt) if s > 0), None)
    results['LongChain-30'] = {
        'q_final': np.mean(succ_q[-500:]),
        'opt_final': np.mean(succ_opt[-500:]),
        'q_first': first_q,
        'opt_first': first_opt,
    }
    
    # 绘图
    plot_learning_curves(succ_q, succ_opt, 'LongChain-30', 
                        f'{output_dir}/longchain30_learning_curves.png')
    plot_chain_policy_evolution(Q_hist_q, Q_hist_opt, C_hist, 30,
                                'LongChain-30', f'{output_dir}/longchain30_policy_evolution.png')
    
    # =========================================================================
    # 实验 2: LongChain-50
    # =========================================================================
    print("\n🔗 实验 2: LongChain-50")
    env = LongChainEnv(n_states=50, slip_prob=0.1)
    P = env.get_transition_matrix()
    
    print("  训练 Q-Learning...")
    succ_q, Q_hist_q = train_q_learning(env, n_episodes=n_episodes)
    
    print("  训练 OptionRL...")
    env = LongChainEnv(n_states=50, slip_prob=0.1)
    succ_opt, Q_hist_opt, C_hist = train_optionrl(env, P, n_episodes=n_episodes)
    
    first_q = next((i for i, s in enumerate(succ_q) if s > 0), None)
    first_opt = next((i for i, s in enumerate(succ_opt) if s > 0), None)
    results['LongChain-50'] = {
        'q_final': np.mean(succ_q[-500:]),
        'opt_final': np.mean(succ_opt[-500:]),
        'q_first': first_q,
        'opt_first': first_opt,
    }
    
    plot_learning_curves(succ_q, succ_opt, 'LongChain-50',
                        f'{output_dir}/longchain50_learning_curves.png')
    plot_chain_policy_evolution(Q_hist_q, Q_hist_opt, C_hist, 50,
                                'LongChain-50', f'{output_dir}/longchain50_policy_evolution.png')
    
    # =========================================================================
    # 实验 3: GridMaze-10x10
    # =========================================================================
    print("\n🗺️ 实验 3: GridMaze-10x10")
    env = GridMazeEnv(size=10)
    P = env.get_transition_matrix()
    
    print("  训练 Q-Learning...")
    succ_q, Q_hist_q = train_q_learning(env, n_episodes=n_episodes)
    
    print("  训练 OptionRL...")
    env = GridMazeEnv(size=10)
    succ_opt, Q_hist_opt, C_hist = train_optionrl(env, P, n_episodes=n_episodes)
    
    first_q = next((i for i, s in enumerate(succ_q) if s > 0), None)
    first_opt = next((i for i, s in enumerate(succ_opt) if s > 0), None)
    results['GridMaze-10x10'] = {
        'q_final': np.mean(succ_q[-500:]),
        'opt_final': np.mean(succ_opt[-500:]),
        'q_first': first_q,
        'opt_first': first_opt,
    }
    
    plot_learning_curves(succ_q, succ_opt, 'GridMaze-10x10',
                        f'{output_dir}/gridmaze_learning_curves.png')
    
    # 最终策略热力图
    Q_final = Q_hist_opt[-1] if Q_hist_opt else np.zeros((100, 4))
    C_final = C_hist[-1] if C_hist else np.zeros(100)
    plot_grid_policy(Q_final, C_final, 10, 'GridMaze-10x10 Final Policy',
                    f'{output_dir}/gridmaze_policy_heatmap.png')
    
    # =========================================================================
    # 汇总图
    # =========================================================================
    plot_summary_comparison(results, f'{output_dir}/summary_comparison.png')
    
    print("\n" + "=" * 70)
    print(" ✅ 所有可视化已完成！")
    print(f" 📁 输出目录: {output_dir}/")
    print("=" * 70)
    print("\n生成的图表:")
    print("  1. longchain30_learning_curves.png   - 学习曲线")
    print("  2. longchain30_policy_evolution.png  - 策略演变")
    print("  3. longchain50_learning_curves.png   - 学习曲线")
    print("  4. longchain50_policy_evolution.png  - 策略演变")
    print("  5. gridmaze_learning_curves.png      - 学习曲线")
    print("  6. gridmaze_policy_heatmap.png       - 策略热力图")
    print("  7. summary_comparison.png            - 汇总对比")


if __name__ == "__main__":
    main()
