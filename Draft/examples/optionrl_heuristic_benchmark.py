
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Tuple

# 设置随机种子
np.random.seed(42)
random.seed(42)

# =============================================================================
# 带障碍物的迷宫环境 (GridMazeWithWalls)
# =============================================================================
class GridMazeWithWalls:
    """
    带墙壁的 10x10 网格迷宫。
    有一个 U 型墙，足以让简单的距离启发式函数失效（陷入局部最优）。
    """
    def __init__(self):
        self.size = 10
        self.n_states = self.size * self.size
        self.n_actions = 4  # 0:Up, 1:Down, 2:Left, 3:Right
        self.start_state = 0  # (0,0)
        self.goal_state = 99  # (9,9)
        self.state = self.start_state
        
        # 定义墙壁：简单的障碍物
        # 墙壁设计：在目标由于 (9,9)，我们在 (8,9), (8,8)... 设置一堵墙，强迫必须绕远路
        # 让我们做一个倒 U 型墙，把终点包围起来，或者把中间隔断
        self.walls = set()
        
        # 构造一个 垂直墙壁，中间留口
        # 在 col=5 处，从 row=0 到 row=8 都是墙，只有 row=9 可以通过
        for r in range(0, 9):
            self.walls.add((r, 5))
            
    def reset(self):
        self.state = self.start_state
        return self.state
    
    def _pos_to_state(self, row, col):
        return row * self.size + col
    
    def _state_to_pos(self, state):
        return state // self.size, state % self.size
    
    def step(self, action: int):
        row, col = self._state_to_pos(self.state)
        old_row, old_col = row, col
        
        if action == 0:   row = max(row - 1, 0)
        elif action == 1: row = min(row + 1, self.size - 1)
        elif action == 2: col = max(col - 1, 0)
        elif action == 3: col = min(col + 1, self.size - 1)
        
        # 碰撞检测
        if (row, col) in self.walls:
            row, col = old_row, old_col
        
        self.state = self._pos_to_state(row, col)
        
        done = (self.state == self.goal_state)
        # 稀疏奖励：只有到达终点才给 1
        reward = 1.0 if done else 0.0
        
        return self.state, reward, done
    
    def get_transition_matrix(self):
        """获取环境的转移矩阵 P，用于 OptionRL 计算价格"""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            # 这是一个确定性环境，所以概率是 1
            # 但我们需要模拟 step 函数的逻辑
            curr_row, curr_col = self._state_to_pos(s)
            
            for a in range(self.n_actions):
                # 模拟移动
                next_r, next_c = curr_row, curr_col
                if a == 0:   next_r = max(curr_row - 1, 0)
                elif a == 1: next_r = min(curr_row + 1, self.size - 1)
                elif a == 2: next_c = max(curr_col - 1, 0)
                elif a == 3: next_c = min(curr_col + 1, self.size - 1)
                
                if (next_r, next_c) in self.walls:
                    next_r, next_c = curr_row, curr_col
                
                s_next = self._pos_to_state(next_r, next_c)
                P[s, a, s_next] = 1.0
        return P

# =============================================================================
# 辅助函数: 势能计算
# =============================================================================

def compute_manhattan_potential(env):
    """计算基于曼哈顿距离的势能: Phi(s)"""
    V = np.zeros(env.n_states)
    g_row, g_col = env._state_to_pos(env.goal_state)
    max_dist = env.size * 2
    
    for s in range(env.n_states):
        r, c = env._state_to_pos(s)
        dist = abs(r - g_row) + abs(c - g_col)
        # 归一化到 [0, 1]，距离越近势能越高
        V[s] = 1.0 - (dist / max_dist)
    
    # 强制终点为 1
    V[env.goal_state] = 1.0
    return V

def compute_option_price_dp(P, goal_state, gamma=0.99, n_iterations=200):
    """基于模型 DP 计算准确的 Option Price"""
    n_states = P.shape[0]
    V = np.zeros(n_states)
    V[goal_state] = 1.0
    
    # 简单的 Value Iteration for Random Policy
    # C(s) = sum_a (1/|A|) * sum_s' P(s'|s,a) * (gamma * C(s'))
    # 注意：这里我们简化处理，假设这是一个“被动扩散过程”
    
    for _ in range(n_iterations):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            if s == goal_state:
                V_new[s] = 1.0
                continue
                
            # Average over all possible next states (Diffusion)
            # 这里简化：假设是随机游走
            # P[s, a, s_next]
            # Sum over actions
            expected_next_v = 0
            for a in range(4):
                # 确定性环境，P[s,a] 只有一个非零元素
                s_next = np.argmax(P[s, a])
                expected_next_v += V[s_next]
            
            V_new[s] = (expected_next_v / 4.0) * gamma # 加上 discount 避免数值爆炸
        V = V_new
    return V

# =============================================================================
# 训练算法
# =============================================================================

def train_agent(env, potential_function=None, n_episodes=500, label="Agent"):
    """
    通用 Q-Learning 训练器，支持 Potential-Based Reward Shaping
    """
    Q = np.zeros((env.n_states, env.n_actions))
    success_rate = []
    
    alpha = 0.1
    gamma = 0.99
    eps = 0.2  # 较高的探索率，因为有墙
    
    # 如果有势能函数，使用 Ng's PBRS: F = gamma * Phi(s') - Phi(s)
    # Target = r + F + gamma * maxQ(s') = r - Phi(s) + gamma * (Phi(s') + maxQ(s'))
    
    window = []
    
    for ep in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                a = np.argmax(Q[s])
            
            s_next, r, done = env.step(a)
            
            # 计算 TD Target
            q_target = r + (0 if done else gamma * np.max(Q[s_next]))
            
            # 应用 Potential-Based Reward Shaping (如果提供)
            if potential_function is not None:
                # F = gamma * Phi(s') - Phi(s)
                phi_s = potential_function[s]
                phi_next = potential_function[s_next]
                shaping = gamma * phi_next - phi_s
                
                # 注意：我们要么把 shaping 加到 reward 里，要么直接加到 TD error
                # Q(s,a) <- Q(s,a) + alpha * (r + shaping + gamma*maxQ - Q)
                td_error = (r + shaping + (0 if done else gamma * np.max(Q[s_next]))) - Q[s, a]
            else:
                td_error = q_target - Q[s, a]
                
            Q[s, a] += alpha * td_error
            s = s_next
            steps += 1
        
        window.append(1 if done else 0)
        if len(window) > 50: window.pop(0)
        success_rate.append(np.mean(window))
        
        if ep % 100 == 0:
            print(f"[{label}] Ep {ep}: Success Rate {np.mean(window):.2f}")
            
    return success_rate, Q

# =============================================================================
# Main Comparison
# =============================================================================

def main():
    print("初始化带墙壁的迷宫 (Wall at col=5, gap at bottom)...")
    env = GridMazeWithWalls()
    P = env.get_transition_matrix()
    
    print("\n1. 计算 Heuristic Potential (Manhattan)...")
    phi_heuristic = compute_manhattan_potential(env)
    
    print("2. 计算 OptionRL Potential (DP Model)...")
    phi_option = compute_option_price_dp(P, env.goal_state)
    
    # 可视化 Potentials
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    im1 = ax[0].imshow(phi_heuristic.reshape(10, 10), cmap='Blues')
    ax[0].set_title("Heuristic (Distance) Potential")
    # 标记墙壁
    for r, c in env.walls:
        ax[0].text(c, r, 'X', ha='center', va='center', color='black', fontweight='bold')
        
    im2 = ax[1].imshow(phi_option.reshape(10, 10), cmap='Greens')
    ax[1].set_title("OptionRL (Model-based) Potential")
    for r, c in env.walls:
        ax[1].text(c, r, 'X', ha='center', va='center', color='black', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('potential_comparison.png')
    print("  已保存势能对比图: potential_comparison.png")
    
    # 简单的验证：起点 (0,0) 的值
    # 曼哈顿距离下，(0,0) 离 (9,9) 很远，但实际上路被堵了，它感知不到
    # OptionRL 下，(0,0) 会因为那堵墙导致“扩散”过来的值变小
    
    print("\n开始训练对比...")
    
    # Run 1: Q-Learning (No Shaping)
    sr_q, _ = train_agent(env, None, label="Q-Learning")
    
    # Run 2: Heuristic PBRS
    sr_h, _ = train_agent(env, phi_heuristic, label="Heuristic PBRS")
    
    # Run 3: OptionRL PBRS
    sr_o, _ = train_agent(env, phi_option, label="OptionRL")
    
    # Plotting Results
    plt.figure(figsize=(10, 6))
    plt.plot(sr_q, label='Q-Learning (Baseline)', color='gray', alpha=0.5)
    plt.plot(sr_h, label='Heuristic PBRS (Manhattan)', color='orange', linewidth=2)
    plt.plot(sr_o, label='OptionRL (Model-based)', color='green', linewidth=2)
    
    plt.title('Performance Comparison in "Trap Maze"')
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate (Moving Avg)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('performance_comparison.png')
    print("  已保存性能对比图: performance_comparison.png")

if __name__ == "__main__":
    main()
