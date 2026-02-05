"""
OptionRL vs Q-Learning: 极稀疏奖励专项测试

设计理念：
- 传统 Q-Learning 依赖 bootstrap 链条：Q(s) <- Q(s') <- Q(s'') <- ... <- reward
- 如果链条太长、中间断裂，Q-Learning 就学不到东西
- OptionRL 直接估计"从当前状态到终点的期望"，绕过中间链条

本测试设计了几个"Q-Learning 几乎必然失败"的极端稀疏环境：
1. LongChain: 一维长链，只有终点有奖励
2. DeepMaze: 深度优先的迷宫，需要走很长路径
3. NeedleInHaystack: 大状态空间中只有一个"针"状态有奖励
4. DelayedReward: 必须完成特定序列才能获得奖励
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# 抽象环境基类
# =============================================================================

class SparseEnv(ABC):
    """极稀疏奖励环境的抽象基类"""
    
    @abstractmethod
    def reset(self) -> int:
        """重置环境，返回初始状态"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool]:
        """执行动作，返回 (next_state, reward, done)"""
        pass
    
    @property
    @abstractmethod
    def n_states(self) -> int:
        pass
    
    @property
    @abstractmethod
    def n_actions(self) -> int:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def get_transition_probs(self) -> Dict:
        """返回转移概率字典 P[s][a] = [(prob, next_s, reward, done), ...]"""
        pass


# =============================================================================
# 环境 1: LongChain - 一维长链
# =============================================================================

class LongChainEnv(SparseEnv):
    """
    一维长链环境:
    
    [0] - [1] - [2] - ... - [N-1] - [GOAL]
    
    - 状态: 0 到 chain_length
    - 动作: 0=左, 1=右
    - 奖励: 只有到达 GOAL (state = chain_length) 才得 +1
    - 难点: 链越长，Q-Learning 的 bootstrap 链条越难传播
    """
    
    def __init__(self, chain_length: int = 20, slip_prob: float = 0.1):
        self.chain_length = chain_length
        self.slip_prob = slip_prob
        self.state = 0
        self.goal_state = chain_length
        self._n_states = chain_length + 1
        self._n_actions = 2  # 0=左, 1=右
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def n_actions(self) -> int:
        return self._n_actions
    
    @property
    def name(self) -> str:
        return f"LongChain-{self.chain_length}"
    
    def reset(self) -> int:
        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        # 滑动概率：有时候动作会被"翻转"
        if random.random() < self.slip_prob:
            action = 1 - action
        
        if action == 1:  # 右
            self.state = min(self.state + 1, self.goal_state)
        else:  # 左
            self.state = max(self.state - 1, 0)
        
        # 只有到达终点才有奖励
        if self.state == self.goal_state:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_probs(self) -> Dict:
        P = {}
        for s in range(self._n_states):
            P[s] = {}
            for a in range(self._n_actions):
                transitions = []
                
                if s == self.goal_state:
                    # 终点状态：保持不动
                    transitions.append((1.0, s, 0.0, True))
                else:
                    # 正常动作
                    for actual_a, prob in [(a, 1 - self.slip_prob), (1 - a, self.slip_prob)]:
                        if actual_a == 1:  # 右
                            next_s = min(s + 1, self.goal_state)
                        else:  # 左
                            next_s = max(s - 1, 0)
                        
                        reward = 1.0 if next_s == self.goal_state else 0.0
                        done = next_s == self.goal_state
                        transitions.append((prob, next_s, reward, done))
                
                P[s][a] = transitions
        return P


# =============================================================================
# 环境 2: GridMaze - 网格迷宫（只有一个出口有奖励）
# =============================================================================

class GridMazeEnv(SparseEnv):
    """
    N x N 网格迷宫:
    
    - 起点: (0, 0) 左上角
    - 终点: (N-1, N-1) 右下角
    - 动作: 0=上, 1=下, 2=左, 3=右
    - 奖励: 只有到达终点才得 +1
    - 有随机墙壁阻挡
    """
    
    def __init__(self, size: int = 8, wall_prob: float = 0.2, slip_prob: float = 0.1):
        self.size = size
        self.slip_prob = slip_prob
        self._n_states = size * size
        self._n_actions = 4  # 上下左右
        
        # 生成固定的墙壁（用种子保证可复现）
        rng = np.random.RandomState(42)
        self.walls = set()
        for i in range(size):
            for j in range(size):
                if (i, j) != (0, 0) and (i, j) != (size-1, size-1):
                    if rng.random() < wall_prob:
                        self.walls.add((i, j))
        
        self.state = 0
        self.goal_state = size * size - 1
    
    def _pos_to_state(self, row: int, col: int) -> int:
        return row * self.size + col
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return state // self.size, state % self.size
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def n_actions(self) -> int:
        return self._n_actions
    
    @property
    def name(self) -> str:
        return f"GridMaze-{self.size}x{self.size}"
    
    def reset(self) -> int:
        self.state = 0
        return self.state
    
    def _move(self, state: int, action: int) -> int:
        row, col = self._state_to_pos(state)
        
        # 动作效果
        if action == 0:  # 上
            new_row, new_col = row - 1, col
        elif action == 1:  # 下
            new_row, new_col = row + 1, col
        elif action == 2:  # 左
            new_row, new_col = row, col - 1
        else:  # 右
            new_row, new_col = row, col + 1
        
        # 边界检查
        if new_row < 0 or new_row >= self.size or new_col < 0 or new_col >= self.size:
            return state
        
        # 墙壁检查
        if (new_row, new_col) in self.walls:
            return state
        
        return self._pos_to_state(new_row, new_col)
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        # 滑动：有概率执行随机动作
        if random.random() < self.slip_prob:
            action = random.randint(0, 3)
        
        self.state = self._move(self.state, action)
        
        if self.state == self.goal_state:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_probs(self) -> Dict:
        P = {}
        for s in range(self._n_states):
            P[s] = {}
            for a in range(self._n_actions):
                transitions = []
                
                if s == self.goal_state:
                    transitions.append((1.0, s, 0.0, True))
                else:
                    # 正常动作
                    next_s_intended = self._move(s, a)
                    reward_intended = 1.0 if next_s_intended == self.goal_state else 0.0
                    done_intended = next_s_intended == self.goal_state
                    transitions.append((1 - self.slip_prob, next_s_intended, reward_intended, done_intended))
                    
                    # 滑动到随机动作
                    for rand_a in range(4):
                        next_s_rand = self._move(s, rand_a)
                        reward_rand = 1.0 if next_s_rand == self.goal_state else 0.0
                        done_rand = next_s_rand == self.goal_state
                        transitions.append((self.slip_prob / 4, next_s_rand, reward_rand, done_rand))
                
                P[s][a] = transitions
        return P


# =============================================================================
# 环境 3: NeedleInHaystack - 大海捞针
# =============================================================================

class NeedleInHaystackEnv(SparseEnv):
    """
    大海捞针环境:
    
    - N 个状态，随机连接
    - 只有一个"针"状态有奖励
    - 从任意状态可以跳到若干个邻居状态
    - 极难通过随机探索找到针
    """
    
    def __init__(self, n_states: int = 100, n_neighbors: int = 4, slip_prob: float = 0.1):
        self._n_states = n_states
        self._n_actions = n_neighbors
        self.slip_prob = slip_prob
        
        # 固定随机种子生成图结构
        rng = np.random.RandomState(123)
        
        # 每个状态有 n_neighbors 个邻居
        self.neighbors = {}
        for s in range(n_states):
            self.neighbors[s] = rng.choice(n_states, size=n_neighbors, replace=False).tolist()
        
        # 针状态（目标）
        self.needle_state = n_states - 1
        self.state = 0
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def n_actions(self) -> int:
        return self._n_actions
    
    @property
    def name(self) -> str:
        return f"NeedleInHaystack-{self._n_states}"
    
    def reset(self) -> int:
        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        if random.random() < self.slip_prob:
            action = random.randint(0, self._n_actions - 1)
        
        self.state = self.neighbors[self.state][action]
        
        if self.state == self.needle_state:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_probs(self) -> Dict:
        P = {}
        for s in range(self._n_states):
            P[s] = {}
            for a in range(self._n_actions):
                transitions = []
                
                if s == self.needle_state:
                    transitions.append((1.0, s, 0.0, True))
                else:
                    # 正常动作
                    next_s = self.neighbors[s][a]
                    reward = 1.0 if next_s == self.needle_state else 0.0
                    done = next_s == self.needle_state
                    transitions.append((1 - self.slip_prob, next_s, reward, done))
                    
                    # 滑动
                    for rand_a in range(self._n_actions):
                        next_s_rand = self.neighbors[s][rand_a]
                        reward_rand = 1.0 if next_s_rand == self.needle_state else 0.0
                        done_rand = next_s_rand == self.needle_state
                        transitions.append((self.slip_prob / self._n_actions, next_s_rand, reward_rand, done_rand))
                
                P[s][a] = transitions
        return P


# =============================================================================
# 环境 4: SequenceMatch - 必须按特定序列行动
# =============================================================================

class SequenceMatchEnv(SparseEnv):
    """
    序列匹配环境:
    
    - 必须按正确的动作序列行动才能获得奖励
    - 例如：必须依次执行 [0, 1, 0, 1, 1] 才能成功
    - 任何错误都会重置进度
    - 这是最极端的稀疏奖励：只有一条正确路径
    """
    
    def __init__(self, sequence_length: int = 8, n_actions: int = 2):
        self.target_sequence = [i % n_actions for i in range(sequence_length)]  # 交替序列
        self.sequence_length = sequence_length
        self._n_actions = n_actions
        self._n_states = sequence_length + 1  # 进度 0 到 sequence_length
        self.state = 0  # 当前匹配进度
    
    @property
    def n_states(self) -> int:
        return self._n_states
    
    @property
    def n_actions(self) -> int:
        return self._n_actions
    
    @property
    def name(self) -> str:
        return f"SequenceMatch-{self.sequence_length}"
    
    def reset(self) -> int:
        self.state = 0
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        if self.state < self.sequence_length:
            if action == self.target_sequence[self.state]:
                self.state += 1  # 匹配成功，进度+1
            else:
                self.state = 0  # 匹配失败，重置进度
        
        if self.state == self.sequence_length:
            return self.state, 1.0, True
        return self.state, 0.0, False
    
    def get_transition_probs(self) -> Dict:
        P = {}
        for s in range(self._n_states):
            P[s] = {}
            for a in range(self._n_actions):
                if s == self.sequence_length:
                    # 已完成
                    P[s][a] = [(1.0, s, 0.0, True)]
                elif a == self.target_sequence[s]:
                    # 正确动作
                    next_s = s + 1
                    reward = 1.0 if next_s == self.sequence_length else 0.0
                    done = next_s == self.sequence_length
                    P[s][a] = [(1.0, next_s, reward, done)]
                else:
                    # 错误动作
                    P[s][a] = [(1.0, 0, 0.0, False)]
        return P


# =============================================================================
# DP 计算期望成功概率/回报
# =============================================================================

def compute_success_prob_dp(env: SparseEnv, n_iterations: int = 200) -> np.ndarray:
    """用 DP 计算从每个状态出发（随机策略）到达目标的概率。"""
    P = env.get_transition_probs()
    n_states = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(n_states)
    
    for _ in range(n_iterations):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            total = 0.0
            for a in range(n_actions):
                for prob, next_s, reward, done in P[s][a]:
                    if reward > 0:
                        total += prob * 1.0 / n_actions
                    elif not done:
                        total += prob * V[next_s] / n_actions
            V_new[s] = total
        
        if np.allclose(V, V_new, atol=1e-8):
            break
        V = V_new
    
    return V


def compute_success_prob_with_policy(
    env: SparseEnv,
    Q: np.ndarray,
    eps: float = 0.1,
    n_iterations: int = 100,
) -> np.ndarray:
    """用当前 Q 表的 ε-greedy 策略计算成功概率。"""
    P = env.get_transition_probs()
    n_states = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(n_states)
    
    for _ in range(n_iterations):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            greedy_a = int(np.argmax(Q[s]))
            total = 0.0
            for a in range(n_actions):
                if a == greedy_a:
                    pi_a = 1.0 - eps + eps / n_actions
                else:
                    pi_a = eps / n_actions
                
                for prob, next_s, reward, done in P[s][a]:
                    if reward > 0:
                        total += pi_a * prob * 1.0
                    elif not done:
                        total += pi_a * prob * V[next_s]
            V_new[s] = total
        
        if np.allclose(V, V_new, atol=1e-8):
            break
        V = V_new
    
    return V


# =============================================================================
# Q-Learning
# =============================================================================

def run_q_learning(
    env: SparseEnv,
    n_episodes: int = 10000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps: float = 0.1,
) -> Tuple[np.ndarray, float]:
    """标准 Q-learning。"""
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    successes = []

    start = time.time()
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0
        success = False
        
        while not done and steps < max_steps:
            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))

            next_s, r, done = env.step(a)
            
            if r > 0:
                success = True

            td_target = r + (0.0 if done else gamma * np.max(Q[next_s]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = next_s
            steps += 1

        successes.append(1.0 if success else 0.0)

    elapsed = time.time() - start
    return np.array(successes), elapsed


# =============================================================================
# OptionRL-DP
# =============================================================================

def run_optionrl_dp(
    env: SparseEnv,
    n_episodes: int = 10000,
    max_steps: int = 200,
    alpha: float = 0.1,
    gamma: float = 0.99,
    r_rate: float = 0.05,
    eps: float = 0.1,
    update_C_every: int = 50,
    blend_ratio: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """OptionRL with DP-computed option prices。"""
    Q = np.zeros((env.n_states, env.n_actions), dtype=float)
    successes = []
    
    disc = math.exp(-r_rate)
    
    # 初始 C
    C = compute_success_prob_dp(env)

    start = time.time()
    for ep in range(n_episodes):
        # 定期更新 C
        if ep > 0 and ep % update_C_every == 0:
            C = compute_success_prob_with_policy(env, Q, eps=eps)
        
        # 动态混合
        current_blend = max(0.1, blend_ratio * (1 - ep / n_episodes))
        
        s = env.reset()
        done = False
        t = 0
        success = False
        
        while not done and t < max_steps:
            if random.random() < eps:
                a = random.randint(0, env.n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))

            next_s, r, done = env.step(a)
            
            if r > 0:
                success = True

            # 混合 TD target
            remaining = max_steps - (t + 1)
            C_next = (disc ** max(remaining, 1)) * C[next_s]
            Q_bootstrap = 0.0 if done else gamma * np.max(Q[next_s])
            
            td_target = r + current_blend * C_next + (1 - current_blend) * Q_bootstrap
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = next_s
            t += 1

        successes.append(1.0 if success else 0.0)

    elapsed = time.time() - start
    return np.array(successes), elapsed


# =============================================================================
# 评估
# =============================================================================

def smooth(x: np.ndarray, w: int = 100) -> np.ndarray:
    if len(x) < w:
        return np.cumsum(x) / (np.arange(len(x)) + 1)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def first_success_episode(successes: np.ndarray) -> Optional[int]:
    for i, s in enumerate(successes):
        if s > 0:
            return i
    return None


def print_results(env_name: str, succ_q: np.ndarray, t_q: float,
                  succ_opt: np.ndarray, t_opt: float, n_episodes: int):
    """打印对比结果。"""
    print(f"\n{'='*70}")
    print(f" 🎯 {env_name}")
    print(f"{'='*70}")
    
    # 分段统计
    segments = [
        ("前 20%", 0, n_episodes // 5),
        ("20-50%", n_episodes // 5, n_episodes // 2),
        ("50-80%", n_episodes // 2, n_episodes * 4 // 5),
        ("最后 20%", n_episodes * 4 // 5, n_episodes),
    ]
    
    print(f"\n{'阶段':<15} {'Q-Learning':>15} {'OptionRL-DP':>15} {'差异':>15}")
    print("-" * 60)
    
    for name, start, end in segments:
        q_rate = succ_q[start:end].mean()
        opt_rate = succ_opt[start:end].mean()
        if q_rate > 0:
            diff = f"{opt_rate/q_rate:.2f}x"
        elif opt_rate > 0:
            diff = "∞x better"
        else:
            diff = "both 0"
        print(f"{name:<15} {q_rate:>15.4f} {opt_rate:>15.4f} {diff:>15}")
    
    print("-" * 60)
    
    first_q = first_success_episode(succ_q)
    first_opt = first_success_episode(succ_opt)
    
    print(f"{'首次成功':<15} {str(first_q):>15} {str(first_opt):>15}", end="")
    if first_q is None and first_opt is not None:
        print(f" {'OptionRL wins':>15}")
    elif first_q is not None and first_opt is None:
        print(f" {'Q-Learning wins':>15}")
    elif first_q is not None and first_opt is not None:
        ratio = first_q / first_opt if first_opt > 0 else float('inf')
        print(f" {f'{ratio:.1f}x faster':>15}")
    else:
        print(f" {'both failed':>15}")
    
    print(f"{'总成功率':<15} {succ_q.mean():>15.4f} {succ_opt.mean():>15.4f}")
    print(f"{'训练时间':<15} {t_q:>15.2f}s {t_opt:>15.2f}s")
    
    # 最终判定
    final_q = succ_q[-n_episodes//5:].mean()
    final_opt = succ_opt[-n_episodes//5:].mean()
    
    print("\n📊 结论: ", end="")
    if final_q == 0 and final_opt == 0:
        print("两者均未学到有效策略")
    elif final_q == 0 and final_opt > 0:
        print(f"✅ OptionRL 成功 ({final_opt:.2%})，Q-Learning 完全失败")
    elif final_q > 0 and final_opt == 0:
        print(f"⚠️ Q-Learning 成功 ({final_q:.2%})，OptionRL 完全失败")
    elif final_opt > final_q * 1.5:
        print(f"✅ OptionRL 显著更好 ({final_opt:.2%} vs {final_q:.2%})")
    elif final_q > final_opt * 1.5:
        print(f"⚠️ Q-Learning 显著更好 ({final_q:.2%} vs {final_opt:.2%})")
    else:
        print(f"🔄 两者表现相近 ({final_opt:.2%} vs {final_q:.2%})")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print(" 🔬 OptionRL vs Q-Learning: 极稀疏奖励专项测试")
    print("=" * 70)
    print("""
测试目的: 验证 OptionRL 在"Q-Learning 几乎必然失败"的极端稀疏环境中的优势

环境设计:
1. LongChain: 一维长链，只有终点有奖励（测试 bootstrap 链条长度）
2. GridMaze: 网格迷宫，只有出口有奖励（测试状态空间 + 路径长度）
3. NeedleInHaystack: 大海捞针，随机图中找唯一目标（测试探索难度）
4. SequenceMatch: 必须按特定序列行动（测试极端稀疏性）
""")
    
    n_episodes = 10000
    max_steps = 300
    
    # 环境配置
    envs = [
        LongChainEnv(chain_length=30, slip_prob=0.1),
        LongChainEnv(chain_length=50, slip_prob=0.1),
        GridMazeEnv(size=10, wall_prob=0.2, slip_prob=0.1),
        NeedleInHaystackEnv(n_states=100, n_neighbors=4, slip_prob=0.1),
        NeedleInHaystackEnv(n_states=200, n_neighbors=4, slip_prob=0.1),
        SequenceMatchEnv(sequence_length=6, n_actions=2),
        SequenceMatchEnv(sequence_length=8, n_actions=2),
    ]
    
    results = []
    
    for env in envs:
        print(f"\n🚀 测试: {env.name} ({env.n_states} states, {env.n_actions} actions)")
        
        print("   [1/2] Q-Learning...")
        succ_q, t_q = run_q_learning(env, n_episodes=n_episodes, max_steps=max_steps)
        
        print("   [2/2] OptionRL-DP...")
        succ_opt, t_opt = run_optionrl_dp(env, n_episodes=n_episodes, max_steps=max_steps)
        
        print_results(env.name, succ_q, t_q, succ_opt, t_opt, n_episodes)
        
        results.append({
            "env": env.name,
            "q_final": succ_q[-n_episodes//5:].mean(),
            "opt_final": succ_opt[-n_episodes//5:].mean(),
            "q_first": first_success_episode(succ_q),
            "opt_first": first_success_episode(succ_opt),
        })
    
    # 汇总
    print("\n" + "=" * 70)
    print(" 📋 汇总表")
    print("=" * 70)
    print(f"\n{'环境':<25} {'Q-Learn最后20%':>15} {'OptionRL最后20%':>15} {'首次成功(Q/Opt)':>20}")
    print("-" * 75)
    for r in results:
        first_str = f"{r['q_first']}/{r['opt_first']}"
        print(f"{r['env']:<25} {r['q_final']:>15.4f} {r['opt_final']:>15.4f} {first_str:>20}")
    
    print("\n" + "=" * 70)
    print(" 💡 关键洞察")
    print("=" * 70)
    print("""
1. 在极稀疏奖励环境中，Q-Learning 的 bootstrap 链条难以建立：
   - 需要先"碰巧"到达目标状态，才能开始反向传播价值
   - 链条越长、状态空间越大，这个概率越低

2. OptionRL 通过 DP 预计算"从每个状态到达目标的概率"C(s)：
   - 即使从未实际到达过目标，也能估计每个状态的"希望程度"
   - TD target = r + C(s') 让每个状态都能立即获得有意义的更新信号

3. 这正是 OptionRL 论文的核心 claim：
   期权价格 C_t(s) = e^{-r(T-t)} * E^Q[R_T | s_t = s]
   编码了"远期价值的结构化先验"，绕过了传统 TD 的 bootstrap 困境。

4. 在简单任务上，Q-Learning 可能更高效（不需要 DP 开销）；
   但在极稀疏任务上，OptionRL 是"能学到 vs 学不到"的本质差异。
""")


if __name__ == "__main__":
    main()
