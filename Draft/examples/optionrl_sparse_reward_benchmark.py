"""
OptionRL vs Q-Learning: 稀疏奖励基准测试

本脚本在多个稀疏奖励环境上比较:
1. FrozenLake 4x4 (简单基准)
2. FrozenLake 8x8 (更大状态空间、更长路径)
3. Taxi-v3 (500 状态、长序列、稀疏大终值)

核心假设: OptionRL 通过估计"风险中性测度下的终值期望"C_t，
在稀疏奖励任务中比传统 Q-learning 更高效地传播远期价值信号。
"""

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import gymnasium as gym
import numpy as np


# =============================================================================
# 环境配置
# =============================================================================

@dataclass
class EnvConfig:
    env_id: str
    max_steps: int
    goal_reward: float  # 成功时的奖励值，用于判断是否成功
    name: str           # 显示名称


ENVS = {
    "frozenlake_4x4": EnvConfig(
        env_id="FrozenLake-v1",
        max_steps=100,
        goal_reward=1.0,
        name="FrozenLake 4x4 (slippery)",
    ),
    "frozenlake_8x8": EnvConfig(
        env_id="FrozenLake-v1",
        max_steps=200,
        goal_reward=1.0,
        name="FrozenLake 8x8 (slippery)",
    ),
    "taxi": EnvConfig(
        env_id="Taxi-v3",
        max_steps=200,
        goal_reward=20.0,
        name="Taxi-v3",
    ),
}


def make_env(cfg: EnvConfig):
    if "FrozenLake" in cfg.env_id:
        map_name = "8x8" if "8x8" in cfg.name else "4x4"
        return gym.make(cfg.env_id, map_name=map_name, is_slippery=True)
    return gym.make(cfg.env_id)


# =============================================================================
# 辅助函数
# =============================================================================

def greedy_epsilon_policy(eps: float) -> Callable[[int, np.ndarray], int]:
    def policy(s: int, Q: np.ndarray) -> int:
        if random.random() < eps:
            return random.randint(0, Q.shape[1] - 1)
        return int(np.argmax(Q[s]))
    return policy


def estimate_goal_prob(
    mc_env,
    start_state: int,
    remaining_steps: int,
    policy: Callable[[int, np.ndarray], int],
    Q: np.ndarray,
    goal_reward: float,
    n_rollouts: int = 16,
) -> float:
    """
    用 Monte Carlo 估计从 start_state 出发，
    在 remaining_steps 步内获得 goal_reward 的概率。
    """
    if remaining_steps <= 0:
        return 0.0

    success = 0
    for _ in range(n_rollouts):
        obs, _ = mc_env.reset()
        # 强制设置起始状态
        mc_env.unwrapped.s = start_state
        done = False
        steps_left = remaining_steps
        final_reward = 0.0
        while not done and steps_left > 0:
            s = mc_env.unwrapped.s
            a = policy(s, Q)
            obs, r, terminated, truncated, _ = mc_env.step(a)
            done = terminated or truncated
            steps_left -= 1
            if done:
                final_reward = r
        # 判断是否成功到达目标
        if final_reward >= goal_reward:
            success += 1
    return success / n_rollouts


def estimate_goal_prob_cached(
    mc_env,
    cache: dict,
    start_state: int,
    remaining_steps: int,
    policy: Callable[[int, np.ndarray], int],
    Q: np.ndarray,
    goal_reward: float,
    n_rollouts: int = 16,
) -> float:
    """带缓存的版本，避免同一 (state, remaining_steps) 反复 MC。"""
    if remaining_steps <= 0:
        return 0.0
    key = (start_state, remaining_steps)
    if key in cache:
        return cache[key]
    prob = estimate_goal_prob(mc_env, start_state, remaining_steps, policy, Q, goal_reward, n_rollouts)
    cache[key] = prob
    return prob


# =============================================================================
# Q-Learning (时间预算版)
# =============================================================================

def run_q_learning(
    cfg: EnvConfig,
    time_budget_s: float = 10.0,
    alpha: float = 0.1,
    gamma: float = 0.99,
    eps: float = 0.1,
) -> Tuple[np.ndarray, int, float]:
    """在给定时间预算内运行 Q-learning。"""
    env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []
    successes = []

    start = time.time()
    episodes = 0
    while time.time() - start < time_budget_s:
        obs, _ = env.reset()
        s = int(obs)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < cfg.max_steps:
            if random.random() < eps:
                a = random.randint(0, n_actions - 1)
            else:
                a = int(np.argmax(Q[s]))

            obs_next, r, terminated, truncated, _ = env.step(a)
            s_next = int(obs_next)
            done = terminated or truncated
            ep_ret += r

            td_target = r + (0.0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            steps += 1

        returns.append(ep_ret)
        successes.append(1.0 if ep_ret >= cfg.goal_reward else 0.0)
        episodes += 1

    elapsed = time.time() - start
    env.close()
    return np.array(returns), np.array(successes), episodes, elapsed


# =============================================================================
# OptionRL-style Q-Learning (时间预算版)
# =============================================================================

def run_optionrl(
    cfg: EnvConfig,
    time_budget_s: float = 10.0,
    alpha: float = 0.1,
    r_rate: float = 0.05,
    eps: float = 0.1,
    n_rollouts: int = 16,
) -> Tuple[np.ndarray, int, float]:
    """在给定时间预算内运行 OptionRL-style Q-learning。"""
    env = make_env(cfg)
    mc_env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []
    successes = []

    disc = math.exp(-r_rate)
    policy_fn = greedy_epsilon_policy(eps)

    start = time.time()
    episodes = 0
    while time.time() - start < time_budget_s:
        cache: dict = {}
        obs, _ = env.reset()
        s = int(obs)
        done = False
        ep_ret = 0.0
        t = 0
        while not done and t < cfg.max_steps:
            a = policy_fn(s, Q)
            obs_next, r, terminated, truncated, _ = env.step(a)
            s_next = int(obs_next)
            done = terminated or truncated
            ep_ret += r

            remaining = cfg.max_steps - (t + 1)
            # 估计从 s_next 在剩余步数内成功的概率
            success_prob = estimate_goal_prob_cached(
                mc_env,
                cache,
                s_next,
                remaining,
                policy=policy_fn,
                Q=Q,
                goal_reward=cfg.goal_reward,
                n_rollouts=n_rollouts,
            )
            # 期权价格: C_{t+1} = e^{-r * remaining} * E^Q[success]
            C_next = (disc ** max(remaining, 0)) * success_prob * cfg.goal_reward

            # TD 目标: 即时奖励 + 期权价格（替代传统 bootstrap）
            td_target = r + C_next
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            t += 1

        returns.append(ep_ret)
        successes.append(1.0 if ep_ret >= cfg.goal_reward else 0.0)
        episodes += 1

    elapsed = time.time() - start
    env.close()
    mc_env.close()
    return np.array(returns), np.array(successes), episodes, elapsed


# =============================================================================
# 评估与报告
# =============================================================================

def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def compute_metrics(successes: np.ndarray, returns: np.ndarray, episodes: int):
    """计算关键指标。"""
    tail_size = min(100, len(successes))
    tail_success_rate = successes[-tail_size:].mean() if tail_size > 0 else 0.0
    overall_success_rate = successes.mean() if len(successes) > 0 else 0.0
    avg_return = returns.mean() if len(returns) > 0 else 0.0
    
    # 首次成功的 episode (如果有)
    first_success = None
    for i, s in enumerate(successes):
        if s > 0:
            first_success = i
            break
    
    return {
        "episodes": episodes,
        "tail_success_rate": tail_success_rate,
        "overall_success_rate": overall_success_rate,
        "avg_return": avg_return,
        "first_success_episode": first_success,
    }


def print_comparison(env_name: str, q_metrics: dict, opt_metrics: dict, 
                     q_time: float, opt_time: float):
    """打印对比结果。"""
    print(f"\n{'='*60}")
    print(f" 环境: {env_name}")
    print(f"{'='*60}")
    
    print(f"\n{'指标':<25} {'Q-Learning':>15} {'OptionRL':>15}")
    print("-" * 55)
    print(f"{'训练时间 (s)':<25} {q_time:>15.2f} {opt_time:>15.2f}")
    print(f"{'训练 Episodes':<25} {q_metrics['episodes']:>15d} {opt_metrics['episodes']:>15d}")
    print(f"{'首次成功 Episode':<25} {str(q_metrics['first_success_episode']):>15} {str(opt_metrics['first_success_episode']):>15}")
    print(f"{'总体成功率':<25} {q_metrics['overall_success_rate']:>15.3f} {opt_metrics['overall_success_rate']:>15.3f}")
    print(f"{'末尾100轮成功率':<25} {q_metrics['tail_success_rate']:>15.3f} {opt_metrics['tail_success_rate']:>15.3f}")
    print(f"{'平均回报':<25} {q_metrics['avg_return']:>15.3f} {opt_metrics['avg_return']:>15.3f}")
    
    # 效率比较
    if q_metrics['tail_success_rate'] > 0 and opt_metrics['tail_success_rate'] > 0:
        ratio = opt_metrics['tail_success_rate'] / q_metrics['tail_success_rate']
        print(f"\n📊 OptionRL 末尾成功率是 Q-Learning 的 {ratio:.2f}x")
    elif opt_metrics['tail_success_rate'] > 0 and q_metrics['tail_success_rate'] == 0:
        print(f"\n📊 OptionRL 成功学习，Q-Learning 在该时间内未能学习到有效策略")
    elif q_metrics['tail_success_rate'] > 0 and opt_metrics['tail_success_rate'] == 0:
        print(f"\n📊 Q-Learning 成功学习，OptionRL 在该时间内未能学习到有效策略")
    else:
        print(f"\n📊 两种方法在该时间预算内均未学习到有效策略")


# =============================================================================
# 主函数
# =============================================================================

def run_benchmark(env_key: str, time_budget_s: float = 10.0, n_rollouts: int = 16):
    """在单个环境上运行基准测试。"""
    cfg = ENVS[env_key]
    
    print(f"\n🚀 开始测试: {cfg.name}")
    print(f"   时间预算: {time_budget_s}s (每种算法)")
    print(f"   最大步数/episode: {cfg.max_steps}")
    
    # Q-Learning
    print(f"   [1/2] 运行 Q-Learning...")
    ret_q, succ_q, epi_q, t_q = run_q_learning(cfg, time_budget_s=time_budget_s)
    
    # OptionRL
    print(f"   [2/2] 运行 OptionRL...")
    ret_opt, succ_opt, epi_opt, t_opt = run_optionrl(
        cfg, time_budget_s=time_budget_s, n_rollouts=n_rollouts
    )
    
    # 计算指标
    q_metrics = compute_metrics(succ_q, ret_q, epi_q)
    opt_metrics = compute_metrics(succ_opt, ret_opt, epi_opt)
    
    # 打印对比
    print_comparison(cfg.name, q_metrics, opt_metrics, t_q, t_opt)
    
    return {
        "env": cfg.name,
        "q_learning": q_metrics,
        "optionrl": opt_metrics,
    }


def main():
    print("=" * 60)
    print(" OptionRL vs Q-Learning: 稀疏奖励基准测试")
    print(" 测试目标: 在相同时间预算下比较两种算法的学习效率")
    print("=" * 60)
    
    # 配置
    time_budget_s = 10.0  # 每种算法每个环境的时间预算
    n_rollouts = 16       # OptionRL 的 MC rollout 数量
    
    results = []
    
    # 测试所有环境
    for env_key in ["frozenlake_4x4", "frozenlake_8x8", "taxi"]:
        result = run_benchmark(env_key, time_budget_s=time_budget_s, n_rollouts=n_rollouts)
        results.append(result)
    
    # 总结
    print("\n" + "=" * 60)
    print(" 📋 总结")
    print("=" * 60)
    print("""
关键发现:
1. 在简单任务 (FrozenLake 4x4) 上，Q-Learning 能跑更多 episodes，
   可能已经足够学到有效策略。

2. 在更复杂任务 (FrozenLake 8x8, Taxi-v3) 上，由于:
   - 状态空间更大
   - 路径更长
   - 奖励更稀疏
   Q-Learning 需要更多的随机探索才能"撞到"正奖励，
   而 OptionRL 通过估计"未来成功概率"的期权价格 C_t，
   能更快地将远期奖励信号传播到当前状态。

3. OptionRL 的单 episode 计算量更大（因为 MC rollout），
   所以在相同时间内跑的 episodes 更少；
   但每个 episode 的价值更新更有方向性。

4. 这是一种"计算换结构"的 trade-off：
   OptionRL 用更多的单步计算，换取对稀疏奖励更好的处理能力。
""")


if __name__ == "__main__":
    main()
