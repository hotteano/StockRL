import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np


@dataclass
class FrozenLakeConfig:
    env_id: str = "FrozenLake-v1"
    map_name: str = "4x4"
    is_slippery: bool = True
    max_steps: int = 50


def make_env(cfg: FrozenLakeConfig):
    env = gym.make(cfg.env_id, map_name=cfg.map_name, is_slippery=cfg.is_slippery)
    return env


def estimate_goal_prob(mc_env, start_state: int, remaining_steps: int,
                       policy: Callable[[int, np.ndarray], int],
                       Q: np.ndarray,
                       n_rollouts: int = 64) -> float:
    """在给定起始状态下，用 MC 估计在剩余步数内到达目标的概率。

    这里用 mc_env 的内部状态 unwrapped.s 直接设置起点，只用于估计，不影响主训练 env。
    """
    if remaining_steps <= 0:
        return 0.0

    success = 0
    for _ in range(n_rollouts):
        obs, _ = mc_env.reset()
        # 直接强制设置起始状态
        mc_env.unwrapped.s = start_state
        done = False
        steps_left = remaining_steps
        while not done and steps_left > 0:
            s = mc_env.unwrapped.s
            a = policy(s, Q)
            obs, r, terminated, truncated, _ = mc_env.step(a)
            done = terminated or truncated
            steps_left -= 1
        # 在 Gymnasium 的 FrozenLake 中，goal 状态的奖励为 1
        if r == 1.0:
            success += 1
    return success / n_rollouts


def estimate_goal_prob_cached(
    mc_env,
    cache: dict,
    start_state: int,
    remaining_steps: int,
    policy: Callable[[int, np.ndarray], int],
    Q: np.ndarray,
    n_rollouts: int = 64,
) -> float:
    """带简单缓存的版本，避免同一 (state, remaining_steps) 反复 MC。"""
    if remaining_steps <= 0:
        return 0.0
    key = (start_state, remaining_steps)
    if key in cache:
        return cache[key]
    prob = estimate_goal_prob(mc_env, start_state, remaining_steps, policy, Q, n_rollouts)
    cache[key] = prob
    return prob


def greedy_epsilon_policy(eps: float) -> Callable[[int, np.ndarray], int]:
    def policy(s: int, Q: np.ndarray) -> int:
        if random.random() < eps:
            return random.randint(0, Q.shape[1] - 1)
        return int(np.argmax(Q[s]))
    return policy


def run_q_learning(cfg: FrozenLakeConfig, n_episodes: int = 2000,
                   alpha: float = 0.1, gamma: float = 0.99,
                   eps: float = 0.1) -> np.ndarray:
    env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []

    for _ in range(n_episodes):
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

    env.close()
    return np.array(returns)


def run_q_learning_time_budget(cfg: FrozenLakeConfig, time_budget_s: float = 5.0,
                               alpha: float = 0.1, gamma: float = 0.99,
                               eps: float = 0.1):
    """在给定时间预算内运行 Q-learning，返回回报序列和实际 episode 数。"""
    env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []

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
        episodes += 1

    elapsed = time.time() - start
    env.close()
    return np.array(returns), episodes, elapsed


def run_optionrl(cfg: FrozenLakeConfig, n_episodes: int = 2000,
                 alpha: float = 0.1, r_rate: float = 0.05,
                 eps: float = 0.1, n_rollouts: int = 64) -> np.ndarray:
    env = make_env(cfg)
    mc_env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []

    disc = math.exp(-r_rate)
    policy_fn = greedy_epsilon_policy(eps)

    for _ in range(n_episodes):
        cache: dict[tuple[int, int], float] = {}
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
            # 估计在风险中性测度下，到终点奖励 1 的概率（带缓存）
            success_prob = estimate_goal_prob_cached(
                mc_env,
                cache,
                s_next,
                remaining,
                policy=policy_fn,
                Q=Q,
                n_rollouts=n_rollouts,
            )
            C_next = (disc ** remaining) * success_prob

            td_target = r + C_next
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            t += 1
        returns.append(ep_ret)

    env.close()
    mc_env.close()
    return np.array(returns)


def run_optionrl_time_budget(cfg: FrozenLakeConfig, time_budget_s: float = 5.0,
                             alpha: float = 0.1, r_rate: float = 0.05,
                             eps: float = 0.1, n_rollouts: int = 64):
    """在给定时间预算内运行 OptionRL-style Q-learning。"""
    env = make_env(cfg)
    mc_env = make_env(cfg)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=float)
    returns = []

    disc = math.exp(-r_rate)
    policy_fn = greedy_epsilon_policy(eps)

    start = time.time()
    episodes = 0
    while time.time() - start < time_budget_s:
        cache: dict[tuple[int, int], float] = {}
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
            success_prob = estimate_goal_prob_cached(
                mc_env,
                cache,
                s_next,
                remaining,
                policy=policy_fn,
                Q=Q,
                n_rollouts=n_rollouts,
            )
            C_next = (disc ** max(remaining, 0)) * success_prob

            td_target = r + C_next
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            t += 1
        returns.append(ep_ret)
        episodes += 1

    elapsed = time.time() - start
    env.close()
    mc_env.close()
    return np.array(returns), episodes, elapsed


def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    cfg = FrozenLakeConfig(map_name="4x4", is_slippery=True, max_steps=50)

    time_budget_s = 5.0

    print(f"Running Q-learning on FrozenLake for ~{time_budget_s}s ...")
    ret_q, epi_q, t_q = run_q_learning_time_budget(cfg, time_budget_s=time_budget_s)

    print(f"Running OptionRL-style Q-learning on FrozenLake for ~{time_budget_s}s ...")
    # 为了控制计算量，这里用较小的 n_rollouts（例如 16）
    ret_opt, epi_opt, t_opt = run_optionrl_time_budget(
        cfg,
        time_budget_s=time_budget_s,
        n_rollouts=16,
    )

    sm_q = smooth(ret_q)
    sm_opt = smooth(ret_opt)

    tail_q = ret_q[-min(200, len(ret_q)):] if len(ret_q) > 0 else ret_q
    tail_opt = ret_opt[-min(200, len(ret_opt)):] if len(ret_opt) > 0 else ret_opt

    print("\nTime & episodes:")
    print(f"  Q-learning     : {t_q:.2f}s, episodes = {epi_q}")
    print(f"  OptionRL-style : {t_opt:.2f}s, episodes = {epi_opt}")

    print("\nAverage success rate over last min(200, episodes) episodes:")
    print(f"  Q-learning     : {tail_q.mean() if len(tail_q)>0 else 0.0:.3f}")
    print(f"  OptionRL-style : {tail_opt.mean() if len(tail_opt)>0 else 0.0:.3f}")

    if len(sm_q) > 0 and len(sm_opt) > 0:
        print("\nSmoothed curves (first/last):")
        print(f"  Q-learning     : {sm_q[0]:.3f} -> {sm_q[-1]:.3f}")
        print(f"  OptionRL-style : {sm_opt[0]:.3f} -> {sm_opt[-1]:.3f}")


if __name__ == "__main__":
    main()
