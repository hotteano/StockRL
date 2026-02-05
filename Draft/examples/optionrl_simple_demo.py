import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class SparseChainEnv:
    """简单稀疏奖励环境：一维链，终点给奖励，其它为 0。

    状态空间: {0, 1, ..., N}
    动作: 0 = 左, 1 = 右
    转移: 带噪声的左右移动，提高不确定性
    回合长度: T 步，最后一步若在 N 则给奖励 1
    """

    n_states: int = 6
    max_steps: int = 10
    p_move_correct: float = 0.7

    def reset(self) -> int:
        self.state = 0
        self.t = 0
        return self.state

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        assert action in (0, 1)
        self.t += 1

        # 真实动作方向：0->左, 1->右
        move = 1 if action == 1 else -1
        if random.random() > self.p_move_correct:
            move = -move  # 以 1-p 的概率反方向

        self.state = max(0, min(self.n_states - 1, self.state + move))

        done = self.t >= self.max_steps
        reward = 1.0 if (done and self.state == self.n_states - 1) else 0.0
        return self.state, reward, done, {}


def estimate_terminal_success_prob(env: SparseChainEnv, start_state: int, remaining_steps: int,
                                   policy="greedy_right", n_rollouts: int = 64) -> float:
    """在给定测度 (这里用真实转移) 下，从 start_state 出发 MC 估计终点奖励=1 的概率。

    这里相当于在风险中性测度 Q 下估计 E_Q[R_T | s_t].
    为简单起见，我们用一个固定策略: 优先向右移动。
    """

    if remaining_steps <= 0:
        # 没步数了，只看当前是否在终点
        return 1.0 if start_state == env.n_states - 1 else 0.0

    success = 0
    for _ in range(n_rollouts):
        # 复制一个环境副本的状态（只克隆必要字段）
        s = start_state
        t_left = remaining_steps
        while t_left > 0:
            # 简单策略: 如果不在最右端就向右，否则保持向右
            if policy == "greedy_right":
                a = 1
            else:
                a = random.randint(0, 1)

            # 手动复刻 env.step 逻辑（避免真正修改 env 对象）
            move = 1 if a == 1 else -1
            if random.random() > env.p_move_correct:
                move = -move
            s = max(0, min(env.n_states - 1, s + move))
            t_left -= 1

        if s == env.n_states - 1:
            success += 1

    return success / n_rollouts


def run_q_learning(env: SparseChainEnv, n_episodes: int = 300, alpha: float = 0.1,
                   gamma: float = 0.99, eps: float = 0.1) -> np.ndarray:
    """标准 Q-learning 基线。"""
    Q = np.zeros((env.n_states, 2), dtype=float)
    returns = []

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = int(np.argmax(Q[s]))

            s_next, r, done, _ = env.step(a)
            ep_return += r

            td_target = r + (0.0 if done else gamma * np.max(Q[s_next]))
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next

        returns.append(ep_return)

    return np.array(returns)


def run_optionrl(env: SparseChainEnv, n_episodes: int = 300, alpha: float = 0.1,
                 r_rate: float = 0.05, eps: float = 0.1, n_rollouts: int = 64) -> np.ndarray:
    """OptionRL 风格的 Q-learning：用期权价 C_{t+1} 作为 bootstrap 目标。

    C_{t+1}(s_{t+1}) \approx e^{-r (T-(t+1))} E^{\mathbb Q}[R_T | s_{t+1}].
    这里通过 MC 在 (与真实环境相同的) 转移下估计概率，然后乘上折现因子。
    """
    Q = np.zeros((env.n_states, 2), dtype=float)
    returns = []
    disc = math.exp(-r_rate)

    for _ in range(n_episodes):
        s = env.reset()
        done = False
        ep_return = 0.0
        t = 0
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = int(np.argmax(Q[s]))

            s_next, r, done, _ = env.step(a)
            ep_return += r

            remaining = env.max_steps - (t + 1)
            # 期权价 = 折现 * 成功概率
            success_prob = estimate_terminal_success_prob(env, s_next, remaining,
                                                          policy="greedy_right",
                                                          n_rollouts=n_rollouts)
            C_next = (disc ** remaining) * success_prob

            td_target = r + C_next
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            t += 1

        returns.append(ep_return)

    return np.array(returns)


def main():
    env = SparseChainEnv(n_states=6, max_steps=10, p_move_correct=0.7)

    print("Running standard Q-learning baseline...")
    ret_q = run_q_learning(env, n_episodes=300)

    print("Running OptionRL-style Q-learning...")
    # 为了公平，每次重新实例化环境，避免内部状态残留
    env2 = SparseChainEnv(n_states=6, max_steps=10, p_move_correct=0.7)
    ret_opt = run_optionrl(env2, n_episodes=300)

    def smooth(x, w=20):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    sm_q = smooth(ret_q)
    sm_opt = smooth(ret_opt)

    print("\nAverage return over last 50 episodes:")
    print(f"  Q-learning     : {ret_q[-50:].mean():.3f}")
    print(f"  OptionRL-style : {ret_opt[-50:].mean():.3f}")

    print("\nSmoothed learning curves (first/last values):")
    print(f"  Q-learning     : {sm_q[0]:.3f} -> {sm_q[-1]:.3f}")
    print(f"  OptionRL-style : {sm_opt[0]:.3f} -> {sm_opt[-1]:.3f}")


if __name__ == "__main__":
    main()
