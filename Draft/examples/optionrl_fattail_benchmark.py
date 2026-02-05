
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random
import math

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# =============================================================================
# 极端稀疏环境: The "Black Swan" Hunter
# =============================================================================

class FatTailGenerator:
    """
    平时波动率极低 (Sigma = 5%)，像死水一样。
    但每年会发生 5 次巨大的正向跳跃 (+15%)。
    
    对于 BS 模型来说，这是一个低波动率市场。
    对于 Merton 模型来说，这是一个高赔率赌场。
    """
    def __init__(self, s0=100.0, dt=1/252):
        self.s0 = s0
        self.dt = dt
        self.sigma = 0.05 # 极低的日常波动
        self.lam = 5.0    # 跳跃频率高
        self.mu_j = 0.15  # 正向跳跃
        self.delta_j = 0.02
        
    def generate_path(self, steps=100):
        # 1. 扩散 (Diffusion) - 平静的水面
        dw = np.random.normal(0, np.sqrt(self.dt), steps)
        
        # 2. 跳跃 (Jump) - 突发的暴利
        n_jumps = np.random.poisson(self.lam * self.dt, steps)
        
        log_ret = np.zeros(steps)
        for i in range(steps):
            jump_sum = 0.0
            if n_jumps[i] > 0:
                jump_sum = np.sum(np.random.normal(self.mu_j, self.delta_j, n_jumps[i]))
            
            # 只有极小的漂移，主要靠跳跃赚钱
            drift = -0.02 * self.dt # 甚至还有点阴跌
            diffusion = self.sigma * dw[i]
            
            log_ret[i] = drift + diffusion + jump_sum
            
        cum_log_ret = np.cumsum(log_ret)
        path = self.s0 * np.exp(np.concatenate(([0], cum_log_ret)))
        return path

# =============================================================================
# 定价模型
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    if T <= 0: return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def merton_call_price(S, K, T, r, sigma, lam, mu_j, delta_j):
    if T <= 0: return max(0.0, S - K)
    k = np.exp(mu_j + 0.5 * delta_j**2) - 1
    lam_prime = lam * (1 + k)
    price = 0.0
    for n in range(15):
        r_n = r - lam * k + (n * np.log(1 + k)) / T
        sigma_n = np.sqrt(sigma**2 + (n * delta_j**2) / T)
        weight = (np.exp(-lam_prime * T) * (lam_prime * T)**n) / math.factorial(n)
        price += weight * bs_call_price(S, K, T, r_n, sigma_n)
    return price

# =============================================================================
# 猎手环境
# =============================================================================

class HuntingEnv:
    """
    任务：捕捉暴涨。
    Action 0: 观望 (Cost = 0)
    Action 1: 埋伏/买入 (Cost = -0.05/day, 资金占用成本)
    
    Reward:
    - 如果 Action=1 且发生了暴涨 (Day Return > 10%) -> Reward = +10.0 (Huge!)
    - 如果 Action=1 且没发生暴涨 -> Reward = -0.05 (Cost)
    - 如果 Action=0 -> Reward = 0
    
    难点：
    - 暴涨是瞬间发生的，必须提前埋伏。
    - 所谓的“瞬间”在离散模拟中就是 t 到 t+1。
    - Q-Learning 很难学，因为它试了几次都亏成本(-0.05)，就再也不敢动了。
    - BS 模型觉得这股票波动率只有 5%，根本不值得花 -0.05 的成本去埋伏。
    - Merton 模型算出这里面有巨大的 Jump Premium，会鼓励 Agent 承受成本去埋伏。
    """
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.gen = FatTailGenerator()
        self.reset()
        
    def reset(self):
        self.path = self.gen.generate_path(self.max_steps + 1)
        self.t = 0
        return self._get_state()
        
    def _get_state(self):
        # 简单状态：只看这一单过去了多久（Time），以及现在的价格相对均线的位置
        # 但在这个随机游走里，最重要的是 Time to Maturity 带来的价值衰减感知
        return self.t
    
    def step(self, action):
        # Action 1 = Long (Betting on Jump)
        self.t += 1
        current_price = self.path[self.t]
        prev_price = self.path[self.t - 1]
        
        ret = (current_price - prev_price) / prev_price
        
        reward = 0.0
        
        if action == 1:
            if ret > 0.10: # 捕捉到了 10% 以上的暴涨
                reward = 10.0 # 大奖！
            else:
                reward = -0.1 # 资金成本/Theta Bleed
        
        done = self.t >= self.max_steps
        return self.t, reward, done

# =============================================================================
# 训练
# =============================================================================

def train_hunter(env, pricing_model=None, episodes=500):
    # Q table: [Time Steps, Actions]
    Q = np.zeros((env.max_steps + 2, 2))
    returns = []
    window = []
    
    alpha = 0.1
    gamma = 0.99
    eps = 0.1 # 降低探索，模拟保守投资者
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = np.argmax(Q[state])
            
            if pricing_model == 'always_long': #不仅是RL，这是一个无脑策略对比
                 a = 1
            if pricing_model == 'always_wait':
                 a = 0
                 
            # 真正的 RL 逻辑
            # Reward Shaping
            shaping = 0.0
            
            # 我们用 Option Price 来代表“在这个时刻持有头寸的潜在价值”
            # 如果 Action=1 (持有)，我们享受这个 Potential。
            # 如果 Action=0 (空仓)，Potential = 0。
            
            # S=100, K=105 (OTM Call), T=left
            T_left = (env.max_steps - state) / 252.0
            
            if pricing_model == 'bs':
                # BS 看到的是 sigma=0.05 的死水
                # 这种情况下 OTM Option 几乎一文不值
                val = bs_call_price(100, 105, T_left, 0.02, 0.05)
            elif pricing_model == 'merton':
                # Merton 看到的是 sigma=0.05 但有 lam=5 的跳跃
                # OTM Option 价值很高
                val = merton_call_price(100, 105, T_left, 0.02, 0.05, 5.0, 0.15, 0.02) 
            else:
                val = 0.0
                
            # Shaping 逻辑：
            # 如果我持有(a=1)，我拥有价值 val。如果我空仓(a=0)，我价值 0。
            # 这不需要复杂的 gamma*phi_next - phi_curr
            # 这是一个策略辅助：给 Action=1 附加一个 Bias
            
            if a == 1 and pricing_model in ['bs', 'merton']:
                 # 将由模型计算出的“每一步持仓的期望收益”加进去
                 # Merton Price 实际上包含了 sum(P_jump * Size_jump)
                 # 我们直接把它转化成 step reward bonus
                 # 粗略估计：Option Price / time_steps ~ drift value
                 bonus = val * 0.1 
                 shaping = bonus
            
            next_state, r, done = env.step(a)
            
            # Q Update
            # r + shaping
            # shaping 这里作为 Policy Bias
            target = (r + shaping) + (0 if done else gamma * np.max(Q[next_state]))
            Q[state, a] += alpha * (target - Q[state, a])
            
            state = next_state
            total_r += r
            
        window.append(total_r)
        if len(window) > 50: window.pop(0)
        returns.append(np.mean(window))
        
    return returns

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("初始化猎手环境: 平时死水微澜(Cost=-0.1)，偶尔暴利(+10.0)...")
    env = HuntingEnv()
    
    print("1. Q-Learning (Naive)...")
    res_q = train_hunter(env)
    
    print("2. OptionRL-BS (Low Vol view)...")
    res_bs = train_hunter(env, pricing_model='bs')
    
    print("3. OptionRL-Merton (Fat Tail view)...")
    res_merton = train_hunter(env, pricing_model='merton')
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_q, label='Q-Learning', color='grey', alpha=0.5)
    plt.plot(res_bs, label='BS Model (Blind to Tails)', color='orange', linestyle='--')
    plt.plot(res_merton, label='Merton Model (Tail Hunter)', color='green', linewidth=2)
    
    plt.title("Hunting for Rare Jumps: The Value of Fat-Tail Models")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Return (Moving Avg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'fat_tail_benchmark.png'
    plt.savefig(save_path)
    print(f"终极对比已保存: {save_path}")
