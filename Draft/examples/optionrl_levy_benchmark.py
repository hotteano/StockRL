
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson
import random
import math

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# =============================================================================
# 1. Levy Process 生成器 (Merton Jump Diffusion)
# =============================================================================

class MertonJumpDiffusionGenerator:
    """
    默顿跳跃扩散模型 (Merton Jump Diffusion Model)
    S_t = S_0 * exp((r - 0.5*sigma^2 - lambda*k) * t + sigma*W_t + sum_{i=1}^{N_t} Y_i)
    
    其中:
    - sigma: 扩散波动率 (Diffusive Volatility)
    - lambda (lam): 跳跃强度 (Jump Intensity, 每年发生跳跃的次数期望)
    - mu_j: 跳跃幅度的对数均值 (Mean of Log Jump Size)
    - delta_j: 跳跃幅度的对数标准差 (Std of Log Jump Size)
    - k: 跳跃的期望相对增量 E[exp(Y)-1] = exp(mu_j + 0.5*delta_j^2) - 1
    """
    def __init__(self, s0=100.0, mu=0.05, sigma=0.2, dt=1/252, 
                 lam=1.0, mu_j=-0.2, delta_j=0.1):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.lam = lam      # 泊松过程参数
        self.mu_j = mu_j    # 跳跃大小均值 (负数代表崩盘)
        self.delta_j = delta_j
        
        # 修正漂移项，使得由 mu 定义总的预期收益率
        # k 是跳跃导致的期望百分比变化
        self.k = np.exp(self.mu_j + 0.5 * self.delta_j**2) - 1
        
    def generate_path(self, steps=100):
        # 预生成随机数
        # 扩散部分
        dw = np.random.normal(0, np.sqrt(self.dt), steps)
        
        # 跳跃部分 (泊松过程)
        # 每一个时间步发生的跳跃次数 (通常是 0 或 1，也可以 >1)
        # Pois(lambda * dt)
        n_jumps = np.random.poisson(self.lam * self.dt, steps)
        
        # 计算路径
        log_ret = np.zeros(steps)
        for i in range(steps):
            jump_sum = 0.0
            if n_jumps[i] > 0:
                # 如果发生了跳跃，生成跳跃幅度
                # Y ~ N(mu_j, delta_j^2)
                curr_jumps = np.random.normal(self.mu_j, self.delta_j, n_jumps[i])
                jump_sum = np.sum(curr_jumps)
            
            # Merton Model Log Return Formula
            # d(ln S) = (mu - lambda*k - 0.5*sigma^2)*dt + sigma*dW + dJ
            drift = (self.mu - self.lam * self.k - 0.5 * self.sigma**2) * self.dt
            diffusion = self.sigma * dw[i]
            
            log_ret[i] = drift + diffusion + jump_sum
            
        # 累加对数收益率重构价格
        cum_log_ret = np.cumsum(log_ret)
        path = self.s0 * np.exp(np.concatenate(([0], cum_log_ret)))
        return path

# =============================================================================
# 2. 定价公式 (BS vs Merton)
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    """标准的 Black-Scholes 定价"""
    if T <= 0: return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def merton_call_price(S, K, T, r, sigma, lam, mu_j, delta_j):
    """
    Merton Jump Diffusion Option Pricing Formula
    它是无穷多个 BS 价格的加权和。
    """
    if T <= 0: return max(0.0, S - K)
    
    # 期望跳跃幅度 k
    k = np.exp(mu_j + 0.5 * delta_j**2) - 1
    
    # 修改后的跳跃强度 lambda'
    lam_prime = lam * (1 + k)
    
    price = 0.0
    # 通常前 10-20 项就收敛了
    for n in range(15):
        # 针对 n 次跳跃调整后的参数
        # r_n: 包含跳跃补偿的无风险利率
        r_n = r - lam * k + (n * np.log(1 + k)) / T
        
        # sigma_n: 包含跳跃方差的调整后波动率
        sigma_n_sq = sigma**2 + (n * delta_j**2) / T
        sigma_n = np.sqrt(sigma_n_sq)
        
        # 泊松概率权重 (注意这里用的是调整后的 lam_prime)
        # P(N_t = n) = e^(-lam'*T) * (lam'*T)^n / n!
        weight = (np.exp(-lam_prime * T) * (lam_prime * T)**n) / math.factorial(n)
        
        bs_val = bs_call_price(S, K, T, r_n, sigma_n)
        price += weight * bs_val
        
    return price

# =============================================================================
# 3. 交易环境 (带跳跃风险)
# =============================================================================

class LevyTradingEnv:
    """
    环境：目标回报 5%，止损 -10%。
    特点：偶尔会发生剧烈的向下跳跃 (Crash)，这对 BS 模型是不可见的黑天鹅。
    """
    def __init__(self, max_steps=60):
        self.max_steps = max_steps
        self.target_ratio = 1.05
        self.stop_loss_ratio = 0.90 # 放宽一点止损，看看跳跃的影响
        
        # 参数配置：主要向下跳跃 (Crash Risk)
        # 每年发生 2 次跳跃，平均每次跳 -10%，波动 5%
        self.gen = MertonJumpDiffusionGenerator(
            mu=0.05, sigma=0.15, 
            lam=2.0, mu_j=-0.10, delta_j=0.05
        )
        self.reset()
        
    def reset(self):
        self.path = self.gen.generate_path(self.max_steps + 10)
        self.t = 0
        self.s0 = self.path[0]
        self.target_price = self.s0 * self.target_ratio
        self.current_price = self.s0
        self.done = False
        return self._get_state()
        
    def _get_state(self):
        price_ratio = self.current_price / self.s0
        time_left = (self.max_steps - self.t) / self.max_steps
        
        # 离散化
        # 0.85 ... 1.15 -> 30 个 bin 细一点
        p_idx = int((price_ratio - 0.85) / 0.01)
        p_idx = max(0, min(29, p_idx))
        
        t_idx = int(time_left * 5)
        return (p_idx, t_idx)
    
    def step(self, action):
        # Action 0: Hold
        # Action 1: Cut
        
        if self.done: return self._get_state(), 0, True
        
        self.t += 1
        self.current_price = self.path[self.t]
        
        reward = 0.0
        
        # 检查边界
        hit_target = self.current_price >= self.target_price
        hit_stop = self.current_price <= (self.s0 * self.stop_loss_ratio)
        time_up = self.t >= self.max_steps
        
        if action == 1: # 主动平仓
            if hit_target: reward = 1.0
            else: reward = -0.1
            self.done = True
            
        elif action == 0: # Hold
            if hit_target:
                reward = 1.0
                self.done = True
            elif hit_stop:
                reward = -1.0 # 止损惩罚
                self.done = True
            elif time_up:
                reward = -0.5 # 超时
                self.done = True
                
        return self._get_state(), reward, self.done

# =============================================================================
# 4. 训练逻辑
# =============================================================================

def train_agent(env, pricing_model=None, episodes=1000):
    # State: 30 price levels x 6 time levels
    Q = np.zeros((30, 6, 2))
    returns = []
    window = []
    
    alpha = 0.1
    gamma = 0.99
    eps = 0.2
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        while not done:
            p_idx, t_idx = state
            
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = np.argmax(Q[p_idx, t_idx])
            
            next_state, r, done = env.step(a)
            next_p, next_t = next_state
            
            # --- Reward Shaping ---
            shaping = 0.0
            if pricing_model == 'bs':
                # BS 只看到扩散风险 sigma=0.15
                func = lambda S, K, T: bs_call_price(S, K, T, 0.02, 0.15)
            elif pricing_model == 'merton':
                # Merton 看到全貌：扩散=0.15, 跳跃 lam=2, mean=-0.10
                func = lambda S, K, T: merton_call_price(S, K, T, 0.02, 0.15, 2.0, -0.10, 0.05)
            else:
                func = None
                
            if func:
                s_curr = env.path[env.t - 1]
                t_curr = (env.max_steps - (env.t - 1)) / 252.0
                s_next = env.current_price
                t_next = (env.max_steps - env.t) / 252.0
                
                phi_curr = func(s_curr, env.target_price, max(0, t_curr))
                phi_next = 0.0 if done else func(s_next, env.target_price, max(0, t_next))
                
                # 缩放系数
                shaping = 0.1 * (gamma * phi_next - phi_curr)
            # ----------------------
            
            target = r + shaping + (0 if done else gamma * np.max(Q[next_p, next_t]))
            Q[p_idx, t_idx, a] += alpha * (target - Q[p_idx, t_idx, a])
            
            state = next_state
            total_r += r
            
        window.append(total_r)
        if len(window) > 100: window.pop(0)
        returns.append(np.mean(window))
        
    return returns

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("初始化 Levy (Merton Jump Diffusion) 环境...")
    print("风险特征: 正常波动 15%, 但每年期望发生2次 -10% 的崩盘跳跃")
    
    env = LevyTradingEnv()
    
    print("\n1. 训练 Q-Learning (无模型)...")
    res_q = train_agent(env, pricing_model=None)
    
    print("2. 训练 OptionRL-BS (使用错误的高斯假设)...")
    res_bs = train_agent(env, pricing_model='bs')
    
    print("3. 训练 OptionRL-Merton (使用正确的跳跃假设)...")
    res_merton = train_agent(env, pricing_model='merton')
    
    # 绘图
    plt.figure(figsize=(12, 7))
    plt.plot(res_q, label='Q-Learning (Baseline)', color='gray', alpha=0.4)
    plt.plot(res_bs, label='OptionRL (Mis-specified: BS Model)', color='#e67e22', linewidth=1.5, linestyle='--')
    plt.plot(res_merton, label='OptionRL (Correct: Merton Model)', color='#27ae60', linewidth=2)
    
    plt.title("Impact of Model Mis-specification in Jump-Diffusion Environments")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Return (Moving Avg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'levy_benchmark.png'
    plt.savefig(save_path)
    print(f"对比图已保存: {save_path}")
