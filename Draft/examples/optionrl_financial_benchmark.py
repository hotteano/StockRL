
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# =============================================================================
# 金融数据生成器
# =============================================================================

class GBMGenerator:
    """几何布朗运动价格生成器"""
    def __init__(self, s0=100.0, mu=0.05, sigma=0.2, dt=1/252):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        
    def generate_path(self, steps=100):
        path = [self.s0]
        s = self.s0
        for _ in range(steps):
            # dS = S * (mu*dt + sigma*dW)
            dw = np.random.normal(0, np.sqrt(self.dt))
            ds = s * (self.mu * self.dt + self.sigma * dw)
            s += ds
            path.append(s)
        return np.array(path)

# =============================================================================
# 期权/概率 计算工具
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes 看涨期权定价"""
    if T <= 0:
        return max(0.0, S - K)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def touch_probability(S, K, T, mu, sigma):
    """
    计算在时间 T 内触及目标价格 K (S < K) 的（真实概率近似）
    这实际上是首达时 (First Hitting Time) 的概率分布积分，或者简化理解。
    为了作为 Potential，我们使用简单的 'In the Money Probability' (ND2) 
    或者 BS Price 本身都可以。
    
    这里为了展示 StockRL 的特性，我们直接用 BS Price 作为“价值势能”。
    """
    return bs_call_price(S, K, T, 0.02, sigma)

# =============================================================================
# 交易环境: Optimal Execution / Stopping
# =============================================================================

class TradingEnv:
    """
    目标：在 T 天内，让价格触及 Target (S0 * 1.05)。
    如果中途触及，自动成功 (Reward=1)。
    如果到期没触及，失败 (Reward=0)。
    
    *这是为了模拟稀疏奖励：只有最终结果，没有中间反馈*
    但为了让 Agent 有操作空间，我们把题目稍微改难一点：
    能够 'Double Down' (加仓) 或 'Give Up' (放弃)？
    
    为了最简单对比，我们做一个纯粹的【持有 vs 止损】环境。
    每天可以决定：继续持有 (Action=0) 或 立即平仓 (Action=1)。
    - 如果 Action=1: 
        如果 P > Target: Reward = 1 (成功止盈)
        如果 P < Target: Reward = -0.1 (止损，小惩罚)
    - 如果持有到最后一天:
        强制平仓，规则同上。
        
    难点：要学会区分“有希望涨上去”和“没希望了赶紧跑”。
    """
    def __init__(self, max_steps=60):
        self.max_steps = max_steps
        self.target_ratio = 1.05
        self.stop_loss_ratio = 0.95
        
        self.gen = GBMGenerator(mu=0.05, sigma=0.2)
        self.reset()
        
    def reset(self):
        # 每次生成一条新路径
        # 为了保证可重复性，可以在外部控制 seed，这里为了训练随机化
        self.path = self.gen.generate_path(self.max_steps + 10)
        self.t = 0
        self.s0 = self.path[0]
        self.target_price = self.s0 * self.target_ratio
        self.current_price = self.s0
        self.done = False
        return self._get_state()
        
    def _get_state(self):
        # 状态: [归一化价格, 剩余时间比例]
        # 离散化状态以便 Q-Learning 使用
        price_ratio = self.current_price / self.s0
        time_left = (self.max_steps - self.t) / self.max_steps
        
        # 简单的离散化
        p_idx = int((price_ratio - 0.9) / 0.02) # 0.90, 0.92 ... 1.10 => 0..10
        p_idx = max(0, min(10, p_idx))
        
        t_idx = int(time_left * 5) # 0..5
        
        return (p_idx, t_idx)
    
    def step(self, action):
        # Action 0: Hold (Wait)
        # Action 1: Cut (Stop/Exercise)
        
        if self.done:
            return self._get_state(), 0, True
            
        reward = 0
        self.t += 1
        self.current_price = self.path[self.t]
        
        # 自动触发判定
        hit_target = self.current_price >= self.target_price
        hit_stop = self.current_price <= (self.s0 * self.stop_loss_ratio)
        time_up = self.t >= self.max_steps
        
        if action == 1: # 主动平仓
            if hit_target:
                reward = 1.0
            else:
                reward = -0.1 # 没到目标就跑了，算是小亏
            self.done = True
            
        elif action == 0: # 继续持有
            # 检查是否触及被动止盈止损
            if hit_target:
                reward = 1.0
                self.done = True
            elif hit_stop:
                reward = -1.0 # 触及止损线，大亏
                self.done = True
            elif time_up:
                reward = -0.5 # 超时未达标
                self.done = True
            else:
                reward = 0.0 # 继续等待
                
        return self._get_state(), reward, self.done

# =============================================================================
# Agents
# =============================================================================

def train_agent_finance(env, potential_func=None, episodes=1000):
    # State space: 11 prices x 6 times = 66 states
    Q = np.zeros((11, 6, 2)) 
    alpha = 0.1
    gamma = 0.99
    eps = 0.2
    
    returns = []
    window = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_r = 0
        
        # 记录路径上的 Potential 以进行 PBRS 更新
        # 这里简化：只在 Transition 时计算 Potential Difference
        
        while not done:
            p_idx, t_idx = state
            
            if random.random() < eps:
                action = random.randint(0, 1)
            else:
                action = np.argmax(Q[p_idx, t_idx])
            
            next_state, r, done = env.step(action)
            next_p, next_t = next_state
            
            # --- Reward Shaping Logic ---
            shaping = 0.0
            if potential_func:
                # 获取当前和下一状态的实际物理值（不仅是索引）
                # 这里为了简化，我们直接用环境内部变量（作弊一点点，为了展示效果）
                # 真实应用中，Agent 会根据 state 估算 Potential
                
                # S, K, T, r, sigma
                # S = current_price
                # K = target_price
                # T = (max_steps - t) * dt
                
                s_curr = env.path[env.t - 1] # 上一步的 S
                t_curr = (env.max_steps - (env.t - 1)) / 252.0
                
                s_next = env.current_price
                t_next = (env.max_steps - env.t) / 252.0
                
                # Potential = BS Price
                # 意义：当前持仓的理论价值
                # 如果 action=1 (Close)，下一状态 potential 为 0
                
                phi_curr = potential_func(s_curr, env.target_price, t_curr, 0.02, 0.2)
                
                if done:
                    phi_next = 0.0
                else:
                    phi_next = potential_func(s_next, env.target_price, t_next, 0.02, 0.2)
                
                # F = gamma * phi_next - phi_curr
                # 缩放一下因为 BS Price 是绝对值 (e.g. 5.0)，Reward 是 (-1, 1)
                # target price ~ 105, s ~ 100. BS ~ 2-3. Scale by 0.1
                factor = 0.1
                shaping = factor * (gamma * phi_next - phi_curr)
            
            # -----------------------------
            
            target = r + shaping + (0 if done else gamma * np.max(Q[next_p, next_t]))
            Q[p_idx, t_idx, action] += alpha * (target - Q[p_idx, t_idx, action])
            
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
    env = TradingEnv()
    
    print("开始金融环境训练对比 (Financial Task: Target Reach)...")
    
    # Baseline: Q-Learning
    print("1. Training Q-Learning...")
    res_q = train_agent_finance(env, potential_func=None)
    
    # OptionRL: BS Potential
    print("2. Training OptionRL (Black-Scholes Potential)...")
    res_opt = train_agent_finance(env, potential_func=bs_call_price)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_q, label='Q-Learning', color='grey', alpha=0.6)
    plt.plot(res_opt, label='OptionRL (BS-Shaping)', color='#c0392b', linewidth=2)
    
    plt.title("Financial Task: Catching the Bull Trend")
    plt.xlabel("Episodes")
    plt.ylabel("Avg Return (Moving Avg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'financial_benchmark.png'
    plt.savefig(save_path)
    print(f"结果已保存至 {save_path}")
