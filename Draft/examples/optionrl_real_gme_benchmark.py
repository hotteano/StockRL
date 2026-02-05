
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
# 真实数据: GME (GameStop) Jan 2021 Short Squeeze
# =============================================================================

# GME Daily Closing Prices (Approximate) from 2021-01-04 to 2021-02-05
# 这段数据包含了人类金融史上最疯狂的跳跃
GME_PRICES = [
    17.25, 17.37, 18.36, 18.08, 17.69, # Jan 4-8 (暴风雨前的宁静)
    19.94, 20.45, 31.40, 39.91, 35.50, # Jan 11-15 (开始异动: Jump!)
    35.50, 39.36, 39.12, 43.03, 65.01, # Jan 19-22 (波动率抬头)
    76.79, 147.98, 347.51, 193.60, 325.00, # Jan 25-29 (THE SQUEEZE: 史诗级崩盘/暴涨)
    225.00, 90.00, 92.41, 53.50, 63.77  # Feb 1-5 (一地鸡毛)
]

# 我们把数据插值加密，模拟小时级别的交易，让 RL 有操作空间
# 将每天拆分为 8 个小时
GME_HOURLY = []
for i in range(len(GME_PRICES)-1):
    start = GME_PRICES[i]
    end = GME_PRICES[i+1]
    # 添加一些随机日内波动
    steps = np.linspace(start, end, 9)[:-1]
    noise = np.random.normal(0, start*0.02, 8)
    GME_HOURLY.extend(steps + noise)
GME_HOURLY = np.array(GME_HOURLY)

# =============================================================================
# 环境: 空头生存挑战 (Short Seller Survival)
# =============================================================================

class RealGMEEnv:
    """
    你是一个持有 GME 空单的机构。
    你的目标是：活下来，并尽可能减少亏损，或者在崩盘(回落)时赚钱。
    
    Action 0: Hold Short (继续持有空单，赌它会跌回来)
    Action 1: Close Short/Cover (平仓止损/止盈)
    Action 2: Re-open Short (加仓空单)
    
    难点：
    - 在 Jan 25-29 那几天，价格每天翻倍 (Jump)。
    - BS 模型看着历史波动率 (Jan 4-8)，觉得 Sigma 很低，会建议你“死扛”，因为“理论上不可能涨这么多”。
    - Merton 模型会因为近期的小跳跃 (Jan 13) 迅速调高 Lambda，警告你“大的要来了”，建议提前止损。
    """
    def __init__(self):
        self.prices = GME_HOURLY
        self.max_steps = len(self.prices) - 1
        self.reset()
        
    def reset(self):
        self.t = 0
        self.position = -1.0 # 初始持有 1 单位空单
        self.cash = 100.0    # 初始保证金
        self.initial_wealth = 100.0
        self.done = False
        self.history = []
        return self._get_state()
    
    def _get_state(self):
        # 归一化价格，时间
        p = self.prices[self.t] / 20.0 
        t = self.t / self.max_steps
        # 简单的持仓状态
        pos = self.position
        return (p, t, pos)
    
    def step(self, action):
        current_price = self.prices[self.t]
        self.t += 1
        next_price = self.prices[self.t]
        
        reward = 0.0
        
        # 1. 计算通过持仓产生的盈亏 (Mark to Market)
        # 空单收益 = (Price_t - Price_t+1) * Position_size
        # 如果价格涨了 (next > current)，空单亏钱
        pnl = (current_price - next_price) * abs(self.position) if self.position < 0 else 0
        
        self.cash += pnl
        
        # 2. 执行动作
        if action == 1: # 平仓 (Cover)
             self.position = 0.0
        elif action == 2: # 加仓/开仓 (Short More)
             self.position = -1.0
        elif action == 0: # Hold
             pass
             
        # 3. 爆仓检测 (Margin Call)
        # 如果保证金跌破 20，强制爆仓，游戏结束
        if self.cash < 20.0:
            reward = -100.0 # 巨大的惩罚
            self.done = True
        else:
            # 存活奖励 + 每日盈亏
            reward = pnl * 0.1 # 缩放
            
        if self.t >= self.max_steps:
            self.done = True
            
        return self._get_state(), reward, self.done

# =============================================================================
# 模型计算 (Rolling Window Estimation)
# =============================================================================

def estimate_metrics(history_prices):
    """简单估算当前的 Sigma 和 Jump Intensity"""
    if len(history_prices) < 10:
        return 0.02, 0 # Default
        
    returns = np.diff(np.log(history_prices[-20:])) # 看最近20小时
    sigma = np.std(returns) * np.sqrt(8*252) # 年化
    
    # 简单的跳跃检测：如果回报率超过 3个标准差，算一次 Jump
    # 这是 Merton 模型的简化在线版本
    jumps = np.sum(np.abs(returns) > 3 * np.std(returns))
    lam = jumps * (252*8 / 20) # 年化跳跃频率
    
    return sigma, lam

def bs_put_price(S, K, T, r, sigma):
    # 空头关注的是 Put Option 价值 (类似持有空单的对冲价值)
    # 或者我们计算 Call Price 作为“风险值” (Risk Metric)
    # 为了统一逻辑，我们计算 Call Price：
    # 如果 Call Price 很高，意味着上涨风险极大，应该平空仓 (Action 1)
    if T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def merton_risk_metric(S, K, T, r, sigma, lam):
    """
    计算 Merton 模型下的上行风险 (Upward Risk)。
    如果 Lambda 很高，这个值会比 BS 高得多。
    """
    # 简化：只加一项跳跃调整
    # 假设跳跃大概率是正向的 (+20%)
    mu_j = 0.2; delta_j = 0.1
    k = np.exp(mu_j + 0.5 * delta_j**2) - 1
    lam_prime = lam * (1 + k)
    
    # Base BS
    price = bs_put_price(S, K, T, r, sigma) * np.exp(-lam_prime * T)
    
    # Add Jump Terms (主要关心发生1次或多次跳跃的风险)
    for n in range(1, 5):
        r_n = r - lam * k + (n * np.log(1 + k)) / T
        sigma_n = np.sqrt(sigma**2 + (n * delta_j**2) / T)
        weight = (np.exp(-lam_prime * T) * (lam_prime * T)**n) / math.factorial(n)
        
        # 这里的关键：当 sigma 变大 (sigma_n)，Risk Metric 应该剧增
        price += weight * bs_put_price(S, K, T, r_n, sigma_n)
        
    return price

# =============================================================================
# 训练脚本
# =============================================================================

def run_simulation(model_type):
    env = RealGMEEnv()
    state = env.reset()
    wealth_history = [env.cash]
    
    # 这里我们不做复杂的 Q-Learning 训练 (样本太少)，
    # 而是直接展示 Policy 在不同 Model 指导下的行为。
    # Model-Based Policy:
    # 比较 "风险值" (Risk) vs "持有收益"
    # 如果 Risk > Threshold, 平仓。
    
    history_prices = [env.prices[0]]
    
    while not env.done:
        curr_price = env.prices[env.t]
        history_prices.append(curr_price)
        
        # 1. 在线估计参数
        sigma, lam = estimate_metrics(history_prices)
        
        # 2. 计算风险 (Risk of Squeeze)
        # S = Current Price
        # K = Cost Basis (假设我们在 20 块开的空单) -> 实际上这里 K 可以是动态的
        # T = 剩余时间
        risk_val = 0.0
        
        # 我们计算一个 OTM Call (S, K=S*1.1) 的价格，代表短期暴涨 10% 的风险成本
        if model_type == 'naive': # 傻瓜空头
            risk_val = 0.0
            
        elif model_type == 'bs':
            # BS 只能看到 Sigma
            # 在 GME 早期，Sigma 很低，BS 会严重低估风险
            risk_val = bs_put_price(curr_price, curr_price*1.1, 0.1, 0.02, sigma)
            
        elif model_type == 'merton':
            # Merton 能看到 Lambda (Frequency of Jumps)
            # 一旦 GME 开始小跳，Lambda 飙升，Merton 风险值会爆炸
            risk_val = merton_risk_metric(curr_price, curr_price*1.1, 0.1, 0.02, sigma, lam)
        
        # 3. 决策逻辑
        action = 0 # Default Hold
        
        # 动态调整阈值
        risk_threshold = 2.0 
        
        if risk_val > risk_threshold:
            action = 1 # Run! (Cover)
        else:
            if env.position == 0 and risk_val < 0.5:
                action = 2 # Re-short if safe
            else:
                action = 0 # Hold
                
        # 强制修正：BS 模型往往反应迟钝
        # 我们让它们都在同一个逻辑下跑，只看 risk_val 的区别
        
        _, _, done = env.step(action)
        wealth_history.append(env.cash)
        
    return wealth_history

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Survival Analysis on Real GME Data (Jan 2021)...")
    
    # 1. Naive Short Seller
    w_naive = run_simulation('naive')
    
    # 2. BS-Guided Seller
    w_bs = run_simulation('bs')
    
    # 3. Merton-Guided Seller
    w_merton = run_simulation('merton')
    
    plt.figure(figsize=(10, 6))
    
    # Plot Wealth
    plt.plot(w_naive, label='Naive Short (Hold forever)', color='gray', linestyle=':')
    plt.plot(w_bs, label='BS Model (Reaction based on Vol)', color='orange', linestyle='--')
    plt.plot(w_merton, label='Merton Model (Reaction based on Jumps)', color='green', linewidth=2)
    
    # Plot Price Background
    ax2 = plt.gca().twinx()
    ax2.plot(GME_HOURLY, color='red', alpha=0.15, label='GME Price')
    ax2.set_ylabel('GME Price ($)', color='red', alpha=0.3)
    
    plt.title("Real-World Fat Tail: GME Short Squeeze Survival")
    plt.xlabel("Hours (Jan 2021)")
    plt.ylabel("Fund Net Value")
    plt.legend(loc='upper left')
    
    save_path = 'real_gme_benchmark.png'
    plt.savefig(save_path)
    print(f"真实数据回测完成: {save_path}")
    print("Merton 模型应当能提前识别出'跳跃频率(Lambda)'的增加，从而在 $300 大崩盘前提前平仓止损。")
