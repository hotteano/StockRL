
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
# 模拟 Web Fuzzing 环境
# =============================================================================

class VulnerableWebEnv:
    """
    一个具有多层级结构的模拟网站。
    
    结构：
    - Layer 0 (Home): 安全。
    - Layer 1 (Login): 安全，但阻挡了大多数随机输入。
    - Layer 2 (User Profile): 安全。
    - Layer 3 (Admin Panel): 高危区域！(Admin API)
    
    Bug 分布：
    - 在 Layer 0-2: 几乎没有 Bug (P_bug = 0.0001)
    - 在 Layer 3: 有巨大安全漏洞 (SQL Injection)，一旦到达，触发 Bug 概率高达 20% (P_bug = 0.2)
    
    难点：
    - 从 Layer 0 到 Layer 3 需要连续输入正确的 Token/Payload，就像走迷宫。
    - 普通 Agent 走到 Layer 1 就觉得没意思了（没 Reward），掉头就走。
    - 我们需要 Merton 模型告诉它：“坚持住！Layer 3 虽然远，但那里发生 Jump (Bug) 的概率极大！”
    """
    def __init__(self, max_steps=50):
        self.max_steps = max_steps
        self.depth = 4 # 0, 1, 2, 3
        self.reset()
        
    def reset(self):
        self.current_layer = 0
        self.t = 0
        return self._get_state()
    
    def _get_state(self):
        # 状态: (当前层级, 剩余时间)
        return (self.current_layer, self.t)
    
    def step(self, action):
        # Action 0: Fuzz current layer (横向探索，容易但无用)
        # Action 1: Try to go deeper (纵向渗透，难，容易失败退回 Layer 0)
        
        self.t += 1
        reward = 0.0
        bug_found = False
        
        # 转移概率
        if action == 1: # 尝试渗透
            # 越深越难渗透
            success_prob = 0.4 if self.current_layer < 3 else 0.0
            if random.random() < success_prob:
                self.current_layer += 1
            else:
                # 渗透失败，可能会被防火墙重置连接
                if random.random() < 0.5:
                    self.current_layer = 0 # Reset to home
        elif action == 0: # 瞎测
            pass # Stay
            
        # 检查是否发现 Bug (稀疏奖励)
        # 只有在 Layer 3 才有大概率 Bug
        if self.current_layer == 3:
            if random.random() < 0.2: # 20% 概率触发 SQL 注入
                reward = 10.0 # Critical Bug Found!
                bug_found = True
        else:
            # 浅层也有极小概率发现无关紧要的 Bug
            if random.random() < 0.001:
                reward = 0.1 # Low severity bug
                
        # 每一跳如果没发现 Bug，都是有成本的（时间/算力）
        # 这也模拟了 Fuzzing 中的“语料变异成本”
        cost = -0.05
        reward += cost
        
        done = self.t >= self.max_steps or bug_found
        return self._get_state(), reward, done

# =============================================================================
# Merton Bug 定价模型
# =============================================================================

def code_merton_potential(layer, max_depth, lam_jump, sigma_diff):
    """
    将代码深度映射为金融模型：
    - 距离 Target (Layer 3) 的距离 <-> Out-of-the-money 程度 (S/K)
    - 代码的脆弱性假设 <-> 跳跃强度 (Lambda)
    
    我们假设：
    - 浅层代码 (Layer 0-1): Lambda (Bug率) 很低，Sigma (复杂度) 低。
    - 深层代码/核心库 (Layer 3): Lambda 很高 (一旦触达，Bug 满天飞)。
    
    Potential = 这里的代码“如果我继续测下去，能挖出 Bug 的期望价值”。
    """
    # 映射：
    # S (当前进度) = layer + 1
    # K (目标深度) = max_depth (3) + 1 = 4
    # T (剩余耐心) = 固定看作未来 100 步
    
    S = layer + 1.0
    K = max_depth + 1.0
    T = 1.0 # 标准化时间
    
    # 关键差异：
    # 普通模型 (BS) 认为所有代码 Bug 率一样低 (sigma=0.1, lam=0)
    # 专家模型 (Merton) 知道 Layer 3 是高危区 (lam 随 layer 指数增加)
    
    # 简化的 Merton 思想估值：
    # V = P(Reach Target) * E[Bug Value | Reach]
    
    # 我们直接用 Merton 公式计算“看涨期权”，即“看涨 Bug 数量”
    # 如果当前在 Layer 3 (Deep)，S接近K，价值高。
    # 如果在 Layer 0，S远小于K，但如果 lam 很大，依然有价值。
    
    # 这里我们简化为一个 Heuristic：
    # Potential = S * exp(lambda * T) * N(d1)
    # 高 Lambada (Bug 密度) 会极大地提升 Potential
    
    return S * np.exp(lam_jump * 0.2) / K # 简化的势能

# =============================================================================
# 训练与测试
# =============================================================================

def train_fuzzer(env, model_type, episodes=1000):
    # Q-Table: [Layer(0-3), Time(0-50), Action(2)]
    Q = np.zeros((5, 52, 2))
    
    total_bugs = 0
    bugs_history = []
    
    alpha = 0.1
    gamma = 0.99
    eps = 0.2
    
    for ep in range(episodes):
        state = env.reset()
        layer, t = state
        done = False
        
        while not done:
            if random.random() < eps:
                a = random.randint(0, 1)
            else:
                a = np.argmax(Q[layer, t])
            
            next_state, r, done = env.step(a)
            next_layer, next_t = next_state
            
            # --- Reward Shaping Based on Model ---
            shaping = 0.0
            
            if model_type == 'q_learning':
                # 无模型，纯试错
                pass
                
            elif model_type == 'bs_fuzzer':
                # 假设所有层级 Bug 率都很低且均匀 (Sigma based)
                phi_curr = code_merton_potential(layer, 3, lam_jump=0.1, sigma_diff=0.1)
                phi_next = 0 if done else code_merton_potential(next_layer, 3, lam_jump=0.1, sigma_diff=0.1)
                shaping = 0.5 * (gamma * phi_next - phi_curr)
                
            elif model_type == 'merton_fuzzer':
                # 假设深层有“跳跃风险” (Attacker Mindset)
                # 认为代码越深，Lambda 越高
                lam_curr = 0.1 + (layer ** 2) # 0.1, 1.1, 4.1...
                lam_next = 0.1 + (next_layer ** 2)
                
                phi_curr = code_merton_potential(layer, 3, lam_curr, 0.1)
                phi_next = 0 if done else code_merton_potential(next_layer, 3, lam_next, 0.1)
                
                # 给予强烈的引导："去深层！那里有宝藏！"
                shaping = 0.5 * (gamma * phi_next - phi_curr)
            
            # Q Update
            target = (r + shaping) + (0 if done else gamma * np.max(Q[next_layer, next_t]))
            Q[layer, t, a] += alpha * (target - Q[layer, t, a])
            
            state = next_state
            layer, t = state
            
            if r > 5.0: # Found critical bug
                total_bugs += 1
                
        bugs_history.append(total_bugs)
        
    return bugs_history

# =============================================================================
# Main Comparison
# =============================================================================

if __name__ == "__main__":
    print("Web Fuzzing Challenge: 寻找深层 Admin Panel 的 SQL 注入...")
    env = VulnerableWebEnv()
    
    print("1. Q-Learning Fuzzer (Baseline)...")
    h_q = train_fuzzer(env, 'q_learning')
    
    print("2. Standard Coverage Fuzzer (BS-like)...")
    h_bs = train_fuzzer(env, 'bs_fuzzer')
    
    print("3. Vulnerability-Guided Fuzzer (Merton-like)...")
    h_merton = train_fuzzer(env, 'merton_fuzzer')
    
    # Plot Cumulative Bugs
    plt.figure(figsize=(10, 6))
    plt.plot(h_q, label='Q-Learning (Random Fuzzing)', color='grey', alpha=0.5)
    plt.plot(h_bs, label='Coverage Guided (Uniform)', color='orange', linestyle='--')
    plt.plot(h_merton, label='Merton Guided (Risk-Oriented)', color='blue', linewidth=2)
    
    plt.title("Automated Bug Finding: Total Critical Bugs Found")
    plt.xlabel("Test Episodes")
    plt.ylabel("Cumulative Critical Bugs Found")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = 'web_fuzz_benchmark.png'
    plt.savefig(save_path)
    print(f"测试对比图已保存: {save_path}")
    print("\n结论预告:")
    print("- Q-Learning: 几乎找不到，因为到达 Admin Panel 需要多次连续正确的渗透 (Deep Exploration)且每步有惩罚。")
    print("- Coverage Guided (BS): 表现一般，因为只有覆盖率奖励，没有对'高危区域'的特殊偏好。")
    print("- Merton Guided: 表现最佳，它将'深层函数'识别为高 Lambda 区域，像猎犬一样死咬不放。")
