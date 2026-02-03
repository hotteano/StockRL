"""
Geometric Poisson Process Option Pricing for MARL

Based on the theoretical derivation in docs_poisson_derivation_v2.md

Core insight: RL agent capability growth is "step-wise" or "eureka-style",
not continuous smooth improvement. Poisson process captures this better
than Brownian motion.

Key formula (analogous to Black-Scholes):
    C = S₀ · Q(k*; λ(1+σ)T) - K·e^{-rT} · Q(k*; λT)

where Q(k; μ) is the Poisson CCDF (complementary cumulative distribution).
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import special
from typing import Tuple, Optional, Dict
import math


def poisson_ccdf(k: int, mu: float) -> float:
    """
    Compute Poisson CCDF: Q(k; μ) = P(X ≥ k | X ~ Poi(μ))
    
    Uses scipy.stats.poisson for accurate computation.
    
    Args:
        k: Threshold value
        mu: Poisson parameter (λT)
    
    Returns:
        P(X ≥ k) where X ~ Poisson(μ)
    """
    from scipy.stats import poisson as poisson_dist
    
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0 if k > 0 else 1.0
    
    # P(X >= k) = 1 - P(X <= k-1) = 1 - CDF(k-1)
    # Or equivalently: sf(k-1) = survival function at k-1
    return float(poisson_dist.sf(k - 1, mu))


def poisson_ccdf_torch(k: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Poisson CCDF approximation using PyTorch.
    
    Uses the relationship between Poisson CDF and incomplete gamma function,
    with a differentiable approximation.
    
    Args:
        k: Threshold values (can be non-integer for differentiability)
        mu: Poisson parameters
    
    Returns:
        Approximate P(X ≥ k) for X ~ Poisson(μ)
    """
    # For differentiability, we use a smooth approximation
    # Based on the normal approximation to Poisson for large μ
    # P(X ≥ k) ≈ Φ((μ - k + 0.5) / √μ)  (continuity correction)
    
    # Ensure numerical stability
    mu_safe = torch.clamp(mu, min=1e-8)
    
    # Normal approximation with continuity correction
    z = (mu_safe - k + 0.5) / torch.sqrt(mu_safe)
    
    # Standard normal CDF
    ccdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    
    return ccdf


class GeometricPoissonPricer(nn.Module):
    """
    Option pricing under Geometric Poisson Process.
    
    Model: dS_t = μ S_t dt + σ S_t dP_t
    
    Solution: S_t = S_0 · e^{μt} · (1+σ)^{P_t}
    
    where P_t is a Poisson process with intensity λ.
    
    Risk-neutral pricing formula:
        C = S_0 · Q(k*; λ(1+σ)T) - K·e^{-rT} · Q(k*; λT)
    
    This is analogous to Black-Scholes but with:
    - Φ(d₁) → Q(k*; λ(1+σ)T)  [Delta]
    - Φ(d₂) → Q(k*; λT)       [ITM probability]
    
    Physical interpretation for MARL:
    - σ: Jump magnitude when a "breakthrough" occurs
    - λ: Rate of breakthroughs (eureka moments)
    - k*: Minimum number of breakthroughs needed to beat baseline K
    """
    
    def __init__(
        self,
        sigma: float = 0.3,       # Jump magnitude (ability gain per breakthrough)
        lambda_rate: float = 1.0, # Jump intensity (breakthroughs per unit time)
        r: float = 0.0,           # Risk-free rate
        use_torch: bool = True    # Use differentiable version
    ):
        """
        Args:
            sigma: Jump magnitude (σ > 0 for positive jumps)
            lambda_rate: Poisson intensity (λ, expected jumps per unit time)
            r: Risk-free rate
            use_torch: If True, use differentiable PyTorch approximation
        """
        super().__init__()
        self.sigma = sigma
        self.lambda_rate = lambda_rate
        self.r = r
        self.use_torch = use_torch
        
        # Validate
        if sigma <= -1:
            raise ValueError("sigma must be > -1 to avoid negative prices")
    
    def critical_jumps(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute critical number of jumps k* for option to be ITM.
        
        k* = ceil[(ln(K/S) - (r - σλ)T) / ln(1+σ)]
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
        
        Returns:
            k*: Minimum jumps needed for S_T > K
        """
        # Risk-neutral drift adjustment
        drift = (self.r - self.sigma * self.lambda_rate) * T
        
        # Solve: S · e^{drift} · (1+σ)^k > K
        # k > [ln(K/S) - drift] / ln(1+σ)
        
        log_ratio = torch.log(K / S)
        log_jump = np.log(1 + self.sigma)
        
        k_star = (log_ratio - drift) / log_jump
        
        # Ceiling and ensure non-negative
        k_star = torch.ceil(torch.clamp(k_star, min=0))
        
        return k_star
    
    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        return_greeks: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute call option price under Geometric Poisson Process.
        
        C = S₀ · Q(k*; λ(1+σ)T) - K·e^{-rT} · Q(k*; λT)
        
        Args:
            S: Spot price(s)
            K: Strike price(s)
            T: Time to maturity
            return_greeks: If True, also return Greeks
        
        Returns:
            C: Call option price
            greeks: Optional dict with Delta, etc.
        """
        # Compute critical jump number
        k_star = self.critical_jumps(S, K, T)
        
        # Poisson parameters
        lambda_T = self.lambda_rate * T
        lambda_prime_T = self.lambda_rate * (1 + self.sigma) * T
        
        # Compute CCDFs
        if self.use_torch:
            Q_asset = poisson_ccdf_torch(k_star, lambda_prime_T)
            Q_strike = poisson_ccdf_torch(k_star, lambda_T)
        else:
            # Use exact scipy implementation (not differentiable)
            Q_asset = torch.tensor([
                poisson_ccdf(int(k.item()), lp.item()) 
                for k, lp in zip(k_star.flatten(), lambda_prime_T.flatten())
            ]).reshape(k_star.shape)
            Q_strike = torch.tensor([
                poisson_ccdf(int(k.item()), lt.item()) 
                for k, lt in zip(k_star.flatten(), lambda_T.flatten())
            ]).reshape(k_star.shape)
        
        # Option price
        discount = torch.exp(-self.r * T)
        C = S * Q_asset - K * discount * Q_strike
        
        # Ensure non-negative
        C = torch.clamp(C, min=0)
        
        if not return_greeks:
            return C, None
        
        # Greeks
        greeks = {
            'delta': Q_asset,           # Δ = Q(k*; λ'T)
            'itm_prob': Q_strike,       # P(S_T > K) = Q(k*; λT)
            'k_star': k_star,           # Critical jump number
            'lambda_T': lambda_T,       # Expected jumps
        }
        
        return C, greeks
    
    def put_price(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute put option price using put-call parity.
        
        P = C - S + K·e^{-rT}
        """
        C, _ = self.forward(S, K, T)
        discount = torch.exp(-self.r * T)
        P = C - S + K * discount
        return torch.clamp(P, min=0)


class TimeVaryingPoissonPricer(nn.Module):
    """
    Geometric Poisson Pricer with time-varying jump intensity.
    
    λ(t) = λ_∞ + (λ_0 - λ_∞) · e^{-γt}
    
    This captures the "前肥后细" pattern:
    - Early training: High λ (frequent breakthroughs during exploration)
    - Late training: Low λ (rare breakthroughs, stable policy)
    """
    
    def __init__(
        self,
        sigma: float = 0.3,
        lambda_0: float = 2.0,      # Initial jump rate (high)
        lambda_inf: float = 0.2,    # Terminal jump rate (low)
        lambda_decay: float = 0.01, # Decay rate
        r: float = 0.0,
        total_steps: int = 10000
    ):
        super().__init__()
        self.sigma = sigma
        self.lambda_0 = lambda_0
        self.lambda_inf = lambda_inf
        self.lambda_decay = lambda_decay
        self.r = r
        self.total_steps = total_steps
        
        self.base_pricer = GeometricPoissonPricer(sigma, lambda_0, r)
    
    def get_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Get time-varying jump intensity."""
        t_norm = t / self.total_steps
        return self.lambda_inf + (self.lambda_0 - self.lambda_inf) * torch.exp(
            -self.lambda_decay * t_norm * self.total_steps
        )
    
    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        current_step: int = 0
    ) -> torch.Tensor:
        """
        Price option with current time-varying λ.
        
        For simplicity, we use the current λ(t) as if it were constant.
        A more accurate approach would integrate over the time-varying intensity.
        """
        # Get current lambda
        t = torch.tensor(current_step, dtype=torch.float32)
        current_lambda = self.get_lambda(t).item()
        
        # Update base pricer's lambda
        self.base_pricer.lambda_rate = current_lambda
        
        # Price
        C, greeks = self.base_pricer(S, K, T, return_greeks=True)
        
        return C, greeks


class PoissonValuationEngine(nn.Module):
    """
    Valuation engine using Geometric Poisson Process for MARL.
    
    Replaces Black-Scholes with Poisson-based pricing, which is more
    appropriate for the discrete, "eureka-style" capability gains in RL.
    """
    
    def __init__(
        self,
        n_agents: int,
        sigma: float = 0.3,         # Jump magnitude
        lambda_0: float = 2.0,      # Initial jump rate
        lambda_inf: float = 0.2,    # Terminal jump rate
        total_steps: int = 10000,
        device: str = 'cpu'
    ):
        super().__init__()
        self.n_agents = n_agents
        self.device = device
        self.current_step = 0
        self.total_steps = total_steps
        
        self.pricer = TimeVaryingPoissonPricer(
            sigma=sigma,
            lambda_0=lambda_0,
            lambda_inf=lambda_inf,
            total_steps=total_steps
        )
        
        # Track agent "stock prices"
        self.register_buffer('prices', torch.ones(n_agents, device=device))
        self.register_buffer('jump_counts', torch.zeros(n_agents, device=device))
    
    def update_prices(self, shapley_values: torch.Tensor) -> torch.Tensor:
        """
        Update agent prices based on Shapley values.
        
        Detect "jumps" when Shapley value exceeds threshold.
        """
        mean_shapley = shapley_values.mean()
        
        # Detect jumps: significant outperformance
        threshold = mean_shapley * 1.5
        jumped = shapley_values > threshold
        
        # Update jump counts
        self.jump_counts += jumped.float()
        
        # Update prices using Poisson model: S = S_0 * (1+σ)^{jumps}
        sigma = self.pricer.sigma
        self.prices = (1 + sigma) ** self.jump_counts
        
        return self.prices
    
    def get_option_values(
        self,
        strikes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute option values for each agent.
        
        Args:
            strikes: Strike prices (default: average price)
        
        Returns:
            Option values
        """
        if strikes is None:
            strikes = self.prices.mean().expand(self.n_agents)
        
        T = torch.tensor(
            (self.total_steps - self.current_step) / self.total_steps
        ).expand(self.n_agents)
        
        C, greeks = self.pricer(
            self.prices, strikes, T, current_step=self.current_step
        )
        
        return C, greeks
    
    def step(self, shapley_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute one valuation step.
        
        Args:
            shapley_values: Credit assignment for this step
        
        Returns:
            Dict with prices, option values, etc.
        """
        self.current_step += 1
        
        # Update prices
        prices = self.update_prices(shapley_values)
        
        # Compute option values
        option_values, greeks = self.get_option_values()
        
        return {
            'prices': prices,
            'option_values': option_values,
            'jump_counts': self.jump_counts,
            'delta': greeks['delta'],
            'itm_prob': greeks['itm_prob'],
            'k_star': greeks['k_star'],
            'current_lambda': self.pricer.get_lambda(
                torch.tensor(self.current_step, dtype=torch.float32)
            ),
        }
    
    def reset(self):
        """Reset engine."""
        self.prices.fill_(1.0)
        self.jump_counts.zero_()
        self.current_step = 0


# ============================================================================
# Comparison: Black-Scholes vs Poisson
# ============================================================================

def compare_models():
    """Compare Black-Scholes and Poisson pricing."""
    print("=" * 60)
    print("Comparison: Black-Scholes vs Geometric Poisson")
    print("=" * 60)
    
    # Import BS
    from .option_pricing import BlackScholesLayer
    
    # Parameters
    S = torch.tensor([100.0])
    K = torch.tensor([100.0])
    T = torch.tensor([1.0])
    r = 0.05
    
    # Black-Scholes
    bs = BlackScholesLayer()
    sigma_bs = torch.tensor([0.2])
    C_bs, greeks_bs = bs(S, K, sigma_bs, T, r, return_greeks=True)
    
    # Poisson
    poisson = GeometricPoissonPricer(sigma=0.2, lambda_rate=1.0, r=r, use_torch=False)
    C_poi, greeks_poi = poisson(S, K, T, return_greeks=True)
    
    print(f"\nParameters: S={S.item()}, K={K.item()}, T={T.item()}, r={r}")
    print(f"\n{'Model':<20} {'Price':>10} {'Delta':>10} {'ITM Prob':>10}")
    print("-" * 50)
    print(f"{'Black-Scholes':<20} {C_bs.item():>10.4f} {greeks_bs['delta'].item():>10.4f} {greeks_bs['delta'].item():>10.4f}")
    print(f"{'Geometric Poisson':<20} {C_poi.item():>10.4f} {greeks_poi['delta'].item():>10.4f} {greeks_poi['itm_prob'].item():>10.4f}")
    print(f"\nPoisson model: k* = {greeks_poi['k_star'].item():.0f} jumps needed to be ITM")


# ============================================================================
# Unit Tests
# ============================================================================

def test_poisson_ccdf():
    """Test Poisson CCDF calculation."""
    print("Testing Poisson CCDF...")
    
    # Test known values
    # P(X >= 0 | μ=5) = 1
    assert abs(poisson_ccdf(0, 5.0) - 1.0) < 1e-6
    
    # P(X >= 10 | μ=5) should be small but not tiny
    p = poisson_ccdf(10, 5.0)
    print(f"  Q(0; 5) = {poisson_ccdf(0, 5.0):.4f}")
    print(f"  Q(5; 5) = {poisson_ccdf(5, 5.0):.4f}")
    print(f"  Q(10; 5) = {poisson_ccdf(10, 5.0):.4f}")
    
    # CCDF should be decreasing
    assert poisson_ccdf(0, 5.0) > poisson_ccdf(5, 5.0) > poisson_ccdf(10, 5.0)
    
    print("  ✓ Poisson CCDF tests passed!\n")


def test_poisson_pricer():
    """Test Geometric Poisson option pricer."""
    print("Testing Geometric Poisson Pricer...")
    
    pricer = GeometricPoissonPricer(sigma=0.3, lambda_rate=1.0, r=0.05, use_torch=False)
    
    S = torch.tensor([100.0])
    K = torch.tensor([100.0])
    T = torch.tensor([1.0])
    
    C, greeks = pricer(S, K, T, return_greeks=True)
    
    print(f"  Option price: {C.item():.4f}")
    print(f"  Delta: {greeks['delta'].item():.4f}")
    print(f"  ITM Probability: {greeks['itm_prob'].item():.4f}")
    print(f"  Critical jumps k*: {greeks['k_star'].item():.0f}")
    
    # Price should be positive
    assert C.item() > 0, "Option price should be positive"
    
    # Delta should be in [0, 1]
    assert 0 <= greeks['delta'].item() <= 1, "Delta should be in [0, 1]"
    
    print("  ✓ Poisson pricer tests passed!\n")


def test_time_varying():
    """Test time-varying jump intensity."""
    print("Testing Time-Varying Poisson...")
    
    pricer = TimeVaryingPoissonPricer(
        sigma=0.3,
        lambda_0=2.0,
        lambda_inf=0.2,
        total_steps=1000
    )
    
    t_early = torch.tensor(100.0)
    t_late = torch.tensor(900.0)
    
    lambda_early = pricer.get_lambda(t_early)
    lambda_late = pricer.get_lambda(t_late)
    
    print(f"  λ(t=100): {lambda_early.item():.4f}")
    print(f"  λ(t=900): {lambda_late.item():.4f}")
    
    assert lambda_early > lambda_late, "Early lambda should be higher (more jumps)"
    
    print("  ✓ Time-varying tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Geometric Poisson Pricing Module Tests")
    print("=" * 60 + "\n")
    
    test_poisson_ccdf()
    test_poisson_pricer()
    test_time_varying()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
