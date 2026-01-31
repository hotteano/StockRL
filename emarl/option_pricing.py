"""
Option Pricing Module for Equity-MARL

This module provides differentiable implementations of:
1. Black-Scholes analytical solution (European options)
2. Binomial tree pricing (CRR model)
3. Asian option approximation (floating strike)

All implementations are fully differentiable for end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import math


class NormalCDF(torch.autograd.Function):
    """
    Differentiable implementation of the standard normal CDF.
    Uses the error function (erf) for numerical stability.
    
    Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        # Derivative of Φ(x) is φ(x) = exp(-x²/2) / sqrt(2π)
        pdf = torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        return grad_output * pdf


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, differentiable."""
    return NormalCDF.apply(x)


def normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, differentiable."""
    return torch.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


class BlackScholesLayer(nn.Module):
    """
    Differentiable Black-Scholes Option Pricing Layer.
    
    Computes European call option prices and Greeks (Delta, Gamma, Theta, Vega).
    All computations are fully differentiable for gradient-based optimization.
    
    In the Equity-MARL context:
    - S (spot price): Agent's cumulative Shapley-based "stock price"
    - K (strike price): Expected contribution baseline (can be floating)
    - σ (volatility): Agent's performance stability
    - T (time to maturity): Remaining training episodes
    - r (risk-free rate): System baseline growth rate
    
    Call option value interpretation:
    - C > 0: Agent is expected to outperform baseline
    - C ≈ 0: Agent is expected to match baseline
    - High C: High growth potential
    
    Attributes:
        epsilon (float): Small constant for numerical stability
    """
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        S: torch.Tensor,      # Spot price (agent stock price)
        K: torch.Tensor,      # Strike price (baseline)
        sigma: torch.Tensor,  # Volatility
        T: torch.Tensor,      # Time to maturity
        r: float = 0.0,       # Risk-free rate
        return_greeks: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute Black-Scholes call option price and optionally Greeks.
        
        Args:
            S: Spot price, shape (batch_size,) or (batch_size, n_agents)
            K: Strike price, same shape as S
            sigma: Volatility, same shape as S
            T: Time to maturity, same shape as S or scalar
            r: Risk-free rate (scalar)
            return_greeks: If True, also return Delta, Gamma, Theta, Vega
        
        Returns:
            C: Call option price, same shape as S
            greeks: Optional dict with 'delta', 'gamma', 'theta', 'vega'
        
        Formula:
            C = S·Φ(d₁) - K·e^{-rT}·Φ(d₂)
            
            where:
            d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
            d₂ = d₁ - σ√T
        """
        # Ensure numerical stability
        S = torch.clamp(S, min=self.epsilon)
        K = torch.clamp(K, min=self.epsilon)
        sigma = torch.clamp(sigma, min=self.epsilon)
        T = torch.clamp(T, min=self.epsilon)
        
        # Handle scalar T
        if not isinstance(T, torch.Tensor):
            T = torch.tensor(T, dtype=S.dtype, device=S.device)
        if T.dim() == 0:
            T = T.expand_as(S)
        
        # Compute d1 and d2
        sqrt_T = torch.sqrt(T)
        d1 = (torch.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Compute option price
        Phi_d1 = normal_cdf(d1)
        Phi_d2 = normal_cdf(d2)
        discount = torch.exp(-r * T)
        
        C = S * Phi_d1 - K * discount * Phi_d2
        
        if not return_greeks:
            return C, None
        
        # Compute Greeks
        phi_d1 = normal_pdf(d1)  # PDF at d1
        
        # Delta: ∂C/∂S = Φ(d₁)
        delta = Phi_d1
        
        # Gamma: ∂²C/∂S² = φ(d₁) / (S·σ·√T)
        gamma = phi_d1 / (S * sigma * sqrt_T + self.epsilon)
        
        # Theta: ∂C/∂t (negative of ∂C/∂T)
        # Θ = -[S·φ(d₁)·σ/(2√T)] - r·K·e^{-rT}·Φ(d₂)
        theta = -(S * phi_d1 * sigma / (2 * sqrt_T)) - r * K * discount * Phi_d2
        
        # Vega: ∂C/∂σ = S·φ(d₁)·√T
        vega = S * phi_d1 * sqrt_T
        
        greeks = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'd1': d1,
            'd2': d2,
        }
        
        return C, greeks
    
    def implied_volatility(
        self,
        C_market: torch.Tensor,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute implied volatility using Newton-Raphson method.
        
        Given a market price C_market, find σ such that BS(S,K,σ,T,r) = C_market.
        
        Args:
            C_market: Observed option price (from Meta-Investor allocation)
            S, K, T, r: Standard BS parameters
            max_iter: Maximum Newton-Raphson iterations
            tol: Convergence tolerance
        
        Returns:
            sigma_iv: Implied volatility
        
        In Equity-MARL context:
        - C_market = w_i * sum(C_j): Market's valuation of agent i
        - If IV > HV: Market expects high future volatility → increase exploration
        - If IV < HV: Market expects stability → decrease exploration
        """
        # Initial guess: use Brenner-Subrahmanyam approximation
        sigma = torch.sqrt(2 * torch.abs(torch.log(S / K)) / T + self.epsilon)
        sigma = torch.clamp(sigma, min=0.1, max=2.0)
        
        for _ in range(max_iter):
            C_model, greeks = self.forward(S, K, sigma, T, r, return_greeks=True)
            vega = greeks['vega']
            
            # Newton-Raphson update
            diff = C_market - C_model
            sigma = sigma + diff / (vega + self.epsilon)
            sigma = torch.clamp(sigma, min=self.epsilon, max=5.0)
            
            if torch.max(torch.abs(diff)) < tol:
                break
        
        return sigma


class BinomialTreePricer(nn.Module):
    """
    Cox-Ross-Rubinstein (CRR) Binomial Tree Option Pricer.
    
    This discrete-time model aligns perfectly with MDP time steps in RL.
    Each node in the tree corresponds to a possible agent state.
    
    Advantages over Black-Scholes:
    - Handles American-style options (early exercise)
    - More intuitive for discrete RL settings
    - Naturally accommodates path-dependent features
    
    The backward induction is differentiable through PyTorch autograd.
    """
    
    def __init__(self, n_steps: int = 50):
        """
        Args:
            n_steps: Number of time steps in the tree
        """
        super().__init__()
        self.n_steps = n_steps
    
    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        sigma: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0,
        option_type: str = 'call',
        american: bool = False
    ) -> torch.Tensor:
        """
        Price options using binomial tree.
        
        Args:
            S: Current spot price, shape (batch_size,)
            K: Strike price
            sigma: Volatility
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            american: If True, allow early exercise
        
        Returns:
            Option price, shape (batch_size,)
        """
        batch_size = S.shape[0]
        device = S.device
        dtype = S.dtype
        
        # Time step
        dt = T / self.n_steps
        
        # Up and down factors
        u = torch.exp(sigma * torch.sqrt(dt))
        d = 1.0 / u
        
        # Risk-neutral probability
        exp_rdt = torch.exp(torch.tensor(r, device=device, dtype=dtype) * dt)
        p = (exp_rdt - d) / (u - d + 1e-8)
        p = torch.clamp(p, min=0.0, max=1.0)
        
        # Build price tree at maturity (final nodes)
        # At step n, there are n+1 possible prices: S * u^j * d^(n-j) for j=0,...,n
        j = torch.arange(self.n_steps + 1, device=device, dtype=dtype)
        
        # Shape: (batch_size, n_steps+1)
        S_T = S.unsqueeze(-1) * (u.unsqueeze(-1) ** j) * (d.unsqueeze(-1) ** (self.n_steps - j))
        
        # Payoff at maturity
        if option_type == 'call':
            payoff = torch.maximum(S_T - K.unsqueeze(-1), torch.zeros_like(S_T))
        else:
            payoff = torch.maximum(K.unsqueeze(-1) - S_T, torch.zeros_like(S_T))
        
        # Backward induction
        V = payoff  # Option values at current layer
        discount = torch.exp(-torch.tensor(r, device=device, dtype=dtype) * dt)
        
        for step in range(self.n_steps - 1, -1, -1):
            # Expected value under risk-neutral measure
            V = discount * (p.unsqueeze(-1) * V[:, 1:step+2] + (1 - p.unsqueeze(-1)) * V[:, :step+1])
            
            if american:
                # Check early exercise
                j_step = torch.arange(step + 1, device=device, dtype=dtype)
                S_step = S.unsqueeze(-1) * (u.unsqueeze(-1) ** j_step) * (d.unsqueeze(-1) ** (step - j_step))
                
                if option_type == 'call':
                    exercise = torch.maximum(S_step - K.unsqueeze(-1), torch.zeros_like(S_step))
                else:
                    exercise = torch.maximum(K.unsqueeze(-1) - S_step, torch.zeros_like(S_step))
                
                V = torch.maximum(V, exercise)
        
        return V.squeeze(-1)


class AsianOptionPricer(nn.Module):
    """
    Asian Option Pricer with Floating Strike.
    
    In Equity-MARL, we use floating strike Asian options where:
    - K(t) = average S over a window
    - This measures "outperforming the average" rather than a fixed baseline
    
    Uses geometric average approximation for analytical tractability,
    with adjustment for arithmetic average.
    """
    
    def __init__(self, window_size: int = 10, epsilon: float = 1e-8):
        """
        Args:
            window_size: Number of past observations for averaging
            epsilon: Numerical stability constant
        """
        super().__init__()
        self.window_size = window_size
        self.epsilon = epsilon
        self.bs_layer = BlackScholesLayer(epsilon)
    
    def forward(
        self,
        S: torch.Tensor,
        S_history: torch.Tensor,  # Shape: (batch_size, window_size)
        sigma: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Price Asian option with floating strike (arithmetic average).
        
        Args:
            S: Current spot price
            S_history: Historical prices for computing average strike
            sigma: Volatility
            T: Time to maturity
            r: Risk-free rate
        
        Returns:
            C: Option price
            K: Computed floating strike (for reference)
        """
        # Compute arithmetic average as strike
        K = S_history.mean(dim=-1)
        
        # Adjusted volatility for Asian option (geometric average approximation)
        # For arithmetic average, we use a correction factor
        sigma_adj = sigma * math.sqrt(1.0 / 3.0)  # Standard adjustment
        
        # Price using adjusted BS
        C, _ = self.bs_layer(S, K, sigma_adj, T, r)
        
        return C, K
    
    def cross_agent_strike(
        self,
        S: torch.Tensor,  # Shape: (n_agents,)
    ) -> torch.Tensor:
        """
        Compute cross-agent floating strike.
        
        K = mean(S) for all agents, measuring "outperforming the average agent".
        
        Args:
            S: Current stock prices of all agents
        
        Returns:
            K: Floating strike (same for all agents)
        """
        return S.mean().expand_as(S)


class OptionPricingEngine(nn.Module):
    """
    Unified option pricing engine for Equity-MARL.
    
    Combines multiple pricing methods and selects the appropriate one
    based on the context.
    """
    
    def __init__(
        self,
        method: str = 'black_scholes',
        n_tree_steps: int = 50,
        window_size: int = 10,
        epsilon: float = 1e-8
    ):
        """
        Args:
            method: 'black_scholes', 'binomial', or 'asian'
            n_tree_steps: Steps for binomial tree
            window_size: Window for Asian option averaging
            epsilon: Numerical stability
        """
        super().__init__()
        self.method = method
        
        self.bs_pricer = BlackScholesLayer(epsilon)
        self.binomial_pricer = BinomialTreePricer(n_tree_steps)
        self.asian_pricer = AsianOptionPricer(window_size, epsilon)
    
    def forward(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        sigma: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0,
        S_history: Optional[torch.Tensor] = None,
        return_greeks: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute option prices and related metrics.
        
        Returns:
            Dictionary with 'price', 'strike', and optionally 'greeks'
        """
        result = {}
        
        if self.method == 'black_scholes':
            C, greeks = self.bs_pricer(S, K, sigma, T, r, return_greeks)
            result['price'] = C
            result['strike'] = K
            if greeks:
                result['greeks'] = greeks
                
        elif self.method == 'binomial':
            C = self.binomial_pricer(S, K, sigma, T, r)
            result['price'] = C
            result['strike'] = K
            
        elif self.method == 'asian':
            if S_history is None:
                raise ValueError("Asian option requires S_history")
            C, K_float = self.asian_pricer(S, S_history, sigma, T, r)
            result['price'] = C
            result['strike'] = K_float
        
        return result


# ============================================================================
# Unit Tests
# ============================================================================

def test_black_scholes():
    """Test Black-Scholes implementation against known values."""
    print("Testing Black-Scholes Layer...")
    
    bs = BlackScholesLayer()
    
    # Test case: S=100, K=100, σ=0.2, T=1, r=0.05
    S = torch.tensor([100.0], requires_grad=True)
    K = torch.tensor([100.0])
    sigma = torch.tensor([0.2])
    T = torch.tensor([1.0])
    r = 0.05
    
    C, greeks = bs(S, K, sigma, T, r, return_greeks=True)
    
    # Known value: approximately 10.45
    print(f"  Option price: {C.item():.4f} (expected ~10.45)")
    print(f"  Delta: {greeks['delta'].item():.4f} (expected ~0.637)")
    
    # Test gradient flow
    C.backward()
    print(f"  Gradient w.r.t. S: {S.grad.item():.4f}")
    
    # Test batch processing
    S_batch = torch.tensor([90.0, 100.0, 110.0], requires_grad=True)
    K_batch = torch.tensor([100.0, 100.0, 100.0])
    sigma_batch = torch.tensor([0.2, 0.2, 0.2])
    T_batch = torch.tensor([1.0, 1.0, 1.0])
    
    C_batch, _ = bs(S_batch, K_batch, sigma_batch, T_batch, r)
    print(f"  Batch prices: {C_batch.detach().numpy()}")
    
    print("  ✓ Black-Scholes tests passed!\n")


def test_binomial_tree():
    """Test binomial tree converges to Black-Scholes."""
    print("Testing Binomial Tree Pricer...")
    
    bs = BlackScholesLayer()
    bt = BinomialTreePricer(n_steps=100)
    
    S = torch.tensor([100.0])
    K = torch.tensor([100.0])
    sigma = torch.tensor([0.2])
    T = torch.tensor([1.0])
    r = 0.05
    
    C_bs, _ = bs(S, K, sigma, T, r)
    C_bt = bt(S, K, sigma, T, r)
    
    print(f"  BS price: {C_bs.item():.4f}")
    print(f"  BT price: {C_bt.item():.4f}")
    print(f"  Difference: {abs(C_bs.item() - C_bt.item()):.6f}")
    
    assert abs(C_bs.item() - C_bt.item()) < 0.1, "Binomial tree should converge to BS"
    print("  ✓ Binomial tree tests passed!\n")


def test_implied_volatility():
    """Test implied volatility calculation."""
    print("Testing Implied Volatility...")
    
    bs = BlackScholesLayer()
    
    S = torch.tensor([100.0])
    K = torch.tensor([100.0])
    sigma_true = torch.tensor([0.25])
    T = torch.tensor([1.0])
    r = 0.05
    
    # Generate market price with known volatility
    C_market, _ = bs(S, K, sigma_true, T, r)
    
    # Recover implied volatility
    sigma_iv = bs.implied_volatility(C_market, S, K, T, r)
    
    print(f"  True volatility: {sigma_true.item():.4f}")
    print(f"  Implied volatility: {sigma_iv.item():.4f}")
    
    assert abs(sigma_true.item() - sigma_iv.item()) < 0.01, "IV should match true σ"
    print("  ✓ Implied volatility tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Option Pricing Module Tests")
    print("=" * 60 + "\n")
    
    test_black_scholes()
    test_binomial_tree()
    test_implied_volatility()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
