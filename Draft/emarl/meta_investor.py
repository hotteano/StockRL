"""
Meta-Investor Module for Equity-MARL

The Meta-Investor is a high-level agent that:
1. Allocates "portfolio weights" to each agent based on option valuations
2. Uses Markowitz Mean-Variance optimization considering correlations
3. Dynamically rebalances based on performance and risk

This implements the "ETF manager" concept from the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque


class MarkowitzOptimizer(nn.Module):
    """
    Markowitz Mean-Variance Portfolio Optimization.
    
    Solves the optimization problem:
    max_w  w'μ - (λ/2)·w'Σw
    s.t.   Σᵢ wᵢ = 1,  wᵢ ≥ 0
    
    where:
    - μ: Expected returns (option values C_i in our context)
    - Σ: Covariance matrix of agent performances
    - λ: Risk aversion parameter
    
    Key insight: This naturally diversifies across agents,
    avoiding over-reliance on a single "star agent".
    """
    
    def __init__(
        self,
        n_agents: int,
        risk_aversion: float = 1.0,
        window_size: int = 50,
        min_weight: float = 0.01,
        max_weight: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            risk_aversion: λ, higher = more risk-averse
            window_size: History window for covariance estimation
            min_weight: Minimum allocation per agent
            max_weight: Maximum allocation per agent
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.risk_aversion = risk_aversion
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.device = device
        
        # History for covariance estimation
        self.shapley_history = deque(maxlen=window_size)
        
        # Running covariance estimate
        self.register_buffer('cov_matrix', torch.eye(n_agents, device=device))
        self.register_buffer('mean_shapley', torch.zeros(n_agents, device=device))
    
    def update_statistics(self, shapley_values: torch.Tensor):
        """Update running statistics for covariance estimation."""
        self.shapley_history.append(shapley_values.clone())
        
        if len(self.shapley_history) >= 2:
            history = torch.stack(list(self.shapley_history))
            
            # Sample mean
            self.mean_shapley = history.mean(dim=0)
            
            # Sample covariance (with regularization)
            centered = history - self.mean_shapley
            self.cov_matrix = (centered.T @ centered) / (len(self.shapley_history) - 1)
            
            # Add regularization for numerical stability
            self.cov_matrix = self.cov_matrix + 0.01 * torch.eye(self.n_agents, device=self.device)
    
    def forward(
        self,
        expected_returns: torch.Tensor,  # μ (option values C_i)
        use_constraints: bool = True
    ) -> torch.Tensor:
        """
        Compute optimal portfolio weights.
        
        Args:
            expected_returns: Expected returns vector μ, shape (n_agents,)
            use_constraints: Whether to apply weight constraints
        
        Returns:
            weights: Optimal portfolio weights, shape (n_agents,)
        """
        mu = expected_returns.to(self.device)
        Sigma = self.cov_matrix
        lam = self.risk_aversion
        
        # Analytical solution (unconstrained): w* = (1/λ)·Σ⁻¹·μ
        try:
            Sigma_inv = torch.linalg.inv(Sigma)
            w_unconstrained = (1.0 / lam) * (Sigma_inv @ mu)
        except:
            # Fallback to pseudo-inverse if singular
            Sigma_inv = torch.linalg.pinv(Sigma)
            w_unconstrained = (1.0 / lam) * (Sigma_inv @ mu)
        
        if use_constraints:
            # Project onto constraints: w ≥ 0, Σw = 1
            weights = self._project_to_simplex(w_unconstrained)
            
            # Apply min/max constraints
            weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
            weights = weights / weights.sum()  # Re-normalize
        else:
            # Just normalize to sum to 1
            weights = F.softmax(w_unconstrained, dim=0)
        
        return weights
    
    def _project_to_simplex(self, v: torch.Tensor) -> torch.Tensor:
        """
        Project vector onto probability simplex (non-negative, sum to 1).
        
        Uses the algorithm from "Efficient Projections onto the l1-Ball"
        by Duchi et al.
        """
        n = len(v)
        
        # Sort in descending order
        u, _ = torch.sort(v, descending=True)
        
        # Find the right threshold
        cssv = torch.cumsum(u, dim=0)
        rho = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        cond = u - (cssv - 1) / rho > 0
        
        # Find last index where condition holds
        rho_star = cond.sum()
        theta = (cssv[rho_star - 1] - 1) / rho_star
        
        # Project
        return torch.maximum(v - theta, torch.zeros_like(v))
    
    def get_diversification_ratio(self, weights: torch.Tensor) -> float:
        """
        Compute diversification ratio.
        
        DR = (Σᵢ wᵢσᵢ) / σ_portfolio
        
        Higher DR = better diversification.
        """
        variances = torch.diag(self.cov_matrix)
        stds = torch.sqrt(variances)
        
        # Weighted average volatility
        weighted_vol = (weights * stds).sum()
        
        # Portfolio volatility
        port_var = weights @ self.cov_matrix @ weights
        port_std = torch.sqrt(port_var + 1e-8)
        
        return (weighted_vol / port_std).item()


class MetaInvestor(nn.Module):
    """
    Complete Meta-Investor module for Equity-MARL.
    
    Combines:
    1. Option-based valuation
    2. Markowitz portfolio optimization
    3. Exploration rate adjustment via implied volatility
    4. Dynamic rebalancing with transaction costs
    
    The Meta-Investor learns to allocate "attention" to agents
    based on their financial metrics.
    """
    
    def __init__(
        self,
        n_agents: int,
        risk_aversion: float = 1.0,
        rebalance_frequency: int = 10,
        transaction_cost: float = 0.001,
        iv_sensitivity: float = 1.0,
        base_exploration: float = 0.1,
        window_size: int = 50,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            risk_aversion: λ in Markowitz optimization
            rebalance_frequency: How often to rebalance (in steps)
            transaction_cost: Cost for changing weights (regularization)
            iv_sensitivity: How strongly IV affects exploration
            base_exploration: Base exploration rate ε
            window_size: History window size
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.iv_sensitivity = iv_sensitivity
        self.base_exploration = base_exploration
        self.device = device
        
        # Portfolio optimizer
        self.optimizer = MarkowitzOptimizer(
            n_agents=n_agents,
            risk_aversion=risk_aversion,
            window_size=window_size,
            device=device
        )
        
        # Current weights
        self.register_buffer('weights', torch.ones(n_agents, device=device) / n_agents)
        self.register_buffer('target_weights', torch.ones(n_agents, device=device) / n_agents)
        
        # For implied volatility calculation
        self.register_buffer('historical_vol', torch.ones(n_agents, device=device) * 0.1)
        
        self.step_count = 0
    
    def forward(
        self,
        valuation_result: Dict[str, torch.Tensor],
        shapley_values: Optional[torch.Tensor] = None,
        smooth_transition: bool = True,
        transition_rate: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Compute portfolio allocation and exploration adjustments.
        
        Args:
            valuation_result: Output from ValuationEngine
            shapley_values: Current Shapley values (for statistics update)
            smooth_transition: Whether to smoothly transition weights
            transition_rate: Rate of weight transition (0-1)
        
        Returns:
            Dictionary with:
            - 'weights': Current portfolio weights
            - 'target_weights': Target weights from optimization
            - 'exploration_rates': Adjusted exploration rates per agent
            - 'diversification_ratio': Portfolio diversification metric
        """
        self.step_count += 1
        
        # Update statistics if Shapley values provided
        if shapley_values is not None:
            self.optimizer.update_statistics(shapley_values)
        
        # Get option values as expected returns
        option_values = valuation_result['option_values']
        
        # Rebalance periodically or on first step
        if self.step_count % self.rebalance_frequency == 0 or self.step_count == 1:
            self.target_weights = self.optimizer(option_values)
        
        # Smooth weight transition (avoid abrupt changes)
        if smooth_transition:
            self.weights = (1 - transition_rate) * self.weights + transition_rate * self.target_weights
        else:
            self.weights = self.target_weights.clone()
        
        # Compute exploration rates based on implied volatility
        exploration_rates = self._compute_exploration_rates(valuation_result)
        
        # Diversification ratio
        div_ratio = self.optimizer.get_diversification_ratio(self.weights)
        
        return {
            'weights': self.weights.clone(),
            'target_weights': self.target_weights.clone(),
            'exploration_rates': exploration_rates,
            'diversification_ratio': div_ratio,
        }
    
    def _compute_exploration_rates(
        self,
        valuation_result: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute exploration rates based on implied volatility.
        
        If IV > HV: Market expects high volatility → explore more
        If IV < HV: Market expects stability → exploit more
        """
        # Historical volatility
        hv = valuation_result['volatilities']
        
        # Implied volatility from option values and weights
        # C_market = w * sum(C) → solve for σ_iv
        option_values = valuation_result['option_values']
        total_value = option_values.sum()
        
        # Market price for each agent based on weight allocation
        c_market = self.weights * total_value
        
        # Approximate IV (simplified - full version would use Newton-Raphson)
        # Higher weight relative to option value → market expects higher volatility
        iv_ratio = (self.weights / (option_values / total_value + 1e-8)).clamp(0.1, 10.0)
        implied_vol = hv * iv_ratio
        
        # Exploration rate adjustment
        # ε_i = ε_base × (IV/HV)^α
        vol_ratio = (implied_vol / (hv + 1e-8)).clamp(0.5, 2.0)
        exploration_rates = self.base_exploration * (vol_ratio ** self.iv_sensitivity)
        
        return exploration_rates.clamp(0.01, 0.5)
    
    def get_weighted_rewards(
        self,
        shapley_values: torch.Tensor,
        delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted rewards for each agent.
        
        r_i^w = w_i × Δ_i × φ_i
        
        Args:
            shapley_values: Shapley values for credit assignment
            delta: Optional Delta values for dynamic discounting
        
        Returns:
            weighted_rewards: Weighted rewards for policy update
        """
        if delta is None:
            delta = torch.ones_like(shapley_values)
        
        return self.weights * delta * shapley_values
    
    def compute_transaction_cost(self, new_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute transaction cost for rebalancing.
        
        Cost = c × Σᵢ |w_new - w_old|
        """
        return self.transaction_cost * torch.abs(new_weights - self.weights).sum()
    
    def reset(self):
        """Reset to equal weights."""
        self.weights.fill_(1.0 / self.n_agents)
        self.target_weights.fill_(1.0 / self.n_agents)
        self.step_count = 0


class LearnableMetaInvestor(nn.Module):
    """
    Neural network-based Meta-Investor.
    
    Instead of analytical Markowitz optimization, learns the allocation
    strategy end-to-end via gradient descent.
    
    Architecture: Attention mechanism over agent embeddings.
    """
    
    def __init__(
        self,
        n_agents: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.device = device
        
        # Embed agent metrics (price, vol, option value, shapley)
        self.metric_embed = nn.Linear(4, embed_dim)
        
        # Self-attention over agents
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
        # Output weights
        self.weight_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
    
    def forward(
        self,
        prices: torch.Tensor,
        volatilities: torch.Tensor,
        option_values: torch.Tensor,
        shapley_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute portfolio weights using learned attention.
        
        Args:
            prices: Stock prices, shape (n_agents,)
            volatilities: Volatilities, shape (n_agents,)
            option_values: Option values, shape (n_agents,)
            shapley_values: Shapley values, shape (n_agents,)
        
        Returns:
            weights: Portfolio weights, shape (n_agents,)
        """
        # Stack metrics: (n_agents, 4)
        metrics = torch.stack([
            prices, volatilities, option_values, shapley_values
        ], dim=-1)
        
        # Add batch dimension: (1, n_agents, 4)
        metrics = metrics.unsqueeze(0)
        
        # Embed: (1, n_agents, embed_dim)
        embeddings = self.metric_embed(metrics)
        
        # Self-attention: agents attend to each other
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        
        # Compute weight logits: (1, n_agents, 1)
        weight_logits = self.weight_head(attended)
        
        # Softmax to get weights: (n_agents,)
        weights = F.softmax(weight_logits.squeeze(), dim=-1)
        
        return weights


# ============================================================================
# Unit Tests
# ============================================================================

def test_markowitz_optimizer():
    """Test Markowitz portfolio optimization."""
    print("Testing Markowitz Optimizer...")
    
    n_agents = 4
    optimizer = MarkowitzOptimizer(n_agents=n_agents, risk_aversion=1.0)
    
    # Simulate history to build covariance
    for _ in range(30):
        shapley = torch.randn(n_agents).abs() + 0.5
        optimizer.update_statistics(shapley)
    
    # Compute optimal weights
    expected_returns = torch.tensor([1.0, 0.8, 0.6, 0.4])
    weights = optimizer(expected_returns)
    
    print(f"  Expected returns: {expected_returns.numpy()}")
    print(f"  Optimal weights: {weights.numpy()}")
    print(f"  Weights sum: {weights.sum().item():.4f}")
    print(f"  Diversification ratio: {optimizer.get_diversification_ratio(weights):.4f}")
    
    # Verify constraints
    assert abs(weights.sum().item() - 1.0) < 1e-5, "Weights should sum to 1"
    assert (weights >= 0).all(), "Weights should be non-negative"
    
    # Higher expected return → higher weight (generally)
    assert weights[0] >= weights[3], "Higher return should get higher weight"
    
    print("  ✓ Markowitz optimizer tests passed!\n")


def test_meta_investor():
    """Test complete Meta-Investor."""
    print("Testing Meta-Investor...")
    
    n_agents = 4
    investor = MetaInvestor(
        n_agents=n_agents,
        risk_aversion=1.0,
        rebalance_frequency=5
    )
    
    # Mock valuation result
    valuation_result = {
        'prices': torch.tensor([1.2, 1.0, 0.9, 0.8]),
        'volatilities': torch.tensor([0.15, 0.1, 0.12, 0.2]),
        'option_values': torch.tensor([0.3, 0.2, 0.15, 0.1]),
    }
    
    # Simulate several steps
    for _ in range(20):
        shapley = torch.randn(n_agents).abs() + 0.5
        result = investor(valuation_result, shapley)
    
    print(f"  Portfolio weights: {result['weights'].numpy()}")
    print(f"  Exploration rates: {result['exploration_rates'].numpy()}")
    print(f"  Diversification: {result['diversification_ratio']:.4f}")
    
    # Verify weights
    assert abs(result['weights'].sum().item() - 1.0) < 1e-5
    
    # Test weighted rewards
    shapley = torch.tensor([1.0, 0.8, 0.6, 0.4])
    delta = torch.tensor([0.9, 0.7, 0.5, 0.3])
    weighted_rewards = investor.get_weighted_rewards(shapley, delta)
    print(f"  Weighted rewards: {weighted_rewards.numpy()}")
    
    print("  ✓ Meta-Investor tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Meta-Investor Module Tests")
    print("=" * 60 + "\n")
    
    test_markowitz_optimizer()
    test_meta_investor()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
