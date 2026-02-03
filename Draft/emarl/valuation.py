"""
Valuation Engine for Equity-MARL

Manages agent "stock prices" based on Shapley values:
1. Maps Shapley values to stock price dynamics
2. Tracks historical prices for volatility estimation
3. Implements decay mechanism to prevent gradient explosion

Key improvement (v2.0):
- Depreciation factor λ to bound stock prices
- Floating strike (Asian option style)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List, Dict
from collections import deque


class StockPriceTracker(nn.Module):
    """
    Tracks and updates agent "stock prices" based on Shapley values.
    
    Stock price evolution:
    S_i(t) = (1-λ)·S_i(t-1)·exp(r_i(t)) + λ·S_base
    
    where:
    - r_i(t) = log(1 + (φ_i - φ̄) / (φ̄ + ε))
    - λ: Depreciation/decay rate
    - S_base: Base stock price (anchor)
    
    This prevents unbounded growth while preserving relative rankings.
    """
    
    def __init__(
        self,
        n_agents: int,
        window_size: int = 50,
        decay_rate: float = 0.02,
        base_price: float = 1.0,
        sensitivity: float = 1.0,
        epsilon: float = 1e-8,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            window_size: History window for volatility estimation
            decay_rate: λ, depreciation rate (0.01-0.1 recommended)
            base_price: S_base, anchor price
            sensitivity: α, controls how strongly Shapley affects price
            epsilon: Numerical stability constant
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.base_price = base_price
        self.sensitivity = sensitivity
        self.epsilon = epsilon
        self.device = device
        
        # Current stock prices
        self.register_buffer('prices', torch.ones(n_agents, device=device) * base_price)
        
        # History buffers
        self.price_history: deque = deque(maxlen=window_size)
        self.shapley_history: deque = deque(maxlen=window_size)
        self.return_history: deque = deque(maxlen=window_size)
        
        # Statistics
        self.register_buffer('volatility', torch.ones(n_agents, device=device) * 0.1)
        self.register_buffer('drift', torch.zeros(n_agents, device=device))
    
    def update(self, shapley_values: torch.Tensor) -> torch.Tensor:
        """
        Update stock prices based on new Shapley values.
        
        Args:
            shapley_values: Tensor of shape (n_agents,)
        
        Returns:
            new_prices: Updated stock prices
        """
        shapley_values = shapley_values.to(self.device)
        
        # Compute relative performance
        mean_shapley = shapley_values.mean()
        std_shapley = shapley_values.std() + self.epsilon
        
        # Log returns (relative to mean)
        relative_shapley = (shapley_values - mean_shapley) / (mean_shapley.abs() + self.epsilon)
        log_returns = torch.log(1 + self.sensitivity * torch.clamp(relative_shapley, min=-0.9, max=10.0))
        
        # Update prices with decay
        # S_new = (1-λ)·S_old·exp(r) + λ·S_base
        old_prices = self.prices.clone()
        self.prices = (
            (1 - self.decay_rate) * self.prices * torch.exp(log_returns) +
            self.decay_rate * self.base_price
        )
        
        # Clamp for numerical stability
        self.prices = torch.clamp(self.prices, min=self.epsilon, max=1000.0)
        
        # Update history
        self.price_history.append(self.prices.clone())
        self.shapley_history.append(shapley_values.clone())
        self.return_history.append(log_returns.clone())
        
        # Update volatility estimate
        self._update_volatility()
        
        return self.prices.clone()
    
    def _update_volatility(self):
        """Update volatility estimate using historical returns."""
        if len(self.return_history) < 2:
            return
        
        returns_tensor = torch.stack(list(self.return_history))
        
        # Sample standard deviation
        self.volatility = returns_tensor.std(dim=0)
        self.volatility = torch.clamp(self.volatility, min=0.01, max=2.0)
        
        # Also compute drift (average return)
        self.drift = returns_tensor.mean(dim=0)
    
    def get_price_history(self) -> torch.Tensor:
        """
        Get price history as tensor.
        
        Returns:
            History tensor of shape (history_len, n_agents)
        """
        if not self.price_history:
            return self.prices.unsqueeze(0)
        return torch.stack(list(self.price_history))
    
    def get_floating_strike(self, mode: str = 'self') -> torch.Tensor:
        """
        Compute floating strike price (Asian option style).
        
        Args:
            mode: 'self' (each agent's own average) or 'cross' (cross-agent average)
        
        Returns:
            strike: Floating strike prices
        """
        history = self.get_price_history()
        
        if mode == 'self':
            # Each agent's average price over window
            return history.mean(dim=0)
        else:  # 'cross'
            # Average across all agents (measures "outperforming the average")
            return history.mean().expand(self.n_agents)
    
    def reset(self):
        """Reset all prices and history."""
        self.prices.fill_(self.base_price)
        self.volatility.fill_(0.1)
        self.drift.zero_()
        self.price_history.clear()
        self.shapley_history.clear()
        self.return_history.clear()


class ValuationEngine(nn.Module):
    """
    Complete valuation engine combining stock price tracking and option pricing.
    
    Computes for each agent:
    1. Current stock price S_i
    2. Historical volatility σ_i
    3. Option value C_i (potential to outperform)
    4. Greeks (Delta, Gamma, etc.) for dynamic adjustments
    
    This is the core component that bridges Shapley values and portfolio allocation.
    """
    
    def __init__(
        self,
        n_agents: int,
        window_size: int = 50,
        decay_rate: float = 0.02,
        base_price: float = 1.0,
        risk_free_rate: float = 0.0,
        strike_mode: str = 'cross',  # 'self' or 'cross'
        option_method: str = 'black_scholes',
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            window_size: History window size
            decay_rate: Price decay rate
            base_price: Base stock price
            risk_free_rate: r in Black-Scholes
            strike_mode: How to compute floating strike
            option_method: 'black_scholes' or 'binomial'
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.risk_free_rate = risk_free_rate
        self.strike_mode = strike_mode
        self.device = device
        
        # Stock price tracker
        self.price_tracker = StockPriceTracker(
            n_agents=n_agents,
            window_size=window_size,
            decay_rate=decay_rate,
            base_price=base_price,
            device=device
        )
        
        # Option pricer (import here to avoid circular dependency)
        from .option_pricing import BlackScholesLayer, BinomialTreePricer, PoissonJumpPricer
        
        if option_method == 'black_scholes':
            self.option_pricer = BlackScholesLayer()
        elif option_method == 'binomial':
            self.option_pricer = BinomialTreePricer()
        elif option_method == 'poisson':
            # Use a default lambda that fits the RL episode scale (e.g., 5-10 breakthroughs expected)
            self.option_pricer = PoissonJumpPricer(lambda_val=5.0)
        else:
            self.option_pricer = BlackScholesLayer()
        
        # Total training steps for time-to-maturity calculation
        self.total_steps = 1000  # Will be updated
        self.current_step = 0
    
    def set_total_steps(self, total_steps: int):
        """Set total training steps for T calculation."""
        self.total_steps = total_steps
    
    def forward(
        self,
        shapley_values: torch.Tensor,
        return_greeks: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute valuations for all agents.
        
        Args:
            shapley_values: Current Shapley values, shape (n_agents,)
            return_greeks: Whether to compute and return Greeks
        
        Returns:
            Dictionary with:
            - 'prices': Stock prices S_i
            - 'strikes': Floating strikes K_i
            - 'volatilities': Historical volatilities σ_i
            - 'option_values': Call option values C_i
            - 'greeks': Dict of Greeks (if requested)
            - 'time_to_maturity': T
        """
        self.current_step += 1
        
        # Update stock prices
        prices = self.price_tracker.update(shapley_values)
        
        # Get volatility
        volatility = self.price_tracker.volatility
        
        # Compute floating strike
        strikes = self.price_tracker.get_floating_strike(mode=self.strike_mode)
        
        # Time to maturity (remaining fraction of training)
        T = max(0.01, (self.total_steps - self.current_step) / self.total_steps)
        T_tensor = torch.tensor(T, device=self.device).expand(self.n_agents)
        
        # Compute option values
        if hasattr(self.option_pricer, 'forward'):
            option_values, greeks = self.option_pricer(
                prices, strikes, volatility, T_tensor,
                r=self.risk_free_rate,
                return_greeks=return_greeks
            )
        else:
            option_values = self.option_pricer(
                prices, strikes, volatility, T_tensor,
                r=self.risk_free_rate
            )
            greeks = None
        
        result = {
            'prices': prices,
            'strikes': strikes,
            'volatilities': volatility,
            'option_values': option_values,
            'time_to_maturity': T_tensor,
        }
        
        if greeks:
            result['greeks'] = greeks
        
        return result
    
    def get_delta_discount(self) -> torch.Tensor:
        """
        Get Delta values to use as dynamic discount factors.
        
        In Equity-MARL, we use Delta for "adaptive discounting":
        - High Delta (good performer): γ_effective ≈ γ (look long-term)
        - Low Delta (poor performer): γ_effective ≈ 0 (focus on immediate)
        
        Returns:
            delta: Delta values for each agent, shape (n_agents,)
        """
        # Get current prices and strikes
        prices = self.price_tracker.prices
        strikes = self.price_tracker.get_floating_strike(self.strike_mode)
        volatility = self.price_tracker.volatility
        T = max(0.01, (self.total_steps - self.current_step) / self.total_steps)
        T_tensor = torch.tensor(T, device=self.device).expand(self.n_agents)
        
        # Compute Greeks
        _, greeks = self.option_pricer(
            prices, strikes, volatility, T_tensor,
            r=self.risk_free_rate, return_greeks=True
        )
        
        if greeks:
            return greeks['delta']
        else:
            # Fallback: simple moneyness-based approximation
            moneyness = prices / (strikes + 1e-8)
            return torch.sigmoid(2 * (moneyness - 1))
    
    def reset(self):
        """Reset the valuation engine."""
        self.price_tracker.reset()
        self.current_step = 0


class RelativeValuation(nn.Module):
    """
    Alternative valuation using relative metrics.
    
    Instead of absolute stock prices, tracks relative performance:
    - Rank-based valuation
    - Z-score valuation
    - Percentile valuation
    
    More robust to scale changes in Shapley values.
    """
    
    def __init__(self, n_agents: int, window_size: int = 50):
        super().__init__()
        self.n_agents = n_agents
        self.shapley_history = deque(maxlen=window_size)
    
    def forward(self, shapley_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute relative valuations.
        
        Returns:
            Dictionary with various relative metrics
        """
        self.shapley_history.append(shapley_values.clone())
        
        # Current relative position
        ranks = self._compute_ranks(shapley_values)
        z_scores = self._compute_z_scores(shapley_values)
        
        # Historical consistency
        if len(self.shapley_history) > 1:
            consistency = self._compute_consistency()
        else:
            consistency = torch.ones_like(shapley_values)
        
        return {
            'ranks': ranks,
            'z_scores': z_scores,
            'consistency': consistency,
            'relative_value': z_scores * consistency  # Combined metric
        }
    
    def _compute_ranks(self, values: torch.Tensor) -> torch.Tensor:
        """Compute normalized ranks (0 to 1)."""
        sorted_indices = torch.argsort(values)
        ranks = torch.zeros_like(values)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        return ranks / (self.n_agents - 1 + 1e-8)
    
    def _compute_z_scores(self, values: torch.Tensor) -> torch.Tensor:
        """Compute z-scores (standardized values)."""
        mean = values.mean()
        std = values.std() + 1e-8
        return (values - mean) / std
    
    def _compute_consistency(self) -> torch.Tensor:
        """
        Compute consistency score (inverse of rank volatility).
        
        Agents with stable rankings get higher scores.
        """
        history = torch.stack(list(self.shapley_history))
        
        # Compute rank at each time step
        ranks = []
        for t in range(len(self.shapley_history)):
            ranks.append(self._compute_ranks(history[t]))
        
        rank_tensor = torch.stack(ranks)
        rank_std = rank_tensor.std(dim=0)
        
        # Inverse of volatility = consistency
        return 1.0 / (rank_std + 0.1)


# ============================================================================
# Unit Tests
# ============================================================================

def test_stock_price_tracker():
    """Test stock price tracking with decay."""
    print("Testing Stock Price Tracker...")
    
    n_agents = 4
    # Use higher decay rate and lower sensitivity to prevent explosion
    tracker = StockPriceTracker(
        n_agents=n_agents, 
        decay_rate=0.1,  # Higher decay
        sensitivity=0.5   # Lower sensitivity
    )
    
    # Simulate several updates
    for t in range(20):
        # Agent 0 consistently outperforms, Agent 3 underperforms
        shapley = torch.tensor([1.5, 1.0, 1.0, 0.5])
        prices = tracker.update(shapley)
    
    print(f"  Final prices: {prices.numpy()}")
    print(f"  Volatilities: {tracker.volatility.numpy()}")
    
    # Agent 0 should have highest price, Agent 3 lowest
    assert prices[0] > prices[3], "Outperformer should have higher price"
    
    # Prices should be bounded (not exploding)
    assert prices.max() < 100, f"Prices should be bounded, got {prices.max()}"
    
    print("  ✓ Stock price tracker tests passed!\n")


def test_valuation_engine():
    """Test complete valuation engine."""
    print("Testing Valuation Engine...")
    
    n_agents = 4
    engine = ValuationEngine(n_agents=n_agents, strike_mode='cross')
    engine.set_total_steps(100)
    
    # Simulate
    for t in range(50):
        shapley = torch.randn(n_agents).abs() + 0.5
        result = engine(shapley)
    
    print(f"  Prices: {result['prices'].numpy()}")
    print(f"  Option values: {result['option_values'].detach().numpy()}")
    print(f"  Delta (discount): {result['greeks']['delta'].numpy()}")
    
    # Option values should be non-negative
    assert (result['option_values'] >= 0).all(), "Option values must be non-negative"
    
    # Delta should be in [0, 1]
    delta = result['greeks']['delta']
    assert (delta >= 0).all() and (delta <= 1).all(), "Delta must be in [0,1]"
    
    print("  ✓ Valuation engine tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Valuation Module Tests")
    print("=" * 60 + "\n")
    
    test_stock_price_tracker()
    test_valuation_engine()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
