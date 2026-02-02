"""
Advanced Option Pricing Models for Non-stationary RL Environments

This module extends the basic Black-Scholes framework to handle:
1. Time-varying volatility σ(t) - "前肥后细" pattern
2. GARCH volatility clustering
3. Jump-diffusion (Merton model) - Poisson jumps
4. Regime-switching models

These models address the core limitations of BS in RL:
- Non-constant volatility
- Non-normal returns (fat tails, jumps)
- Time-dependent parameters
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Callable
from scipy.stats import norm
import math


class TimeVaryingVolatility(nn.Module):
    """
    Time-varying volatility model: σ(t) = σ_∞ + (σ_0 - σ_∞) * exp(-λt)
    
    Models the "前肥后细" (front-heavy, back-thin) pattern in RL training:
    - Early training: High volatility (exploration)
    - Late training: Low volatility (convergence)
    
    This is the simplest extension to handle non-stationarity.
    """
    
    def __init__(
        self,
        sigma_0: float = 0.5,      # Initial volatility (high)
        sigma_inf: float = 0.1,    # Terminal volatility (low)
        decay_rate: float = 0.01,  # Decay speed λ
        total_steps: int = 10000
    ):
        """
        Args:
            sigma_0: Initial volatility (exploration phase)
            sigma_inf: Asymptotic volatility (convergence phase)
            decay_rate: Exponential decay rate
            total_steps: Total training steps
        """
        super().__init__()
        self.sigma_0 = sigma_0
        self.sigma_inf = sigma_inf
        self.decay_rate = decay_rate
        self.total_steps = total_steps
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute volatility at time t.
        
        Args:
            t: Time step(s), can be scalar or tensor
        
        Returns:
            sigma(t): Time-varying volatility
        """
        # Normalize t to [0, 1]
        t_normalized = t / self.total_steps
        
        sigma_t = self.sigma_inf + (self.sigma_0 - self.sigma_inf) * torch.exp(
            -self.decay_rate * t_normalized * self.total_steps
        )
        return sigma_t
    
    def get_integrated_variance(self, t1: float, t2: float) -> float:
        """
        Compute integrated variance ∫_{t1}^{t2} σ²(s) ds
        
        Needed for option pricing with time-varying volatility.
        """
        # Analytical solution for exponential decay
        a = self.sigma_inf
        b = self.sigma_0 - self.sigma_inf
        lam = self.decay_rate
        
        # ∫ (a + b*e^{-λs})² ds = a²s + 2ab/λ(1-e^{-λs}) + b²/2λ(1-e^{-2λs})
        def antiderivative(s):
            return (
                a**2 * s +
                2*a*b/lam * (1 - np.exp(-lam*s)) +
                b**2/(2*lam) * (1 - np.exp(-2*lam*s))
            )
        
        return antiderivative(t2) - antiderivative(t1)


class GARCHVolatility(nn.Module):
    """
    GARCH(1,1) volatility model for capturing volatility clustering.
    
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    
    In RL context:
    - High volatility tends to follow high volatility (exploration bursts)
    - Volatility gradually mean-reverts (learning stabilization)
    
    Properties:
    - Unconditional variance: σ² = ω / (1 - α - β)
    - Persistence: α + β (closer to 1 = more persistent)
    """
    
    def __init__(
        self,
        omega: float = 0.01,    # Base variance
        alpha: float = 0.1,     # ARCH coefficient (shock impact)
        beta: float = 0.8,      # GARCH coefficient (persistence)
        n_agents: int = 1
    ):
        """
        Args:
            omega: Base variance level
            alpha: Weight on past squared return (shock impact)
            beta: Weight on past variance (persistence)
            n_agents: Number of agents to track
        """
        super().__init__()
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.n_agents = n_agents
        
        # Check stationarity condition
        if alpha + beta >= 1:
            print(f"Warning: α + β = {alpha + beta} >= 1, GARCH is not stationary")
        
        # Unconditional variance
        self.long_run_var = omega / (1 - alpha - beta) if alpha + beta < 1 else omega
        
        # State: current variance estimates
        self.register_buffer('variance', torch.ones(n_agents) * self.long_run_var)
        self.register_buffer('prev_return', torch.zeros(n_agents))
    
    def update(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Update GARCH variance given new returns.
        
        Args:
            returns: Log returns, shape (n_agents,)
        
        Returns:
            new_variance: Updated variance estimates
        """
        # GARCH(1,1) update
        self.variance = (
            self.omega +
            self.alpha * self.prev_return ** 2 +
            self.beta * self.variance
        )
        
        self.prev_return = returns.clone()
        
        return self.variance
    
    def get_volatility(self) -> torch.Tensor:
        """Get current volatility (sqrt of variance)."""
        return torch.sqrt(self.variance)
    
    def forecast(self, h: int = 1) -> torch.Tensor:
        """
        Forecast variance h steps ahead.
        
        E[σ²_{t+h} | I_t] = σ² + (α+β)^h * (σ²_t - σ²)
        
        Args:
            h: Forecast horizon
        
        Returns:
            Forecasted variance
        """
        persistence = self.alpha + self.beta
        return self.long_run_var + (persistence ** h) * (self.variance - self.long_run_var)
    
    def reset(self):
        """Reset to unconditional variance."""
        self.variance.fill_(self.long_run_var)
        self.prev_return.zero_()


class MertonJumpDiffusion(nn.Module):
    """
    Merton (1976) Jump-Diffusion Model.
    
    dS/S = (μ - λκ) dt + σ dW + (e^Y - 1) dN(λ)
    
    where:
    - dW: Brownian motion (continuous diffusion)
    - dN(λ): Poisson process with intensity λ (discrete jumps)
    - Y ~ N(μ_J, σ_J²): Jump size distribution
    - κ = E[e^Y - 1] = e^{μ_J + σ_J²/2} - 1
    
    This models the RL scenario where:
    - Most returns are small (diffusion): normal policy updates
    - Occasional large jumps (Poisson): exploration or breakthrough
    
    The "前肥后细" pattern is captured by time-varying λ(t).
    """
    
    def __init__(
        self,
        sigma: float = 0.2,        # Diffusion volatility
        lambda_0: float = 1.0,     # Initial jump intensity (per unit time)
        lambda_inf: float = 0.1,   # Terminal jump intensity
        lambda_decay: float = 0.01,# Jump intensity decay rate
        mu_J: float = 0.0,         # Mean jump size (log scale)
        sigma_J: float = 0.3,      # Jump size volatility
        total_steps: int = 10000
    ):
        """
        Args:
            sigma: Diffusion volatility
            lambda_0: Initial jump intensity (high for exploration)
            lambda_inf: Asymptotic jump intensity (low for convergence)
            lambda_decay: Decay rate for jump intensity
            mu_J: Mean of jump size distribution
            sigma_J: Std of jump size distribution
            total_steps: Total training steps
        """
        super().__init__()
        self.sigma = sigma
        self.lambda_0 = lambda_0
        self.lambda_inf = lambda_inf
        self.lambda_decay = lambda_decay
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.total_steps = total_steps
        
        # Expected relative jump: κ = E[e^Y - 1]
        self.kappa = np.exp(mu_J + 0.5 * sigma_J**2) - 1
    
    def jump_intensity(self, t: torch.Tensor) -> torch.Tensor:
        """
        Time-varying jump intensity: λ(t) = λ_∞ + (λ_0 - λ_∞) * e^{-γt}
        
        Early training: High λ (frequent jumps from exploration)
        Late training: Low λ (rare jumps, stable policy)
        """
        t_normalized = t / self.total_steps
        return self.lambda_inf + (self.lambda_0 - self.lambda_inf) * torch.exp(
            -self.lambda_decay * t_normalized * self.total_steps
        )
    
    def option_price(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0,
        t: float = 0.0,
        n_terms: int = 20
    ) -> torch.Tensor:
        """
        Merton's jump-diffusion option pricing formula.
        
        C = Σ_{n=0}^{∞} [e^{-λ'T} (λ'T)^n / n!] * BS(S, K, r_n, σ_n, T)
        
        where:
        - λ' = λ(1 + κ)
        - r_n = r - λκ + n*ln(1+κ)/T
        - σ_n² = σ² + n*σ_J²/T
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            t: Current time (for time-varying λ)
            n_terms: Number of terms in series (truncation)
        
        Returns:
            Option price
        """
        # Get current jump intensity
        lambda_t = self.jump_intensity(torch.tensor(t)).item()
        
        # Adjusted intensity
        lambda_prime = lambda_t * (1 + self.kappa)
        
        # Series expansion
        price = torch.zeros_like(S)
        
        for n in range(n_terms):
            # Poisson weight: e^{-λ'T} (λ'T)^n / n!
            log_weight = -lambda_prime * T + n * torch.log(lambda_prime * T + 1e-10) - math.lgamma(n + 1)
            weight = torch.exp(log_weight)
            
            # Adjusted parameters for term n
            r_n = r - lambda_t * self.kappa + n * np.log(1 + self.kappa + 1e-10) / (T + 1e-10)
            sigma_n = torch.sqrt(self.sigma**2 + n * self.sigma_J**2 / (T + 1e-10))
            
            # Black-Scholes price with adjusted parameters
            bs_price = self._black_scholes(S, K, sigma_n, T, r_n)
            
            price = price + weight * bs_price
        
        return price
    
    def _black_scholes(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        sigma: torch.Tensor,
        T: torch.Tensor,
        r: float
    ) -> torch.Tensor:
        """Standard Black-Scholes call price."""
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * torch.sqrt(T) + 1e-8)
        d2 = d1 - sigma * torch.sqrt(T)
        
        Phi = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        
        C = S * Phi(d1) - K * torch.exp(-r * T) * Phi(d2)
        return C
    
    def simulate_path(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int = 1000
    ) -> torch.Tensor:
        """
        Simulate price paths using jump-diffusion.
        
        Args:
            S0: Initial price
            T: Total time
            n_steps: Number of time steps
            n_paths: Number of paths to simulate
        
        Returns:
            paths: Simulated paths, shape (n_paths, n_steps + 1)
        """
        dt = T / n_steps
        paths = torch.zeros(n_paths, n_steps + 1)
        paths[:, 0] = S0
        
        for i in range(n_steps):
            t = i * dt
            lambda_t = self.jump_intensity(torch.tensor(t)).item()
            
            # Diffusion component
            dW = torch.randn(n_paths) * np.sqrt(dt)
            diffusion = self.sigma * dW
            
            # Jump component
            n_jumps = torch.poisson(torch.ones(n_paths) * lambda_t * dt)
            jump_sizes = torch.zeros(n_paths)
            for j in range(n_paths):
                if n_jumps[j] > 0:
                    jumps = torch.randn(int(n_jumps[j].item())) * self.sigma_J + self.mu_J
                    jump_sizes[j] = jumps.sum()
            
            # Update path
            drift = (0.0 - lambda_t * self.kappa) * dt
            paths[:, i + 1] = paths[:, i] * torch.exp(drift + diffusion + jump_sizes)
        
        return paths


class TimeVaryingJumpDiffusion(nn.Module):
    """
    Complete time-varying jump-diffusion model for RL.
    
    dS/S = μ(t) dt + σ(t) dW + dJ(t)
    
    All parameters decay from initial to terminal values:
    - σ(t): Volatility (exploration → stability)
    - λ(t): Jump intensity (frequent → rare)
    
    This is the most comprehensive model for RL's "前肥后细" pattern.
    """
    
    def __init__(
        self,
        # Volatility parameters
        sigma_0: float = 0.4,
        sigma_inf: float = 0.1,
        sigma_decay: float = 0.005,
        # Jump intensity parameters
        lambda_0: float = 2.0,
        lambda_inf: float = 0.1,
        lambda_decay: float = 0.01,
        # Jump size parameters
        mu_J: float = 0.0,
        sigma_J: float = 0.3,
        # Time parameters
        total_steps: int = 10000,
        device: str = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.total_steps = total_steps
        
        # Initialize sub-models
        self.volatility_model = TimeVaryingVolatility(
            sigma_0, sigma_inf, sigma_decay, total_steps
        )
        self.jump_model = MertonJumpDiffusion(
            sigma_0, lambda_0, lambda_inf, lambda_decay,
            mu_J, sigma_J, total_steps
        )
    
    def get_parameters(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get all time-varying parameters at time t.
        
        Returns:
            Dict with 'sigma', 'lambda', 'mu_J', 'sigma_J'
        """
        return {
            'sigma': self.volatility_model(t),
            'lambda': self.jump_model.jump_intensity(t),
            'mu_J': self.jump_model.mu_J,
            'sigma_J': self.jump_model.sigma_J,
        }
    
    def option_price(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: float = 0.0,
        current_step: int = 0
    ) -> torch.Tensor:
        """
        Price option under time-varying jump-diffusion.
        
        Uses the jump-diffusion formula with time-varying parameters.
        """
        return self.jump_model.option_price(S, K, T, r, t=current_step)


class RegimeSwitchingVolatility(nn.Module):
    """
    Regime-switching (Hidden Markov) volatility model.
    
    The system switches between discrete regimes:
    - Regime 1 (Exploration): High μ, high σ
    - Regime 2 (Exploitation): Low μ, low σ
    
    Transition probabilities determine switching behavior.
    
    This naturally captures the exploration-exploitation trade-off!
    """
    
    def __init__(
        self,
        n_regimes: int = 2,
        regime_params: Optional[Dict] = None,
        n_agents: int = 1
    ):
        """
        Args:
            n_regimes: Number of regimes
            regime_params: Dict with 'mu', 'sigma' for each regime
            n_agents: Number of agents
        """
        super().__init__()
        self.n_regimes = n_regimes
        self.n_agents = n_agents
        
        # Default: exploration vs exploitation regimes
        if regime_params is None:
            regime_params = {
                'mu': [0.1, 0.02],      # High drift (exploration) vs low (exploitation)
                'sigma': [0.4, 0.1],    # High vol (exploration) vs low (exploitation)
            }
        
        self.mu = torch.tensor(regime_params['mu'])
        self.sigma = torch.tensor(regime_params['sigma'])
        
        # Transition matrix (row = from, col = to)
        # Default: tend to stay in current regime
        self.transition = nn.Parameter(torch.tensor([
            [0.95, 0.05],  # From exploration: 95% stay, 5% switch
            [0.02, 0.98],  # From exploitation: 2% switch, 98% stay
        ]))
        
        # Current regime probabilities for each agent
        self.register_buffer('regime_probs', torch.ones(n_agents, n_regimes) / n_regimes)
    
    def update(self, returns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update regime probabilities given observed returns (Bayes filter).
        
        Args:
            returns: Observed log returns
        
        Returns:
            regime_probs: Updated regime probabilities
            expected_sigma: Expected volatility under regime mixture
        """
        # Likelihood of returns under each regime (Gaussian)
        likelihoods = torch.zeros(self.n_agents, self.n_regimes)
        for k in range(self.n_regimes):
            likelihoods[:, k] = torch.exp(
                -0.5 * ((returns - self.mu[k]) / self.sigma[k])**2
            ) / (self.sigma[k] * np.sqrt(2 * np.pi))
        
        # Prediction step: P(S_t | y_{1:t-1})
        predicted = self.regime_probs @ torch.softmax(self.transition, dim=1)
        
        # Update step: P(S_t | y_{1:t})
        updated = predicted * likelihoods
        self.regime_probs = updated / (updated.sum(dim=1, keepdim=True) + 1e-8)
        
        # Expected volatility
        expected_sigma = (self.regime_probs * self.sigma).sum(dim=1)
        
        return self.regime_probs, expected_sigma
    
    def get_current_regime(self) -> torch.Tensor:
        """Get most likely current regime for each agent."""
        return self.regime_probs.argmax(dim=1)
    
    def get_volatility(self) -> torch.Tensor:
        """Get expected volatility under regime mixture."""
        return (self.regime_probs * self.sigma).sum(dim=1)


# ============================================================================
# Integration with E-MARL
# ============================================================================

class AdvancedValuationEngine(nn.Module):
    """
    Extended valuation engine with advanced volatility models.
    
    Supports:
    - Standard (constant σ)
    - Time-varying σ(t)
    - GARCH
    - Jump-diffusion
    - Regime-switching
    """
    
    def __init__(
        self,
        n_agents: int,
        model_type: str = 'time_varying',  # 'standard', 'time_varying', 'garch', 'jump', 'regime'
        total_steps: int = 10000,
        device: str = 'cpu',
        **kwargs
    ):
        super().__init__()
        self.n_agents = n_agents
        self.model_type = model_type
        self.total_steps = total_steps
        self.device = device
        self.current_step = 0
        
        # Initialize appropriate volatility model
        if model_type == 'time_varying':
            self.vol_model = TimeVaryingVolatility(
                total_steps=total_steps, **kwargs
            )
        elif model_type == 'garch':
            self.vol_model = GARCHVolatility(n_agents=n_agents, **kwargs)
        elif model_type == 'jump':
            self.vol_model = TimeVaryingJumpDiffusion(
                total_steps=total_steps, **kwargs
            )
        elif model_type == 'regime':
            self.vol_model = RegimeSwitchingVolatility(n_agents=n_agents, **kwargs)
        else:
            self.vol_model = None  # Standard constant volatility
    
    def get_volatility(self, returns: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get current volatility estimate.
        
        Args:
            returns: Recent returns (needed for GARCH/regime)
        
        Returns:
            Volatility for each agent
        """
        if self.model_type == 'time_varying':
            t = torch.tensor(self.current_step, dtype=torch.float32)
            sigma = self.vol_model(t)
            return sigma.expand(self.n_agents)
        
        elif self.model_type == 'garch':
            if returns is not None:
                self.vol_model.update(returns)
            return self.vol_model.get_volatility()
        
        elif self.model_type == 'jump':
            t = torch.tensor(self.current_step, dtype=torch.float32)
            params = self.vol_model.get_parameters(t)
            return params['sigma'].expand(self.n_agents)
        
        elif self.model_type == 'regime':
            if returns is not None:
                _, sigma = self.vol_model.update(returns)
                return sigma
            return self.vol_model.get_volatility()
        
        else:
            return torch.ones(self.n_agents) * 0.2
    
    def step(self):
        """Advance time step."""
        self.current_step += 1


# ============================================================================
# Unit Tests
# ============================================================================

def test_time_varying_volatility():
    """Test time-varying volatility model."""
    print("Testing Time-Varying Volatility...")
    
    model = TimeVaryingVolatility(sigma_0=0.5, sigma_inf=0.1, decay_rate=0.001, total_steps=1000)
    
    t_early = torch.tensor(100.0)
    t_late = torch.tensor(900.0)
    
    sigma_early = model(t_early)
    sigma_late = model(t_late)
    
    print(f"  σ(t=100): {sigma_early.item():.4f}")
    print(f"  σ(t=900): {sigma_late.item():.4f}")
    
    assert sigma_early > sigma_late, "Early volatility should be higher"
    print("  ✓ Time-varying volatility tests passed!\n")


def test_garch():
    """Test GARCH model."""
    print("Testing GARCH Volatility...")
    
    model = GARCHVolatility(omega=0.01, alpha=0.1, beta=0.8, n_agents=4)
    
    # Simulate updates
    for _ in range(20):
        returns = torch.randn(4) * 0.1
        model.update(returns)
    
    vol = model.get_volatility()
    print(f"  Current volatility: {vol.numpy()}")
    
    # Forecast
    forecast = model.forecast(h=10)
    print(f"  10-step forecast: {forecast.numpy()}")
    
    print("  ✓ GARCH tests passed!\n")


def test_jump_diffusion():
    """Test Merton jump-diffusion model."""
    print("Testing Jump-Diffusion Model...")
    
    model = MertonJumpDiffusion(
        sigma=0.2, lambda_0=1.0, lambda_inf=0.1,
        mu_J=0.0, sigma_J=0.3, total_steps=1000
    )
    
    # Test jump intensity decay
    lambda_early = model.jump_intensity(torch.tensor(100.0))
    lambda_late = model.jump_intensity(torch.tensor(900.0))
    
    print(f"  λ(t=100): {lambda_early.item():.4f}")
    print(f"  λ(t=900): {lambda_late.item():.4f}")
    
    assert lambda_early > lambda_late, "Early jump intensity should be higher"
    
    # Test option pricing
    S = torch.tensor([100.0])
    K = torch.tensor([100.0])
    T = torch.tensor([1.0])
    
    price = model.option_price(S, K, T, r=0.05, t=500)
    print(f"  Option price: {price.item():.4f}")
    
    print("  ✓ Jump-diffusion tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Advanced Pricing Models Tests")
    print("=" * 60 + "\n")
    
    test_time_varying_volatility()
    test_garch()
    test_jump_diffusion()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
