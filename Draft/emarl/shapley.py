"""
Shapley Value Computation Module for Equity-MARL

Implements various methods for computing Shapley values in MARL:
1. Exact computation (for small n)
2. Monte Carlo approximation (scalable)
3. Permutation sampling (efficient)

The Shapley value provides a game-theoretic optimal solution for credit assignment.

Key property (Efficiency): Σᵢ φᵢ = v(N) = Total team reward
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from itertools import permutations, combinations
import math


class ShapleyCalculator:
    """
    Exact Shapley Value Calculator.
    
    Computes the exact Shapley value using the definition:
    φᵢ(v) = Σₛ⊆ₙ₋ᵢ [|S|!(n-|S|-1)!/n!] × [v(S∪{i}) - v(S)]
    
    Complexity: O(2ⁿ) - only feasible for n ≤ 12 agents.
    
    In Equity-MARL:
    - v(S): Expected team reward when only coalition S participates
    - φᵢ: Agent i's marginal contribution (fair share of team reward)
    """
    
    def __init__(self, n_agents: int, value_function: Optional[Callable] = None):
        """
        Args:
            n_agents: Number of agents
            value_function: Function v(S) -> float, where S is a set of agent indices
        """
        self.n_agents = n_agents
        self.value_function = value_function
        
        # Precompute factorial weights
        self._precompute_weights()
    
    def _precompute_weights(self):
        """Precompute Shapley weights for all coalition sizes."""
        n = self.n_agents
        self.weights = {}
        
        for s in range(n):
            # Weight for coalition of size s when computing marginal contribution of one agent
            # w(s) = s! × (n-s-1)! / n!
            self.weights[s] = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
    
    def compute(
        self,
        value_function: Optional[Callable] = None,
        values_cache: Optional[dict] = None
    ) -> np.ndarray:
        """
        Compute exact Shapley values for all agents.
        
        Args:
            value_function: v(S) -> float (overrides init value_function)
            values_cache: Optional precomputed {frozenset: value} dictionary
        
        Returns:
            shapley_values: Array of shape (n_agents,)
        """
        v = value_function or self.value_function
        if v is None and values_cache is None:
            raise ValueError("Must provide either value_function or values_cache")
        
        n = self.n_agents
        shapley_values = np.zeros(n)
        all_agents = set(range(n))
        
        for i in range(n):
            phi_i = 0.0
            others = all_agents - {i}
            
            # Iterate over all subsets S of N \ {i}
            for s in range(n):  # |S| from 0 to n-1
                for S in combinations(others, s):
                    S_set = frozenset(S)
                    S_with_i = S_set | {i}
                    
                    # Get coalition values
                    if values_cache:
                        v_S = values_cache.get(S_set, 0.0)
                        v_S_i = values_cache.get(S_with_i, 0.0)
                    else:
                        v_S = v(S_set)
                        v_S_i = v(S_with_i)
                    
                    # Marginal contribution
                    marginal = v_S_i - v_S
                    
                    # Add weighted contribution
                    phi_i += self.weights[s] * marginal
            
            shapley_values[i] = phi_i
        
        return shapley_values
    
    def verify_efficiency(self, shapley_values: np.ndarray, total_value: float, tol: float = 1e-6) -> bool:
        """
        Verify the efficiency axiom: Σᵢ φᵢ = v(N)
        
        Args:
            shapley_values: Computed Shapley values
            total_value: v(N), the grand coalition value
            tol: Tolerance for numerical comparison
        
        Returns:
            True if efficiency holds
        """
        return abs(np.sum(shapley_values) - total_value) < tol


class MonteCarloShapley(nn.Module):
    """
    Monte Carlo Shapley Value Approximation.
    
    Uses permutation sampling to estimate Shapley values in O(M × n) time,
    where M is the number of samples.
    
    Algorithm:
    1. Sample random permutations of agents
    2. For each permutation, compute marginal contribution of each agent
    3. Average over all samples
    
    Theorem: This is an unbiased estimator of Shapley values.
    With M = O(n log n) samples, achieves high accuracy.
    """
    
    def __init__(
        self,
        n_agents: int,
        n_samples: int = 100,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            n_samples: Number of permutation samples (higher = more accurate)
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_samples = n_samples
        self.device = device
        
        # For tracking running statistics
        self.register_buffer('running_mean', torch.zeros(n_agents))
        self.register_buffer('running_var', torch.ones(n_agents))
        self.register_buffer('n_updates', torch.tensor(0))
    
    def forward(
        self,
        value_function: Callable[[torch.Tensor], torch.Tensor],
        state: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        team_reward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Shapley values using Monte Carlo sampling.
        
        Args:
            value_function: Function that takes a binary mask (n_agents,) and returns
                           the coalition value. mask[i]=1 means agent i participates.
            state: Optional state tensor for context
            actions: Optional actions tensor (n_agents, action_dim)
            team_reward: Optional team reward (used for rescaling)
        
        Returns:
            shapley_values: Tensor of shape (n_agents,)
        """
        n = self.n_agents
        device = self.device
        
        # Accumulator for Shapley values
        shapley_sum = torch.zeros(n, device=device)
        
        for _ in range(self.n_samples):
            # Sample a random permutation
            perm = torch.randperm(n, device=device)
            
            # Track coalition as we add agents
            mask = torch.zeros(n, device=device)
            prev_value = value_function(mask)  # v(∅)
            
            for i in range(n):
                agent_idx = perm[i].item()
                
                # Add agent to coalition
                mask[agent_idx] = 1.0
                curr_value = value_function(mask)
                
                # Marginal contribution - ensure scalar
                marginal = curr_value - prev_value
                if isinstance(marginal, torch.Tensor):
                    marginal = marginal.squeeze().item() if marginal.numel() == 1 else marginal.squeeze()
                shapley_sum[agent_idx] += marginal
                
                prev_value = curr_value
        
        # Average over samples
        shapley_values = shapley_sum / self.n_samples
        
        # Optionally rescale to match team reward exactly
        if team_reward is not None:
            current_sum = shapley_values.sum()
            if current_sum.abs() > 1e-8:
                shapley_values = shapley_values * (team_reward / current_sum)
        
        # Update running statistics
        self._update_running_stats(shapley_values)
        
        return shapley_values
    
    def _update_running_stats(self, shapley_values: torch.Tensor, momentum: float = 0.1):
        """Update running mean and variance for normalization."""
        with torch.no_grad():
            self.running_mean = (1 - momentum) * self.running_mean + momentum * shapley_values
            self.running_var = (1 - momentum) * self.running_var + momentum * (shapley_values - self.running_mean) ** 2
            self.n_updates += 1
    
    def normalize(self, shapley_values: torch.Tensor) -> torch.Tensor:
        """Normalize Shapley values using running statistics."""
        return (shapley_values - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)


class DifferentiableShapley(nn.Module):
    """
    Differentiable Shapley Value Approximation using Neural Networks.
    
    Instead of computing Shapley values exactly, we train a network to predict them.
    This allows end-to-end gradient flow through the credit assignment process.
    
    Architecture: MLP that takes (state, actions, reward) and outputs Shapley values.
    
    Loss function:
    1. Efficiency constraint: Σᵢ φᵢ = R (sum equals team reward)
    2. Regression to MC estimates: ||φ_pred - φ_MC||²
    """
    
    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128
    ):
        """
        Args:
            n_agents: Number of agents
            state_dim: Dimension of state
            action_dim: Dimension of each agent's action
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.n_agents = n_agents
        
        input_dim = state_dim + n_agents * action_dim + 1  # +1 for team reward
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents),
        )
        
        # Output normalization to satisfy efficiency constraint
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        team_reward: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict Shapley values.
        
        Args:
            state: State tensor, shape (batch_size, state_dim)
            actions: Actions tensor, shape (batch_size, n_agents, action_dim)
            team_reward: Team reward, shape (batch_size,)
        
        Returns:
            shapley_values: Shape (batch_size, n_agents)
            Guaranteed to sum to team_reward (efficiency constraint)
        """
        batch_size = state.shape[0]
        
        # Flatten inputs
        actions_flat = actions.view(batch_size, -1)
        reward_expanded = team_reward.unsqueeze(-1)
        
        x = torch.cat([state, actions_flat, reward_expanded], dim=-1)
        
        # Predict proportions
        raw_output = self.network(x)
        proportions = self.softmax(raw_output)  # Sum to 1
        
        # Scale to satisfy efficiency: Σᵢ φᵢ = R
        shapley_values = proportions * team_reward.unsqueeze(-1)
        
        return shapley_values
    
    def compute_loss(
        self,
        pred_shapley: torch.Tensor,
        target_shapley: torch.Tensor,
        team_reward: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.
        
        Args:
            pred_shapley: Predicted values
            target_shapley: MC-estimated values
            team_reward: Team reward
        
        Returns:
            loss: Total loss
            metrics: Dictionary of individual loss components
        """
        # Regression loss
        regression_loss = F.mse_loss(pred_shapley, target_shapley)
        
        # Efficiency constraint (should already be satisfied by softmax, but add as regularizer)
        efficiency_loss = (pred_shapley.sum(dim=-1) - team_reward).pow(2).mean()
        
        # Optional: symmetry regularization (similar agents → similar values)
        
        loss = regression_loss + 0.01 * efficiency_loss
        
        metrics = {
            'regression_loss': regression_loss.item(),
            'efficiency_loss': efficiency_loss.item(),
        }
        
        return loss, metrics


class MARLValueFunction:
    """
    Helper class to create value functions for MARL environments.
    
    Provides different strategies for computing coalition values v(S):
    1. Mask-based: Set non-participating agents to default action
    2. Counterfactual: Remove non-participating agents from the environment
    3. Baseline subtraction: v(S) = R(S) - baseline
    """
    
    def __init__(
        self,
        env,
        agents: List,
        default_action: Union[int, torch.Tensor] = 0,
        baseline: str = 'zero'  # 'zero', 'mean', 'random'
    ):
        """
        Args:
            env: MARL environment
            agents: List of agent policies
            default_action: Action for non-participating agents
            baseline: Baseline strategy for non-participants
        """
        self.env = env
        self.agents = agents
        self.n_agents = len(agents)
        self.default_action = default_action
        self.baseline = baseline
    
    def __call__(self, coalition: Union[set, frozenset, torch.Tensor]) -> float:
        """
        Compute value of coalition S.
        
        Args:
            coalition: Set of participating agent indices, or binary mask
        
        Returns:
            Coalition value (expected reward)
        """
        if isinstance(coalition, torch.Tensor):
            mask = coalition
        else:
            mask = torch.zeros(self.n_agents)
            for i in coalition:
                mask[i] = 1.0
        
        # Get current state
        state = self.env.get_state()
        
        # Compute actions
        actions = []
        for i, agent in enumerate(self.agents):
            if mask[i] > 0.5:
                # Participating: use learned policy
                action = agent.act(state)
            else:
                # Non-participating: use default/baseline
                if self.baseline == 'zero':
                    action = self.default_action
                elif self.baseline == 'random':
                    action = self.env.action_space[i].sample()
                else:
                    action = self.default_action
            actions.append(action)
        
        # Execute and get reward
        _, rewards, _, _, _ = self.env.step(actions)
        
        # Team reward
        if isinstance(rewards, (list, tuple)):
            return sum(rewards)
        return rewards


# ============================================================================
# Unit Tests
# ============================================================================

def test_exact_shapley():
    """Test exact Shapley value computation with a known game."""
    print("Testing Exact Shapley Calculator...")
    
    # Simple 3-player game where v(S) = |S|² (superadditive)
    def value_func(S):
        return len(S) ** 2
    
    calc = ShapleyCalculator(n_agents=3, value_function=value_func)
    shapley = calc.compute()
    
    # For v(S) = |S|², Shapley values should be equal due to symmetry
    print(f"  Shapley values: {shapley}")
    print(f"  Sum: {shapley.sum():.4f} (should be {value_func({0,1,2})})")
    
    # Verify efficiency
    assert calc.verify_efficiency(shapley, value_func({0, 1, 2}))
    
    # Verify symmetry (all agents have equal Shapley values)
    assert np.allclose(shapley, shapley[0])
    
    print("  ✓ Exact Shapley tests passed!\n")


def test_monte_carlo_shapley():
    """Test Monte Carlo Shapley approximation."""
    print("Testing Monte Carlo Shapley...")
    
    n_agents = 5
    
    # Simple value function: sum of participating agents' indices
    def value_func(mask: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(len(mask), dtype=torch.float32)
        return (mask * indices).sum()
    
    mc_shapley = MonteCarloShapley(n_agents=n_agents, n_samples=1000)
    shapley = mc_shapley(value_func)
    
    print(f"  Shapley values: {shapley.numpy()}")
    print(f"  Sum: {shapley.sum().item():.4f}")
    
    # For v(S) = Σᵢ∈S i, Shapley value of agent i should be exactly i
    expected = torch.arange(n_agents, dtype=torch.float32)
    print(f"  Expected: {expected.numpy()}")
    
    assert torch.allclose(shapley, expected, atol=0.5), "MC estimate should be close to exact"
    print("  ✓ Monte Carlo Shapley tests passed!\n")


def test_efficiency_constraint():
    """Test that Shapley values sum to total reward."""
    print("Testing Efficiency Constraint...")
    
    n_agents = 4
    team_reward = 10.0
    
    def value_func(mask: torch.Tensor) -> torch.Tensor:
        # Normalize to match team reward when all participate
        return mask.sum() / n_agents * team_reward
    
    mc_shapley = MonteCarloShapley(n_agents=n_agents, n_samples=500)
    shapley = mc_shapley(value_func, team_reward=torch.tensor(team_reward))
    
    print(f"  Shapley values: {shapley.numpy()}")
    print(f"  Sum: {shapley.sum().item():.4f} (should be {team_reward})")
    
    assert abs(shapley.sum().item() - team_reward) < 0.1
    print("  ✓ Efficiency constraint tests passed!\n")


if __name__ == "__main__":
    import torch.nn.functional as F
    
    print("=" * 60)
    print("Shapley Value Module Tests")
    print("=" * 60 + "\n")
    
    test_exact_shapley()
    test_monte_carlo_shapley()
    test_efficiency_constraint()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
