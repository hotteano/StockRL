"""
Equity-MARL (E-MARL) Framework

Complete integration of all components:
1. Shapley Value for credit assignment
2. Option pricing for dynamic valuation
3. Meta-Investor for portfolio optimization
4. Bubble detection for self-correction

This is the main entry point for using E-MARL in your MARL experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np

from .shapley import MonteCarloShapley, ShapleyCalculator
from .option_pricing import BlackScholesLayer, OptionPricingEngine
from .valuation import ValuationEngine, StockPriceTracker
from .meta_investor import MetaInvestor, MarkowitzOptimizer
from .bubble_detector import BubbleDetector, ParameterRestructurer


class EquityMARL(nn.Module):
    """
    Equity-MARL: A Financial Dynamics Framework for Multi-Agent Reinforcement Learning
    
    This framework transforms the MARL problem into a "portfolio management" problem:
    - Each agent is treated as a "stock"
    - Shapley values determine "fundamental value"
    - Option pricing determines "market valuation"
    - A Meta-Investor allocates "positions" (attention weights)
    - Bubble detection provides self-correction
    
    Key advantages:
    1. Theoretically grounded credit assignment (Shapley = game-theoretic optimal)
    2. Risk-aware (volatility consideration via option pricing)
    3. Dynamic attention (Markowitz-style portfolio optimization)
    4. Self-correcting (bubble detection prevents overreliance on single agent)
    5. Adaptive exploration (implied volatility adjusts exploration rates)
    
    Usage:
        emarl = EquityMARL(n_agents=4)
        
        # In training loop:
        team_reward = env.step(actions)
        result = emarl.step(team_reward, state, actions)
        
        # Use for policy update:
        weighted_rewards = result['weighted_rewards']
        exploration_rates = result['exploration_rates']
    """
    
    def __init__(
        self,
        n_agents: int,
        # Shapley configuration
        shapley_samples: int = 100,
        # Valuation configuration
        window_size: int = 50,
        decay_rate: float = 0.02,
        base_price: float = 1.0,
        risk_free_rate: float = 0.0,
        strike_mode: str = 'cross',  # 'self' or 'cross'
        option_method: str = 'black_scholes',
        # Meta-Investor configuration
        risk_aversion: float = 1.0,
        rebalance_frequency: int = 10,
        transaction_cost: float = 0.001,
        base_exploration: float = 0.1,
        iv_sensitivity: float = 1.0,
        # Bubble detection configuration
        bubble_threshold: float = 1.5,
        crash_threshold: float = 0.5,
        restructure_strategy: str = 'soft_reset',
        reset_ratio: float = 0.5,
        # General configuration
        total_steps: int = 10000,
        device: str = 'cpu'
    ):
        """
        Initialize the Equity-MARL framework.
        
        Args:
            n_agents: Number of agents in the MARL system
            
            Shapley Config:
            - shapley_samples: Monte Carlo samples for Shapley estimation
            
            Valuation Config:
            - window_size: History window for statistics
            - decay_rate: Stock price decay rate (λ)
            - base_price: Anchor stock price
            - risk_free_rate: r in Black-Scholes
            - strike_mode: 'self' (agent's own average) or 'cross' (all agents average)
            - option_method: 'black_scholes' or 'binomial'
            
            Meta-Investor Config:
            - risk_aversion: λ in Markowitz optimization
            - rebalance_frequency: Steps between rebalancing
            - transaction_cost: Cost for changing weights
            - base_exploration: Base exploration rate ε
            - iv_sensitivity: How IV affects exploration
            
            Bubble Detection Config:
            - bubble_threshold: Threshold for bubble detection
            - crash_threshold: Shapley drop ratio for crash
            - restructure_strategy: 'soft_reset', 'hard_reset', 'clone', 'distill'
            - reset_ratio: Interpolation ratio for soft reset
            
            General Config:
            - total_steps: Total training steps (for time-to-maturity)
            - device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.device = device
        self.total_steps = total_steps
        
        # Initialize components
        self.shapley_calculator = MonteCarloShapley(
            n_agents=n_agents,
            n_samples=shapley_samples,
            device=device
        )
        
        self.valuation_engine = ValuationEngine(
            n_agents=n_agents,
            window_size=window_size,
            decay_rate=decay_rate,
            base_price=base_price,
            risk_free_rate=risk_free_rate,
            strike_mode=strike_mode,
            option_method=option_method,
            device=device
        )
        self.valuation_engine.set_total_steps(total_steps)
        
        self.meta_investor = MetaInvestor(
            n_agents=n_agents,
            risk_aversion=risk_aversion,
            rebalance_frequency=rebalance_frequency,
            transaction_cost=transaction_cost,
            iv_sensitivity=iv_sensitivity,
            base_exploration=base_exploration,
            window_size=window_size,
            device=device
        )
        
        self.bubble_detector = BubbleDetector(
            n_agents=n_agents,
            bubble_threshold=bubble_threshold,
            crash_threshold=crash_threshold,
            device=device
        )
        
        self.restructurer = ParameterRestructurer(
            strategy=restructure_strategy,
            reset_ratio=reset_ratio
        )
        
        # Statistics tracking
        self.step_count = 0
        self.history = {
            'shapley_values': [],
            'option_values': [],
            'weights': [],
            'bubble_coeffs': [],
            'crashes': [],
        }
    
    def compute_shapley_values(
        self,
        team_reward: torch.Tensor,
        value_function: Optional[Callable] = None,
        state: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Shapley values for credit assignment.
        
        Args:
            team_reward: Total team reward
            value_function: Function v(mask) -> coalition value
            state: Current state (optional, for context)
            actions: Agent actions (optional, for context)
        
        Returns:
            shapley_values: Credit assignment for each agent
        """
        if value_function is None:
            # Default: uniform distribution (as baseline)
            value_function = lambda mask: mask.sum() / self.n_agents * team_reward
        
        shapley = self.shapley_calculator(
            value_function=value_function,
            state=state,
            actions=actions,
            team_reward=team_reward
        )
        
        return shapley
    
    def step(
        self,
        team_reward: torch.Tensor,
        value_function: Optional[Callable] = None,
        state: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        agents: Optional[List[nn.Module]] = None
    ) -> Dict[str, Any]:
        """
        Execute one step of the E-MARL framework.
        
        This is the main interface for integrating E-MARL into your training loop.
        
        Args:
            team_reward: Total team reward from environment
            value_function: Function to compute coalition values v(S)
            state: Current state (optional)
            actions: Agent actions (optional)
            agents: List of agent modules (optional, for restructuring)
        
        Returns:
            Dictionary containing:
            - 'shapley_values': Credit assignment
            - 'weighted_rewards': Rewards weighted by portfolio allocation
            - 'weights': Current portfolio weights
            - 'exploration_rates': Per-agent exploration rates
            - 'option_values': Agent valuations
            - 'delta': Delta values for dynamic discounting
            - 'bubble_info': Bubble detection results
            - 'crash_mask': Which agents crashed (if any)
        """
        self.step_count += 1
        
        # Ensure tensors are on correct device
        if not isinstance(team_reward, torch.Tensor):
            team_reward = torch.tensor(team_reward, device=self.device)
        
        # Step 1: Compute Shapley values (credit assignment)
        shapley_values = self.compute_shapley_values(
            team_reward, value_function, state, actions
        )
        
        # Step 2: Update valuations (stock prices, option values)
        valuation_result = self.valuation_engine(shapley_values, return_greeks=True)
        
        # Step 3: Portfolio optimization (Meta-Investor)
        investor_result = self.meta_investor(valuation_result, shapley_values)
        
        # Step 4: Bubble detection
        bubble_info = self.bubble_detector(
            investor_result['weights'],
            valuation_result['option_values'],
            shapley_values
        )
        
        # Step 5: Handle crashes (parameter restructuring)
        if bubble_info['crash_mask'].any() and agents is not None:
            self.restructurer.restructure(
                agents,
                bubble_info['crash_mask'],
                all_agents=agents,
                best_agent_idx=valuation_result['option_values'].argmax().item()
            )
        
        # Compute weighted rewards
        delta = valuation_result['greeks']['delta'] if 'greeks' in valuation_result else None
        weighted_rewards = self.meta_investor.get_weighted_rewards(shapley_values, delta)
        
        # Update history
        self._update_history(shapley_values, valuation_result, investor_result, bubble_info)
        
        return {
            'shapley_values': shapley_values,
            'weighted_rewards': weighted_rewards,
            'weights': investor_result['weights'],
            'exploration_rates': investor_result['exploration_rates'],
            'option_values': valuation_result['option_values'],
            'prices': valuation_result['prices'],
            'volatilities': valuation_result['volatilities'],
            'delta': delta,
            'bubble_info': bubble_info,
            'crash_mask': bubble_info['crash_mask'],
            'diversification_ratio': investor_result['diversification_ratio'],
        }
    
    def _update_history(
        self,
        shapley_values: torch.Tensor,
        valuation_result: Dict,
        investor_result: Dict,
        bubble_info: Dict
    ):
        """Update internal history for analysis."""
        self.history['shapley_values'].append(shapley_values.detach().cpu().numpy())
        self.history['option_values'].append(valuation_result['option_values'].detach().cpu().numpy())
        self.history['weights'].append(investor_result['weights'].detach().cpu().numpy())
        self.history['bubble_coeffs'].append(bubble_info['bubble_coeff'].detach().cpu().numpy())
        self.history['crashes'].append(bubble_info['crash_mask'].cpu().numpy())
    
    def get_effective_discount(self, base_gamma: float = 0.99) -> torch.Tensor:
        """
        Get effective discount factors for each agent.
        
        γ_effective = γ × Δ
        
        High Delta (good performer): full discounting (look long-term)
        Low Delta (poor performer): reduced discounting (focus on immediate)
        
        Args:
            base_gamma: Base discount factor
        
        Returns:
            Effective discount factors per agent
        """
        delta = self.valuation_engine.get_delta_discount()
        return base_gamma * delta
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the E-MARL system.
        
        Returns:
            Dictionary with various statistics
        """
        if len(self.history['shapley_values']) == 0:
            return {}
        
        shapley_arr = np.array(self.history['shapley_values'])
        weights_arr = np.array(self.history['weights'])
        option_arr = np.array(self.history['option_values'])
        crashes_arr = np.array(self.history['crashes'])
        
        return {
            'total_steps': self.step_count,
            'mean_shapley': shapley_arr.mean(axis=0).tolist(),
            'std_shapley': shapley_arr.std(axis=0).tolist(),
            'mean_weights': weights_arr.mean(axis=0).tolist(),
            'final_weights': weights_arr[-1].tolist() if len(weights_arr) > 0 else None,
            'total_crashes': crashes_arr.sum(axis=0).tolist(),
            'mean_option_value': option_arr.mean(axis=0).tolist(),
        }
    
    def reset(self):
        """Reset the E-MARL system."""
        self.step_count = 0
        self.valuation_engine.reset()
        self.meta_investor.reset()
        self.bubble_detector.reset()
        self.history = {
            'shapley_values': [],
            'option_values': [],
            'weights': [],
            'bubble_coeffs': [],
            'crashes': [],
        }


class EasyEquityMARL:
    """
    Simplified interface for Equity-MARL.
    
    Provides a minimal API for quick integration with existing MARL code.
    
    Usage:
        emarl = EasyEquityMARL(n_agents=4)
        
        # In training loop:
        rewards, weights = emarl.process_reward(team_reward)
        
        # Update each agent with weighted reward
        for i, agent in enumerate(agents):
            agent.learn(rewards[i])
    """
    
    def __init__(self, n_agents: int, device: str = 'cpu', **kwargs):
        """
        Args:
            n_agents: Number of agents
            device: Computation device
            **kwargs: Additional arguments passed to EquityMARL
        """
        self.emarl = EquityMARL(n_agents=n_agents, device=device, **kwargs)
        self.n_agents = n_agents
        self.device = device
    
    def process_reward(
        self,
        team_reward: float,
        value_function: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process team reward and return individual weighted rewards.
        
        Args:
            team_reward: Total team reward
            value_function: Optional coalition value function
        
        Returns:
            rewards: Weighted rewards for each agent
            weights: Current portfolio weights
        """
        result = self.emarl.step(
            torch.tensor(team_reward, device=self.device),
            value_function=value_function
        )
        
        rewards = result['weighted_rewards'].detach().cpu().numpy()
        weights = result['weights'].detach().cpu().numpy()
        
        return rewards, weights
    
    def get_exploration_rates(self) -> np.ndarray:
        """Get current exploration rates for each agent."""
        return self.emarl.meta_investor.forward(
            self.emarl.valuation_engine.forward(
                torch.zeros(self.n_agents, device=self.device)
            )
        )['exploration_rates'].detach().cpu().numpy()
    
    def reset(self):
        """Reset the system."""
        self.emarl.reset()


# ============================================================================
# Integration Examples
# ============================================================================

def example_integration():
    """
    Example: Integrating E-MARL with a simple MARL training loop.
    """
    print("Example: E-MARL Integration")
    print("=" * 60)
    
    n_agents = 4
    n_episodes = 100
    steps_per_episode = 50
    
    # Initialize E-MARL
    emarl = EquityMARL(
        n_agents=n_agents,
        total_steps=n_episodes * steps_per_episode,
        device='cpu'
    )
    
    # Simulate training loop
    for episode in range(n_episodes):
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # Simulate team reward (in practice, this comes from environment)
            team_reward = torch.randn(1).abs().item() * 10
            
            # Define value function (in practice, run agents with coalition masks)
            def value_function(mask):
                # Simplified: contribution proportional to mask
                base = torch.tensor([1.0, 0.8, 0.6, 0.4])
                return (mask * base).sum() * team_reward / base.sum()
            
            # E-MARL step
            result = emarl.step(
                team_reward=torch.tensor(team_reward),
                value_function=value_function
            )
            
            episode_reward += team_reward
            
            # In practice: update agents with weighted rewards
            # for i, agent in enumerate(agents):
            #     agent.update(result['weighted_rewards'][i])
        
        if (episode + 1) % 20 == 0:
            stats = emarl.get_summary_statistics()
            print(f"Episode {episode+1}:")
            print(f"  Mean weights: {[f'{w:.3f}' for w in stats['mean_weights']]}")
            print(f"  Total crashes: {stats['total_crashes']}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Final Statistics:")
    stats = emarl.get_summary_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Final weights: {[f'{w:.3f}' for w in stats['final_weights']]}")
    print(f"  Mean Shapley: {[f'{s:.3f}' for s in stats['mean_shapley']]}")


# ============================================================================
# Unit Tests
# ============================================================================

def test_equity_marl():
    """Test complete Equity-MARL framework."""
    print("Testing Equity-MARL Framework...")
    
    n_agents = 4
    emarl = EquityMARL(
        n_agents=n_agents,
        total_steps=1000,
        device='cpu'
    )
    
    # Simulate several steps
    for _ in range(50):
        team_reward = torch.randn(1).abs() * 10
        
        def value_function(mask):
            return mask.sum() / n_agents * team_reward
        
        result = emarl.step(team_reward, value_function)
    
    # Check outputs
    assert 'shapley_values' in result
    assert 'weighted_rewards' in result
    assert 'weights' in result
    assert 'exploration_rates' in result
    
    # Weights should sum to 1
    assert abs(result['weights'].sum().item() - 1.0) < 1e-5
    
    # Get statistics
    stats = emarl.get_summary_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Mean weights: {stats['mean_weights']}")
    
    print("  ✓ Equity-MARL framework tests passed!\n")


def test_easy_interface():
    """Test simplified interface."""
    print("Testing Easy Interface...")
    
    emarl = EasyEquityMARL(n_agents=4)
    
    for _ in range(20):
        rewards, weights = emarl.process_reward(10.0)
    
    assert len(rewards) == 4
    assert len(weights) == 4
    assert abs(sum(weights) - 1.0) < 1e-5
    
    print(f"  Rewards: {rewards}")
    print(f"  Weights: {weights}")
    print("  ✓ Easy interface tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Equity-MARL Framework Tests")
    print("=" * 60 + "\n")
    
    test_equity_marl()
    test_easy_interface()
    
    print("=" * 60)
    print("Running Integration Example...")
    print("=" * 60 + "\n")
    
    example_integration()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
