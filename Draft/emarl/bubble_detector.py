"""
Bubble Detection and Parameter Restructuring for Equity-MARL

Implements market-inspired mechanisms for:
1. Detecting overvalued agents (bubbles)
2. Triggering "bankruptcy restructuring" when bubbles burst
3. Parameter reset strategies to escape local optima

This provides a self-correcting mechanism for the learning process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import numpy as np


class BubbleDetector(nn.Module):
    """
    Detects valuation bubbles in agent performance.
    
    A bubble is defined as:
    Bubble_i(t) = w_i(t) / relative_C_i(t) > threshold
    
    meaning the actual allocation exceeds what the option valuation suggests.
    
    A bubble "bursts" when:
    1. Bubble coefficient was high
    2. Shapley value drops sharply
    
    This signals that the agent was overvalued and underperformed.
    """
    
    def __init__(
        self,
        n_agents: int,
        bubble_threshold: float = 1.5,
        crash_threshold: float = 0.5,
        lookback_window: int = 10,
        cooldown_period: int = 20,
        device: str = 'cpu'
    ):
        """
        Args:
            n_agents: Number of agents
            bubble_threshold: Bubble coefficient threshold
            crash_threshold: Shapley drop ratio to trigger crash
            lookback_window: Window to detect sharp drops
            cooldown_period: Steps before an agent can crash again
            device: Computation device
        """
        super().__init__()
        self.n_agents = n_agents
        self.bubble_threshold = bubble_threshold
        self.crash_threshold = crash_threshold
        self.lookback_window = lookback_window
        self.cooldown_period = cooldown_period
        self.device = device
        
        # History tracking
        self.bubble_history = deque(maxlen=lookback_window)
        self.shapley_history = deque(maxlen=lookback_window)
        
        # Cooldown counters (prevent repeated restructuring)
        self.register_buffer('cooldown', torch.zeros(n_agents, device=device))
        
        # Statistics
        self.register_buffer('crash_count', torch.zeros(n_agents, device=device))
        self.register_buffer('total_crashes', torch.tensor(0, device=device))
    
    def compute_bubble_coefficient(
        self,
        weights: torch.Tensor,
        option_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bubble coefficient for each agent.
        
        Bubble_i = w_i / (C_i / ΣC_j)
        
        Interpretation:
        - Bubble = 1: Fair valuation (weight matches option value proportion)
        - Bubble > 1: Overvalued (allocated more than deserved)
        - Bubble < 1: Undervalued (allocated less than deserved)
        
        Args:
            weights: Portfolio weights, shape (n_agents,)
            option_values: Option values, shape (n_agents,)
        
        Returns:
            bubble_coeff: Bubble coefficients, shape (n_agents,)
        """
        total_value = option_values.sum() + 1e-8
        relative_values = option_values / total_value
        
        bubble_coeff = weights / (relative_values + 1e-8)
        
        return bubble_coeff
    
    def detect_crash(
        self,
        shapley_values: torch.Tensor,
        bubble_coeff: torch.Tensor
    ) -> torch.Tensor:
        """
        Detect bubble crashes (burst events).
        
        A crash occurs when:
        1. Agent was in bubble territory (Bubble > threshold)
        2. Shapley value dropped sharply (< crash_threshold × recent_max)
        3. Agent is not in cooldown period
        
        Args:
            shapley_values: Current Shapley values
            bubble_coeff: Current bubble coefficients
        
        Returns:
            crash_mask: Boolean mask, True for agents experiencing crash
        """
        # Update history
        self.bubble_history.append(bubble_coeff.clone())
        self.shapley_history.append(shapley_values.clone())
        
        # Decrease cooldown
        self.cooldown = torch.clamp(self.cooldown - 1, min=0)
        
        if len(self.shapley_history) < 2:
            return torch.zeros(self.n_agents, dtype=torch.bool, device=self.device)
        
        # Check conditions
        history_tensor = torch.stack(list(self.shapley_history))
        recent_max = history_tensor.max(dim=0).values
        
        # Condition 1: Was in bubble
        bubble_history_tensor = torch.stack(list(self.bubble_history))
        was_in_bubble = bubble_history_tensor.max(dim=0).values > self.bubble_threshold
        
        # Condition 2: Sharp drop
        sharp_drop = shapley_values < self.crash_threshold * recent_max
        
        # Condition 3: Not in cooldown
        not_in_cooldown = self.cooldown == 0
        
        # Combine conditions
        crash_mask = was_in_bubble & sharp_drop & not_in_cooldown
        
        # Update cooldown for crashed agents
        self.cooldown[crash_mask] = self.cooldown_period
        
        # Update statistics
        self.crash_count += crash_mask.float()
        self.total_crashes += crash_mask.sum()
        
        return crash_mask
    
    def forward(
        self,
        weights: torch.Tensor,
        option_values: torch.Tensor,
        shapley_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full bubble detection pipeline.
        
        Returns:
            Dictionary with:
            - 'bubble_coeff': Current bubble coefficients
            - 'crash_mask': Which agents are crashing
            - 'is_bubble': Which agents are currently in bubble territory
            - 'is_undervalued': Which agents are undervalued
        """
        bubble_coeff = self.compute_bubble_coefficient(weights, option_values)
        crash_mask = self.detect_crash(shapley_values, bubble_coeff)
        
        return {
            'bubble_coeff': bubble_coeff,
            'crash_mask': crash_mask,
            'is_bubble': bubble_coeff > self.bubble_threshold,
            'is_undervalued': bubble_coeff < 1.0 / self.bubble_threshold,
        }
    
    def get_bubble_penalty(
        self,
        bubble_coeff: torch.Tensor,
        penalty_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Compute penalty term for bubble regularization.
        
        L_bubble = β × Σᵢ max(0, Bubble_i - θ)²
        
        This can be added to the loss function to discourage bubbles.
        """
        excess = torch.clamp(bubble_coeff - self.bubble_threshold, min=0)
        return penalty_scale * (excess ** 2).sum()
    
    def reset(self):
        """Reset detector state."""
        self.bubble_history.clear()
        self.shapley_history.clear()
        self.cooldown.zero_()


class ParameterRestructurer:
    """
    Implements parameter restructuring strategies for crashed agents.
    
    Strategies:
    1. Soft Reset: Add noise to parameters, partially randomizing
    2. Hard Reset: Reset policy head, keep feature extractor
    3. Knowledge Distillation: Learn from successful agents
    4. Clone: Copy parameters from best performing agent
    """
    
    def __init__(
        self,
        strategy: str = 'soft_reset',
        reset_ratio: float = 0.5,
        noise_scale: float = 0.1,
        distill_steps: int = 100,
        distill_lr: float = 0.001
    ):
        """
        Args:
            strategy: 'soft_reset', 'hard_reset', 'distill', or 'clone'
            reset_ratio: Interpolation ratio for soft reset
            noise_scale: Scale of noise for soft reset
            distill_steps: Steps for knowledge distillation
            distill_lr: Learning rate for distillation
        """
        self.strategy = strategy
        self.reset_ratio = reset_ratio
        self.noise_scale = noise_scale
        self.distill_steps = distill_steps
        self.distill_lr = distill_lr
    
    def restructure(
        self,
        agent: nn.Module,
        crash_mask: torch.Tensor,
        all_agents: Optional[List[nn.Module]] = None,
        best_agent_idx: Optional[int] = None,
        replay_buffer: Optional[object] = None
    ):
        """
        Apply restructuring to crashed agents.
        
        Args:
            agent: The agent module (or list of agents)
            crash_mask: Boolean mask indicating which agents crashed
            all_agents: List of all agents (for clone/distill strategies)
            best_agent_idx: Index of best performing agent
            replay_buffer: Experience buffer for distillation
        """
        if isinstance(agent, list):
            for i, ag in enumerate(agent):
                if crash_mask[i]:
                    self._restructure_single(
                        ag, all_agents, best_agent_idx, replay_buffer
                    )
        else:
            # Assume agent has sub-networks for each agent index
            # This is a simplified version
            pass
    
    def _restructure_single(
        self,
        agent: nn.Module,
        all_agents: Optional[List[nn.Module]] = None,
        best_agent_idx: Optional[int] = None,
        replay_buffer: Optional[object] = None
    ):
        """Apply restructuring to a single agent."""
        if self.strategy == 'soft_reset':
            self._soft_reset(agent)
        elif self.strategy == 'hard_reset':
            self._hard_reset(agent)
        elif self.strategy == 'clone':
            if all_agents and best_agent_idx is not None:
                self._clone(agent, all_agents[best_agent_idx])
        elif self.strategy == 'distill':
            if all_agents and best_agent_idx is not None and replay_buffer:
                self._distill(agent, all_agents[best_agent_idx], replay_buffer)
    
    def _soft_reset(self, agent: nn.Module):
        """
        Soft reset: interpolate between current and random parameters.
        
        θ_new = (1 - α)·θ_old + α·(θ_old + noise)
        
        This partially randomizes while retaining some learned structure.
        """
        with torch.no_grad():
            for name, param in agent.named_parameters():
                # Skip feature extractor if present
                if 'feature' in name.lower() or 'encoder' in name.lower():
                    continue
                
                noise = torch.randn_like(param) * param.std() * self.noise_scale
                param.data = (1 - self.reset_ratio) * param.data + self.reset_ratio * noise
    
    def _hard_reset(self, agent: nn.Module):
        """
        Hard reset: reinitialize policy head parameters.
        
        Keeps feature extractor frozen, resets only the decision layers.
        """
        for name, module in agent.named_modules():
            # Reset policy/action heads
            if 'policy' in name.lower() or 'action' in name.lower() or 'head' in name.lower():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    
    def _clone(self, target_agent: nn.Module, source_agent: nn.Module):
        """
        Clone: copy parameters from best agent.
        
        θ_target = θ_source + small_noise
        """
        with torch.no_grad():
            for (name_t, param_t), (name_s, param_s) in zip(
                target_agent.named_parameters(),
                source_agent.named_parameters()
            ):
                if name_t != name_s:
                    continue
                
                # Copy with small perturbation for diversity
                noise = torch.randn_like(param_s) * 0.01
                param_t.data = param_s.data + noise
    
    def _distill(
        self,
        student: nn.Module,
        teacher: nn.Module,
        replay_buffer
    ):
        """
        Knowledge distillation: train student to match teacher outputs.
        
        This preserves the teacher's learned behavior while giving
        the student a fresh start on the optimization landscape.
        """
        optimizer = torch.optim.Adam(student.parameters(), lr=self.distill_lr)
        
        teacher.eval()
        student.train()
        
        for _ in range(self.distill_steps):
            # Sample from replay buffer
            if hasattr(replay_buffer, 'sample'):
                batch = replay_buffer.sample(32)
                states = batch.states
            else:
                # Fallback for simple list buffer
                indices = np.random.choice(len(replay_buffer), 32)
                states = torch.stack([replay_buffer[i][0] for i in indices])
            
            # Get teacher outputs
            with torch.no_grad():
                teacher_out = teacher(states)
            
            # Get student outputs
            student_out = student(states)
            
            # Distillation loss (KL divergence or MSE)
            if isinstance(teacher_out, tuple):
                teacher_out = teacher_out[0]
                student_out = student_out[0]
            
            loss = F.mse_loss(student_out, teacher_out)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        student.eval()


class RestructuringScheduler:
    """
    Schedules and manages restructuring events.
    
    Features:
    - Tracks restructuring history
    - Prevents excessive restructuring
    - Adapts thresholds based on learning progress
    """
    
    def __init__(
        self,
        n_agents: int,
        max_restructures_per_agent: int = 5,
        min_interval: int = 50,
        adaptive_threshold: bool = True
    ):
        """
        Args:
            n_agents: Number of agents
            max_restructures_per_agent: Maximum restructures before giving up
            min_interval: Minimum steps between restructures for same agent
            adaptive_threshold: Whether to adapt bubble threshold over time
        """
        self.n_agents = n_agents
        self.max_restructures = max_restructures_per_agent
        self.min_interval = min_interval
        self.adaptive_threshold = adaptive_threshold
        
        self.restructure_counts = np.zeros(n_agents)
        self.last_restructure_step = np.zeros(n_agents)
        self.current_step = 0
    
    def should_restructure(
        self,
        agent_idx: int,
        crash_detected: bool
    ) -> bool:
        """
        Determine if restructuring should proceed.
        
        Args:
            agent_idx: Index of the agent
            crash_detected: Whether a crash was detected
        
        Returns:
            True if restructuring should proceed
        """
        self.current_step += 1
        
        if not crash_detected:
            return False
        
        # Check if max restructures reached
        if self.restructure_counts[agent_idx] >= self.max_restructures:
            return False
        
        # Check minimum interval
        steps_since_last = self.current_step - self.last_restructure_step[agent_idx]
        if steps_since_last < self.min_interval:
            return False
        
        # Approve restructuring
        self.restructure_counts[agent_idx] += 1
        self.last_restructure_step[agent_idx] = self.current_step
        
        return True
    
    def get_adaptive_threshold(self, base_threshold: float, progress: float) -> float:
        """
        Adapt bubble threshold based on training progress.
        
        Early training: Higher threshold (more tolerance for exploration)
        Late training: Lower threshold (stricter valuation)
        
        Args:
            base_threshold: Base threshold value
            progress: Training progress in [0, 1]
        
        Returns:
            Adapted threshold
        """
        if not self.adaptive_threshold:
            return base_threshold
        
        # Linear decay from 2×base to base
        return base_threshold * (2.0 - progress)


# ============================================================================
# Unit Tests
# ============================================================================

def test_bubble_detector():
    """Test bubble detection."""
    print("Testing Bubble Detector...")
    
    n_agents = 4
    detector = BubbleDetector(
        n_agents=n_agents,
        bubble_threshold=1.5,
        crash_threshold=0.5
    )
    
    # Scenario: Agent 0 is overvalued (weight >> option value)
    weights = torch.tensor([0.5, 0.2, 0.2, 0.1])  # Agent 0 has 50%
    option_values = torch.tensor([0.1, 0.3, 0.3, 0.3])  # But only 10% of value
    
    bubble_coeff = detector.compute_bubble_coefficient(weights, option_values)
    print(f"  Bubble coefficients: {bubble_coeff.numpy()}")
    
    assert bubble_coeff[0] > detector.bubble_threshold, "Agent 0 should be in bubble"
    
    # Simulate crash: Agent 0's Shapley drops
    shapley_t1 = torch.tensor([1.0, 0.8, 0.7, 0.6])
    result1 = detector(weights, option_values, shapley_t1)
    
    shapley_t2 = torch.tensor([0.3, 0.8, 0.7, 0.6])  # Agent 0 drops to 30%
    result2 = detector(weights, option_values, shapley_t2)
    
    print(f"  Crash mask: {result2['crash_mask'].numpy()}")
    
    # Agent 0 should crash (was in bubble + sharp drop)
    assert result2['crash_mask'][0], "Agent 0 should crash"
    
    print("  ✓ Bubble detector tests passed!\n")


def test_parameter_restructurer():
    """Test parameter restructuring."""
    print("Testing Parameter Restructurer...")
    
    # Create a simple network
    agent = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 4)
    )
    
    # Save original parameters
    original_params = {name: p.clone() for name, p in agent.named_parameters()}
    
    # Apply soft reset
    restructurer = ParameterRestructurer(strategy='soft_reset', reset_ratio=0.5)
    crash_mask = torch.tensor([True])  # Dummy
    
    restructurer._soft_reset(agent)
    
    # Check parameters changed
    changed = False
    for name, p in agent.named_parameters():
        if not torch.allclose(p, original_params[name]):
            changed = True
            break
    
    assert changed, "Parameters should have changed after soft reset"
    print("  ✓ Parameter restructurer tests passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Bubble Detection Module Tests")
    print("=" * 60 + "\n")
    
    test_bubble_detector()
    test_parameter_restructurer()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
