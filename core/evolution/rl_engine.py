"""
J.A.R.V.I.S — Reinforcement Learning Engine
Deep Q-Network with experience replay and target network.
This is the self-evolution core of J.A.R.V.I.S.

Evolution Loop:
STATE → ACTION → REWARD → LEARN → OPTIMIZE → REPEAT
"""

import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


@dataclass
class Experience:
    """A single learning experience (s, a, r, s', done)."""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    timestamp: float = field(default_factory=time.time)


class DQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture.
    Separates state value and action advantage for better learning stability.
    """

    def __init__(self, state_dim: int = 16, action_dim: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


class ExperienceReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    Samples experiences weighted by their learning value (TD error).
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self._max_priority = 1.0

    def push(self, experience: Experience, priority: float = None):
        """Add experience to buffer."""
        if priority is None:
            priority = self._max_priority
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample experiences (uniform for now, prioritized optionally)."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        priorities_arr = np.array(self.priorities)
        probs = priorities_arr / priorities_arr.sum()

        indices = np.random.choice(
            len(self.buffer), size=batch_size, replace=False, p=probs
        )
        return [self.buffer[i] for i in indices]

    def update_priority(self, idx: int, priority: float):
        if 0 <= idx < len(self.priorities):
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size


class StrategyComparator:
    """Compares strategies and tracks which approaches work best."""

    def __init__(self):
        self.strategy_scores: Dict[str, List[float]] = {}

    def record(self, strategy: str, reward: float):
        if strategy not in self.strategy_scores:
            self.strategy_scores[strategy] = []
        self.strategy_scores[strategy].append(reward)
        # Keep only last 50 scores per strategy
        if len(self.strategy_scores[strategy]) > 50:
            self.strategy_scores[strategy].pop(0)

    def best_strategy(self) -> Optional[str]:
        if not self.strategy_scores:
            return None
        return max(
            self.strategy_scores,
            key=lambda s: np.mean(self.strategy_scores[s])
        )

    def get_rankings(self) -> List[Tuple[str, float]]:
        rankings = [
            (s, np.mean(scores))
            for s, scores in self.strategy_scores.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)


class RLEngine:
    """
    Deep Q-Learning Engine — the self-evolution core.
    Implements:
    - Dueling DQN with experience replay
    - Target network for training stability
    - ε-greedy exploration
    - Forced evolution cycles
    - Strategy comparison
    - Self-benchmarking
    """

    ACTION_TYPES = [
        "respond", "analyze", "create_action", "execute_action",
        "delete_action", "system_action", "generate_help",
        "halt", "chat_response", "plan_action"
    ]

    def __init__(self, config: dict):
        self.config = config
        self.state_dim = config.get("state_dim", 16)
        self.action_dim = len(self.ACTION_TYPES)

        # Networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.get("learning_rate", 0.001)
        )
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=500, gamma=0.9
        )

        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=config.get("replay_buffer_size", 10000)
        )
        self.batch_size = config.get("batch_size", 32)

        # RL hyperparameters
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.gamma = config.get("discount_factor", 0.95)
        self.train_frequency = config.get("train_frequency", 4)
        self.target_update_freq = config.get("target_update_frequency", 100)

        # Tracking
        self._step_count = 0
        self._episode_count = 0
        self._wins = 0
        self._total_loss = 0.0
        self._loss_history: List[float] = []

        # Strategy comparison
        self.comparator = StrategyComparator()

        logger.info(
            f"RL Engine online: DQN({self.state_dim}→{self.action_dim}) | "
            f"ε={self.epsilon:.2f} | γ={self.gamma}"
        )

    async def learn(self, state: List[float], action_choice, reward: float,
                    exec_result) -> float:
        """
        Core learning step:
        1. Store experience
        2. Train when buffer ready
        3. Update target network
        4. Decay epsilon
        """
        self._step_count += 1
        self._episode_count += 1

        if reward > 0.5:
            self._wins += 1

        # Map action type to index
        action_idx = self._action_to_idx(action_choice.action_type)

        # Record in strategy comparator
        self.comparator.record(action_choice.strategy, reward)

        # Create experience (simplified next state = same state for now)
        next_state = state  # Will be updated on next world model tick
        done = reward < 0  # Episode "failed"

        experience = Experience(
            state=state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=done
        )

        # Priority: higher reward → higher priority
        priority = abs(reward) + 0.01
        self.replay_buffer.push(experience, priority)

        # Train if ready
        loss = 0.0
        if (self._step_count % self.train_frequency == 0 and
                self.replay_buffer.is_ready(self.batch_size)):
            loss = await self._train_step()
            self._total_loss += loss
            self._loss_history.append(loss)
            if len(self._loss_history) > 100:
                self._loss_history.pop(0)

        # Update target network
        if self._step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.debug("Target network synchronized.")

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

        return loss

    async def _train_step(self) -> float:
        """Sample a batch and perform a gradient descent step."""
        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.BoolTensor([e.done for e in batch])

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            target_q = rewards + self.gamma * next_q * (~dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return float(loss.item())

    async def forced_evolution(self, iterations: int = 50) -> int:
        """Trigger intensive training on existing replay buffer."""
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0

        for i in range(iterations):
            loss = await self._train_step()
            self._loss_history.append(loss)

        # Sync target
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.info(f"Forced evolution: {iterations} iterations complete.")
        return iterations

    def get_stats(self) -> Dict[str, Any]:
        avg_loss = np.mean(self._loss_history) if self._loss_history else 0.0
        return {
            "epsilon": round(self.epsilon, 4),
            "episodes": self._episode_count,
            "wins": self._wins,
            "win_rate": round(self._wins / max(1, self._episode_count), 3),
            "replay_buffer": len(self.replay_buffer),
            "avg_loss": round(avg_loss, 6),
            "best_strategy": self.comparator.best_strategy() or "none",
            "steps": self._step_count
        }

    def get_policy_state(self) -> Dict:
        """Serialize policy for persistence."""
        return {
            "q_network_state": {
                k: v.tolist() for k, v in self.q_network.state_dict().items()
            },
            "epsilon": self.epsilon,
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "wins": self._wins
        }

    def load_policy(self, state: Dict):
        """Restore policy from serialized state."""
        try:
            network_state = {
                k: torch.tensor(v)
                for k, v in state.get("q_network_state", {}).items()
            }
            self.q_network.load_state_dict(network_state)
            self.target_network.load_state_dict(network_state)
            self.epsilon = state.get("epsilon", self.epsilon)
            self._step_count = state.get("step_count", 0)
            self._episode_count = state.get("episode_count", 0)
            self._wins = state.get("wins", 0)
            logger.info(f"Policy restored: ε={self.epsilon:.3f}, steps={self._step_count}")
        except Exception as e:
            logger.warning(f"Policy load failed: {e}")

    def _action_to_idx(self, action_type: str) -> int:
        try:
            return self.ACTION_TYPES.index(action_type)
        except ValueError:
            return 0  # Default to "respond"
