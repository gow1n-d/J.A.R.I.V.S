"""
J.A.R.V.I.S — World Model
Internal simulation of environment, system state, and dependencies.
Provides state vectors for the RL decision engine.
"""

import time
import psutil
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from loguru import logger


@dataclass
class WorldState:
    """Complete snapshot of the current world state."""
    timestamp: float
    system_load: float          # CPU usage 0-1
    memory_usage: float         # RAM usage 0-1
    task_queue_depth: int
    recent_success_rate: float  # last 10 tasks
    conversation_depth: int     # turns in current conversation
    active_mode: str
    last_action_risk: str
    last_action_success: bool
    user_satisfaction: float    # estimated from sentiment
    entropy: float              # exploration tendency


class WorldModel:
    """
    Maintains an internal simulation of the world.
    Updates on every interaction and provides state vectors for RL.
    """

    STATE_DIM = 16  # Dimensionality of state vector

    def __init__(self, memory):
        self.memory = memory
        self.state = WorldState(
            timestamp=time.time(),
            system_load=0.0,
            memory_usage=0.0,
            task_queue_depth=0,
            recent_success_rate=0.5,
            conversation_depth=0,
            active_mode="adaptive",
            last_action_risk="low",
            last_action_success=True,
            user_satisfaction=0.7,
            entropy=0.5
        )
        self._history: List[WorldState] = []
        self._success_window: List[bool] = []
        logger.debug("World Model online.")

    async def update(self, percept, goal):
        """Update world state based on new percept and goal."""
        # System stats
        try:
            self.state.system_load = psutil.cpu_percent(interval=0.1) / 100.0
            self.state.memory_usage = psutil.virtual_memory().percent / 100.0
        except Exception:
            self.state.system_load = 0.2
            self.state.memory_usage = 0.3

        # Conversation tracking
        self.state.conversation_depth += 1
        self.state.timestamp = time.time()

        # User satisfaction from sentiment
        sentiment_map = {"positive": 0.9, "neutral": 0.7, "negative": 0.3}
        self.state.user_satisfaction = sentiment_map.get(percept.sentiment, 0.7)

        # Snapshot history (keep last 100)
        self._history.append(WorldState(**self.state.__dict__))
        if len(self._history) > 100:
            self._history.pop(0)

        logger.debug(
            f"World updated: cpu={self.state.system_load:.2f} "
            f"mem={self.state.memory_usage:.2f} "
            f"depth={self.state.conversation_depth}"
        )

    def record_outcome(self, success: bool, risk: str):
        """Record the outcome of the last action for state tracking."""
        self.state.last_action_success = success
        self.state.last_action_risk = risk

        # Rolling success rate
        self._success_window.append(success)
        if len(self._success_window) > 10:
            self._success_window.pop(0)
        self.state.recent_success_rate = (
            sum(self._success_window) / len(self._success_window)
        )

    async def get_state_vector(self) -> List[float]:
        """
        Convert world state to a fixed-size vector for RL.
        All values normalized to [0, 1].
        """
        mode_encoding = {"engineer": 0.33, "execution": 0.66, "adaptive": 1.0}
        risk_encoding = {"low": 0.0, "medium": 0.5, "high": 1.0}

        vector = [
            self.state.system_load,
            self.state.memory_usage,
            min(1.0, self.state.task_queue_depth / 10.0),
            self.state.recent_success_rate,
            min(1.0, self.state.conversation_depth / 50.0),
            mode_encoding.get(self.state.active_mode, 0.5),
            risk_encoding.get(self.state.last_action_risk, 0.0),
            1.0 if self.state.last_action_success else 0.0,
            self.state.user_satisfaction,
            self.state.entropy,
            # Derived features
            self.state.system_load * self.state.memory_usage,         # Resource pressure
            self.state.recent_success_rate * self.state.user_satisfaction,  # Quality score
            min(1.0, self.state.conversation_depth / 100.0),
            1.0 - self.state.system_load,                             # Available capacity
            self.state.recent_success_rate,
            0.5  # Bias term
        ]

        # Ensure correct dimension
        while len(vector) < self.STATE_DIM:
            vector.append(0.0)

        return vector[:self.STATE_DIM]

    def get_world_summary(self) -> Dict[str, Any]:
        return {
            "cpu": f"{self.state.system_load*100:.1f}%",
            "ram": f"{self.state.memory_usage*100:.1f}%",
            "success_rate": f"{self.state.recent_success_rate*100:.1f}%",
            "conversation_turns": self.state.conversation_depth,
            "user_satisfaction": f"{self.state.user_satisfaction*100:.0f}%"
        }
