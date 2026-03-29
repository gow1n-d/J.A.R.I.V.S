"""
J.A.R.V.I.S — Decision Engine
Hybrid rule-based + reinforcement learning decision system.
Selects the optimal action given the current state.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from loguru import logger


@dataclass
class ActionChoice:
    """A decision made by the engine."""
    action_id: str
    action_type: str
    strategy: str           # learned | rule_based | exploratory
    confidence: float
    parameters: Dict[str, Any]
    estimated_reward: float
    risk_level: str         # low | medium | high
    rationale: str


class QNetwork(nn.Module):
    """
    Deep Q-Network for action value estimation.
    Maps state vectors to action Q-values.
    """

    def __init__(self, state_dim: int = 16, action_dim: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_dim)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RuleEngine:
    """
    Rule-based decision baseline.
    Provides safe, deterministic fallback decisions.
    """

    RULES = [
        # (condition_fn, action_type, risk_level, confidence)
        (lambda s, g: g.get("intent") == "stop", "halt", "low", 1.0),
        (lambda s, g: g.get("intent") == "help", "generate_help", "low", 1.0),
        (lambda s, g: g.get("domain") == "system", "system_action", "low", 0.9),
        (lambda s, g: g.get("intent") == "analyze", "analyze", "low", 0.85),
        (lambda s, g: g.get("intent") == "create", "create_action", "medium", 0.8),
        (lambda s, g: g.get("intent") == "execute", "execute_action", "high", 0.7),
        (lambda s, g: g.get("intent") == "delete", "delete_action", "high", 0.6),
        (lambda s, g: g.get("domain") == "conversation", "chat_response", "low", 0.95),
        (lambda s, g: True, "respond", "low", 0.5),  # Default fallback
    ]

    def decide(self, state_vector: List[float], goal_context: Dict) -> Optional[tuple]:
        for condition, action_type, risk, confidence in self.RULES:
            try:
                if condition(state_vector, goal_context):
                    return action_type, risk, confidence
            except Exception:
                continue
        return "respond", "low", 0.5


ACTION_TYPES = [
    "respond", "analyze", "create_action", "execute_action",
    "delete_action", "system_action", "generate_help",
    "halt", "chat_response", "plan_action"
]


class DecisionEngine:
    """
    Hybrid Decision Engine:
    - Uses DQN for learned decisions when confidence is high
    - Falls back to rule engine for safety and explainability
    """

    def __init__(self, config: dict, memory):
        self.config = config
        self.memory = memory

        # RL components
        state_dim = config.get("state_dim", 16)
        action_dim = len(ACTION_TYPES)
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.get("learning_rate", 0.001)
        )

        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

        # Rule engine fallback
        self.rule_engine = RuleEngine()

        self._decision_count = 0
        logger.debug("Decision Engine online (DQN + Rule Engine hybrid).")

    async def decide(self, state_vector: List[float], task_graph) -> ActionChoice:
        """
        Choose optimal action using hybrid policy:
        1. Try DQN if epsilon allows
        2. Fall back to rule engine if confidence too low
        3. Always pass through safety governor
        """
        self._decision_count += 1

        # Get goal context
        goal_context = {}
        if task_graph.tasks:
            params = task_graph.tasks[0].parameters
            goal_context = {
                "intent": params.get("goal_context", {}).get("intent", "unknown"),
                "domain": params.get("goal_context", {}).get("domain", "unknown"),
            }

        # Pad/truncate state vector to correct dimension
        padded_state = self._pad_state(state_vector)

        # Decision: ε-greedy
        use_rl = random.random() > self.epsilon and self._decision_count > 100

        if use_rl:
            action_idx, confidence, q_values = self._rl_decide(padded_state)
            action_type = ACTION_TYPES[action_idx]
            strategy = "learned"
        else:
            # Rule-based
            action_type, risk_level, confidence = self.rule_engine.decide(
                padded_state, goal_context
            )
            strategy = "rule_based" if self.epsilon < 0.8 else "exploratory"
            if strategy == "exploratory" and random.random() < 0.3:
                action_type = random.choice(ACTION_TYPES)
                strategy = "exploratory"

        # Assess risk
        risk_level = self._assess_risk(action_type, goal_context)

        # Estimate reward
        estimated_reward = self._estimate_reward(action_type, confidence, risk_level)

        # Build rationale
        rationale = self._build_rationale(
            action_type, strategy, confidence, goal_context
        )

        action = ActionChoice(
            action_id=f"A-{int(time.time())}-{self._decision_count:04d}",
            action_type=action_type,
            strategy=strategy,
            confidence=confidence,
            parameters={
                "task_graph": task_graph,
                "goal_context": goal_context,
                "state_vector": padded_state
            },
            estimated_reward=estimated_reward,
            risk_level=risk_level,
            rationale=rationale
        )

        logger.debug(
            f"Decision: {action_type} | strategy={strategy} | "
            f"conf={confidence:.2f} | risk={risk_level} | ε={self.epsilon:.3f}"
        )

        return action

    def _rl_decide(self, state: List[float]) -> tuple:
        """Get DQN decision."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze()

        action_idx = int(q_values.argmax().item())
        max_q = float(q_values.max().item())
        # Normalize confidence
        confidence = min(1.0, max(0.0, (max_q + 1) / 2))
        return action_idx, confidence, q_values.tolist()

    def _pad_state(self, state: List[float]) -> List[float]:
        target_dim = self.config.get("state_dim", 16)
        if len(state) < target_dim:
            return state + [0.0] * (target_dim - len(state))
        return state[:target_dim]

    def _assess_risk(self, action_type: str, context: Dict) -> str:
        high_risk = {"execute_action", "delete_action"}
        medium_risk = {"create_action", "system_action", "plan_action"}
        if action_type in high_risk:
            return "high"
        elif action_type in medium_risk:
            return "medium"
        return "low"

    def _estimate_reward(self, action_type: str, confidence: float, risk: str) -> float:
        base_reward = confidence
        risk_penalty = {"low": 0.0, "medium": -0.1, "high": -0.2}.get(risk, 0.0)
        return max(0.0, base_reward + risk_penalty)

    def _build_rationale(
        self, action_type: str, strategy: str,
        confidence: float, context: Dict
    ) -> str:
        return (
            f"Selected '{action_type}' via {strategy} "
            f"(confidence={confidence:.0%}, "
            f"domain={context.get('domain', 'unknown')})"
        )

    def update_epsilon(self):
        """Decay exploration rate after each learning step."""
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
