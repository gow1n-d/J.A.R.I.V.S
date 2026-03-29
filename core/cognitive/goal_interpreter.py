"""
J.A.R.V.I.S — Goal Interpreter
Converts perceived user intent into structured, executable goals.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from loguru import logger


@dataclass
class Goal:
    """A structured goal derived from user intent."""
    id: str
    raw_intent: str
    goal_type: str          # achieve | query | maintain | avoid
    description: str
    priority: float         # 0.0 - 1.0
    constraints: List[str]
    success_criteria: List[str]
    context: Dict[str, Any]
    estimated_complexity: str  # simple | moderate | complex
    timestamp: float = field(default_factory=time.time)


class GoalInterpreter:
    """
    Transforms percepts into well-defined Goal structures.
    Uses memory context to enrich goal interpretation.
    """

    GOAL_TYPE_MAP = {
        "create": "achieve",
        "update": "achieve",
        "execute": "achieve",
        "plan": "achieve",
        "delete": "achieve",
        "analyze": "query",
        "read": "query",
        "learn": "query",
        "status": "query",
        "help": "query",
        "remember": "maintain",
        "stop": "avoid",
        "unknown": "query",
        "predict": "query",
    }

    COMPLEXITY_MAP = {
        "simple": ["show", "get", "read", "list", "status", "help"],
        "moderate": ["create", "update", "analyze", "plan", "remember"],
        "complex": ["execute", "deploy", "build", "train", "optimize"],
    }

    def __init__(self, memory):
        self.memory = memory
        self._goal_counter = 0
        logger.debug("Goal Interpreter online.")

    async def interpret(self, percept) -> Goal:
        """Convert a Percept into a structured Goal."""
        self._goal_counter += 1
        goal_id = f"G-{int(time.time())}-{self._goal_counter:04d}"

        # Determine goal type
        goal_type = self.GOAL_TYPE_MAP.get(percept.intent, "query")

        # Estimate complexity
        complexity = self._estimate_complexity(percept)

        # Build success criteria
        success_criteria = self._build_success_criteria(percept, goal_type)

        # Extract constraints from entities and context
        constraints = self._extract_constraints(percept)

        # Build description
        description = self._synthesize_description(percept)

        # Priority based on urgency + domain
        priority = self._calculate_priority(percept)

        # Context enrichment from memory
        context = await self._enrich_context(percept)

        goal = Goal(
            id=goal_id,
            raw_intent=percept.raw_input,
            goal_type=goal_type,
            description=description,
            priority=priority,
            constraints=constraints,
            success_criteria=success_criteria,
            context=context,
            estimated_complexity=complexity
        )

        logger.debug(f"Goal: {goal_id} | {goal_type} | {complexity} | p={priority:.2f}")
        return goal

    def _estimate_complexity(self, percept) -> str:
        text = percept.raw_input.lower()
        for level, keywords in self.COMPLEXITY_MAP.items():
            for kw in keywords:
                if kw in text:
                    return level
        # Multi-step detection
        if any(w in text for w in ["then", "after", "next", "and also", "also"]):
            return "complex"
        if len(percept.keywords) > 5:
            return "moderate"
        return "simple"

    def _build_success_criteria(self, percept, goal_type: str) -> List[str]:
        criteria = []
        if goal_type == "achieve":
            criteria.append("Task completes without errors")
            criteria.append("Output matches expected format")
            if percept.entities.get("files"):
                criteria.append("Target files successfully modified")
        elif goal_type == "query":
            criteria.append("Accurate response generated")
            criteria.append("Response addresses user question completely")
        elif goal_type == "maintain":
            criteria.append("State successfully preserved")
        elif goal_type == "avoid":
            criteria.append("Operation halted cleanly")
            criteria.append("No data loss occurred")
        return criteria

    def _extract_constraints(self, percept) -> List[str]:
        constraints = []
        if percept.urgency == "critical":
            constraints.append("Execute immediately")
        if percept.urgency == "high":
            constraints.append("Prioritize speed")
        if percept.entities.get("paths"):
            constraints.append(f"Operate within paths: {percept.entities['paths']}")
        return constraints

    def _synthesize_description(self, percept) -> str:
        intent = percept.intent.replace("_", " ").title()
        keywords = ", ".join(percept.keywords[:4]) if percept.keywords else "general request"
        return f"{intent}: {keywords}"

    def _calculate_priority(self, percept) -> float:
        urgency_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        domain_boost = {"task": 0.1, "system": 0.05, "query": 0.0, "conversation": -0.1}

        base = urgency_scores.get(percept.urgency, 0.3)
        boost = domain_boost.get(percept.domain, 0.0)
        return min(1.0, max(0.0, base + boost))

    async def _enrich_context(self, percept) -> Dict[str, Any]:
        context = {
            "intent": percept.intent,
            "domain": percept.domain,
            "entities": percept.entities,
            "keywords": percept.keywords
        }
        try:
            recent = self.memory.short_term.get_all()
            if recent:
                context["previous_input"] = recent[-1].get("input", "")
        except Exception:
            pass
        return context
