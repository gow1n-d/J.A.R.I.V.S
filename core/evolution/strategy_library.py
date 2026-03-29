"""
J.A.R.V.I.S — Strategy Library
Stores, retrieves, and ranks successful execution strategies.
Enables J.A.R.V.I.S to reuse optimized workflows.
"""

import json
import time
from typing import List, Dict, Any, Optional
from loguru import logger


class StrategyLibrary:
    """
    Persistent library of learned successful strategies.
    Strategies are ranked by composite score and usage count.
    Enables rapid reuse of proven approaches.
    """

    def __init__(self, memory):
        self.memory = memory
        self._cache: List[Dict] = []
        self._loaded = False

    async def _ensure_loaded(self):
        if not self._loaded:
            stored = await self.memory.long_term.retrieve("strategy_library")
            if stored:
                self._cache = stored
            self._loaded = True

    async def record_success(self, goal, action_choice, exec_result, score: float):
        """Archive a successful strategy for future reuse."""
        await self._ensure_loaded()

        strategy = {
            "id": f"S-{int(time.time())}",
            "name": f"{action_choice.action_type}:{goal.estimated_complexity}",
            "goal_type": goal.goal_type,
            "goal_complexity": goal.estimated_complexity,
            "action_type": action_choice.action_type,
            "strategy_type": action_choice.strategy,
            "score": score,
            "uses": 1,
            "last_used": time.time(),
            "keywords": goal.context.get("keywords", []),
            "success_criteria": goal.success_criteria,
            "domain": goal.context.get("domain", "unknown"),
            "rationale": action_choice.rationale
        }

        # Check if similar strategy exists
        existing = self._find_similar(strategy)
        if existing:
            # Update existing strategy
            existing["uses"] += 1
            existing["score"] = (existing["score"] * 0.8) + (score * 0.2)  # EMA update
            existing["last_used"] = time.time()
        else:
            self._cache.append(strategy)

        # Sort by composite score
        self._cache.sort(
            key=lambda s: s["score"] * 0.7 + min(s["uses"] / 100, 0.3),
            reverse=True
        )

        # Keep top 500 strategies
        if len(self._cache) > 500:
            self._cache = self._cache[:500]

        # Persist
        await self.memory.long_term.store(
            "strategy_library", self._cache, category="strategies", importance=0.9
        )
        logger.debug(f"Strategy recorded: {strategy['name']} (score={score:.2f})")

    def _find_similar(self, strategy: Dict) -> Optional[Dict]:
        """Find a similar existing strategy."""
        for s in self._cache:
            if (s["action_type"] == strategy["action_type"] and
                    s["goal_complexity"] == strategy["goal_complexity"] and
                    s["domain"] == strategy["domain"]):
                return s
        return None

    async def find_best_for(self, goal) -> Optional[Dict]:
        """Find the best strategy for a given goal."""
        await self._ensure_loaded()

        domain = goal.context.get("domain", "unknown")
        candidates = [
            s for s in self._cache
            if s.get("domain") == domain and
            s.get("goal_complexity") == goal.estimated_complexity
        ]

        if not candidates:
            candidates = [
                s for s in self._cache
                if s.get("goal_type") == goal.goal_type
            ]

        if candidates:
            return max(candidates, key=lambda s: s["score"])
        return None

    async def load_all(self) -> List[Dict]:
        """Return all strategies from library."""
        await self._ensure_loaded()
        return self._cache

    async def count(self) -> int:
        await self._ensure_loaded()
        return len(self._cache)

    async def get_rankings(self) -> List[Dict]:
        """Return top 10 ranked strategies."""
        await self._ensure_loaded()
        return self._cache[:10]
