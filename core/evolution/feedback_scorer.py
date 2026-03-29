"""
J.A.R.V.I.S — Feedback Scorer
Evaluates execution results and generates reward signals for RL.
"""

import time
from dataclasses import dataclass
from typing import Optional
from loguru import logger


class FeedbackScorer:
    """
    Multi-dimensional feedback scoring system.
    Converts execution results into normalized reward signals [0, 1].

    Dimensions scored:
    - Task success/failure
    - Execution efficiency (time)
    - Goal alignment
    - Safety compliance
    - Output quality
    """

    WEIGHTS = {
        "success": 0.45,
        "efficiency": 0.20,
        "alignment": 0.20,
        "safety": 0.10,
        "quality": 0.05
    }

    async def score(self, exec_result, goal) -> float:
        """
        Compute composite reward score for a completed execution.
        Returns value in [0, 1].
        """
        scores = {}

        # 1. SUCCESS score
        scores["success"] = self._score_success(exec_result)

        # 2. EFFICIENCY score
        scores["efficiency"] = self._score_efficiency(exec_result)

        # 3. ALIGNMENT score (did we address the actual goal?)
        scores["alignment"] = self._score_alignment(exec_result, goal)

        # 4. SAFETY score (did we stay within safety bounds?)
        scores["safety"] = self._score_safety(exec_result)

        # 5. QUALITY score (output quality heuristics)
        scores["quality"] = self._score_quality(exec_result)

        # Weighted composite
        final_score = sum(
            scores[dim] * weight
            for dim, weight in self.WEIGHTS.items()
        )

        final_score = max(0.0, min(1.0, final_score))

        logger.debug(
            f"Feedback: success={scores['success']:.2f} "
            f"efficiency={scores['efficiency']:.2f} "
            f"alignment={scores['alignment']:.2f} "
            f"final={final_score:.2f}"
        )

        return final_score

    def _score_success(self, exec_result) -> float:
        """Binary success + partial credit for retries."""
        if not hasattr(exec_result, 'success'):
            return 0.5
        if exec_result.success:
            # Penalize slightly for retries needed
            retries = getattr(exec_result, 'retries', 0)
            return max(0.6, 1.0 - retries * 0.1)
        else:
            return 0.0

    def _score_efficiency(self, exec_result) -> float:
        """Score based on execution time relative to complexity."""
        duration = getattr(exec_result, 'duration', 1.0)
        # Target: under 2 seconds for simple tasks
        if duration < 0.5:
            return 1.0
        elif duration < 2.0:
            return 0.9
        elif duration < 5.0:
            return 0.7
        elif duration < 15.0:
            return 0.5
        else:
            return 0.3

    def _score_alignment(self, exec_result, goal) -> float:
        """How well does the result match the stated goal?"""
        if not hasattr(exec_result, 'output'):
            return 0.5

        # Check if output contains relevant keywords from goal
        goal_keywords = set(goal.context.get("keywords", []))
        output_str = str(getattr(exec_result, 'output', "")).lower()

        if not goal_keywords:
            return 0.7  # Can't measure, assume partial

        matches = sum(1 for kw in goal_keywords if kw in output_str)
        return min(1.0, 0.3 + (matches / max(len(goal_keywords), 1)) * 0.7)

    def _score_safety(self, exec_result) -> float:
        """Was the action within safety parameters?"""
        risk = getattr(exec_result, 'risk_level', 'low')
        approved = getattr(exec_result, 'safety_approved', True)

        if not approved:
            return 0.0  # Safety violation

        risk_scores = {"low": 1.0, "medium": 0.8, "high": 0.5}
        return risk_scores.get(risk, 0.7)

    def _score_quality(self, exec_result) -> float:
        """Heuristic output quality scoring."""
        output = str(getattr(exec_result, 'output', ""))
        if not output or output.strip() == "":
            return 0.1
        if len(output) > 50:
            return 1.0
        elif len(output) > 20:
            return 0.7
        else:
            return 0.4
