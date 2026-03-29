"""
J.A.R.V.I.S — Response Formatter
Formats execution results into personality-consistent responses.
"""

import random
from typing import Optional
from loguru import logger


class ResponseFormatter:
    """
    Converts raw execution results into polished J.A.R.V.I.S responses.
    Applies personality mode, formatting, and contextual additions.
    """

    def __init__(self, personality):
        self.personality = personality

    async def format_response(
        self, exec_result, goal, score: float, mode: str
    ) -> str:
        """Format an execution result into the final response."""
        output = str(getattr(exec_result, 'output', ""))
        success = getattr(exec_result, 'success', True)
        duration = getattr(exec_result, 'duration', 0.0)
        risk = getattr(exec_result, 'risk_level', 'low')

        profile = self.personality.get_profile()
        self.personality.increment_interaction()

        # Base response
        if not output or output.strip() == "":
            output = self._fallback_response(goal, mode)

        # Mode-specific formatting
        if mode == "engineer":
            response = self._engineer_format(output, exec_result, score, duration)
        elif mode == "execution":
            response = self._execution_format(output, success, duration)
        else:
            response = self._adaptive_format(output, success, score)

        # Warnings from safety
        warnings = getattr(exec_result, 'metadata', {})

        # Witty remark (adaptive mode, occasional)
        if self.personality.should_add_wit() and mode == "adaptive":
            response += f"\n\n_{self.personality.get_wit_line()}_"

        return response

    def _engineer_format(self, output: str, result, score: float, duration: float) -> str:
        completed = getattr(result, 'completed_tasks', 1)
        total = getattr(result, 'task_count', 1)
        retries = getattr(result, 'retries', 0)
        return (
            f"{output}\n\n"
            f"---\n"
            f"**Execution Report** | Tasks: {completed}/{total} | "
            f"Duration: {duration:.2f}s | Score: {score:.2f} | "
            f"Retries: {retries} | Strategy: {result.metadata.get('strategy', 'N/A') if hasattr(result, 'metadata') else 'N/A'}"
        )

    def _execution_format(self, output: str, success: bool, duration: float) -> str:
        status = "✅ DONE" if success else "⚠ PARTIAL"
        return f"{status} | {output} | {duration:.2f}s"

    def _adaptive_format(self, output: str, success: bool, score: float) -> str:
        return output

    def _fallback_response(self, goal, mode: str) -> str:
        return (
            f"Directive received and processed: `{goal.description}`\n\n"
            f"Goal type: **{goal.goal_type}** | Priority: {goal.priority:.0%} | "
            f"Complexity: **{goal.estimated_complexity}**\n\n"
            f"All systems engaged. Ready for next directive."
        )

    def format_safety_block(self, safety_result) -> str:
        """Format a safety block message."""
        msg = (
            f"**⛔ Safety Governor Intervened**\n\n"
            f"{safety_result.blocked_reason}\n"
        )
        if safety_result.warnings:
            msg += "\n**Warnings:**\n" + "\n".join(f"• {w}" for w in safety_result.warnings)
        if safety_result.safe_alternative:
            msg += f"\n\n**Suggested Alternative:** {safety_result.safe_alternative}"
        return msg

    def format_error(self, error: str) -> str:
        """Format a system error response."""
        return (
            f"**⚠ System Exception**\n\n"
            f"An internal error occurred during processing:\n"
            f"`{error}`\n\n"
            f"The incident has been logged. Stability maintained. "
            f"Please rephrase or try again."
        )
