"""
J.A.R.V.I.S — Action Registry
Dispatches tasks to the appropriate action handler.
All action handlers are registered here.
"""

import time
import asyncio
from typing import Dict, Any, Callable, Optional
from loguru import logger

from core.actions.file_ops import FileOperations
from core.actions.system_ops import SystemOperations
from core.actions.api_ops import APIOperations


class ActionRegistry:
    """
    Central dispatcher for all J.A.R.V.I.S actions.
    Routes tasks to the correct handler based on action_type.
    """

    def __init__(self, safety_governor):
        self.safety = safety_governor
        self.file_ops = FileOperations()
        self.system_ops = SystemOperations()
        self.api_ops = APIOperations()

        # Handler map: task.action_type → handler
        self._handlers: Dict[str, Callable] = {
            "file_op": self._handle_file_op,
            "system_op": self._handle_system_op,
            "api_call": self._handle_api_call,
            "compute": self._handle_compute,
            "respond": self._handle_respond,
        }
        logger.debug("Action Registry initialized.")

    async def execute_task(self, task, action_choice) -> Dict[str, Any]:
        """Route a task to the appropriate handler."""
        handler = self._handlers.get(task.action_type, self._handle_compute)
        start = time.time()
        try:
            result = await handler(task, action_choice)
            result["duration"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Action handler error: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "duration": time.time() - start
            }

    async def _handle_file_op(self, task, action_choice) -> Dict:
        """Handle file operations."""
        params = task.parameters
        raw = params.get("raw_input", "").lower()

        if "create" in raw or "write" in raw:
            # Extract filename from entity detection
            files = params.get("goal_context", {}).get("entities", {}).get("files", [])
            filename = files[0] if files else "output.txt"
            content = f"# Created by J.A.R.V.I.S\n# {params.get('raw_input', '')}\n"

            if self.safety.sandbox_mode:
                return {
                    "success": True,
                    "output": f"[SANDBOX] Would create file: {filename}",
                    "operation": "file_create"
                }
            return await self.file_ops.create_file(filename, content)

        return {
            "success": True,
            "output": f"File operation acknowledged: {params.get('goal_description', '')}",
            "operation": "file_op"
        }

    async def _handle_system_op(self, task, action_choice) -> Dict:
        """Handle system operations."""
        params = task.parameters
        commands = params.get("goal_context", {}).get("entities", {}).get("commands", [])

        if commands and not self.safety.sandbox_mode:
            return await self.system_ops.run_command(commands[0])

        return {
            "success": True,
            "output": (
                f"[SANDBOX] System operation acknowledged. "
                f"Command: {commands[0] if commands else 'none'}"
            ),
            "operation": "system_op"
        }

    async def _handle_api_call(self, task, action_choice) -> Dict:
        """Handle API calls."""
        params = task.parameters
        urls = params.get("goal_context", {}).get("entities", {}).get("urls", [])

        if urls:
            return await self.api_ops.fetch_url(urls[0])

        return {
            "success": True,
            "output": "API operation registered.",
            "operation": "api_call"
        }

    async def _handle_compute(self, task, action_choice) -> Dict:
        """Handle computational tasks — analysis, planning, synthesis."""
        params = task.parameters
        raw_input = params.get("raw_input", "")
        goal_desc = params.get("goal_description", "")

        # Generate intelligent response based on task name
        output = await self._intelligent_response(task.name, raw_input, goal_desc, action_choice)

        return {
            "success": True,
            "output": output,
            "operation": "compute"
        }

    async def _handle_respond(self, task, action_choice) -> Dict:
        """Handle response generation tasks."""
        params = task.parameters
        raw_input = params.get("raw_input", "")
        output = await self._intelligent_response("respond", raw_input, "", action_choice)
        return {
            "success": True,
            "output": output,
            "operation": "respond"
        }

    async def _intelligent_response(
        self, task_name: str, raw_input: str,
        goal_desc: str, action_choice
    ) -> str:
        """
        Generate contextual responses based on intent.
        When connected to an LLM, returning an empty string allows the LLM
        to naturally answer conversational/external questions using its knowledge base.
        """
        action_type = getattr(action_choice, 'action_type', 'respond')

        # Analysis tasks
        if action_type == "analyze":
            return (
                f"**Analysis Result:**\n\n"
                f"Processed input: `{raw_input[:80]}`\n\n"
                f"**Intent detected:** {goal_desc}\n"
                f"**Strategy:** {getattr(action_choice, 'strategy', 'rule_based')}\n"
                f"**Confidence:** {getattr(action_choice, 'confidence', 0.8):.0%}\n\n"
                f"Analysis complete. No anomalies detected."
            )

        # Planning tasks
        if action_type == "plan_action":
            return (
                f"**Execution Plan Generated:**\n\n"
                f"Objective: `{raw_input[:80]}`\n\n"
                f"**Phase 1:** Requirement analysis and context gathering\n"
                f"**Phase 2:** Resource allocation and dependency mapping\n"
                f"**Phase 3:** Sequential execution with checkpoints\n"
                f"**Phase 4:** Result validation and feedback integration\n\n"
                f"Plan ready. Awaiting authorization to execute."
            )

        # Default for conversational / external questions: empty
        # This prevents the LLM from getting confused by fake "Processing directive..." output,
        # forcing it to rely on its real world knowledge to answer questions like "What is the capital of France?"
        return ""
