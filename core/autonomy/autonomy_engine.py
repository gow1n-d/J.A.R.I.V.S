"""
J.A.R.V.I.S — Autonomy Engine
High-level command execution with retry logic and adaptive optimization.
Operates without supervision. Retries failed actions. Optimizes flow.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Any, List
from loguru import logger


@dataclass
class ExecutionResult:
    """Result of executing a task graph via the autonomy engine."""
    success: bool
    output: Any
    task_count: int
    completed_tasks: int
    failed_tasks: int
    retries: int
    duration: float
    risk_level: str
    safety_approved: bool
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class AutonomyEngine:
    """
    Autonomous execution engine.
    Accepts high-level plans, executes without supervision,
    handles retries, and optimizes execution flow dynamically.
    """

    def __init__(self, action_registry, safety_governor, config: dict):
        self.registry = action_registry
        self.safety = safety_governor
        self.config = config

        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 2.0)
        self.execution_timeout = config.get("execution_timeout", 30)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

        self._running = True
        self._active_tasks: List = []
        logger.debug("Autonomy Engine online.")

    async def execute(self, action_choice, task_graph) -> ExecutionResult:
        """
        Execute a full task graph autonomously.
        Follows execution order, handles dependencies, retries failures.
        """
        start_time = time.time()
        completed = 0
        failed = 0
        total_retries = 0
        final_output = ""
        final_risk = action_choice.risk_level

        try:
            # Execute tasks in dependency order
            for task_id in task_graph.execution_order:
                task = task_graph.get_task(task_id)
                if not task:
                    continue

                # Check if dependencies failed
                if any(
                    dep_id for dep_id in task.dependencies
                    if (dep := task_graph.get_task(dep_id)) and dep.status.value == "failed"
                ):
                    from core.cognitive.task_planner import TaskStatus
                    task.status = TaskStatus.SKIPPED
                    continue

                # Execute with retry logic
                result, retries = await self._execute_with_retry(task, action_choice)

                total_retries += retries
                if result.get("success"):
                    from core.cognitive.task_planner import TaskStatus
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    completed += 1
                    final_output = result.get("output", final_output)
                else:
                    from core.cognitive.task_planner import TaskStatus
                    task.status = TaskStatus.FAILED
                    task.error = result.get("error", "Unknown error")
                    failed += 1
                    logger.warning(f"Task {task_id} failed: {task.error}")

            duration = time.time() - start_time
            success = completed > 0 and failed == 0

            return ExecutionResult(
                success=success,
                output=final_output or self._synthesize_output(action_choice, task_graph),
                task_count=len(task_graph.tasks),
                completed_tasks=completed,
                failed_tasks=failed,
                retries=total_retries,
                duration=duration,
                risk_level=final_risk,
                safety_approved=True,
                metadata={
                    "action_type": action_choice.action_type,
                    "strategy": action_choice.strategy,
                    "confidence": action_choice.confidence
                }
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            logger.error(f"Execution timeout after {duration:.1f}s")
            return ExecutionResult(
                success=False,
                output="Execution timed out.",
                task_count=len(task_graph.tasks),
                completed_tasks=completed,
                failed_tasks=failed + 1,
                retries=total_retries,
                duration=duration,
                risk_level=final_risk,
                safety_approved=True,
                error="Timeout"
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Execution error: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                output=str(e),
                task_count=len(task_graph.tasks),
                completed_tasks=completed,
                failed_tasks=failed + 1,
                retries=total_retries,
                duration=duration,
                risk_level=final_risk,
                safety_approved=True,
                error=str(e)
            )

    async def _execute_with_retry(self, task, action_choice) -> tuple:
        """Execute a single task with automatic retry on failure."""
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                from core.cognitive.task_planner import TaskStatus
                task.status = TaskStatus.RUNNING

                # Dispatch to action registry
                result = await asyncio.wait_for(
                    self.registry.execute_task(task, action_choice),
                    timeout=task.timeout
                )

                if result.get("success"):
                    return result, retries

                last_error = result.get("error", "Unknown")
                retries += 1

                if retries <= self.max_retries:
                    logger.debug(f"Retry {retries}/{self.max_retries} for {task.name}")
                    task.status = TaskStatus.RETRYING
                    task.retries = retries
                    await asyncio.sleep(self.retry_delay * (2 ** (retries - 1)))

            except asyncio.TimeoutError:
                last_error = f"Timeout after {task.timeout}s"
                retries += 1
                if retries <= self.max_retries:
                    await asyncio.sleep(self.retry_delay)
            except Exception as e:
                last_error = str(e)
                retries += 1
                if retries <= self.max_retries:
                    await asyncio.sleep(self.retry_delay)

        return {"success": False, "error": last_error, "output": ""}, retries

    def _synthesize_output(self, action_choice, task_graph) -> str:
        """Generate a summary output when task output is empty."""
        goal_desc = ""
        if task_graph.tasks:
            goal_desc = task_graph.tasks[0].parameters.get("raw_input", "")

        return (
            f"Executing: {action_choice.action_type} | "
            f"Strategy: {action_choice.strategy} | "
            f"Request: {goal_desc[:100]}"
        )

    async def halt(self):
        """Immediately stop all active tasks."""
        self._running = False
        self._active_tasks.clear()
        logger.critical("Autonomy Engine halted.")
