"""
J.A.R.V.I.S — Task Planner
Breaks goals into sub-tasks and creates execution graphs.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from loguru import logger


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class Task:
    """A single executable task unit."""
    id: str
    name: str
    description: str
    action_type: str        # file_op | system_op | api_call | compute | respond
    parameters: Dict[str, Any]
    dependencies: List[str]  # IDs of tasks that must complete first
    priority: float
    max_retries: int = 2
    timeout: float = 30.0
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0


@dataclass
class TaskGraph:
    """Directed acyclic graph of tasks for goal execution."""
    goal_id: str
    tasks: List[Task]
    execution_order: List[str]  # topologically sorted task IDs
    estimated_duration: float
    created_at: float = field(default_factory=time.time)

    def get_task(self, task_id: str) -> Optional[Task]:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    def ready_tasks(self) -> List[Task]:
        """Return tasks whose dependencies are all complete."""
        completed_ids = {
            t.id for t in self.tasks
            if t.status == TaskStatus.COMPLETED
        }
        return [
            t for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def is_complete(self) -> bool:
        return all(
            t.status in {TaskStatus.COMPLETED, TaskStatus.SKIPPED}
            for t in self.tasks
        )

    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks)


class TaskPlanner:
    """
    Decomposes goals into executable task graphs.
    Handles dependency resolution and execution ordering.
    """

    # Templates for common goal types
    PLAN_TEMPLATES = {
        "create_file": [
            ("validate_path", "Validate target path", "compute", {}),
            ("create_file", "Create file with content", "file_op", {}),
            ("verify_creation", "Verify file created", "compute", {}),
        ],
        "execute_command": [
            ("risk_check", "Assess command risk", "compute", {}),
            ("execute", "Execute command", "system_op", {}),
            ("capture_output", "Capture and format output", "compute", {}),
        ],
        "analyze_text": [
            ("preprocess", "Preprocess input text", "compute", {}),
            ("analyze", "Run analysis", "compute", {}),
            ("synthesize", "Synthesize results", "compute", {}),
        ],
        "respond": [
            ("context_gather", "Gather context", "compute", {}),
            ("generate_response", "Generate response", "compute", {}),
        ],
        "default": [
            ("interpret", "Interpret request", "compute", {}),
            ("execute", "Execute action", "compute", {}),
            ("respond", "Format response", "respond", {}),
        ]
    }

    def __init__(self, world_model):
        self.world_model = world_model
        self._plan_counter = 0
        logger.debug("Task Planner online.")

    async def plan(self, goal) -> TaskGraph:
        """Create an execution graph for the given goal."""
        self._plan_counter += 1

        # Select appropriate template
        template_key = self._select_template(goal)
        template = self.PLAN_TEMPLATES.get(template_key, self.PLAN_TEMPLATES["default"])

        # Build tasks from template
        tasks = []
        prev_id = None

        for i, (name, desc, action_type, params) in enumerate(template):
            task_id = f"{goal.id}-T{i:02d}-{name}"
            task = Task(
                id=task_id,
                name=name,
                description=desc,
                action_type=action_type,
                parameters={
                    **params,
                    "goal_context": goal.context,
                    "goal_description": goal.description,
                    "raw_input": goal.raw_intent
                },
                dependencies=[prev_id] if prev_id else [],
                priority=goal.priority,
                timeout=self._estimate_timeout(action_type, goal.estimated_complexity)
            )
            tasks.append(task)
            prev_id = task_id

        # Topological sort (sequential for now, parallel supported)
        execution_order = self._topological_sort(tasks)

        # Estimate total duration
        estimated_duration = sum(
            self._estimate_timeout(t.action_type, goal.estimated_complexity)
            for t in tasks
        )

        graph = TaskGraph(
            goal_id=goal.id,
            tasks=tasks,
            execution_order=execution_order,
            estimated_duration=estimated_duration
        )

        logger.debug(
            f"Plan created: {len(tasks)} tasks for goal {goal.id} "
            f"(template={template_key})"
        )
        return graph

    def _select_template(self, goal) -> str:
        intent = goal.context.get("intent", "unknown")
        raw = goal.raw_intent.lower()

        if intent == "create":
            if any(w in raw for w in ["file", ".txt", ".py", ".json"]):
                return "create_file"
            return "default"
        elif intent in {"execute", "run"}:
            return "execute_command"
        elif intent in {"analyze", "read", "status"}:
            return "analyze_text"
        elif intent in {"help", "learn", "predict"}:
            return "respond"
        else:
            return "default"

    def _estimate_timeout(self, action_type: str, complexity: str) -> float:
        base = {"file_op": 5.0, "system_op": 15.0, "api_call": 10.0,
                "compute": 3.0, "respond": 2.0}.get(action_type, 5.0)
        multiplier = {"simple": 1.0, "moderate": 1.5, "complex": 2.5}.get(complexity, 1.0)
        return base * multiplier

    def _topological_sort(self, tasks: List[Task]) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree = {t.id: len(t.dependencies) for t in tasks}
        graph = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in graph:
                    graph[dep].append(task.id)

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order
