"""
J.A.R.V.I.S — Test Suite
Validates all core modules independently.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ─── Perception Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_perception_basic():
    from core.perception import PerceptionLayer
    p = PerceptionLayer()
    percept = await p.process("Create a file called test.txt")
    assert percept.intent == "create"
    assert percept.action_required == True
    assert "test.txt" in percept.entities.get("files", [])

@pytest.mark.asyncio
async def test_perception_question():
    from core.perception import PerceptionLayer
    p = PerceptionLayer()
    percept = await p.process("What is the status of the system?")
    assert percept.question == True
    assert percept.intent in {"status", "analyze", "read"}

@pytest.mark.asyncio
async def test_perception_urgency():
    from core.perception import PerceptionLayer
    p = PerceptionLayer()
    percept = await p.process("URGENT: run this immediately!")
    assert percept.urgency in {"high", "critical"}

@pytest.mark.asyncio
async def test_perception_empty():
    from core.perception import PerceptionLayer
    p = PerceptionLayer()
    percept = await p.process("")
    assert percept.intent == "unknown"


# ─── Memory Tests ─────────────────────────────────────────────────────────────

def test_short_term_memory():
    from core.memory.short_term import ShortTermMemory
    stm = ShortTermMemory(capacity=5)
    for i in range(7):
        stm.store({"input": f"item_{i}"})
    # capacity is 5 — should have only 5 items
    assert len(stm) == 5

def test_short_term_search():
    from core.memory.short_term import ShortTermMemory
    stm = ShortTermMemory()
    stm.store({"input": "analyze performance metrics"})
    stm.store({"input": "create a report"})
    results = stm.search("analyze")
    assert len(results) >= 1

@pytest.mark.asyncio
async def test_long_term_memory():
    from core.memory.long_term import LongTermMemory
    ltm = LongTermMemory(db_path=":memory:")
    # Use in-memory SQLite for tests
    ltm._conn = __import__('sqlite3').connect(":memory:")
    ltm._conn.row_factory = __import__('sqlite3').Row
    await ltm._create_tables()

    await ltm.store("test_key", {"value": 42})
    result = await ltm.retrieve("test_key")
    assert result == {"value": 42}

    count = await ltm.count()
    assert count == 1

    await ltm.delete("test_key")
    assert await ltm.retrieve("test_key") is None


# ─── Safety Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_safety_low_risk_approved():
    from core.safety.safety_governor import SafetyGovernor
    from dataclasses import dataclass

    @dataclass
    class MockAction:
        action_type = "respond"
        risk_level = "low"
        rationale = "test"
        parameters = {}

    gov = SafetyGovernor({"sandbox_mode": True, "max_risk_level": "medium"})
    result = await gov.evaluate(MockAction())
    assert result.approved == True

@pytest.mark.asyncio
async def test_safety_blocked_pattern():
    from core.safety.safety_governor import SafetyGovernor
    from dataclasses import dataclass

    @dataclass
    class DangerousAction:
        action_type = "execute_action"
        risk_level = "high"
        rationale = "test"
        parameters = {"cmd": "rm -rf /"}

    gov = SafetyGovernor({"sandbox_mode": True, "max_risk_level": "medium"})
    result = await gov.evaluate(DangerousAction())
    assert result.approved == False
    assert result.risk_level == "critical"


# ─── RL Engine Tests ──────────────────────────────────────────────────────────

def test_rl_engine_init():
    from core.evolution.rl_engine import RLEngine
    config = {
        "learning_rate": 0.001,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "discount_factor": 0.95,
        "replay_buffer_size": 100,
        "batch_size": 4,
        "train_frequency": 2,
        "target_update_frequency": 10
    }
    rl = RLEngine(config)
    stats = rl.get_stats()
    assert stats["epsilon"] == 1.0
    assert stats["episodes"] == 0

def test_experience_replay_buffer():
    from core.evolution.rl_engine import ExperienceReplayBuffer, Experience
    buf = ExperienceReplayBuffer(capacity=10)
    for i in range(15):
        exp = Experience(
            state=[0.0]*16, action=0, reward=float(i),
            next_state=[0.0]*16, done=False
        )
        buf.push(exp)
    assert len(buf) == 10  # capped at capacity

def test_rl_policy_serialization():
    from core.evolution.rl_engine import RLEngine
    config = {"learning_rate": 0.001, "epsilon": 0.5, "epsilon_min": 0.01,
              "epsilon_decay": 0.99, "discount_factor": 0.95,
              "replay_buffer_size": 100, "batch_size": 4,
              "train_frequency": 2, "target_update_frequency": 10}
    rl = RLEngine(config)
    state = rl.get_policy_state()
    assert "epsilon" in state
    assert "q_network_state" in state
    # Load back
    rl2 = RLEngine(config)
    rl2.load_policy(state)
    assert abs(rl2.epsilon - 0.5) < 0.001


# ─── Cognitive Tests ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_goal_interpreter():
    from core.perception import PerceptionLayer
    from core.memory.short_term import ShortTermMemory

    class MockMemory:
        short_term = ShortTermMemory()
        async def retrieve(self, k): return None

    from core.cognitive.goal_interpreter import GoalInterpreter
    gi = GoalInterpreter(MockMemory())
    perc = await PerceptionLayer().process("Create a new Python script")
    goal = await gi.interpret(perc)
    assert goal.goal_type == "achieve"
    assert goal.priority > 0

@pytest.mark.asyncio
async def test_task_planner():
    from core.perception import PerceptionLayer
    from core.cognitive.goal_interpreter import GoalInterpreter
    from core.cognitive.task_planner import TaskPlanner
    from core.memory.short_term import ShortTermMemory

    class MockMemory:
        short_term = ShortTermMemory()
        async def retrieve(self, k): return None

    class MockWorldModel:
        pass

    gi = GoalInterpreter(MockMemory())
    tp = TaskPlanner(MockWorldModel())

    perc = await PerceptionLayer().process("Analyze the system performance")
    goal = await gi.interpret(perc)
    graph = await tp.plan(goal)

    assert len(graph.tasks) > 0
    assert len(graph.execution_order) == len(graph.tasks)


# ─── Integration Test ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jarvis_boot_and_respond():
    """Full integration test: boot J.A.R.V.I.S and process a simple message."""
    from core.jarvis import JARVIS
    j = JARVIS()
    await j.initialize()
    response = await j.process("Hello, what can you do?")
    assert response is not None
    assert len(response) > 0
    await j.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
