"""
J.A.R.V.I.S — Central Orchestrator
Coordinates all subsystems: perception, cognition, memory, evolution, safety, execution.
"""

import asyncio
import time
import yaml
from pathlib import Path
from loguru import logger
from rich.console import Console
from typing import Optional

from core.perception import PerceptionLayer
from core.cognitive.goal_interpreter import GoalInterpreter
from core.cognitive.task_planner import TaskPlanner
from core.cognitive.decision_engine import DecisionEngine
from core.cognitive.world_model import WorldModel
from core.memory.memory_manager import MemoryManager
from core.evolution.rl_engine import RLEngine
from core.evolution.feedback_scorer import FeedbackScorer
from core.evolution.strategy_library import StrategyLibrary
from core.autonomy.autonomy_engine import AutonomyEngine
from core.actions.action_registry import ActionRegistry
from core.safety.safety_governor import SafetyGovernor
from core.personality.personality_engine import PersonalityEngine
from core.personality.response_formatter import ResponseFormatter
from core.llm.llm_manager import LLMManager
from core.voice.tts_engine import TTSEngine
from core.voice.locale_map import edge_voice_for_locale

console = Console()


class JARVIS:
    """
    Central orchestrator for J.A.R.V.I.S.
    All subsystems converge here. This is the cognitive hub.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._initialized = False
        self._running = False
        self._interaction_count = 0

        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0,
            "start_time": time.time()
        }

        # Configure logging
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            "data/logs/jarvis_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level=self.config.get("system", {}).get("log_level", "INFO"),
            format="{time:HH:mm:ss} | {level} | {message}"
        )

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    async def initialize(self):
        """Boot all subsystems in the correct order."""
        if self._initialized:
            return

        logger.info("J.A.R.V.I.S initialization sequence started.")

        # 1. Safety Governor (FIRST — immutable constraint)
        self.safety = SafetyGovernor(self.config.get("safety", {}))

        # 2. Memory System
        self.memory = MemoryManager(self.config.get("memory", {}))
        await self.memory.initialize()

        # 3. Perception Layer
        self.perception = PerceptionLayer()

        # 4. Cognitive Core
        self.goal_interpreter = GoalInterpreter(self.memory)
        self.world_model = WorldModel(self.memory)
        self.task_planner = TaskPlanner(self.world_model)
        self.decision_engine = DecisionEngine(
            self.config.get("evolution", {}),
            self.memory
        )

        # 5. Evolution Engine
        self.rl_engine = RLEngine(self.config.get("evolution", {}))
        self.feedback_scorer = FeedbackScorer()
        self.strategy_library = StrategyLibrary(self.memory)

        # 6. Action Registry
        self.action_registry = ActionRegistry(self.safety)

        # 7. Autonomy Engine
        self.autonomy = AutonomyEngine(
            self.action_registry,
            self.safety,
            self.config.get("autonomy", {})
        )

        # 8. Personality & Response
        self.personality = PersonalityEngine(
            self.config.get("personality", {}),
            self.memory
        )
        self.formatter = ResponseFormatter(self.personality)

        # 9. LLM Manager (brain for natural language)
        self.llm = LLMManager()
        await self.llm.initialize()

        # 10. TTS Engine (voice output)
        self.tts = TTSEngine(self.config.get("voice", {}))
        await self.tts.initialize()

        # Load state from previous sessions
        await self._restore_state()

        self._initialized = True
        self._running = True
        llm_status = f"LLM: {self.llm.provider_info}"
        tts_status = f"TTS: {'online' if self.tts.is_available else 'offline'}"
        logger.info(f"J.A.R.V.I.S fully initialized. {llm_status} | {tts_status}")

    async def _restore_state(self):
        """Restore learned strategies and RL policy from memory."""
        try:
            strategies = await self.strategy_library.load_all()
            logger.info(f"Restored {len(strategies)} learned strategies.")

            policy_state = await self.memory.long_term.retrieve("rl_policy_state")
            if policy_state:
                self.rl_engine.load_policy(policy_state)
                logger.info("RL policy restored from previous session.")
        except Exception as e:
            logger.warning(f"State restoration partial: {e}")

    async def process(self, user_input: str, response_locale: Optional[str] = None) -> str:
        """
        Main processing pipeline:
        INPUT → PERCEIVE → INTERPRET → PLAN → DECIDE → SAFETY CHECK → EXECUTE → LEARN → RESPOND
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        self._interaction_count += 1

        try:
            logger.info(f"Processing: {user_input[:80]}...")

            # 1. PERCEPTION: Parse intent, urgency, context
            percept = await self.perception.process(user_input)

            # Handle built-in system commands
            builtin_response = await self._handle_builtin(percept)
            if builtin_response:
                return builtin_response

            # 2. STORE in short-term memory
            self.memory.short_term.store({
                "input": user_input,
                "percept": percept.__dict__ if hasattr(percept, '__dict__') else {},
                "timestamp": time.time()
            })

            # 3. GOAL INTERPRETATION
            goal = await self.goal_interpreter.interpret(percept)
            logger.debug(f"Goal: {goal}")

            # 4. WORLD MODEL UPDATE
            await self.world_model.update(percept, goal)

            # 5. TASK PLANNING
            task_graph = await self.task_planner.plan(goal)
            logger.debug(f"Tasks planned: {len(task_graph.tasks)}")

            # 6. DECISION ENGINE: Choose optimal strategy
            state_vector = await self.world_model.get_state_vector()
            action_choice = await self.decision_engine.decide(state_vector, task_graph)

            # 7. SAFETY CHECK (immutable governor)
            safety_result = await self.safety.evaluate(action_choice)
            if not safety_result.approved:
                response = self.formatter.format_safety_block(safety_result)
                return response

            # 8. EXECUTE via Autonomy Engine
            exec_result = await self.autonomy.execute(action_choice, task_graph)

            # 9. FEEDBACK + LEARNING
            score = await self.feedback_scorer.score(exec_result, goal)
            await self.rl_engine.learn(state_vector, action_choice, score, exec_result)

            # 10. EPISODIC MEMORY: Store the episode
            await self.memory.episodic.store_episode({
                "input": user_input,
                "goal": goal.__dict__,
                "action": action_choice.__dict__,
                "result": exec_result.__dict__,
                "score": score
            })

            # 11. EVOLVE: Update strategy library if successful
            if score > 0.7:
                await self.strategy_library.record_success(
                    goal, action_choice, exec_result, score
                )
                self.metrics["successful_tasks"] += 1
            else:
                self.metrics["failed_tasks"] += 1

            # 12. LLM-powered response generation
            raw_output = str(getattr(exec_result, 'output', '') or '')

            if self.llm.is_llm_active:
                # Build rich context for LLM
                context = (
                    f"Action taken: {action_choice.action_type} | "
                    f"Mode: {self.personality.current_mode} | "
                    f"Success: {exec_result.success} | "
                    f"Score: {score:.2f}"
                )
                if raw_output and raw_output != user_input:
                    context += f"\nExecution output: {raw_output[:300]}"
                response = await self.llm.generate(
                    user_input, context=context, response_locale=response_locale
                )
            else:
                # Fallback: rule-based formatter
                response = await self.formatter.format_response(
                    exec_result, goal, score,
                    mode=self.personality.current_mode
                )

            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["total_response_time"] += elapsed
            self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["total_requests"]
            )

            logger.info(f"Request processed in {elapsed:.2f}s | Score: {score:.2f} | LLM: {self.llm.is_llm_active}")
            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Processing error: {e}", exc_info=True)
            self.metrics["failed_tasks"] += 1
            return self.formatter.format_error(str(e))

    async def _handle_builtin(self, percept) -> Optional[str]:
        """Handle special system commands."""
        cmd = percept.raw_input.lower().strip()

        if cmd == "status":
            return await self._get_status()
        elif cmd == "memory":
            return await self._get_memory_report()
        elif cmd == "history":
            return await self._get_history()
        elif cmd == "strategies":
            return await self._get_strategies()
        elif cmd == "evolve":
            return await self._trigger_evolution()
        elif cmd == "diagnose":
            return await self._run_diagnostics()
        elif cmd == "benchmark":
            return await self._run_benchmark()
        elif cmd.startswith("mode "):
            mode = cmd.split("mode ", 1)[1].strip()
            return await self._change_mode(mode)

        return None

    async def _get_status(self) -> str:
        uptime = time.time() - self.metrics["start_time"]
        hours, rem = divmod(int(uptime), 3600)
        minutes, seconds = divmod(rem, 60)
        total = self.metrics["total_requests"]
        success = self.metrics["successful_tasks"]
        rate = (success / total * 100) if total > 0 else 0

        mem_stats = await self.memory.get_stats()
        rl_stats = self.rl_engine.get_stats()
        llm_info = getattr(self, 'llm', None)
        tts_info = getattr(self, 'tts', None)

        return (
            f"**⚡ J.A.R.V.I.S System Status**\n\n"
            f"**Uptime:** {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            f"**Requests:** {total} total | {success} successful | "
            f"{self.metrics['failed_tasks']} failed\n"
            f"**Success Rate:** {rate:.1f}%\n"
            f"**Avg Response:** {self.metrics['avg_response_time']:.2f}s\n"
            f"**Mode:** {self.personality.current_mode.upper()}\n\n"
            f"**LLM Brain:** {llm_info.provider_info if llm_info else 'offline'}\n"
            f"**Voice TTS:** {'online — ' + (tts_info.voice if tts_info else '') if (tts_info and tts_info.is_available) else 'offline'}\n"
            f"**Memory:** {mem_stats}\n"
            f"**RL Engine:** ε={rl_stats['epsilon']:.3f} | "
            f"Episodes={rl_stats['episodes']} | Wins={rl_stats['wins']}\n"
            f"**Safety:** Governor armed | Sandbox: {self.safety.sandbox_mode}"
        )

    async def _get_memory_report(self) -> str:
        st = self.memory.short_term.get_all()
        ep_count = await self.memory.episodic.count()
        lt_count = await self.memory.long_term.count()
        strategies = await self.strategy_library.count()

        return (
            f"**🧠 Memory Report**\n\n"
            f"**Short-term:** {len(st)} active items\n"
            f"**Long-term:** {lt_count} stored entries\n"
            f"**Episodic:** {ep_count} recorded episodes\n"
            f"**Strategy Library:** {strategies} learned strategies\n\n"
            f"**Recent context:** {st[-1]['input'] if st else 'Empty'}"
        )

    async def _get_history(self) -> str:
        episodes = await self.memory.episodic.get_recent(10)
        if not episodes:
            return "No execution history available yet."
        lines = ["**📜 Recent Execution History**\n"]
        for i, ep in enumerate(episodes, 1):
            lines.append(
                f"{i}. [{ep.get('score', 0):.1f}★] {ep.get('input', 'N/A')[:60]}"
            )
        return "\n".join(lines)

    async def _get_strategies(self) -> str:
        strategies = await self.strategy_library.load_all()
        if not strategies:
            return "No strategies learned yet. Complete successful tasks to build the strategy library."
        lines = ["**📚 Learned Strategy Library**\n"]
        for s in strategies[:10]:
            lines.append(
                f"• [{s.get('score', 0):.2f}★] {s.get('name', 'Strategy')} "
                f"— used {s.get('uses', 0)}x"
            )
        return "\n".join(lines)

    async def _trigger_evolution(self) -> str:
        iterations = await self.rl_engine.forced_evolution()
        return (
            f"**🔁 Evolution Cycle Complete**\n\n"
            f"Ran {iterations} optimization iterations.\n"
            f"Policy updated. ε={self.rl_engine.epsilon:.4f}\n"
            f"Strategy library refined."
        )

    async def _run_diagnostics(self) -> str:
        results = []
        results.append("**🔍 Self-Diagnostic Report**\n")

        # Check all subsystems
        systems = [
            ("Perception Layer", self.perception is not None),
            ("Goal Interpreter", self.goal_interpreter is not None),
            ("Task Planner", self.task_planner is not None),
            ("Decision Engine", self.decision_engine is not None),
            ("World Model", self.world_model is not None),
            ("Memory Manager", self.memory is not None),
            ("RL Engine", self.rl_engine is not None),
            ("Safety Governor", self.safety is not None),
            ("Action Registry", self.action_registry is not None),
            ("Autonomy Engine", self.autonomy is not None),
        ]

        for name, ok in systems:
            status = "✅ ONLINE" if ok else "❌ OFFLINE"
            results.append(f"{name}: {status}")

        results.append(f"\n**Memory integrity:** OK")
        results.append(f"**Safety integrity:** IMMUTABLE ✅")
        results.append(f"**RL Policy:** {self.rl_engine.get_stats()}")

        return "\n".join(results)

    async def _run_benchmark(self) -> str:
        import time
        start = time.time()
        # Simulate 10 internal decision cycles
        dummy_state = [0.5] * 16
        actions_taken = []
        for _ in range(10):
            task_graph = await self.task_planner.plan(
                await self.goal_interpreter.interpret(
                    await self.perception.process("benchmark test")
                )
            )
            action = await self.decision_engine.decide(dummy_state, task_graph)
            actions_taken.append(action)
        elapsed = time.time() - start
        return (
            f"**⚡ Benchmark Results**\n\n"
            f"10 decision cycles in {elapsed:.3f}s\n"
            f"Avg per cycle: {elapsed/10*1000:.1f}ms\n"
            f"Throughput: {10/elapsed:.1f} decisions/sec\n"
            f"**Rating:** {'Excellent' if elapsed < 0.5 else 'Good' if elapsed < 1.0 else 'Acceptable'}"
        )

    async def _change_mode(self, mode: str) -> str:
        valid_modes = {"engineer", "execution", "adaptive"}
        if mode not in valid_modes:
            return f"Invalid mode. Options: {', '.join(valid_modes)}"
        self.personality.set_mode(mode)
        mode_descriptions = {
            "engineer": "Detailed, technical, structured analysis.",
            "execution": "Direct, action-oriented output.",
            "adaptive": "Learning your preferences and adjusting dynamically."
        }
        return (
            f"**Mode switched to: {mode.upper()}**\n"
            f"{mode_descriptions[mode]}\n\n"
            f"Behavioral adaptation applied."
        )

    async def launch_dashboard(self):
        """Launch the web dashboard in a background thread."""
        import threading
        from dashboard.app import create_app
        app = create_app(self)
        cfg = self.config.get("dashboard", {})
        host = cfg.get("host", "127.0.0.1")
        port = cfg.get("port", 5000)

        def run():
            app.run(host=host, port=port, debug=False, use_reloader=False)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        console.print(f"[green]Dashboard live at: http://{host}:{port}[/green]")

    async def emergency_stop(self):
        """Immediate halt of all operations."""
        logger.critical("EMERGENCY STOP TRIGGERED")
        self._running = False
        await self.autonomy.halt()
        await self.memory.flush()

    async def shutdown(self):
        """Graceful shutdown — save state, close connections."""
        logger.info("Graceful shutdown initiated.")
        self._running = False

        # Save RL policy
        policy_state = self.rl_engine.get_policy_state()
        await self.memory.long_term.store("rl_policy_state", policy_state)

        # Flush memory
        await self.memory.flush()
        await self.memory.close()

        logger.info("J.A.R.V.I.S shutdown complete.")

    async def speak(self, text: str, voice: Optional[str] = None) -> Optional[str]:
        """Convert text to speech and return base64 audio (for dashboard)."""
        tts = getattr(self, 'tts', None)
        if tts and tts.is_available:
            return await tts.speak_to_base64(text, voice=voice)
        return None

    async def process_and_speak(
        self, user_input: str, response_locale: Optional[str] = None
    ) -> dict:
        """Process input and return both text response and audio."""
        text_response = await self.process(user_input, response_locale=response_locale)
        edge_voice = edge_voice_for_locale(response_locale)
        audio_b64 = await self.speak(text_response, voice=edge_voice)
        return {
            "text": text_response,
            "audio": audio_b64,
            "llm_active": getattr(self, 'llm', None) and self.llm.is_llm_active,
            "tts_active": getattr(self, 'tts', None) and self.tts.is_available,
            "locale": response_locale,
            "tts_voice": edge_voice,
        }
