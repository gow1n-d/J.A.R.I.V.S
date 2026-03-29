# 🧠 J.A.R.V.I.S — System Architecture

> **Just A Rather Very Intelligent System** — Autonomous • Self-Evolving • Controlled

---

## ⚡ Processing Pipeline

```
USER INPUT
  ↓ PerceptionLayer      → intent, urgency, entities, sentiment, domain
  ↓ GoalInterpreter      → goal_type, priority, constraints, success_criteria
  ↓ WorldModel.update()  → 16D state vector (CPU, RAM, success_rate...)
  ↓ TaskPlanner          → Task DAG, topological order, timeouts
  ↓ DecisionEngine       → ActionChoice (DQN + Rules, epsilon-greedy)
  ↓ SafetyGovernor       → risk check, confirmation gate, audit log
  ↓ AutonomyEngine       → execute with retry + timeout handling
  ↓ ActionRegistry       → route to FileOps / SystemOps / APIops / Compute
  ↓ FeedbackScorer       → composite reward [0,1]
  ↓ RLEngine.learn()     → DQN gradient step, epsilon decay
  ↓ EpisodicMemory       → store full episode to SQLite
  ↓ StrategyLibrary      → archive successful patterns
  ↓ ResponseFormatter    → personality-mode-aware formatted response
  ↓ USER RESPONSE
```

---

## 📁 Folder Structure

```
J.A.R.V.I.S/
├── main.py                       # Rich CLI entry point
├── config.yaml                   # System configuration
├── requirements.txt              # Python dependencies
├── core/
│   ├── jarvis.py                 # Central orchestrator
│   ├── perception.py             # Intent/urgency/entity extraction
│   ├── cognitive/
│   │   ├── goal_interpreter.py   # Percept → Goal
│   │   ├── task_planner.py       # Goal → Task DAG
│   │   ├── decision_engine.py    # DQN + Rule Engine hybrid
│   │   └── world_model.py        # 16D state vector
│   ├── memory/
│   │   ├── memory_manager.py     # Memory coordinator
│   │   ├── short_term.py         # Ring buffer
│   │   ├── long_term.py          # SQLite persistent store
│   │   ├── episodic.py           # Episode storage + replay
│   │   └── semantic.py           # Embeddings + vector search
│   ├── evolution/
│   │   ├── rl_engine.py          # Dueling Double DQN
│   │   ├── feedback_scorer.py    # Multi-dim reward
│   │   └── strategy_library.py  # Learned strategy archive
│   ├── autonomy/
│   │   └── autonomy_engine.py    # Retry + timeout + deps
│   ├── actions/
│   │   ├── action_registry.py    # Action dispatcher
│   │   ├── file_ops.py           # File operations
│   │   ├── system_ops.py         # System commands
│   │   └── api_ops.py            # HTTP operations
│   ├── safety/
│   │   └── safety_governor.py    # IMMUTABLE safety core
│   └── personality/
│       ├── personality_engine.py # 3-mode personality
│       └── response_formatter.py # Mode-aware formatting
├── dashboard/
│   ├── app.py                    # Flask + SocketIO
│   └── templates/index.html      # HUD dashboard
└── tests/
    └── test_suite.py             # 15 tests
```

---

## 🤖 RL Engine

| Component | Implementation |
|-----------|---------------|
| Algorithm | Double Dueling DQN |
| State space | 16-dimensional continuous vector |
| Action space | 10 discrete types |
| Architecture | Feature(128) → Value(1) + Advantage(10) |
| Optimizer | Adam lr=0.001 + StepLR |
| Exploration | epsilon-greedy (1.0 → 0.01) |
| Replay | Prioritized Experience Replay (10k) |
| Target sync | Every 100 steps |
| Persistence | JSON policy saved on shutdown |

---

## 🛡 Safety System

| Check | Mechanism |
|-------|-----------|
| Pattern blocking | Regex blocklist (rm -rf, DROP TABLE...) |
| Risk classification | low / medium / high / critical |
| Max risk gate | Configurable in config.yaml |
| Confirmation | Interactive CLI prompt for medium+ risk |
| Sandbox | High-risk actions simulated only |
| Audit log | Immutable 10,000-entry ring buffer |
| User override | STOP command always honored |

---

## 📊 Feedback Scoring

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Success | 45% | Task completed without error |
| Efficiency | 20% | Execution time vs complexity |
| Alignment | 20% | Output matches goal keywords |
| Safety | 10% | Within risk bounds |
| Quality | 5% | Output richness |

---

## 🚀 Roadmap

### Phase 1 (COMPLETE)
- Multi-layer perception (intent, urgency, entities, sentiment)
- Task graph planning with topological sort
- Hybrid Dueling DQN + rule-based decisions
- 4-type memory system (STM / LTM / Episodic / Semantic)
- Prioritized experience replay + Double DQN
- 5-dimension feedback scorer
- Immutable safety governor + audit log
- Autonomy engine with retry/timeout/dependency
- 3-mode personality engine
- Futuristic HUD web dashboard
- Rich CLI with banner

### Phase 2 (Next)
- FAISS vector index for semantic search
- spaCy NLP integration
- Predictive intelligence
- Extended action library

### Phase 3
- Meta-learning (optimize hyperparameters)
- Curiosity-driven exploration (ICM)
- Policy self-distillation

### Phase 4
- Plugin architecture
- Voice interface (Whisper + TTS)
- Multi-agent coordination

---

## 🖥 Quick Start

```bash
cd J.A.R.V.I.S
pip install -r requirements.txt
python main.py
```

**Commands:** `status` | `diagnose` | `benchmark` | `evolve` | `memory` | `history` | `strategies` | `dashboard` | `STOP`
