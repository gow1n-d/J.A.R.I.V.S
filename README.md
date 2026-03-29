# 🧠 J.A.R.V.I.S
### Just A Rather Very Intelligent System
**Autonomous • Self-Evolving • Controlled • Stark-Class Intelligence**

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## 📁 Project Structure

```
J.A.R.V.I.S/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── config.yaml                # System configuration
├── .env                       # Environment variables
│
├── core/
│   ├── jarvis.py              # Central orchestrator
│   ├── perception.py          # Perception layer
│   ├── cognitive/
│   │   ├── goal_interpreter.py
│   │   ├── task_planner.py
│   │   ├── decision_engine.py
│   │   └── world_model.py
│   ├── memory/
│   │   ├── memory_manager.py
│   │   ├── short_term.py
│   │   ├── long_term.py
│   │   ├── episodic.py
│   │   └── semantic.py
│   ├── evolution/
│   │   ├── rl_engine.py
│   │   ├── feedback_scorer.py
│   │   ├── experience_replay.py
│   │   └── strategy_library.py
│   ├── autonomy/
│   │   ├── autonomy_engine.py
│   │   └── executor.py
│   ├── actions/
│   │   ├── action_registry.py
│   │   ├── file_ops.py
│   │   ├── system_ops.py
│   │   └── api_ops.py
│   ├── safety/
│   │   ├── safety_governor.py
│   │   ├── risk_classifier.py
│   │   └── permission_system.py
│   └── personality/
│       ├── personality_engine.py
│       └── response_formatter.py
│
├── dashboard/
│   ├── app.py                 # Web dashboard (Flask)
│   ├── static/
│   └── templates/
│
├── data/
│   ├── jarvis.db              # SQLite memory database
│   ├── strategies/            # Saved strategies
│   └── logs/                  # Execution logs
│
└── tests/
    └── test_suite.py
```

## 🛡️ Safety First
J.A.R.V.I.S has an immutable safety layer. The user always has full control.
Type `STOP` at any time to halt all operations immediately.
