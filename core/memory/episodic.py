"""
J.A.R.V.I.S — Episodic Memory
Records sequences of actions and their outcomes.
Enables J.A.R.V.I.S to learn from "experiences" like biological episodic recall.
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class EpisodicMemory:
    """
    Episodic memory stores complete interaction episodes:
    (input → goal → action → result → score).
    Used for experience replay in RL and pattern recognition.
    """

    def __init__(self, db_path: str = "data/jarvis.db", max_episodes: int = 1000):
        self.db_path = db_path
        self.max_episodes = max_episodes
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self):
        """Connect and create episodes table."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        await self._create_tables()
        logger.debug("Episodic memory initialized.")

    async def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input TEXT,
                goal TEXT,
                action TEXT,
                result TEXT,
                score REAL DEFAULT 0.0,
                success INTEGER DEFAULT 0,
                duration REAL DEFAULT 0.0,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ep_score ON episodes(score)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ep_time ON episodes(timestamp)"
        )
        self._conn.commit()

    async def store_episode(self, episode: Dict[str, Any]):
        """Store a complete interaction episode."""
        self._conn.execute("""
            INSERT INTO episodes (input, goal, action, result, score, success, duration, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(episode.get("input", "")),
            json.dumps(episode.get("goal", {}), default=str),
            json.dumps(episode.get("action", {}), default=str),
            json.dumps(episode.get("result", {}), default=str),
            float(episode.get("score", 0.0)),
            1 if episode.get("score", 0.0) > 0.5 else 0,
            float(episode.get("duration", 0.0)),
            time.time()
        ))
        self._conn.commit()

        # Enforce max episodes limit
        total = await self.count()
        if total > self.max_episodes:
            await self._prune_oldest()

        logger.debug(f"Episode stored (score={episode.get('score', 0):.2f})")

    async def get_recent(self, n: int = 10) -> List[Dict]:
        """Get the N most recent episodes."""
        cursor = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
        )
        rows = cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_top_scoring(self, n: int = 10) -> List[Dict]:
        """Get the N highest-scoring episodes (for experience replay)."""
        cursor = self._conn.execute(
            "SELECT * FROM episodes ORDER BY score DESC LIMIT ?", (n,)
        )
        rows = cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def get_failures(self, n: int = 10) -> List[Dict]:
        """Get recent failed episodes for failure analysis."""
        cursor = self._conn.execute(
            "SELECT * FROM episodes WHERE success = 0 ORDER BY timestamp DESC LIMIT ?",
            (n,)
        )
        rows = cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def sample_random(self, n: int = 32) -> List[Dict]:
        """Sample N random episodes (for experience replay batches)."""
        cursor = self._conn.execute(
            "SELECT * FROM episodes ORDER BY RANDOM() LIMIT ?", (n,)
        )
        rows = cursor.fetchall()
        return [self._row_to_dict(r) for r in rows]

    async def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM episodes")
        return cursor.fetchone()["cnt"]

    async def get_success_rate(self) -> float:
        cursor = self._conn.execute(
            "SELECT AVG(success) as rate FROM (SELECT success FROM episodes ORDER BY timestamp DESC LIMIT 100)"
        )
        row = cursor.fetchone()
        return float(row["rate"] or 0.0)

    def _row_to_dict(self, row) -> Dict:
        return {
            "id": row["id"],
            "input": row["input"],
            "goal": json.loads(row["goal"]) if row["goal"] else {},
            "action": json.loads(row["action"]) if row["action"] else {},
            "result": json.loads(row["result"]) if row["result"] else {},
            "score": row["score"],
            "success": bool(row["success"]),
            "timestamp": row["timestamp"]
        }

    async def _prune_oldest(self):
        """Remove oldest episodes to maintain max capacity."""
        self._conn.execute("""
            DELETE FROM episodes WHERE id IN (
                SELECT id FROM episodes ORDER BY timestamp ASC LIMIT 10
            )
        """)
        self._conn.commit()

    async def close(self):
        if self._conn:
            self._conn.close()
