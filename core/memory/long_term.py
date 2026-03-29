"""
J.A.R.V.I.S — Long-Term Memory
Persistent key-value store backed by SQLite.
Stores learned knowledge, RL policy state, and user preferences.
"""

import json
import time
import sqlite3
import asyncio
from pathlib import Path
from typing import Any, Optional, List
from loguru import logger


class LongTermMemory:
    """
    Persistent SQLite-backed long-term memory.
    Handles serialization of complex Python objects via JSON.
    """

    def __init__(self, db_path: str = "data/jarvis.db"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    async def initialize(self):
        """Create tables and connect to database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        await self._create_tables()
        logger.debug(f"Long-term memory initialized at {self.db_path}")

    async def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_category ON memory(category)"
        )
        self._conn.commit()

    async def store(self, key: str, value: Any, category: str = "general",
                    importance: float = 0.5):
        """Persist a key-value pair to long-term memory."""
        now = time.time()
        serialized = json.dumps(value, default=str)
        self._conn.execute("""
            INSERT OR REPLACE INTO memory
            (key, value, category, importance, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (key, serialized, category, importance, now, now))
        self._conn.commit()
        logger.debug(f"LTM stored: {key} [{category}]")

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key."""
        cursor = self._conn.execute(
            "SELECT value FROM memory WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row:
            # Increment access count
            self._conn.execute(
                "UPDATE memory SET access_count = access_count + 1 WHERE key = ?",
                (key,)
            )
            self._conn.commit()
            return json.loads(row["value"])
        return None

    async def search(self, query: str, category: str = None,
                     limit: int = 10) -> List[dict]:
        """Search memory by key pattern or category."""
        if category:
            cursor = self._conn.execute(
                "SELECT * FROM memory WHERE category = ? ORDER BY importance DESC LIMIT ?",
                (category, limit)
            )
        else:
            cursor = self._conn.execute(
                "SELECT * FROM memory WHERE key LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{query}%", limit)
            )
        rows = cursor.fetchall()
        return [
            {
                "key": r["key"],
                "value": json.loads(r["value"]),
                "category": r["category"],
                "importance": r["importance"]
            }
            for r in rows
        ]

    async def delete(self, key: str):
        """Remove a key from memory."""
        self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        self._conn.commit()

    async def count(self) -> int:
        """Return total number of stored entries."""
        cursor = self._conn.execute("SELECT COUNT(*) as cnt FROM memory")
        return cursor.fetchone()["cnt"]

    async def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
