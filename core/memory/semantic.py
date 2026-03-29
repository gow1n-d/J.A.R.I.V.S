"""
J.A.R.V.I.S — Semantic Memory
Learned knowledge store with vector similarity search.
Uses sentence embeddings for semantic retrieval.
Falls back to keyword search if embeddings unavailable.
"""

import json
import time
import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class SemanticMemory:
    """
    Semantic memory stores learned facts, concepts, and knowledge.
    Supports vector similarity search using sentence embeddings.
    Falls back gracefully to keyword matching if transformers unavailable.
    """

    def __init__(self, db_path: str = "data/jarvis.db",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.model_name = model_name
        self._conn: Optional[sqlite3.Connection] = None
        self._encoder = None
        self._use_embeddings = False

    async def initialize(self):
        """Connect to database and load embedding model if available."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        await self._create_tables()

        # Try loading sentence transformer
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
            self._use_embeddings = True
            logger.info(f"Semantic memory: embeddings enabled ({self.model_name})")
        except ImportError:
            logger.warning("sentence-transformers not available. Using keyword search.")
            self._use_embeddings = False

    async def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concept TEXT NOT NULL,
                knowledge TEXT NOT NULL,
                embedding BLOB,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 0.8,
                source TEXT DEFAULT 'learned',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sem_concept ON semantic_memory(concept)"
        )
        self._conn.commit()

    async def store(self, concept: str, knowledge: str,
                    category: str = "general", confidence: float = 0.8,
                    source: str = "learned"):
        """Store a learned concept and its knowledge."""
        now = time.time()
        embedding = None
        if self._use_embeddings and self._encoder:
            emb = self._encoder.encode(knowledge)
            embedding = emb.tobytes()

        # Check if concept already exists (update if so)
        cursor = self._conn.execute(
            "SELECT id FROM semantic_memory WHERE concept = ?", (concept,)
        )
        existing = cursor.fetchone()

        if existing:
            self._conn.execute("""
                UPDATE semantic_memory
                SET knowledge = ?, embedding = ?, confidence = ?, updated_at = ?
                WHERE concept = ?
            """, (knowledge, embedding, confidence, now, concept))
        else:
            self._conn.execute("""
                INSERT INTO semantic_memory
                (concept, knowledge, embedding, category, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (concept, knowledge, embedding, category, confidence, source, now, now))

        self._conn.commit()
        logger.debug(f"Semantic: stored '{concept}' [{category}]")

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant knowledge using semantic or keyword search."""
        if self._use_embeddings and self._encoder:
            return await self._semantic_search(query, top_k)
        else:
            return await self._keyword_search(query, top_k)

    async def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Vector similarity search using cosine distance."""
        query_emb = self._encoder.encode(query)

        cursor = self._conn.execute(
            "SELECT * FROM semantic_memory WHERE embedding IS NOT NULL"
        )
        rows = cursor.fetchall()

        scored = []
        for row in rows:
            stored_emb = np.frombuffer(row["embedding"], dtype=np.float32)
            # Cosine similarity
            norm_q = np.linalg.norm(query_emb)
            norm_s = np.linalg.norm(stored_emb)
            if norm_q > 0 and norm_s > 0:
                similarity = float(np.dot(query_emb, stored_emb) / (norm_q * norm_s))
                scored.append((similarity, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"concept": r["concept"], "knowledge": r["knowledge"],
             "confidence": r["confidence"], "similarity": s}
            for s, r in scored[:top_k]
        ]

    async def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword search."""
        words = query.lower().split()
        results = []
        for word in words:
            cursor = self._conn.execute(
                "SELECT * FROM semantic_memory WHERE concept LIKE ? OR knowledge LIKE ? "
                "ORDER BY confidence DESC LIMIT ?",
                (f"%{word}%", f"%{word}%", top_k)
            )
            for row in cursor.fetchall():
                if row["concept"] not in [r["concept"] for r in results]:
                    results.append({
                        "concept": row["concept"],
                        "knowledge": row["knowledge"],
                        "confidence": row["confidence"],
                        "similarity": 0.5
                    })
        return results[:top_k]

    async def count(self) -> int:
        cursor = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM semantic_memory"
        )
        return cursor.fetchone()["cnt"]

    async def close(self):
        if self._conn:
            self._conn.close()
