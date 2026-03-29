"""
J.A.R.V.I.S — Memory Manager
Coordinates all memory types: short-term, long-term, episodic, semantic.
"""

from core.memory.short_term import ShortTermMemory
from core.memory.long_term import LongTermMemory
from core.memory.episodic import EpisodicMemory
from core.memory.semantic import SemanticMemory
from loguru import logger


class MemoryManager:
    """
    Central coordinator for J.A.R.V.I.S's multi-type memory system.
    Manages the complete memory lifecycle: store, retrieve, forget, consolidate.
    """

    def __init__(self, config: dict):
        self.config = config
        self.short_term = ShortTermMemory(
            capacity=config.get("short_term_capacity", 50)
        )
        self.long_term = LongTermMemory(
            db_path=config.get("long_term_db", "data/jarvis.db")
        )
        self.episodic = EpisodicMemory(
            db_path=config.get("long_term_db", "data/jarvis.db"),
            max_episodes=config.get("episodic_max_episodes", 1000)
        )
        self.semantic = SemanticMemory(
            db_path=config.get("long_term_db", "data/jarvis.db"),
            model_name=config.get("semantic_embedding_model", "all-MiniLM-L6-v2")
        )

    async def initialize(self):
        """Initialize all memory subsystems."""
        await self.long_term.initialize()
        await self.episodic.initialize()
        await self.semantic.initialize()
        logger.info("Memory system fully initialized.")

    async def consolidate(self):
        """
        Memory consolidation: Move important short-term items to long-term.
        Similar to sleep-based memory consolidation in biological systems.
        """
        items = self.short_term.get_all()
        consolidated = 0
        for item in items:
            if item.get("importance", 0.5) > 0.7:
                await self.long_term.store(
                    f"consolidated_{item.get('timestamp', 0)}",
                    item
                )
                consolidated += 1
        logger.info(f"Consolidated {consolidated} items to long-term memory.")

    async def get_stats(self) -> str:
        st = len(self.short_term.get_all())
        lt = await self.long_term.count()
        ep = await self.episodic.count()
        return f"ST:{st} LT:{lt} EP:{ep}"

    async def flush(self):
        """Persist all in-memory state to disk."""
        self.short_term.flush()
        logger.info("Memory flushed to disk.")

    async def close(self):
        """Close all database connections."""
        await self.long_term.close()
        await self.episodic.close()
        await self.semantic.close()
        logger.info("Memory connections closed.")
