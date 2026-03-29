"""
J.A.R.V.I.S — Short-Term Memory
Active context window. Holds the most recent N interactions.
Implements a fast ring buffer with importance scoring.
"""

import time
from collections import deque
from typing import List, Dict, Any, Optional
from loguru import logger


class ShortTermMemory:
    """
    Fast, in-memory ring buffer for active context.
    Analogous to working memory in cognitive psychology.
    """

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._importance_scores: Dict[int, float] = {}
        self._access_count: Dict[int, int] = {}
        logger.debug(f"Short-term memory online (capacity={capacity}).")

    def store(self, item: Dict[str, Any], importance: float = 0.5) -> int:
        """Store an item in short-term memory. Returns item index."""
        enriched = {
            **item,
            "timestamp": time.time(),
            "importance": importance,
            "access_count": 0
        }
        self._buffer.append(enriched)
        idx = len(self._buffer) - 1
        logger.debug(f"STM stored item (importance={importance:.2f})")
        return idx

    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the N most recent items."""
        items = list(self._buffer)
        return items[-n:] if len(items) >= n else items

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all items in short-term memory."""
        return list(self._buffer)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Simple keyword search over STM."""
        query_lower = query.lower()
        results = []
        for item in self._buffer:
            text = str(item.get("input", "")).lower()
            if any(word in text for word in query_lower.split()):
                results.append(item)
        return results[-top_k:]

    def update_importance(self, idx: int, delta: float):
        """Boost or reduce importance of an item."""
        if 0 <= idx < len(self._buffer):
            items = list(self._buffer)
            items[idx]["importance"] = min(
                1.0, max(0.0, items[idx].get("importance", 0.5) + delta)
            )

    def clear(self):
        """Clear all short-term memory."""
        self._buffer.clear()
        logger.info("Short-term memory cleared.")

    def flush(self):
        """No-op for STM (in-memory only)."""
        pass

    def __len__(self) -> int:
        return len(self._buffer)
