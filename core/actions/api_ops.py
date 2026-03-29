"""
J.A.R.V.I.S — API Operations
Handles external API calls with rate limiting and error handling.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class APIOperations:
    """External API operations with rate limiting."""

    def __init__(self):
        self._last_call_time: Dict[str, float] = {}
        self._rate_limit = 1.0  # seconds between calls to same host

    async def fetch_url(self, url: str, method: str = "GET",
                        headers: dict = None, data: dict = None) -> Dict[str, Any]:
        """Fetch data from a URL with rate limiting."""
        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "error": "requests not installed",
                "output": ""
            }

        # Extract host for rate limiting
        try:
            from urllib.parse import urlparse
            host = urlparse(url).netloc
        except Exception:
            host = url

        # Rate limiting
        now = time.time()
        if host in self._last_call_time:
            elapsed = now - self._last_call_time[host]
            if elapsed < self._rate_limit:
                await asyncio.sleep(self._rate_limit - elapsed)

        self._last_call_time[host] = time.time()

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(url, headers=headers or {}, timeout=10)
            )

            return {
                "success": response.status_code < 400,
                "output": response.text[:500],
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return {"success": False, "error": str(e), "output": ""}
