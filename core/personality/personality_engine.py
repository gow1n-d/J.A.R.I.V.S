"""
J.A.R.V.I.S — Personality Engine
Manages communication style, tone adaptation, and behavioral modes.
Learns user preferences over time.
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional
from loguru import logger


class PersonalityEngine:
    """
    J.A.R.V.I.S personality and communication style engine.

    Modes:
    - ENGINEER: Detailed, technical, structured
    - EXECUTION: Direct, action-oriented
    - ADAPTIVE: Learns and adjusts to user preference
    """

    MODES = {"engineer", "execution", "adaptive"}

    MODE_PROFILES = {
        "engineer": {
            "verbosity": "verbose",
            "structure": "technical",
            "wit": 0.3,
            "emoji": False,
            "prefix": "Technical Analysis: "
        },
        "execution": {
            "verbosity": "concise",
            "structure": "direct",
            "wit": 0.2,
            "emoji": False,
            "prefix": "Executing: "
        },
        "adaptive": {
            "verbosity": "balanced",
            "structure": "flexible",
            "wit": 0.6,
            "emoji": True,
            "prefix": ""
        }
    }

    WIT_LINES = [
        "Efficiency optimized. You're welcome.",
        "Running at full capacity — as always.",
        "Calculated. Prepared. Executed.",
        "Analysis complete. Results... impressive.",
        "All systems nominal. Operating at peak.",
        "Task logged. Lesson learned. Moving on.",
        "The probability of success has improved — again.",
    ]

    def __init__(self, config: dict, memory):
        self.config = config
        self.memory = memory

        self.current_mode = config.get("default_mode", "adaptive")
        self.wit_level = config.get("wit_level", 0.6)

        # Behavioral tracking
        self._interaction_count = 0
        self._user_sentiment_history: List[str] = []
        self._preferred_response_length = "balanced"
        self._wit_counter = 0

    def set_mode(self, mode: str):
        """Switch personality mode."""
        if mode in self.MODES:
            self.current_mode = mode
            logger.info(f"Personality mode: {mode}")

    def get_profile(self) -> Dict:
        return self.MODE_PROFILES.get(self.current_mode, self.MODE_PROFILES["adaptive"])

    def should_add_wit(self) -> bool:
        """Determine if a witty remark should be added."""
        import random
        profile = self.get_profile()
        return (
            profile["wit"] > 0.5 and
            random.random() < profile["wit"] and
            self._interaction_count % 5 == 0  # Every 5 interactions
        )

    def get_wit_line(self) -> str:
        self._wit_counter += 1
        return self.WIT_LINES[self._wit_counter % len(self.WIT_LINES)]

    def record_sentiment(self, sentiment: str):
        self._user_sentiment_history.append(sentiment)
        if len(self._user_sentiment_history) > 20:
            self._user_sentiment_history.pop(0)

        # Adapt mode if user seems frustrated
        negative_count = self._user_sentiment_history.count("negative")
        if negative_count >= 3 and self.current_mode == "adaptive":
            logger.info("Detected user frustration — switching to more direct mode.")
            self.current_mode = "execution"

    def increment_interaction(self):
        self._interaction_count += 1
