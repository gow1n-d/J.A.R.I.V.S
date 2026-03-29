"""
J.A.R.V.I.S — Perception Layer
Converts raw user input into structured internal state.
Detects intent, urgency, entities, and context.
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from loguru import logger


@dataclass
class Percept:
    """Structured internal representation of user input."""
    raw_input: str
    intent: str
    intent_confidence: float
    urgency: str            # low | medium | high | critical
    entities: Dict[str, List[str]]
    keywords: List[str]
    question: bool
    action_required: bool
    domain: str             # task | query | system | conversation | unknown
    timestamp: float = field(default_factory=time.time)
    sentiment: str = "neutral"  # positive | neutral | negative


class IntentClassifier:
    """Rule-based + pattern intent classifier."""

    INTENT_PATTERNS = {
        "create": [r"\b(create|make|build|generate|write|produce|design)\b"],
        "read": [r"\b(read|show|display|get|fetch|retrieve|find|search|look)\b"],
        "update": [r"\b(update|modify|change|edit|fix|improve|upgrade|refactor)\b"],
        "delete": [r"\b(delete|remove|erase|drop|clear|destroy)\b"],
        "execute": [r"\b(run|execute|start|launch|deploy|activate|trigger)\b"],
        "analyze": [r"\b(analyze|analyse|evaluate|assess|check|inspect|review|diagnose)\b"],
        "plan": [r"\b(plan|schedule|organize|arrange|prepare|roadmap)\b"],
        "learn": [r"\b(learn|teach|train|study|understand|explain|how)\b"],
        "status": [r"\b(status|health|metrics|stats|performance|monitor)\b"],
        "stop": [r"\b(stop|halt|pause|cancel|abort|quit|exit)\b"],
        "help": [r"\b(help|assist|support|guide|tutorial|what can)\b"],
        "remember": [r"\b(remember|save|store|keep|note|record)\b"],
        "forget": [r"\b(forget|clear memory|reset|wipe)\b"],
        "predict": [r"\b(predict|forecast|anticipate|estimate|expect|will)\b"],
    }

    URGENCY_PATTERNS = {
        "critical": [r"\b(emergency|critical|urgent|asap|immediately|now!)\b"],
        "high": [r"\b(urgent|important|priority|quickly|fast|hurry)\b"],
        "medium": [r"\b(soon|when you can|please|need to)\b"],
        "low": [r"\b(whenever|eventually|sometime|maybe|perhaps)\b"],
    }

    DOMAIN_PATTERNS = {
        "task": [r"\b(create|make|build|run|execute|deploy|write|generate)\b"],
        "query": [r"\b(what|when|where|who|how|why|which|tell me|show me)\b"],
        "system": [r"\b(status|mode|memory|history|diagnose|benchmark|evolve)\b"],
        "conversation": [r"\b(hello|hi|hey|thanks|thank you|great|nice|cool)\b"],
    }

    def classify_intent(self, text: str) -> tuple[str, float]:
        text_lower = text.lower()
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            if score > 0:
                scores[intent] = score

        if not scores:
            return "unknown", 0.3

        best = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = min(scores[best] / max(total, 1), 1.0)

        return best, confidence

    def classify_urgency(self, text: str) -> str:
        text_lower = text.lower()
        for urgency, patterns in self.URGENCY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return urgency
        return "low"

    def classify_domain(self, text: str) -> str:
        text_lower = text.lower()
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return domain
        return "unknown"

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {
            "files": [],
            "urls": [],
            "numbers": [],
            "paths": [],
            "commands": [],
        }

        # Files
        entities["files"] = re.findall(r'\b[\w-]+\.\w{2,4}\b', text)

        # URLs
        entities["urls"] = re.findall(
            r'https?://[^\s]+', text
        )

        # Numbers
        entities["numbers"] = re.findall(r'\b\d+(?:\.\d+)?\b', text)

        # File paths
        entities["paths"] = re.findall(
            r'(?:[A-Za-z]:)?(?:[/\\][\w\s.-]+)+', text
        )

        # Quoted commands
        entities["commands"] = re.findall(r'`([^`]+)`', text)

        return {k: v for k, v in entities.items() if v}

    def extract_keywords(self, text: str) -> List[str]:
        stop_words = {
            "the", "a", "an", "is", "it", "to", "for", "of", "in",
            "on", "at", "by", "or", "and", "but", "with", "from",
            "that", "this", "can", "will", "be", "do", "i", "me",
            "my", "you", "your", "please", "want", "need"
        }
        words = re.findall(r'\b[a-zA-Z]\w{2,}\b', text.lower())
        return [w for w in words if w not in stop_words][:10]

    def detect_sentiment(self, text: str) -> str:
        positive = re.findall(
            r'\b(great|good|excellent|perfect|thanks|love|amazing|awesome|well done)\b',
            text.lower()
        )
        negative = re.findall(
            r'\b(bad|wrong|error|fail|broken|terrible|awful|problem|issue)\b',
            text.lower()
        )
        if len(positive) > len(negative):
            return "positive"
        elif len(negative) > len(positive):
            return "negative"
        return "neutral"


class PerceptionLayer:
    """
    Converts raw inputs into structured Percept objects.
    Acts as the sensory cortex of J.A.R.V.I.S.
    """

    def __init__(self):
        self.classifier = IntentClassifier()
        self._processed_count = 0
        logger.debug("Perception Layer online.")

    async def process(self, raw_input: str) -> Percept:
        """Convert raw text input into structured Percept."""
        self._processed_count += 1

        if not raw_input or not raw_input.strip():
            return Percept(
                raw_input="",
                intent="unknown",
                intent_confidence=0.0,
                urgency="low",
                entities={},
                keywords=[],
                question=False,
                action_required=False,
                domain="unknown",
                sentiment="neutral"
            )

        text = raw_input.strip()

        # Classify
        intent, confidence = self.classifier.classify_intent(text)
        urgency = self.classifier.classify_urgency(text)
        domain = self.classifier.classify_domain(text)
        entities = self.classifier.extract_entities(text)
        keywords = self.classifier.extract_keywords(text)
        sentiment = self.classifier.detect_sentiment(text)

        # Detect question
        is_question = (
            text.endswith("?") or
            bool(re.match(r'^(what|when|where|who|how|why|which|can|could|would|is|are)', text.lower()))
        )

        # Determine if action is needed
        action_required = intent in {
            "create", "update", "delete", "execute", "plan", "remember", "forget"
        }

        percept = Percept(
            raw_input=text,
            intent=intent,
            intent_confidence=confidence,
            urgency=urgency,
            entities=entities,
            keywords=keywords,
            question=is_question,
            action_required=action_required,
            domain=domain,
            sentiment=sentiment
        )

        logger.debug(
            f"Percept: intent={intent}({confidence:.2f}) | "
            f"urgency={urgency} | domain={domain}"
        )

        return percept
