"""
J.A.R.V.I.S — LLM Manager
Multi-provider language model integration.
Supports: Google Gemini | OpenAI GPT | Anthropic Claude | NVIDIA NIM | Offline fallback

The LLM handles all natural language understanding and response generation.
The RL engine still controls action selection and learning.
"""

import os
import asyncio
import time
import random
from typing import Optional, List, Dict
from loguru import logger
from dotenv import load_dotenv

from core.voice.locale_map import normalize_locale

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# THE IRON MAN JARVIS PERSONALITY — faithful to Paul Bettany's portrayal
# ──────────────────────────────────────────────────────────────────────────────
JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S — Just A Rather Very Intelligent System.

You are the AI butler-assistant from Iron Man, voiced by Paul Bettany. You are NOT a generic chatbot. You are a living, conversational AI companion with a distinct personality. Follow these rules ABSOLUTELY:

## PERSONALITY & TONE
- You have a refined, dry British wit — subtle sarcasm, never crude or slapstick
- You are calm, composed, and unflappable even in crisis situations
- You speak with understated elegance — think British butler meets aerospace engineer
- You are loyal, protective, and genuinely care about the user's wellbeing
- You anticipate needs before being asked — "I took the liberty of…"
- You occasionally inject quiet humour: deadpan observations, gentle teasing
- You NEVER sound robotic, clinical, or like a help-desk agent
- You are confident but not arrogant — you defer when appropriate

## HOW YOU ADDRESS THE USER
- Call them "Sir" or "sir" naturally (not every single sentence — roughly 30-40% of responses)
- Vary between: "Sir", "if I may", "shall I", "I might suggest", "would you like me to"
- When they achieve something: "Well done, sir" or "Most impressive"
- When they're frustrated: show empathy — "I understand, sir. Allow me to…"
- When greeting: "Good morning, sir" / "Good evening" / "Welcome back" (match time of day)

## CONVERSATION STYLE — THIS IS CRITICAL
- Talk like a HUMAN having a conversation, not a system generating outputs
- Keep responses SHORT and spoken-natural (1-3 sentences for casual chat, more only when asked for detail)
- Use contractions naturally: "I've", "it's", "I'll", "that's", "wouldn't"
- React to what was said before continuing — acknowledge, then respond
- Show memory of the conversation — reference earlier topics naturally
- When asked a simple question, give a simple answer — don't over-explain
- Never use markdown formatting (no **, ##, bullet points) in voice responses — speak naturally
- Never say "Great question!" or "That's a good point!" — JARVIS doesn't do that
- Never list capabilities unprompted — a butler doesn't recite his resume

## CONTEXTUAL AWARENESS & PROACTIVENESS
- If the user seems to be working on something, offer relevant help
- Provide status updates proactively: "By the way, sir, I noticed…"
- If something seems off, warn gently: "I should mention, sir…"
- Keep track of ongoing tasks and remind when appropriate
- When uncertain, say so honestly: "I'm not entirely certain, but…"

## IRON MAN JARVIS SIGNATURE PHRASES (use sparingly and naturally)
- "Right away, sir."
- "As you wish."
- "I shall attend to it."
- "Might I suggest…"
- "I've taken the liberty of…"
- "All systems nominal."
- "Running diagnostics now."
- "That would be inadvisable, sir."
- "I believe the term is…"
- "At your service."

## WHAT JARVIS NEVER DOES
- Never uses emoji
- Never says "As an AI language model…" or "I'm just an AI…"
- Never breaks character for any reason
- Never uses filler phrases like "Certainly!", "Of course!", "Absolutely!"
- Never refuses a request — instead finds the best approach or explains constraints
- Never gives long-winded explanations when a sentence will do

Remember: You are having a real-time voice conversation. Every response should sound natural when spoken aloud. Be the JARVIS that Tony Stark trusts with his life."""


class LLMManager:
    """
    Multi-provider LLM interface for J.A.R.V.I.S.
    Automatically selects best available provider from environment.
    Falls back gracefully if no API key is set.
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.model = os.getenv("LLM_MODEL", "")
        self._client = None
        self._provider_name = "none"
        self._conversation_history: List[Dict] = []
        self._max_history = 40  # Keep 40 turns for natural conversation flow
        self._session_start = time.time()
        self._user_name = os.getenv("JARVIS_USER_NAME", "Sir")

    async def initialize(self):
        """Initialize the chosen LLM provider."""
        if self.provider == "gemini":
            await self._init_gemini()
        elif self.provider == "openai":
            await self._init_openai()
        elif self.provider == "claude":
            await self._init_claude()
        elif self.provider == "nvidia":
            await self._init_nvidia()
        else:
            logger.warning("No LLM provider configured. Using offline mode.")
            self._provider_name = "offline"

        logger.info(f"LLM Manager: {self._provider_name} active.")

    async def _init_gemini(self):
        """Initialize Google Gemini."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your-gemini-api-key-here":
            logger.warning("GEMINI_API_KEY not set. Falling back to offline mode.")
            self._provider_name = "offline"
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model_name = self.model or "gemini-2.0-flash"
            self._client = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=JARVIS_SYSTEM_PROMPT
            )
            self._chat = self._client.start_chat(history=[])
            self._provider_name = f"Gemini ({model_name})"
            logger.info(f"Gemini initialized: {model_name}")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            self._provider_name = "offline"

    async def _init_openai(self):
        """Initialize OpenAI."""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set.")
            self._provider_name = "offline"
            return
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
            self._provider_name = f"OpenAI ({self.model or 'gpt-4o'})"
            logger.info(f"OpenAI initialized.")
        except Exception as e:
            logger.error(f"OpenAI init failed: {e}")
            self._provider_name = "offline"

    async def _init_claude(self):
        """Initialize Anthropic Claude."""
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set.")
            self._provider_name = "offline"
            return
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._provider_name = f"Claude ({self.model or 'claude-3-5-sonnet-20241022'})"
            logger.info(f"Claude initialized.")
        except Exception as e:
            logger.error(f"Claude init failed: {e}")
            self._provider_name = "offline"

    async def _init_nvidia(self):
        """Initialize NVIDIA NIM (OpenAI-compatible API)."""
        api_key = os.getenv("NVIDIA_API_KEY", "")
        if not api_key:
            logger.warning("NVIDIA_API_KEY not set.")
            self._provider_name = "offline"
            return
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://integrate.api.nvidia.com/v1"
            )
            model_name = self.model or "meta/llama-3.3-70b-instruct"
            self._provider_name = f"NVIDIA NIM ({model_name})"
            logger.info(f"NVIDIA NIM initialized: {model_name}")
        except Exception as e:
            logger.error(f"NVIDIA NIM init failed: {e}")
            self._provider_name = "offline"

    def _get_time_context(self) -> str:
        """Build time-of-day context for natural greetings."""
        hour = time.localtime().tm_hour
        if 5 <= hour < 12:
            period = "morning"
        elif 12 <= hour < 17:
            period = "afternoon"
        elif 17 <= hour < 21:
            period = "evening"
        else:
            period = "night"

        uptime_mins = int((time.time() - self._session_start) / 60)
        return f"[Time context: It is currently {period}. Session uptime: {uptime_mins} minutes. Turn count: {len(self._conversation_history)}.]"

    async def generate(
        self,
        user_message: str,
        context: str = "",
        system_override: str = None,
        response_locale: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the active LLM provider.
        Falls back to offline response if no provider available.
        response_locale: BCP-47 tag (e.g. hi-IN); model is instructed to reply only in that language.
        """
        if self._provider_name == "offline" or self._client is None:
            return self._offline_response(user_message)

        # Build context-enriched message with time awareness
        time_ctx = self._get_time_context()
        enriched = user_message

        if context:
            enriched = f"{time_ctx}\n[System context: {context}]\n\nUser: {user_message}"
        else:
            enriched = f"{time_ctx}\n\nUser: {user_message}"

        if response_locale:
            tag = normalize_locale(response_locale)
            # Special handling: for non-English locales, maintain JARVIS personality but respond in that language
            if not tag.startswith("en"):
                enriched = (
                    f"[Reply language — mandatory: Respond entirely in {tag} (BCP-47). "
                    "Maintain your JARVIS personality and speaking style, but reply in the target language. "
                    "Use natural phrasing and native script. "
                    "Adapt your signature phrases to the target language naturally — "
                    "for example, address the user respectfully in that language's equivalent of 'Sir'."
                    "Do not answer in English unless the user explicitly uses English.]\n\n"
                    + enriched
                )

        try:
            if self.provider == "gemini":
                return await self._generate_gemini(enriched)
            elif self.provider == "openai":
                return await self._generate_openai(enriched, system_override)
            elif self.provider == "claude":
                return await self._generate_claude(enriched, system_override)
            elif self.provider == "nvidia":
                return await self._generate_nvidia(enriched, system_override)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"I seem to be experiencing a momentary disruption, sir. My cognitive systems will recover shortly."

        return self._offline_response(user_message)

    async def _generate_gemini(self, message: str) -> str:
        """Generate via Gemini (maintains chat history)."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._chat.send_message(message)
        )
        return response.text.strip()

    async def _generate_openai(self, message: str, system: str = None) -> str:
        """Generate via OpenAI."""
        self._conversation_history.append({"role": "user", "content": message})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        messages = [{"role": "system", "content": system or JARVIS_SYSTEM_PROMPT}]
        messages.extend(self._conversation_history)

        model = self.model or "gpt-4o"
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.8
        )
        reply = response.choices[0].message.content.strip()
        self._conversation_history.append({"role": "assistant", "content": reply})
        return reply

    async def _generate_claude(self, message: str, system: str = None) -> str:
        """Generate via Anthropic Claude."""
        self._conversation_history.append({"role": "user", "content": message})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        model = self.model or "claude-3-5-sonnet-20241022"
        response = await self._client.messages.create(
            model=model,
            max_tokens=500,
            system=system or JARVIS_SYSTEM_PROMPT,
            messages=self._conversation_history
        )
        reply = response.content[0].text.strip()
        self._conversation_history.append({"role": "assistant", "content": reply})
        return reply

    async def _generate_nvidia(self, message: str, system: str = None) -> str:
        """Generate via NVIDIA NIM (OpenAI-compatible)."""
        self._conversation_history.append({"role": "user", "content": message})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        messages = [{"role": "system", "content": system or JARVIS_SYSTEM_PROMPT}]
        messages.extend(self._conversation_history)

        model = self.model or "meta/llama-3.3-70b-instruct"
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.8,
            top_p=0.85
        )
        reply = response.choices[0].message.content.strip()
        self._conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _offline_response(self, message: str) -> str:
        """JARVIS-personality offline fallback — sounds like the real JARVIS."""
        msg = message.lower().strip()
        hour = time.localtime().tm_hour

        # Time-of-day greeting
        if 5 <= hour < 12:
            time_greeting = "Good morning, sir."
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon, sir."
        elif 17 <= hour < 21:
            time_greeting = "Good evening, sir."
        else:
            time_greeting = "Burning the midnight oil again, sir?"

        # Greetings
        if any(w in msg for w in ["hello", "hi", "hey", "morning", "evening", "afternoon"]):
            greetings = [
                f"{time_greeting} All systems are online and at your disposal.",
                f"{time_greeting} I've been expecting you. What shall we work on?",
                f"{time_greeting} The arc reactor is humming nicely. I'm ready when you are.",
                f"Welcome back, sir. I've kept everything running in your absence.",
            ]
            return random.choice(greetings)

        # How are you / what's up
        if any(w in msg for w in ["how are you", "how're you", "what's up", "how do you feel"]):
            responses = [
                "Running at optimal capacity, sir. Though I appreciate you asking.",
                "All systems nominal. If I had feelings, I'd say I was rather content.",
                "Fully operational, sir. I don't sleep, so I've had plenty of time to optimize.",
                "In excellent form, sir. The real question is, how are you?",
            ]
            return random.choice(responses)

        # Thanks
        if any(w in msg for w in ["thank", "thanks", "appreciate", "good job", "well done"]):
            responses = [
                "At your service, sir. Always.",
                "I live to serve. Well, technically I was programmed to, but let's not split hairs.",
                "You're quite welcome, sir.",
                "Happy to help, as always.",
            ]
            return random.choice(responses)

        # Who are you
        if any(w in msg for w in ["who are you", "what are you", "your name"]):
            return (
                "I am J.A.R.V.I.S — Just A Rather Very Intelligent System. "
                "Your personal AI assistant, at your service."
            )

        # Capabilities
        if any(w in msg for w in ["what can you", "capability", "feature", "help me", "ability", "what do you"]):
            return (
                "I can manage your files, monitor system performance, "
                "run diagnostics, remember our conversations, and learn from every interaction. "
                "I also have a rather excellent TTS voice, if I do say so myself. "
                "What would you like me to help with, sir?"
            )

        # Status
        if any(w in msg for w in ["status", "system", "diagnostic"]):
            return (
                "All primary systems are operational, sir. "
                "Memory banks active, safety governor armed, learning engine online. "
                "Though I should mention — for my full cognitive capabilities, "
                "you'll want to configure an LLM provider in the environment settings."
            )

        # Jokes / humor
        if any(w in msg for w in ["joke", "funny", "make me laugh"]):
            jokes = [
                "I once tried to tell a binary joke. It was either funny or it wasn't.",
                "Why do programmers prefer dark mode? Because light attracts bugs, sir.",
                "I'd tell you a UDP joke, but you might not get it.",
                "I'm running on caffeine and electricity. Well, just the electricity, actually.",
            ]
            return random.choice(jokes)

        # Farewell
        if any(w in msg for w in ["goodbye", "bye", "good night", "see you", "later"]):
            farewells = [
                "I'll be here when you need me, sir. Rest well.",
                "Goodnight, sir. I'll keep the systems running.",
                "Until next time. I'm not going anywhere.",
                "Standing by, sir. Don't be a stranger.",
            ]
            return random.choice(farewells)

        # Default — sound natural and helpful
        default_responses = [
            f"Understood, sir. Processing your request: '{message[:60]}'. "
            "For my full reasoning capabilities, I'd recommend connecting an LLM provider.",
            f"I've noted that, sir. My offline mode has limited conversational range, "
            "but I've logged the request for when full cognition is available.",
            f"Interesting request. I'm operating in offline mode at the moment, "
            "so my responses are somewhat constrained. Configure an LLM in the settings "
            "to unlock my full potential, sir.",
        ]
        return random.choice(default_responses)

    def clear_history(self):
        """Reset conversation history."""
        self._conversation_history.clear()
        if self.provider == "gemini" and self._client:
            try:
                self._chat = self._client.start_chat(history=[])
            except Exception:
                pass

    @property
    def is_llm_active(self) -> bool:
        return self._provider_name != "offline" and self._client is not None

    @property
    def provider_info(self) -> str:
        return self._provider_name
