"""
J.A.R.V.I.S — Text-to-Speech Engine
Uses Microsoft Edge TTS (edge-tts) for neural voice synthesis.
Default: en-GB-RyanNeural — British male, very JARVIS-like.
Streams audio to dashboard or saves to file for playback.
"""

import asyncio
import io
import os
import tempfile
import base64
from pathlib import Path
from typing import Optional
from loguru import logger


class TTSEngine:
    """
    Neural text-to-speech engine using Microsoft Edge TTS.
    Produces high-quality British accent audio — no API key needed.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.voice = os.getenv("TTS_VOICE", config.get("voice", "en-GB-RyanNeural"))
        self.rate = os.getenv("TTS_RATE", config.get("rate", "+20%"))
        self.pitch = config.get("pitch", "+0Hz")
        self.enabled = os.getenv("VOICE_ENABLED", "true").lower() == "true"
        self._available = False

        # Output dir for audio files
        self._audio_dir = Path("data/audio")
        self._audio_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Check if edge-tts is available."""
        try:
            import edge_tts
            self._available = True
            logger.info(f"TTS Engine online: {self.voice} | rate={self.rate}")
        except ImportError:
            self._available = False
            logger.warning("edge-tts not installed. Run: pip install edge-tts")

    async def speak_to_bytes(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """
        Convert text to speech and return audio bytes (MP3).
        Used for streaming to the web browser.
        `voice`: optional Edge neural voice id; defaults to instance/env voice.
        """
        if not self._available or not self.enabled:
            return None

        # Strip markdown formatting for cleaner speech
        clean_text = self._clean_for_speech(text)
        if not clean_text.strip():
            return None

        voice_id = voice or self.voice
        try:
            import edge_tts
            communicate = edge_tts.Communicate(
                text=clean_text,
                voice=voice_id,
                rate=self.rate,
                pitch=self.pitch
            )
            audio_data = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data.write(chunk["data"])

            audio_bytes = audio_data.getvalue()
            logger.debug(f"TTS synthesized: {len(audio_bytes)} bytes for {len(clean_text)} chars")
            return audio_bytes if audio_bytes else None

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    async def speak_to_base64(self, text: str, voice: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech and return as base64-encoded MP3.
        Used for sending audio to browser via JSON API.
        """
        audio_bytes = await self.speak_to_bytes(text, voice=voice)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("utf-8")
        return None

    async def speak_to_file(self, text: str, filename: str = None) -> Optional[Path]:
        """Save speech to MP3 file."""
        if not filename:
            filename = f"jarvis_{id(text) % 100000}.mp3"
        output_path = self._audio_dir / filename

        audio_bytes = await self.speak_to_bytes(text)
        if audio_bytes:
            output_path.write_bytes(audio_bytes)
            return output_path
        return None

    def _clean_for_speech(self, text: str) -> str:
        """Remove markdown and clean text for natural speech synthesis."""
        import re
        # Remove markdown bold/italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        # Remove emoji (keep it clean for TTS)
        text = re.sub(r'[⚡🧠🛡⌨💻📊🔁🔍✅❌⛔🎯📋🔬📁🚀✍🎭🌐🔧⚙️🤖📼💡🔑⚠️🔔💾📄]', '', text)
        # Remove bullet points
        text = re.sub(r'^[•·\-\*]\s+', '', text, flags=re.MULTILINE)
        # Remove multiple newlines → single pause
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n', ', ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', 'link', text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Limit length for TTS
        if len(text) > 500:
            text = text[:500] + "... further details available on display."
        return text

    async def list_voices(self) -> list:
        """List all available edge-tts voices."""
        if not self._available:
            return []
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [v for v in voices if 'en' in v.get('Locale', '').lower()]
        except Exception:
            return []

    @property
    def is_available(self) -> bool:
        return self._available and self.enabled
