"""
BCP-47 locale tags → Microsoft Edge neural TTS voice IDs (edge-tts).
Used for multilingual voice replies. Unknown tags fall back to en-GB JARVIS-style voice.
"""

from __future__ import annotations

# Lowercase keys: language or language-region
EDGE_TTS_VOICE_BY_LOCALE: dict[str, str] = {
    "en": "en-GB-RyanNeural",
    "en-gb": "en-GB-RyanNeural",
    "en-us": "en-US-GuyNeural",
    "en-in": "en-IN-PrabhatNeural",
    "hi": "hi-IN-MadhurNeural",
    "hi-in": "hi-IN-MadhurNeural",
    "es": "es-ES-AlvaroNeural",
    "es-es": "es-ES-AlvaroNeural",
    "es-mx": "es-MX-JorgeNeural",
    "es-us": "es-US-AlonsoNeural",
    "fr": "fr-FR-HenriNeural",
    "fr-fr": "fr-FR-HenriNeural",
    "fr-ca": "fr-CA-AntoineNeural",
    "de": "de-DE-KillianNeural",
    "de-de": "de-DE-KillianNeural",
    "it": "it-IT-DiegoNeural",
    "it-it": "it-IT-DiegoNeural",
    "pt": "pt-BR-AntonioNeural",
    "pt-br": "pt-BR-AntonioNeural",
    "pt-pt": "pt-PT-DuarteNeural",
    "ja": "ja-JP-KeitaNeural",
    "ja-jp": "ja-JP-KeitaNeural",
    "ko": "ko-KR-InJoonNeural",
    "ko-kr": "ko-KR-InJoonNeural",
    "zh": "zh-CN-YunxiNeural",
    "zh-cn": "zh-CN-YunxiNeural",
    "zh-tw": "zh-TW-HsiaoChenNeural",
    "ar": "ar-SA-HamedNeural",
    "ar-sa": "ar-SA-HamedNeural",
    "ta": "ta-IN-ValluvarNeural",
    "ta-in": "ta-IN-ValluvarNeural",
    "te": "te-IN-MohanNeural",
    "te-in": "te-IN-MohanNeural",
    "mr": "mr-IN-ManoharNeural",
    "mr-in": "mr-IN-ManoharNeural",
    "bn": "bn-IN-BashkarNeural",
    "bn-in": "bn-IN-BashkarNeural",
    "ru": "ru-RU-DmitryNeural",
    "ru-ru": "ru-RU-DmitryNeural",
    "nl": "nl-NL-MaartenNeural",
    "nl-nl": "nl-NL-MaartenNeural",
    "pl": "pl-PL-MarekNeural",
    "pl-pl": "pl-PL-MarekNeural",
    "tr": "tr-TR-AhmetNeural",
    "tr-tr": "tr-TR-AhmetNeural",
    "vi": "vi-VN-NamMinhNeural",
    "vi-vn": "vi-VN-NamMinhNeural",
    "th": "th-TH-NiwatNeural",
    "th-th": "th-TH-NiwatNeural",
    "id": "id-ID-ArdiNeural",
    "id-id": "id-ID-ArdiNeural",
    "uk": "uk-UA-OstapNeural",
    "uk-ua": "uk-UA-OstapNeural",
    "sv": "sv-SE-MattiasNeural",
    "sv-se": "sv-SE-MattiasNeural",
    "da": "da-DK-JeppeNeural",
    "da-dk": "da-DK-JeppeNeural",
    "fi": "fi-FI-HarriNeural",
    "fi-fi": "fi-FI-HarriNeural",
    "nb": "nb-NO-FinnNeural",
    "nb-no": "nb-NO-FinnNeural",
    "el": "el-GR-NestorasNeural",
    "el-gr": "el-GR-NestorasNeural",
    "he": "he-IL-AvriNeural",
    "he-il": "he-IL-AvriNeural",
}


def normalize_locale(tag: str | None) -> str:
    if not tag or not str(tag).strip():
        return "en-gb"
    t = str(tag).strip().lower().replace("_", "-")
    if t == "en":
        t = "en-gb"
    return t


def edge_voice_for_locale(tag: str | None) -> str:
    """Resolve an Edge TTS voice id for browser/API locale tag."""
    t = normalize_locale(tag)
    if t in EDGE_TTS_VOICE_BY_LOCALE:
        return EDGE_TTS_VOICE_BY_LOCALE[t]
    if "-" in t:
        lang = t.split("-")[0]
        if lang in EDGE_TTS_VOICE_BY_LOCALE:
            return EDGE_TTS_VOICE_BY_LOCALE[lang]
    return "en-GB-RyanNeural"
