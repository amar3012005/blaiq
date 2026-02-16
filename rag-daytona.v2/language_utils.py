"""
Language resolution utility.

Single source of truth for mapping language codes/names to standard full names.
All system prompts, context architecture, and response generation use this
to determine the target language — keeping RAG content in English and
controlling output language purely through prompt instructions.
"""

import os

# Default response language (configurable per deployment)
DEFAULT_LANGUAGE = os.getenv("TARA_DEFAULT_LANGUAGE", "te")

LANGUAGE_MAP = {
    # Telugu
    "te": "Telugu", "tel": "Telugu", "telugu": "Telugu",
    # English
    "en": "English", "eng": "English", "english": "English",
    # German
    "de": "German", "deu": "German", "ger": "German",
    "german": "German", "deutsch": "German",
    # Hindi
    "hi": "Hindi", "hin": "Hindi", "hindi": "Hindi",
    # Tamil
    "ta": "Tamil", "tam": "Tamil", "tamil": "Tamil",
    # Kannada
    "kn": "Kannada", "kan": "Kannada", "kannada": "Kannada",
    # Malayalam
    "ml": "Malayalam", "mal": "Malayalam", "malayalam": "Malayalam",
    # French
    "fr": "French", "fra": "French", "french": "French",
    # Spanish
    "es": "Spanish", "spa": "Spanish", "spanish": "Spanish",
    # Japanese
    "ja": "Japanese", "jpn": "Japanese", "japanese": "Japanese",
    # Korean
    "ko": "Korean", "kor": "Korean", "korean": "Korean",
    # Portuguese
    "pt": "Portuguese", "por": "Portuguese", "portuguese": "Portuguese",
    # Arabic
    "ar": "Arabic", "ara": "Arabic", "arabic": "Arabic",
}


def resolve_language(lang: str) -> str:
    """Resolve a language code or name to its standard full English name.

    Examples:
        resolve_language("te")      -> "Telugu"
        resolve_language("english") -> "English"
        resolve_language("de")      -> "German"
        resolve_language("xyz")     -> "Xyz"  (fallback: capitalize)
    """
    if not lang:
        return resolve_language(DEFAULT_LANGUAGE)
    return LANGUAGE_MAP.get(lang.lower().strip(), lang.strip().capitalize())


def is_english(lang: str) -> bool:
    """Check if the resolved language is English."""
    return resolve_language(lang) == "English"
