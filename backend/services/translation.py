import os
import re
import json
import logging
import requests
from typing import Optional, Dict

logger = logging.getLogger("careflow.translation")

class ClinicalTranslationService:
    """
    Zero-VRAM Pure Google Translate Clinical Translation Pipeline.
    Uses Google Translate direct API (zero external packages, zero API keys required).
    Supports:
    - Hindi (Devanagari) -> English
    - Hinglish (Romanized Hindi) -> English
    - English -> Devanagari Hindi
    - English -> Natural Hinglish
    """

    def __init__(self):
        self._cache: Dict[str, str] = {}

    def _google_translate_direct(self, text: str, source: str = "auto", target: str = "hi") -> Optional[str]:
        """Calls Google Translate direct API without third-party dependencies."""
        if not text or not text.strip():
            return ""
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": source,
                "tl": target,
                "dt": "t",
                "q": text.strip()
            }
            resp = requests.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and data[0]:
                    translated = "".join([part[0] for part in data[0] if part and part[0]]).strip()
                    if translated:
                        return translated
        except Exception as e:
            logger.debug(f"Direct Google Translate error ({source}->{target}): {e}")
        return None

    def translate_to_english(self, text: str) -> str:
        """
        Translates Hindi (Devanagari) or Romanized Hinglish into clear clinical English via Google Translate.
        """
        if not text or not text.strip():
            return ""

        raw = text.strip()
        cache_key = f"en::{raw}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        res = self._google_translate_direct(raw, source="auto", target="en")
        if res and res.strip():
            self._cache[cache_key] = res.strip()
            return res.strip()

        return raw

    def translate_to_hindi(self, english_text: str, default_hindi: Optional[str] = None) -> str:
        """
        Translates an English clinical question to natural Devanagari Hindi via Google Translate.
        """
        if not english_text or not english_text.strip():
            return default_hindi or ""

        raw = english_text.strip()
        cache_key = f"hi::{raw}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # If model provided valid Devanagari, use it
        if default_hindi and re.search(r'[\u0900-\u097F]', default_hindi):
            self._cache[cache_key] = default_hindi.strip()
            return default_hindi.strip()

        res = self._google_translate_direct(raw, source="en", target="hi")
        if res and res.strip():
            self._cache[cache_key] = res.strip()
            return res.strip()

        return default_hindi or raw

    def translate_to_hinglish(self, english_text: str, default_hinglish: Optional[str] = None) -> str:
        """
        Translates an English clinical question to natural colloquial Hinglish.
        First translates to Hindi via Google Translate, then transliterates to Roman script.
        """
        if not english_text or not english_text.strip():
            return default_hinglish or ""

        raw = english_text.strip()
        cache_key = f"hinglish::{raw}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if default_hinglish and default_hinglish.strip():
            self._cache[cache_key] = default_hinglish.strip()
            return default_hinglish.strip()

        # Translate to Hindi first via Google Translate
        hi_text = self.translate_to_hindi(raw)
        if hi_text and hi_text != raw:
            # Transliterate Devanagari to Latin via Google Input Tools
            try:
                import urllib.parse
                enc = urllib.parse.quote(hi_text)
                url = f"https://inputtools.google.com/request?text={enc}&itc=hi-t-i0-und&num=1"
                resp = requests.get(url, timeout=3)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 1 and data[1]:
                        cand = data[1][0][1][0]
                        if cand:
                            self._cache[cache_key] = cand
                            return cand
            except Exception:
                pass
            # Fallback to Hindi text
            self._cache[cache_key] = hi_text
            return hi_text

        return raw

# Global singleton instance
translation_service = ClinicalTranslationService()
