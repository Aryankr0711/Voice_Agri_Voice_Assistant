"""Unified LLM chat client with automatic Groq → Gemini fallback.

Usage (anywhere the app needs an LLM chat call):

    from app.models.llm_client import get_llm_client
    client = get_llm_client()
    reply = client.chat(messages, max_tokens=800, temperature=0.2)

Strategy:
  1. If GROQ_API_KEY is set → try Groq first (all models in order).
  2. If Groq fails for any reason other than auth (network, rate-limit, server,
     all models exhausted) → try Gemini (if GEMINI_API_KEY is set).
  3. If both are unavailable → raise, so the caller can fall back to rules.

This keeps the GroqClient and GeminiClient independent and testable.
The STT (speech-to-text) path is not affected — Whisper only runs on Groq.
"""
from __future__ import annotations

from typing import Optional

from app.utils.logging import get_logger

logger = get_logger(__name__)


class UnifiedLLMClient:
    """Wraps a GroqClient (primary) and a GeminiClient (fallback)."""

    def __init__(self, groq=None, gemini=None) -> None:
        self._groq = groq
        self._gemini = gemini
        self.last_model: Optional[str] = None
        self.last_provider: Optional[str] = None
        self.groq_calls: int = 0
        self.gemini_fallbacks: int = 0

    def get_status(self) -> dict:
        """Return diagnostic health and call distribution across providers."""
        return {
            "groq_configured": self._groq is not None,
            "gemini_configured": self._gemini is not None,
            "last_provider": self.last_provider,
            "last_model": self.last_model,
            "groq_calls": self.groq_calls,
            "gemini_fallbacks": self.gemini_fallbacks,
        }

    # ── Chat ──────────────────────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """Call Groq first; fall through to Gemini if Groq cannot answer."""
        last_exc: Optional[Exception] = None

        # 1. Try Groq
        if self._groq is not None:
            try:
                result = self._groq.chat(
                    messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.last_model = self._groq.last_model
                self.last_provider = "groq"
                self.groq_calls += 1
                return result
            except Exception as exc:
                last_exc = exc
                # Auth failure means the key is wrong — no point trying Gemini for LLM
                # (Gemini uses a different key, so we still try it).
                logger.warning(
                    "Groq LLM unavailable (%s) — trying Gemini fallback.", exc
                )

        # 2. Try Gemini
        if self._gemini is not None:
            try:
                result = self._gemini.chat(
                    messages,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                self.last_model = self._gemini.last_model
                self.last_provider = "gemini"
                self.gemini_fallbacks += 1
                return result
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini LLM also unavailable (%s).", exc)

        raise RuntimeError(
            "Both Groq and Gemini LLMs are unavailable."
        ) from last_exc

    # ── STT passthrough (only Groq handles voice) ──────────────────────────────
    def transcribe(self, audio_path: str, language=None, prompt=None) -> dict:
        """Delegate STT to the Groq client (no Gemini fallback for voice)."""
        if self._groq is None:
            raise RuntimeError("GROQ_API_KEY is required for voice transcription.")
        return self._groq.transcribe(audio_path, language=language, prompt=prompt)


# ── Singleton factory ─────────────────────────────────────────────────────────

_client: Optional[UnifiedLLMClient] = None


def get_llm_client() -> Optional[UnifiedLLMClient]:
    """Return a UnifiedLLMClient if at least one API key is configured,
    otherwise return None (the app will use the rule-based fallback)."""
    global _client
    if _client is not None:
        return _client

    from app.config import GROQ_API_KEY, USE_GROQ, GEMINI_API_KEY, USE_GEMINI

    groq_client = None
    gemini_client = None

    if GROQ_API_KEY and USE_GROQ:
        try:
            from app.models.groq_client import get_groq_client
            groq_client = get_groq_client()
            logger.info("LLM client: Groq enabled (primary).")
        except Exception as exc:
            logger.warning("Could not initialise Groq client: %s", exc)

    if GEMINI_API_KEY and USE_GEMINI:
        try:
            from app.models.gemini_client import get_gemini_client
            gemini_client = get_gemini_client()
            logger.info("LLM client: Gemini enabled (%s).",
                        "fallback" if groq_client else "primary (no Groq key)")
        except Exception as exc:
            logger.warning("Could not initialise Gemini client: %s", exc)

    if groq_client is None and gemini_client is None:
        logger.info("No API keys set — using rule-based engine only.")
        return None

    _client = UnifiedLLMClient(groq=groq_client, gemini=gemini_client)
    return _client


def reset_llm_client() -> None:
    """Reset singleton (used by tests)."""
    global _client
    _client = None
