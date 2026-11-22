"""Google Gemini client — fallback LLM when Groq is unavailable or rate-limited.

Used when GEMINI_API_KEY is set and the Groq client raises an error.
Only needs `requests`, so it runs on the 512 MB Render Lite host.

Chat calls walk GEMINI_LLM_MODELS in order:
  • a model this account cannot use is skipped permanently;
  • a 429 (quota) triggers a short wait then the next model.

The API key is only ever sent as a query parameter (as the Gemini REST API
requires) — never logged.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import requests

from app.utils.logging import get_logger

logger = get_logger(__name__)

# Default Gemini models tried in order
_DEFAULT_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_LLM_MODELS",
        "gemini-1.5-flash,gemini-1.5-pro,gemini-2.0-flash-exp",
    ).split(",")
    if m.strip()
]

_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
_TIMEOUT = 40.0


class GeminiError(RuntimeError):
    """A failed Gemini API call. `kind` is one of:
    auth · rate_limit · unavailable · bad_request · server · network
    """

    def __init__(self, kind: str, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after


class GeminiClient:
    """Minimal Gemini REST client for chat completions (fallback LLM)."""

    def __init__(
        self,
        api_key: str = "",
        models: Optional[list[str]] = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.models = list(models or _DEFAULT_MODELS)
        self.timeout = timeout
        self.last_model: Optional[str] = None
        self._dead: set[str] = set()
        self._session = requests.Session()

    # ── HTTP ──────────────────────────────────────────────────────────────────
    def _post(self, model: str, payload: dict) -> requests.Response:
        if not self.api_key:
            raise GeminiError("auth", "GEMINI_API_KEY is not set")
        url = f"{_GEMINI_API_BASE}/{model}:generateContent"
        try:
            return self._session.post(
                url,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise GeminiError(
                "network", f"Gemini request failed: {type(exc).__name__}"
            ) from exc

    @staticmethod
    def _raise_for(resp: requests.Response, model: str) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
            msg = str(body.get("error", {}).get("message", ""))
            status_code = body.get("error", {}).get("code", resp.status_code)
        except ValueError:
            msg, status_code = resp.text[:200], resp.status_code

        low = msg.lower()
        if resp.status_code == 401 or "api_key" in low or "invalid" in low:
            kind = "auth"
        elif resp.status_code == 429 or "quota" in low or "rate" in low:
            kind = "rate_limit"
        elif resp.status_code in (403, 404) or "not found" in low or "not supported" in low:
            kind = "unavailable"
        elif resp.status_code >= 500:
            kind = "server"
        else:
            kind = "bad_request"

        try:
            retry_after = float(resp.headers.get("retry-after") or 0)
        except ValueError:
            retry_after = 0.0

        raise GeminiError(
            kind, f"Gemini {resp.status_code} ({model}): {msg[:200]}", retry_after
        )

    # ── Chat ──────────────────────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        *,
        json_mode: bool = False,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        """Reply text from the first Gemini model that can answer.

        `messages` uses OpenAI format: [{role, content}, …]. The system
        message (role=system) is extracted and sent as systemInstruction.
        """
        last: Optional[GeminiError] = None
        for attempt in range(2):
            waits = []
            for model in [m for m in self.models if m not in self._dead]:
                try:
                    text = self._chat_once(model, messages, json_mode, max_tokens, temperature)
                    self.last_model = model
                    return text
                except GeminiError as exc:
                    last = exc
                    logger.warning("Gemini chat: %s", exc)
                    if exc.kind in ("auth", "network"):
                        raise
                    if exc.kind == "unavailable":
                        self._dead.add(model)
                    elif exc.kind == "rate_limit":
                        waits.append(exc.retry_after or 2.0)

            # All models failed — if they were rate-limited wait with exponential backoff and retry.
            if attempt == 0 and last and last.kind == "rate_limit" and waits:
                delay = min(max(min(waits), 1.0), 6.0)
                logger.info("Gemini rate limited; waiting %.1fs before secondary attempt.", delay)
                time.sleep(delay)
                continue
            break

        raise last or GeminiError("unavailable", "no Gemini model is available")

    def _chat_once(
        self,
        model: str,
        messages: list[dict],
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> str:
        # Split system prompt from conversation
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        conv = [m for m in messages if m.get("role") != "system"]

        # Build Gemini contents
        contents = []
        for m in conv:
            role = "user" if m.get("role") == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n".join(system_parts)}]
            }

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        t0 = time.time()
        resp = self._post(model, payload)
        self._raise_for(resp, model)

        body = resp.json()
        try:
            text = (
                body["candidates"][0]["content"]["parts"][0]["text"] or ""
            ).strip()
        except (KeyError, IndexError):
            text = ""

        if not text:
            raise GeminiError("server", f"{model} returned an empty reply")

        logger.info(
            "Gemini chat | %s | %.1fs | %d chars", model, time.time() - t0, len(text)
        )
        return text


# ── Singleton ─────────────────────────────────────────────────────────────────

_client: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def set_gemini_client(client: Optional[GeminiClient]) -> None:
    """Swap the shared client (tests use a fake one)."""
    global _client
    _client = client
