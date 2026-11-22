"""Test script for UnifiedLLMClient dual-provider fallback.

Verifies:
  1. Groq is called first when configured.
  2. If Groq raises an exception (or is simulated down), Gemini takes over.
  3. Status reporting and provider call counters track failovers accurately.

Usage:
    python scripts/test_gemini_fallback.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.llm_client import UnifiedLLMClient


class FakeFailingClient:
    def chat(self, *args, **kwargs):
        raise RuntimeError("Simulated Groq 429 quota exhaustion")


class FakeWorkingClient:
    def __init__(self, name="fake-gemini"):
        self.last_model = name

    def chat(self, messages, *args, **kwargs):
        return "Simulated response from Gemini fallback"


def main():
    print("Testing UnifiedLLMClient failover mechanism...")
    unified = UnifiedLLMClient(groq=FakeFailingClient(), gemini=FakeWorkingClient("gemini-1.5-flash"))

    messages = [{"role": "user", "content": "Hello"}]
    reply = unified.chat(messages)

    assert reply == "Simulated response from Gemini fallback", f"Unexpected reply: {reply}"
    assert unified.last_provider == "gemini", f"Expected gemini, got {unified.last_provider}"
    assert unified.gemini_fallbacks == 1, f"Expected 1 fallback, got {unified.gemini_fallbacks}"

    status = unified.get_status()
    assert status["gemini_fallbacks"] == 1
    assert status["last_provider"] == "gemini"

    print("PASS: UnifiedLLMClient successfully falls over from Groq to Gemini.")
    print("Status:", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
