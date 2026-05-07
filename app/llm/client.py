from __future__ import annotations

import json
import logging
from typing import Any

import requests

from app.core.config import settings

log = logging.getLogger(__name__)


class LLMClient:
    """
    Very small OpenAI-compatible chat client.

    Design choice: keep a tiny surface area (chat(messages)->str), so swapping providers
    doesn't ripple through RAG/agent code.
    """

    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()

        if self.provider != "openai":
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for llm_provider=openai")

        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.timeout = settings.llm_timeout_s

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        log.debug("LLM request model=%s base_url=%s", self.model, self.base_url)
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

