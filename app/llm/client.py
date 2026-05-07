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

        self.timeout = settings.llm_timeout_s

        if self.provider == "mock":
            return

        if self.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required for llm_provider=openai")

            self.base_url = settings.openai_base_url.rstrip("/")
            self.api_key = settings.openai_api_key
            self.model = settings.openai_model
            return

        if self.provider == "gemini":
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is required for llm_provider=gemini")

            self.base_url = settings.gemini_base_url.rstrip("/")
            self.api_key = settings.gemini_api_key
            self.model = settings.gemini_model
            return

        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    def chat(self, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        if self.provider == "mock":
            last_user = next(
                (str(m.get("content", "")).strip() for m in reversed(messages) if m.get("role") == "user"),
                "",
            )
            if not last_user:
                last_user = "Hello! (No user message found.)"
            return (
                "MOCK LLM (no API key configured).\n\n"
                "I can’t call a real model yet, but your pipeline is wired correctly.\n\n"
                f"Your last question:\n{last_user}"
            )

        if self.provider == "openai":
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

            log.debug("LLM request provider=openai model=%s base_url=%s", self.model, self.base_url)
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        if self.provider == "gemini":
            # Gemini's API uses "contents" with roles "user" and "model". This app's messages
            # are OpenAI-style: {"role": "...", "content": "..."}.
            system_prefix = ""
            contents: list[dict[str, Any]] = []

            for m in messages:
                role = str(m.get("role", "user")).lower()
                content = str(m.get("content", ""))
                if not content:
                    continue

                if role == "system":
                    system_prefix += content.strip() + "\n"
                    continue

                gemini_role = "user" if role == "user" else "model"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

            if system_prefix:
                system_text = system_prefix.strip()
                if contents and contents[0]["role"] == "user":
                    contents[0]["parts"][0]["text"] = f"{system_text}\n\n{contents[0]['parts'][0]['text']}"
                else:
                    contents.insert(0, {"role": "user", "parts": [{"text": system_text}]})

            url = f"{self.base_url}/models/{self.model}:generateContent"
            params = {"key": self.api_key}
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": contents,
                "generationConfig": {"temperature": temperature},
            }

            log.debug("LLM request provider=gemini model=%s base_url=%s", self.model, self.base_url)
            resp = requests.post(
                url, params=params, headers=headers, data=json.dumps(payload), timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Gemini returned no candidates: {data}")
            content = (candidates[0].get("content") or {}).get("parts") or []
            if not content:
                raise RuntimeError(f"Gemini returned empty content: {data}")
            text = content[0].get("text")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(f"Gemini returned non-text content: {data}")
            return text

        raise ValueError(f"Unsupported LLM provider: {self.provider}")

