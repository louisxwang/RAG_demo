from __future__ import annotations

import logging
import re

from app.agent.tools import ToolError, calculator
from app.llm.client import LLMClient
from app.rag.retriever import Retriever

log = logging.getLogger(__name__)


class Orchestrator:
    """
    Simple multi-step agent:
      1) retrieve context
      2) summarize context
      3) answer with context + optional tool usage

    Design choice: explicit steps list for debuggability and low overhead.
    """

    def __init__(self) -> None:
        self.retriever = Retriever()
        self.llm = LLMClient()

    def run(self, question: str) -> tuple[str, list[str], list[str]]:
        steps: list[str] = []

        # Tool calling (minimal): if user asks to calculate, do it deterministically.
        tool_result: str | None = None
        m = re.search(r"(?:calc(?:ulate)?\s*[:\-]?\s*)(.+)$", question.strip(), flags=re.I)
        if m:
            expr = m.group(1).strip()
            try:
                val = calculator(expr)
                tool_result = f"Calculator({expr}) = {val}"
                steps.append(f"tool: {tool_result}")
            except ToolError:
                steps.append("tool: calculator failed (unsupported expression)")

        retrieved = self.retriever.search(question)
        context = [f"[{c.source}] {c.text}" for c in retrieved]
        steps.append(f"retrieve: {len(context)} chunks")

        context_block = "\n\n".join(context) if context else "(no context found)"
        summary = self.llm.chat(
            [
                {"role": "system", "content": "Summarize the provided context for answering a user question. Be concise."},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{question}"},
            ],
            temperature=0.2,
        )
        steps.append("summarize: done")

        extra = f"\n\nTool result:\n{tool_result}" if tool_result else ""
        answer = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI assistant. Use the context summary to answer. "
                        "If context is insufficient, say what is missing. Keep it brief."
                    ),
                },
                {"role": "user", "content": f"Context summary:\n{summary}{extra}\n\nUser question:\n{question}"},
            ],
            temperature=0.2,
        )
        steps.append("answer: done")

        return answer.strip(), [c.text for c in retrieved], steps

