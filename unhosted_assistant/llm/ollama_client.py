"""
Ollama client — the local "brain". Talks to a model running on your own machine
via Ollama's HTTP API (default http://localhost:11434). Stdlib only (urllib), so
the agent has no cloud dependency and no heavy SDK.

Supports tool/function calling: pass `tools` (a list of tool specs) and the model
may return `tool_calls` in its reply, which the agent loop executes.

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


@dataclass
class ChatMessage:
    role: str                       # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    # for role="tool" replies, the name of the tool that produced this content
    tool_name: str | None = None

    def to_api(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.tool_name:
            msg["tool_name"] = self.tool_name
        return msg


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        temperature: float = 0.4,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    # ---- health -------------------------------------------------------------
    def is_up(self) -> bool:
        try:
            self._get("/api/tags", timeout=4.0)
            return True
        except OllamaError:
            return False

    def available_models(self) -> list[str]:
        try:
            data = self._get("/api/tags", timeout=4.0)
        except OllamaError:
            return []
        return [m.get("name", "") for m in data.get("models", [])]

    def ensure_ready(self) -> None:
        """Raise a friendly OllamaError if the server is down or the model is missing."""
        if not self.is_up():
            raise OllamaError(
                "Ollama isn't running. Start it with `ollama serve`, then try again."
            )
        models = self.available_models()
        # model names can carry a :tag; match on the base name too
        base = self.model.split(":")[0]
        if models and not any(m == self.model or m.split(":")[0] == base for m in models):
            raise OllamaError(
                f"Model '{self.model}' isn't pulled. Run `ollama pull {self.model}` "
                f"(installed: {', '.join(models) or 'none'})."
            )

    # ---- chat ---------------------------------------------------------------
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """One non-streaming chat turn. Returns the assistant message (which may
        contain tool_calls the caller should execute)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        data = self._post("/api/chat", payload)
        msg = data.get("message", {}) or {}
        return ChatMessage(
            role=msg.get("role", "assistant"),
            content=msg.get("content", "") or "",
            tool_calls=msg.get("tool_calls", []) or [],
        )

    # ---- transport ----------------------------------------------------------
    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama request to {path} failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama returned invalid JSON from {path}: {e}") from e

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        req = urllib.request.Request(self.host + path)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OllamaError(f"Ollama request to {path} failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OllamaError(f"Ollama returned invalid JSON from {path}: {e}") from e
