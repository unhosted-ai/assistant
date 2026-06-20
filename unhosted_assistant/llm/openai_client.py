"""
OpenAI-compatible client — a second local "brain" backend.

Talks to any server that speaks the OpenAI Chat Completions API:
  - LM Studio        (http://localhost:1234/v1)   ← the common one
  - llama.cpp server (--api, http://localhost:8080/v1)
  - Jan, vLLM, LocalAI, text-generation-webui (OpenAI extension), …

It duck-types `OllamaClient` (same `model`, `host`, `is_up()`,
`available_models()`, `ensure_ready()`, `chat()` surface) so the agent loop,
router, CLI, and voice server use it without caring which backend is behind it.
Stdlib only (urllib) — no SDK, no cloud dependency, in the spirit of the project.

Tool/function calling is translated both ways: the agent speaks Ollama's
``ChatMessage`` shape; this client converts to/from OpenAI's
``tool_calls`` / ``tool_call_id`` format.

Docs: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

# Reuse the shared message type so the agent loop sees one consistent shape.
from .ollama_client import ChatMessage


class OpenAIError(RuntimeError):
    """Raised when the OpenAI-compatible server is unreachable or errors."""


# LM Studio and most local servers ignore the API key, but the OpenAI client
# library and some servers expect the header to be present.
_PLACEHOLDER_KEY = "lm-studio"


class OpenAIClient:
    def __init__(
        self,
        model: str = "",
        host: str = "http://localhost:1234/v1",
        timeout: float = 120.0,
        temperature: float = 0.4,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        # Accept either ".../v1" or a bare host; normalize to include /v1 once.
        self.host = self._normalize_host(host)
        self.timeout = timeout
        self.temperature = temperature
        self.api_key = api_key or _PLACEHOLDER_KEY

    @staticmethod
    def _normalize_host(host: str) -> str:
        h = host.rstrip("/")
        if not h.endswith("/v1"):
            h = h + "/v1"
        return h

    # ---- health -------------------------------------------------------------
    def is_up(self) -> bool:
        try:
            self._get("/models", timeout=4.0)
            return True
        except OpenAIError:
            return False

    def available_models(self) -> list[str]:
        try:
            data = self._get("/models", timeout=4.0)
        except OpenAIError:
            return []
        # OpenAI shape: {"data": [{"id": "..."}, ...]}
        items = data.get("data", []) if isinstance(data, dict) else []
        return [m.get("id", "") for m in items if m.get("id")]

    def ensure_ready(self) -> None:
        """Friendly error if the server is down or the model isn't loaded."""
        if not self.is_up():
            raise OpenAIError(
                "No OpenAI-compatible server reachable at "
                f"{self.host}. In LM Studio, start the local server "
                "(Developer ▸ Start Server), then try again."
            )
        models = self.available_models()
        if self.model and models and self.model not in models:
            base = self.model.split(":")[0]
            if not any(m == self.model or m.split(":")[0] == base for m in models):
                raise OpenAIError(
                    f"Model '{self.model}' isn't loaded on the server "
                    f"(available: {', '.join(models) or 'none'})."
                )

    # ---- chat ---------------------------------------------------------------
    def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """One non-streaming chat turn. Returns the assistant message (which may
        carry tool_calls in Ollama's shape for the agent loop to execute)."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._to_openai_messages(messages),
            "stream": False,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        data = self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        msg = (choices[0].get("message") if choices else {}) or {}
        return ChatMessage(
            role=msg.get("role", "assistant") or "assistant",
            content=msg.get("content") or "",
            tool_calls=self._from_openai_tool_calls(msg.get("tool_calls")),
        )

    # ---- message translation ------------------------------------------------
    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        """Convert the agent's Ollama-shaped messages into OpenAI's format.

        The tricky part is tool replies: OpenAI requires each role="tool" message
        to carry the `tool_call_id` of the assistant tool call it answers. The
        agent's ChatMessage only carries `tool_name`, so we remember the ids the
        most recent assistant turn emitted and match by function name.
        """
        out: list[dict[str, Any]] = []
        # name -> list of tool_call_ids still awaiting a reply (FIFO per name)
        pending_ids: dict[str, list[str]] = {}

        for m in messages:
            if m.role == "tool":
                name = m.tool_name or ""
                call_id = ""
                ids = pending_ids.get(name)
                if ids:
                    call_id = ids.pop(0)
                entry: dict[str, Any] = {"role": "tool", "content": m.content}
                if call_id:
                    entry["tool_call_id"] = call_id
                # OpenAI also accepts a name field on tool messages.
                if name:
                    entry["name"] = name
                out.append(entry)
                continue

            entry = {"role": m.role, "content": m.content or ""}
            if m.role == "assistant" and m.tool_calls:
                oai_calls = self._to_openai_tool_calls(m.tool_calls)
                entry["tool_calls"] = oai_calls
                # assistant messages with tool_calls should have null content
                if not m.content:
                    entry["content"] = None
                # register the ids so the following tool replies can reference them
                for c in oai_calls:
                    fn_name = c.get("function", {}).get("name", "")
                    pending_ids.setdefault(fn_name, []).append(c["id"])
            out.append(entry)
        return out

    @staticmethod
    def _to_openai_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Ollama tool_call -> OpenAI tool_call (arguments must be a JSON string)."""
        result: list[dict[str, Any]] = []
        for i, call in enumerate(calls):
            fn = call.get("function", call) or {}
            args = fn.get("arguments", {})
            if not isinstance(args, str):
                args = json.dumps(args)
            result.append({
                "id": call.get("id") or f"call_{i}",
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            })
        return result

    @staticmethod
    def _from_openai_tool_calls(calls: Any) -> list[dict[str, Any]]:
        """OpenAI tool_calls -> Ollama shape the agent loop understands.

        The loop's `_parse_tool_call` already accepts arguments as a dict or a
        JSON string, so we keep arguments as the model returned them but preserve
        the id so multi-call turns can be matched on the way back."""
        if not calls:
            return []
        out: list[dict[str, Any]] = []
        for call in calls:
            fn = call.get("function", {}) or {}
            out.append({
                "id": call.get("id", ""),
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments", "{}"),
                },
            })
        return out

    # ---- transport ----------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=body, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001 - best-effort error detail
                pass
            raise OpenAIError(f"server error {e.code} from {path}: {detail}") from e
        except urllib.error.URLError as e:
            raise OpenAIError(f"request to {path} failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OpenAIError(f"invalid JSON from {path}: {e}") from e

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        req = urllib.request.Request(self.host + path, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise OpenAIError(f"request to {path} failed: {e}") from e
        except json.JSONDecodeError as e:
            raise OpenAIError(f"invalid JSON from {path}: {e}") from e
