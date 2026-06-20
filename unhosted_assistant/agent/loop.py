"""
The agent loop — wires a local model to the skill registry: send the system
prompt + conversation + tool specs to the model, execute any tool calls it makes,
feed results back, and iterate until it answers or hits the step bound (a
deterministic guardrail).

This is the clean, general assistant loop — no personal-twin layers (no persona
files, memory, or voice). The model client is injected, so the loop is unit
testable with a mock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..llm.ollama_client import ChatMessage
from ..skills.base import SkillRegistry, default_registry
from .router import RouteDecision, Router


class ModelClient(Protocol):
    def chat(self, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None) -> ChatMessage: ...


DEFAULT_SYSTEM = (
    "You are a helpful, local-first AI assistant running on the user's own "
    "hardware. Be concise and accurate. Use the provided tools when they help; "
    "otherwise answer directly. Start with the answer, then any caveats."
)


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    route: RouteDecision | None = None


class Agent:
    def __init__(
        self,
        client: ModelClient,
        registry: SkillRegistry | None = None,
        max_steps: int = 6,
        system: str | None = None,
        router: Router | None = None,
    ) -> None:
        self.client = client
        self.registry = registry or default_registry
        self.max_steps = max_steps
        self.system = system if system is not None else DEFAULT_SYSTEM
        self.router = router

    def run(self, user_input: str) -> AgentResult:
        decision: RouteDecision | None = None
        if self.router is not None:
            decision = self.router.route(user_input)
            if hasattr(self.client, "model"):
                self.client.model = decision.model  # type: ignore[attr-defined]

        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system),
            ChatMessage(role="user", content=user_input),
        ]
        tools = self.registry.tool_specs()
        used: list[tuple[str, dict[str, Any]]] = []

        for step in range(1, self.max_steps + 1):
            reply = self.client.chat(messages, tools=tools)
            messages.append(reply)

            if not reply.tool_calls:
                return AgentResult(answer=reply.content.strip(), steps=step,
                                   tool_calls=used, route=decision)

            for call in reply.tool_calls:
                name, args = _parse_tool_call(call)
                result = self.registry.dispatch(name, args)
                used.append((name, args))
                messages.append(ChatMessage(role="tool", tool_name=name, content=result))

        last = next((m for m in reversed(messages) if m.role == "assistant"), None)
        answer = (last.content.strip() if last and last.content else
                  "[stopped] reached the step limit before finishing.")
        return AgentResult(answer=answer, steps=self.max_steps, tool_calls=used, route=decision)


def _parse_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    fn = call.get("function", call) or {}
    name = fn.get("name", "")
    raw = fn.get("arguments", {})
    if isinstance(raw, str):
        try:
            args = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw, dict):
        args = raw
    else:
        args = {}
    return name, args
