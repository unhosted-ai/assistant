"""
Tests for the Unhosted Assistant — agent loop, router, and skill gates.
All offline (a mock model client; no live model/Ollama needed).

Run:  python tests/test_assistant.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unhosted_assistant.agent.loop import Agent, DEFAULT_SYSTEM  # noqa: E402
from unhosted_assistant.agent.router import Router, classify  # noqa: E402
from unhosted_assistant.llm.ollama_client import ChatMessage  # noqa: E402
from unhosted_assistant.skills.base import SkillRegistry, Skill  # noqa: E402


# ---------- agent loop ----------
class ScriptedClient:
    """Returns pre-scripted assistant turns; records the system prompt + any
    tool results fed back, so we can assert the loop's plumbing."""
    def __init__(self, turns):
        self.turns = turns
        self.calls = 0
        self.model = "mock"
        self.system_seen = ""
        self.tool_results = []

    def chat(self, messages, tools=None):
        for m in messages:
            if m.role == "system":
                self.system_seen = m.content
            if m.role == "tool":
                self.tool_results.append(m.content)
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        return turn


def _tc(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_tool_calling_loop():
    reg = SkillRegistry()
    reg.register(Skill("add", "add two ints",
                       {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
                       run=lambda a, b: str(int(a) + int(b))))
    client = ScriptedClient([
        ChatMessage(role="assistant", content="", tool_calls=[_tc("add", {"a": 2, "b": 3})]),
        ChatMessage(role="assistant", content="The answer is 5."),
    ])
    agent = Agent(client=client, registry=reg)
    res = agent.run("what is 2+3?")
    assert res.answer == "The answer is 5.", res.answer
    assert res.steps == 2
    assert ("add", {"a": 2, "b": 3}) in res.tool_calls
    assert "5" in client.tool_results[0]
    print("✓ loop: model → tool → result → answer")


def test_default_system_prompt():
    client = ScriptedClient([ChatMessage(role="assistant", content="hi")])
    agent = Agent(client=client, registry=SkillRegistry())
    agent.run("hello")
    assert client.system_seen == DEFAULT_SYSTEM
    assert "local-first" in client.system_seen
    print("✓ loop: uses the default local-first system prompt")


def test_step_bound_stops_runaway():
    reg = SkillRegistry()
    reg.register(Skill("noop", "no-op", {"type": "object", "properties": {}}, run=lambda: "ok"))
    # always asks for a tool → never answers → must stop at max_steps
    client = ScriptedClient([ChatMessage(role="assistant", content="", tool_calls=[_tc("noop", {})])])
    agent = Agent(client=client, registry=reg, max_steps=4)
    res = agent.run("loop forever")
    assert res.steps == 4
    print("✓ loop: step bound stops a runaway tool loop")


def test_unknown_tool_recovers():
    reg = SkillRegistry()
    client = ScriptedClient([
        ChatMessage(role="assistant", content="", tool_calls=[_tc("nope", {})]),
        ChatMessage(role="assistant", content="recovered."),
    ])
    agent = Agent(client=client, registry=reg)
    res = agent.run("call a bad tool")
    assert res.answer == "recovered."
    assert "unknown skill" in client.tool_results[0]
    print("✓ loop: unknown tool → error fed back, loop recovers")


# ---------- router ----------
POLICY = {
    "models": {
        "primary": {"name": "qwen3:14b"},
        "fastFallback": {"name": "qwen3:8b"},
        "deepPlanner": {"name": "deepseek-r1"},
    },
    "routingRules": [
        {"id": "rule_low_power", "when": {"deviceState": ["battery_saver"]}, "useModel": "fastFallback"},
        {"id": "rule_high_risk", "when": {"riskLevel": ["high"]}, "useModel": "deepPlanner"},
        {"id": "rule_deep_path", "when": {"taskComplexity": ["high"]}, "useModel": "deepPlanner"},
        {"id": "rule_fast_path", "when": {"taskComplexity": ["low", "medium"], "riskLevel": ["low", "medium"]}, "useModel": "primary"},
        {"id": "rule_default", "when": {}, "useModel": "primary"},
    ],
}


def test_classify():
    # classify() returns (complexity, risk, reasons) as strings
    c, risk, _ = classify("what's the date?")
    assert c == "low" and risk == "low"
    _, risk2, _ = classify("delete the production database")
    assert risk2 == "high"
    print("✓ router: classify simple vs destructive")


def test_route_safety_order():
    r = Router(POLICY)
    assert r.route("what's the date?").rule_id == "rule_fast_path"
    assert r.route("delete production now").rule_id == "rule_high_risk"
    assert r.route("what's the date?", device_state="battery_saver").rule_id == "rule_low_power"
    # nothing falls through
    for p in ["hi", "explain why in detail", "rm -rf server", "summarize this"]:
        assert r.route(p).rule_id != "none"
    print("✓ router: safety-first ordering + no fall-through")


# ---------- skill gates ----------
def test_web_off_by_default():
    os.environ.pop("UA_WEB", None); os.environ.pop("CTWIN_WEB", None)
    # fresh import of builtin against a clean registry
    from unhosted_assistant.skills import builtin  # noqa: F401
    from unhosted_assistant.skills.base import default_registry as R
    assert "[web disabled]" in R.dispatch("web_search", {"query": "x"})
    assert "[web disabled]" in R.dispatch("fetch_url", {"url": "https://example.com"})
    print("✓ skills: web search/fetch refuse when UA_WEB is unset")


def test_file_sandbox():
    os.environ["UA_WORKSPACE"] = tempfile.mkdtemp()
    from unhosted_assistant.skills import builtin  # noqa: F401
    from unhosted_assistant.skills.base import default_registry as R
    out = R.dispatch("read_file", {"path": "../../../../etc/passwd"})
    assert "outside the workspace" in out
    print("✓ skills: file read is sandboxed (path escape blocked)")


def test_now_skill():
    from unhosted_assistant.skills import builtin  # noqa: F401
    from unhosted_assistant.skills.base import default_registry as R
    out = R.dispatch("now", {})
    assert any(d in out for d in ("202", "20"))  # has a year
    print("✓ skills: now returns a date")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall assistant tests passed.")
