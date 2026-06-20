"""
Unhosted Assistant — a clean, local-first AI assistant CLI.

  ua "what's the date?"            one-shot
  ua                              interactive REPL
  ua --model llama3.2 "..."       pick a model
  ua --route-explain "..."        show which model the policy picked

Talks to a local model: your own Unhosted cluster if it's running (auto-detected),
else Ollama, else any OpenAI-compatible server (LM Studio, llama.cpp, Jan).
Local-first by design — no cloud, no API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent.loop import Agent
from .agent.router import Router
from .llm.ollama_client import OllamaError
from .llm.openai_client import OpenAIError
from .llm import providers
from . import skills  # noqa: F401
from .skills import builtin  # noqa: F401  (registers built-in skills)
from .skills.base import default_registry

LLM_ERRORS = (OllamaError, OpenAIError)


def _load_config() -> dict:
    for p in (Path.cwd() / "assistant_config.json",):
        try:
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def build_agent(model: str | None = None, *, route: bool = True) -> Agent:
    cfg = _load_config()
    model = model or os.environ.get("UA_MODEL") or cfg.get("model") or "qwen2.5:7b"
    host = os.environ.get("UA_OLLAMA_HOST") or cfg.get("host") or "http://localhost:11434"
    backend = providers.MultiBackend(
        ollama_host=host,
        openai_base=providers.openai_base_url(cfg),
        openai_label=providers.openai_label(cfg),
    )
    client = backend.client_for(model)
    if route and backend.is_openai_model(model):
        route = False
    router = Router() if route else None
    agent = Agent(client=client, registry=default_registry, router=router)
    agent.configured_model = model  # type: ignore[attr-defined]
    agent.backend = backend  # type: ignore[attr-defined]
    return agent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ua", description="Local-first AI assistant (Unhosted).")
    ap.add_argument("prompt", nargs="*", help="one-shot prompt; omit for a REPL")
    ap.add_argument("--model", help="model id (pins it, disables routing)")
    ap.add_argument("--skills", action="store_true", help="list skills and exit")
    ap.add_argument("--models", action="store_true", help="list available models and exit")
    ap.add_argument("--no-route", action="store_true", help="disable policy routing")
    ap.add_argument("--route-explain", action="store_true", help="print the routing decision")
    args = ap.parse_args(argv)

    if args.skills:
        for n in default_registry.names():
            sk = default_registry.get(n)
            print(f"  {n:<14} {sk.description if sk else ''}")
        return 0

    use_routing = not args.no_route and not args.model
    agent = build_agent(args.model, route=use_routing)

    if args.models:
        backend = getattr(agent, "backend", None)
        models = backend.list_models() if backend else []
        print("\n".join(models) if models else "(no models found — start Ollama / Unhosted / LM Studio)")
        return 0

    def ask(prompt: str) -> None:
        try:
            result = agent.run(prompt)
        except LLM_ERRORS as e:
            print(f"⚠ {e}", file=sys.stderr); return
        if args.route_explain and result.route is not None:
            print(result.route.explain(), file=sys.stderr)
        print(result.answer)

    if args.prompt:
        ask(" ".join(args.prompt))
        return 0

    print("Unhosted Assistant · local. Ctrl-D or 'exit' to quit.\n")
    while True:
        try:
            line = input("» ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return 0
        if line in {"exit", "quit"}:
            return 0
        if line:
            ask(line)


if __name__ == "__main__":
    raise SystemExit(main())
