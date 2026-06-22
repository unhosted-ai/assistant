# Unhosted Assistant

A clean, **local-first AI assistant** that runs on your own hardware. It talks to
a local model — your own [Unhosted](https://github.com/unhosted-ai) cluster if
it's running (auto-detected), else [Ollama](https://ollama.com), else any
OpenAI-compatible server (LM Studio, llama.cpp, Jan) — and uses **skills** (tools)
to do real things: tell the time, read files, search the web, fetch a page.

**No cloud. No API key. No telemetry.** By design.

> Part of the [UnhostedAI](https://github.com/unhosted-ai) family — frontier AI on
> hardware you own.

---

## What it does

- **Answers with a local model** — picks the right one per request via a small
  routing policy (a fast model for quick asks, a deeper one for complex/risky ones).
- **Uses your Unhosted cluster automatically** — if the Unhosted daemon is up, it
  routes inference there (your pooled hardware), no setup.
- **Tools** — `now`, `read_file`, `list_dir` (sandboxed), and opt-in web
  `web_search` + `fetch_url`.
- **Bounded, safe loop** — a step limit, and a skill that errors is fed back to the
  model to recover from rather than crashing the run.

This is the general assistant. (The personal "digital twin" — voice cloning,
persona, private memory — lives in a separate project.)

## Quick start

```bash
# 1. a local model: your Unhosted daemon, or Ollama:
ollama pull qwen2.5:7b        # or llama3.2, etc.

# 2. run it
python -m unhosted_assistant "what's the date?"
python -m unhosted_assistant                       # interactive REPL
python -m unhosted_assistant --models              # list available models
python -m unhosted_assistant --route-explain "..." # show which model was chosen
UA_WEB=1 python -m unhosted_assistant "what's new in AI today?"   # web on
```

## Model backends

- **Unhosted** — auto-detected at `127.0.0.1:7777` (override `UA_UNHOSTED_BASE`).
- **Ollama** — `UA_OLLAMA_HOST` (default `http://localhost:11434`).
- **OpenAI-compatible** (LM Studio etc.) — set `UA_OPENAI_BASE` or
  `UA_USE_LMSTUDIO=1`.

Routing applies to local Ollama models; an explicit `--model` pins one and turns
routing off.

## Adding a skill

```python
from unhosted_assistant.skills.base import default_registry as R

@R.add("weather", "Get the weather for a city.",
       {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
def weather(city: str) -> str:
    return f"It's pleasant in {city}."   # call a real API here
```

## Privacy

Local-first. Web access is **off by default** (`UA_WEB=1` to enable). The file
tools are sandboxed to `~/.unhosted-assistant/workspace`. Nothing is uploaded.

## License

MIT.
