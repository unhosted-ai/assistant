"""
Multi-backend model discovery + selection.

The twin can drive more than one local backend at once:
  - Ollama                 (http://localhost:11434)        — default
  - OpenAI-compatible      (LM Studio, llama.cpp, Jan, …)  — opt-in

This module probes whichever backends are reachable, merges their models into a
single list for the picker, and builds the right client for a chosen model.

Model id scheme (kept backward-compatible):
  - Ollama models keep their bare name, e.g. ``llama3.2`` or ``qwen2.5:3b``.
  - OpenAI-backend models are prefixed with the provider label and a slash,
    e.g. ``lmstudio/qwen2.5-7b-instruct``. The label is configurable; the
    prefix is what /api/model uses to route to the right client.

Everything is local-first and opt-in: the OpenAI backend is only probed when a
base URL is configured (CTWIN_OPENAI_BASE / config), so default installs that
only use Ollama are unaffected.
"""

from __future__ import annotations

import os
from typing import Any

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

# Separates the provider label from the model name in a tagged id.
# A slash is safe: Ollama names use ':' for tags but not '/'.
SEP = "/"

DEFAULT_OPENAI_BASE = "http://localhost:1234/v1"  # LM Studio default
DEFAULT_OPENAI_LABEL = "lmstudio"


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in {"1", "true", "yes", "on"}


# Unhosted (https://github.com/unhosted-ai) runs a local, OpenAI-compatible
# inference endpoint when its daemon is up. If we find it, Anita can use the
# user's own pooled hardware as a model backend — local-first, no setup.
UNHOSTED_BASE = "http://127.0.0.1:7777/v1"


def unhosted_base_url() -> str | None:
    """Return Unhosted's OpenAI-compatible base URL IF its daemon is reachable,
    else None. Overridable with CTWIN_UNHOSTED_BASE; disable auto-detect entirely
    with CTWIN_NO_UNHOSTED=1 (used in tests for deterministic behaviour)."""
    if _truthy(os.environ.get("CTWIN_NO_UNHOSTED")):
        return None
    base = os.environ.get("CTWIN_UNHOSTED_BASE", "").strip() or UNHOSTED_BASE
    try:
        import urllib.request
        root = base.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + "/v1/models", timeout=2) as r:
            if r.status == 200:
                return base
    except Exception:
        pass
    return None


def openai_base_url(cfg: dict[str, Any] | None = None) -> str | None:
    """Return the configured OpenAI-compatible base URL, or None if not enabled.

    Priority: explicit env/config → a running **Unhosted** daemon (auto) → LM
    Studio (if CTWIN_USE_LMSTUDIO). Returns None otherwise so the OpenAI backend
    stays off by default.
    """
    env = os.environ.get("CTWIN_OPENAI_BASE")
    if env and env.strip():
        return env.strip()
    cfg = cfg or {}
    for key in ("openai_base", "openaiBase"):
        if isinstance(cfg.get(key), str) and cfg[key].strip():
            return cfg[key].strip()
    block = cfg.get("openai") or cfg.get("lmstudio")
    if isinstance(block, dict):
        base = block.get("base") or block.get("host") or block.get("base_url")
        if isinstance(base, str) and base.strip():
            return base.strip()
    # auto: use Unhosted if it's running (the user owns the hardware)
    unh = unhosted_base_url()
    if unh:
        return unh
    if _truthy(os.environ.get("CTWIN_USE_LMSTUDIO")):
        return DEFAULT_OPENAI_BASE
    return None


def openai_label(cfg: dict[str, Any] | None = None) -> str:
    """The short provider label used as the model-id prefix. 'unhosted' when the
    backend is a running Unhosted daemon, else config/env, else 'lmstudio'."""
    env = os.environ.get("CTWIN_OPENAI_LABEL")
    if env and env.strip():
        return env.strip()
    cfg = cfg or {}
    block = cfg.get("openai") or cfg.get("lmstudio")
    if isinstance(block, dict) and isinstance(block.get("label"), str) and block["label"].strip():
        return block["label"].strip()
    # if we resolved to Unhosted's endpoint, label it as such
    base = openai_base_url(cfg)
    if base and base == (os.environ.get("CTWIN_UNHOSTED_BASE", "").strip() or UNHOSTED_BASE):
        return "unhosted"
    return DEFAULT_OPENAI_LABEL


def split_model_id(model_id: str) -> tuple[str | None, str]:
    """Split a (possibly tagged) model id into (provider_label, bare_name).

    'lmstudio/qwen2.5-7b' -> ('lmstudio', 'qwen2.5-7b')
    'llama3.2'            -> (None, 'llama3.2')
    'qwen2.5:3b'         -> (None, 'qwen2.5:3b')   (the ':' tag is not a provider)
    """
    if SEP in model_id:
        label, _, name = model_id.partition(SEP)
        if label and name:
            return label, name
    return None, model_id


class MultiBackend:
    """Holds the configured backends and resolves models/clients across them."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        openai_base: str | None = None,
        openai_label: str = DEFAULT_OPENAI_LABEL,
        timeout: float = 120.0,
    ) -> None:
        self.ollama_host = ollama_host
        self.openai_base = openai_base
        self.openai_label = openai_label
        self.timeout = timeout

    # ---- discovery ----------------------------------------------------------
    def list_models(self) -> list[str]:
        """Merged, provider-tagged model list for the picker. Ollama models keep
        their bare names; OpenAI-backend models are prefixed with the label."""
        models: list[str] = []
        try:
            models.extend(OllamaClient(host=self.ollama_host).available_models())
        except Exception:  # noqa: BLE001 - a down backend just contributes nothing
            pass
        if self.openai_base:
            try:
                oai = OpenAIClient(host=self.openai_base, timeout=self.timeout)
                for name in oai.available_models():
                    models.append(f"{self.openai_label}{SEP}{name}")
            except Exception:  # noqa: BLE001
                pass
        # de-dupe while preserving order
        seen: set[str] = set()
        out: list[str] = []
        for m in models:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    # ---- client construction ------------------------------------------------
    def client_for(self, model_id: str, *, temperature: float = 0.4):
        """Build the right client for a model id, switching backend by prefix."""
        label, name = split_model_id(model_id)
        if label is not None and self.openai_base and label == self.openai_label:
            return OpenAIClient(
                model=name,
                host=self.openai_base,
                timeout=self.timeout,
                temperature=temperature,
            )
        # default: Ollama (bare names, or unknown prefixes fall back here)
        return OllamaClient(
            model=name,
            host=self.ollama_host,
            timeout=self.timeout,
            temperature=temperature,
        )

    def is_openai_model(self, model_id: str) -> bool:
        label, _ = split_model_id(model_id)
        return label is not None and label == self.openai_label
