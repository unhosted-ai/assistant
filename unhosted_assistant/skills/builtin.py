"""
Built-in skills for the general assistant — safe, local, useful. Importing this
module registers them on the default registry.

  now         current date/time
  read_file   read a UTF-8 text file (sandboxed to a workspace)
  list_dir    list a folder in the workspace
  web_search  search the web (opt-in: CTWIN_WEB=1)
  fetch_url   fetch a page as readable text (opt-in)
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from .base import default_registry as R


def _workspace() -> Path:
    root = Path(os.environ.get("UA_WORKSPACE", Path.home() / ".unhosted-assistant" / "workspace"))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _safe_path(rel: str) -> Path:
    root = _workspace()
    p = (root / rel).resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"path '{rel}' is outside the workspace")
    return p


@R.add("now", "Get the current date and time (local).")
def now() -> str:
    return _dt.datetime.now().strftime("%A, %B %d, %Y · %H:%M")


@R.add("list_dir", "List files in a folder inside the workspace.",
       {"type": "object", "properties": {"path": {"type": "string"}}})
def list_dir(path: str = "") -> str:
    p = _safe_path(path or ".")
    if not p.exists():
        return f"[empty] '{path or '.'}' does not exist"
    items = sorted(os.listdir(p))
    return "\n".join(items) if items else "[empty]"


@R.add("read_file", "Read a UTF-8 text file from the workspace (first ~8 KB).",
       {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
def read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.is_file():
        return f"[not found] '{path}'"
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[:8000] + ("\n…[truncated]" if len(text) > 8000 else "")


def _web_enabled() -> bool:
    return os.environ.get("UA_WEB", os.environ.get("CTWIN_WEB", "")).strip() in {"1", "true", "yes", "on"}


@R.add("web_search", "Search the internet and return the top results (title, URL, "
       "snippet). Use for current events or facts you don't know. Requires web access.",
       {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
def web_search(query: str) -> str:
    if not _web_enabled():
        return "[web disabled] Internet is off (local-first default). Enable with UA_WEB=1."
    import html as _h, re, urllib.parse as up, urllib.request as ur, urllib.error as ue
    q = (query or "").strip()
    if not q:
        return "[refused] empty query."
    data = up.urlencode({"q": q}).encode()
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    try:
        r = ur.Request("https://lite.duckduckgo.com/lite/", data=data, headers={"User-Agent": ua})
        with ur.urlopen(r, timeout=15) as resp:
            page = resp.read(400_000).decode("utf-8", "replace")
    except (ue.URLError, ue.HTTPError) as e:
        return f"[error] search failed: {e}"

    def clean(s: str) -> str:
        return _h.unescape(re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", "", s))).strip()

    links = re.findall(r'<a[^>]*href="(http[^"]+)"[^>]*class=[\'"]result-link[\'"][^>]*>(.*?)</a>', page, re.S)
    snips = re.findall(r'<td[^>]*class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', page, re.S)
    out = []
    for i, (href, title) in enumerate(links[:5]):
        out.append(f"{i+1}. {clean(title)}\n   {href}\n   {clean(snips[i]) if i < len(snips) else ''}")
    return ("Top results for “%s”:\n\n" % q + "\n\n".join(out)) if out else f"[no results] for '{q}'."


@R.add("fetch_url", "Fetch a web page and return its readable text. Requires web access.",
       {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]})
def fetch_url(url: str) -> str:
    if not _web_enabled():
        return "[web disabled] Internet is off (local-first default). Enable with UA_WEB=1."
    import re, urllib.request as ur, urllib.error as ue
    url = (url or "").strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "[refused] Only http(s) URLs are allowed."
    try:
        req = ur.Request(url, headers={"User-Agent": "UnhostedAssistant/0.1"})
        with ur.urlopen(req, timeout=15) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(600_000)
        text = raw.decode("utf-8", "replace")
        if "html" in ctype.lower() or text.lstrip().startswith("<"):
            text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return text[:4000] + ("…[truncated]" if len(text) > 4000 else "")
    except (ue.URLError, ue.HTTPError) as e:
        return f"[error] couldn't fetch {url}: {e}"
