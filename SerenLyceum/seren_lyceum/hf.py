"""
seren_lyceum.hf
════════════════════════════════════════════════════════════════════════
The Hugging Face login, for the models that need one (Llama, Gemma: a login
that has accepted the model's licence).

It lives where Hugging Face keeps it - $HF_HOME/token, what `hf auth login`
writes - so there is one login per box, shared by every tool on it, and node
setup or a person at a shell can fill it in exactly as the Lyceum does. The
Lyceum looks there first (or at $HF_TOKEN in its environment) and only asks
when it is missing. A token given here is written there, owner-only, and is
never sent back, logged or put on a command line: the trainer reads it the
way every Hugging Face tool does.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

_TOKEN = re.compile(r"^hf_[A-Za-z0-9]{20,}$")


def hf_home(configured: str = "") -> Path:
    raw = configured or os.environ.get("HF_HOME") or "~/.cache/huggingface"
    return Path(os.path.expanduser(raw))


def login(configured_home: str = "") -> dict[str, Any]:
    """Whether there is a login, and where it comes from - never the token."""
    if os.environ.get("HF_TOKEN"):
        return {"found": True, "where": "the HF_TOKEN environment variable"}
    p = hf_home(configured_home) / "token"
    if p.is_file() and p.read_text(encoding="utf-8", errors="replace").strip():
        return {"found": True, "where": f"this box's Hugging Face login ({p})"}
    return {"found": False, "where": None, "path": str(p)}


def save(token: str, configured_home: str = "") -> dict[str, Any]:
    token = (token or "").strip()
    if not _TOKEN.match(token):
        raise ValueError("that does not look like a Hugging Face token (they start with hf_)")
    home = hf_home(configured_home)
    home.mkdir(parents=True, exist_ok=True)
    p = home / "token"
    tmp = home / f".token.{os.getpid()}"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    os.replace(tmp, p)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass
    return login(configured_home)


def needs_login(model_id: str, gated: Optional[set[str]] = None) -> bool:
    """A rough read of whether a model wants a login: the gated families,
    plus anything the caller marks."""
    m = (model_id or "").lower()
    return m.startswith(("meta-llama/", "google/gemma", "mistralai/mistral-large")) or model_id in (gated or set())
