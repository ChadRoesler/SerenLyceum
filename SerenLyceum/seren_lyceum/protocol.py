"""
seren_lyceum.protocol
════════════════════════════════════════════════════════════════════════
The trainer's lines, the same five the old engine's RunPod trainer printed
(SerenCommandCenter, 2026), so anything that read those reads these:

    [SEREN:STATUS] train_started
    [SEREN:PROGRESS] step=40/300 epoch=0.53 loss=1.2345 lr=1.80e-04
    [SEREN:METRIC] eval_loss=0.9876
    [SEREN:ERROR] the dataset has no rows
    [SEREN:COMPLETE] out/model-Q5_K_M.gguf

STATUS is a word for where the run is; PROGRESS and METRIC carry key=value
pairs; ERROR and COMPLETE carry free text. A line that is not one of these is
ordinary log output and is kept as such. The trainer prints them to its log;
status.json in the run directory is what the service reads at a glance, and
these lines are the story of how it got there.
"""
from __future__ import annotations

import re
from typing import Any, Optional

KINDS = ("STATUS", "PROGRESS", "METRIC", "ERROR", "COMPLETE")
_LINE = re.compile(r"^\s*\[SEREN:(STATUS|PROGRESS|METRIC|ERROR|COMPLETE)\]\s?(.*)$")
_PAIR = re.compile(r"([A-Za-z_][\w.]*)=(\S+)")


def _value(raw: str) -> Any:
    """A number when it reads as one, else the text."""
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def pairs(text: str) -> dict[str, Any]:
    """key=value pairs; `step=40/300` becomes step 40 and total 300."""
    out: dict[str, Any] = {}
    for k, v in _PAIR.findall(text or ""):
        if k == "step" and "/" in v:
            a, _, b = v.partition("/")
            out["step"], out["total"] = _value(a), _value(b)
        else:
            out[k] = _value(v)
    return out


def parse(line: str) -> Optional[tuple[str, Any]]:
    """(kind, payload) for a protocol line, None for anything else. PROGRESS
    and METRIC payloads are dicts; the rest are the text."""
    m = _LINE.match(line or "")
    if not m:
        return None
    kind, rest = m.group(1), m.group(2).strip()
    return (kind, pairs(rest)) if kind in ("PROGRESS", "METRIC") else (kind, rest)


def line(kind: str, payload: Any = "") -> str:
    """One protocol line. A dict payload becomes key=value pairs, with
    step and total written as step=N/T the way the old engine wrote them."""
    if kind not in KINDS:
        raise ValueError(f"not a protocol kind: {kind}")
    if isinstance(payload, dict):
        d = dict(payload)
        bits = []
        if "step" in d:
            total = d.pop("total", None)
            step = d.pop("step")
            bits.append(f"step={step}/{total}" if total is not None else f"step={step}")
        for k, v in d.items():
            if isinstance(v, float):
                v = f"{v:.2e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.4f}"
            bits.append(f"{k}={str(v).replace(' ', '_')}")
        payload = " ".join(bits)
    return f"[SEREN:{kind}] {payload}".rstrip()
