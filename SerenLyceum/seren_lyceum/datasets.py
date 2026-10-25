"""
seren_lyceum.datasets
════════════════════════════════════════════════════════════════════════
What you want it to learn. A dataset is a JSONL file, one example per line,
in whichever of the common shapes you already have:

    chat         {"messages": [{"role": "user", "content": ...},
                               {"role": "assistant", "content": ...}]}
    sharegpt     {"conversations": [{"from": "human", "value": ...}, ...]}
    alpaca       {"instruction": ..., "input": ..., "output": ...}
    prompt       {"prompt": ..., "completion": ...}
    text         {"text": ...}           (plain text, learned as it reads)

Everything but plain text is turned into chat messages, so the trainer has
one shape to handle. The file is read once on the way in and the person is
told, in words, what was found: how many examples, the shape, the lines that
could not be used and why, how long they run. Where it came from is not
asked and not checked: we just train.

A dataset is a folder too: data.jsonl (normalised) and dataset.json (what
was found). A job copies data.jsonl in, so replacing a dataset never
changes a run that already used it.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .jobs import _read_json, _write_json, slug

ROLES = {"system", "user", "assistant"}
_SHAREGPT_ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                  "system": "system", "chatgpt": "assistant", "bing": "assistant", "bot": "assistant"}
SHAPE_SAY = {
    "chat": "chat conversations",
    "sharegpt": "ShareGPT conversations (turned into chat)",
    "alpaca": "instruction / response pairs (turned into chat)",
    "prompt": "prompt / completion pairs (turned into chat)",
    "text": "plain text",
}
MAX_BAD_SHOWN = 8


class DatasetError(ValueError):
    pass


def _shape(row: dict[str, Any]) -> Optional[str]:
    if isinstance(row.get("messages"), list):
        return "chat"
    if isinstance(row.get("conversations"), list):
        return "sharegpt"
    if "instruction" in row and "output" in row:
        return "alpaca"
    if "prompt" in row and "completion" in row:
        return "prompt"
    if isinstance(row.get("text"), str):
        return "text"
    return None


def _as_messages(row: dict[str, Any], shape: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """The row in chat (or text) form, or why it cannot be used."""
    if shape == "text":
        t = row.get("text") or ""
        return ({"text": t}, None) if t.strip() else (None, "the text is empty")
    if shape == "chat":
        msgs = row["messages"]
    elif shape == "sharegpt":
        msgs = []
        for turn in row["conversations"]:
            if not isinstance(turn, dict):
                return None, "a turn is not an object"
            role = _SHAREGPT_ROLE.get(str(turn.get("from", "")).lower())
            if role is None:
                return None, f"unknown speaker {turn.get('from')!r}"
            msgs.append({"role": role, "content": turn.get("value", "")})
    elif shape == "alpaca":
        ask = str(row.get("instruction") or "")
        if str(row.get("input") or "").strip():
            ask = f"{ask}\n\n{row['input']}"
        msgs = [{"role": "user", "content": ask}, {"role": "assistant", "content": str(row.get("output") or "")}]
        if str(row.get("system") or "").strip():
            msgs.insert(0, {"role": "system", "content": str(row["system"])})
    else:  # prompt
        p, c = row["prompt"], row["completion"]
        if isinstance(p, list) and isinstance(c, list):     # TRL's conversational prompt/completion
            msgs = [*p, *c]
        else:
            msgs = [{"role": "user", "content": str(p)}, {"role": "assistant", "content": str(c)}]
    clean = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            return None, f"a message has role {m.get('role') if isinstance(m, dict) else m!r}, not system/user/assistant"
        content = m.get("content")
        if not isinstance(content, str):
            return None, "a message's content is not text"
        clean.append({"role": m["role"], "content": content})
    if not any(m["role"] == "assistant" and m["content"].strip() for m in clean):
        return None, "there is no answer (assistant message) to learn from"
    if clean[-1]["role"] != "assistant":
        return None, "it does not end with an answer (the last message is not the assistant's)"
    return {"messages": clean}, None


def inspect_lines(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read JSONL lines: the usable rows (normalised) and what was found."""
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    shapes: dict[str, int] = {}
    chars: list[int] = []
    n_bad = 0
    for i, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue

        def miss(why: str) -> None:
            nonlocal n_bad
            n_bad += 1
            if len(bad) < MAX_BAD_SHOWN:
                bad.append({"line": i, "why": why})
        try:
            row = json.loads(raw)
        except ValueError:
            miss("not valid JSON")
            continue
        if not isinstance(row, dict):
            miss("not a JSON object")
            continue
        shape = _shape(row)
        if shape is None:
            miss("not a shape the Lyceum knows (messages, conversations, instruction/output, prompt/completion or text)")
            continue
        norm, why = _as_messages(row, shape)
        if norm is None:
            miss(why or "unusable")
            continue
        shapes[shape] = shapes.get(shape, 0) + 1
        rows.append(norm)
        chars.append(len(norm["text"]) if "text" in norm else sum(len(m["content"]) for m in norm["messages"]))
    kinds = {"text" if s == "text" else "chat" for s in shapes}
    found = {
        "rows": len(rows),
        "unusable": n_bad,
        "unusable_shown": bad,
        "shapes": shapes,
        "shape_say": ", ".join(f"{n} {SHAPE_SAY[s]}" for s, n in sorted(shapes.items(), key=lambda kv: -kv[1])),
        "chars_avg": round(sum(chars) / len(chars)) if chars else 0,
        "chars_max": max(chars) if chars else 0,
        # a rough count a person can reason with; the trainer counts for real
        "tokens_est": round(sum(chars) / 3.6),
        "mixed": len(kinds) > 1,
    }
    return rows, found


def _question_row(raw: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """One line of a question set as {"messages": [... user, (assistant)]}:
    any shape the learning sets take, with the answer optional, or a bare
    question as a line of plain text."""
    try:
        row = json.loads(raw)
    except ValueError:
        return {"messages": [{"role": "user", "content": raw}]}, None
    if isinstance(row, str):
        return ({"messages": [{"role": "user", "content": row}]}, None) if row.strip() else (None, "it is empty")
    if not isinstance(row, dict):
        return None, "not a question (a line of text, or a JSON object)"
    for k in ("question", "q", "ask"):
        if isinstance(row.get(k), str) and row[k].strip():
            msgs = [{"role": "user", "content": row[k]}]
            a = row.get("answer") or row.get("expected") or row.get("a")
            if isinstance(a, str) and a.strip():
                msgs.append({"role": "assistant", "content": a})
            return {"messages": msgs}, None
    shape = _shape(row)
    if shape == "text":
        return {"messages": [{"role": "user", "content": row["text"]}]}, None
    if "instruction" in row:
        ask = str(row.get("instruction") or "")
        if str(row.get("input") or "").strip():
            ask = f"{ask}\n\n{row['input']}"
        msgs = [{"role": "user", "content": ask}]
        if str(row.get("output") or "").strip():
            msgs.append({"role": "assistant", "content": str(row["output"])})
        return {"messages": msgs}, None
    if shape == "prompt" or ("prompt" in row and "completion" not in row):
        p = row["prompt"]
        msgs = list(p) if isinstance(p, list) else [{"role": "user", "content": str(p)}]
        c = row.get("completion")
        if c:
            msgs += list(c) if isinstance(c, list) else [{"role": "assistant", "content": str(c)}]
        return {"messages": [m for m in msgs if isinstance(m, dict)]}, None
    if shape in ("chat", "sharegpt"):
        norm, why = _as_messages(row, shape)
        if norm is not None:
            return norm, None
        msgs = row.get("messages") or []
        if msgs and all(isinstance(m, dict) and m.get("role") in ROLES for m in msgs) \
                and any(m["role"] == "user" for m in msgs):
            return {"messages": [{"role": m["role"], "content": str(m.get("content") or "")} for m in msgs]}, None
        return None, why
    return None, "not a question the Lyceum can read (a line of text, question/answer, messages, prompt/completion)"


def inspect_questions(lines: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a question set: the usable questions and what was found."""
    rows: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    n_bad = 0
    for i, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        row, why = _question_row(raw)
        if row is None or not any(m["role"] == "user" and m["content"].strip() for m in row["messages"]):
            n_bad += 1
            if len(bad) < MAX_BAD_SHOWN:
                bad.append({"line": i, "why": why or "there is no question in it"})
            continue
        rows.append(row)
    answered = sum(1 for r in rows if r["messages"][-1]["role"] == "assistant")
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
    return rows, {
        "rows": len(rows), "unusable": n_bad, "unusable_shown": bad, "shapes": {"chat": len(rows)} if rows else {},
        "answered": answered,
        "shape_say": (f"questions, {answered} with the answer you expect" if answered else "questions"),
        "chars_avg": round(sum(chars) / len(chars)) if chars else 0, "chars_max": max(chars) if chars else 0,
        "tokens_est": round(sum(chars) / 3.6), "mixed": False,
    }


class DatasetStore:
    """The datasets folder: one folder per dataset, named by the person."""

    def __init__(self, root: Path):
        self.root = Path(root) / "datasets"
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, name: str) -> Path:
        d = self.root / slug(name, 60)
        if not d.is_dir():
            raise KeyError(name)
        return d

    def add(self, name: str, lines: Iterable[str], source: str = "upload", note: str = "",
            use: str = "learn") -> dict[str, Any]:
        """Keep a dataset: normalised rows and what was found. Refused, with
        the reason, when nothing in it can be used or it mixes plain text with
        conversations (one run learns one way). use "questions" is a question
        set: what a training is tested on, the answers optional (24 Oct 2026:
        "upload a series of questions and use that for the drop down")."""
        name = (name or "").strip()
        if not name:
            raise DatasetError("give the dataset a name")
        if use not in ("learn", "questions"):
            raise DatasetError("a dataset is to learn from or questions to test with")
        rows, found = inspect_questions(lines) if use == "questions" else inspect_lines(lines)
        if not rows:
            why = found["unusable_shown"][0]["why"] if found["unusable_shown"] else "the file is empty"
            raise DatasetError(f"nothing in it can be used ({why})")
        if found["mixed"]:
            raise DatasetError("it mixes plain text with conversations; split them into two datasets")
        d = self.root / slug(name, 60)
        tmp = self.root / f".{slug(name, 60)}.{int(time.time() * 1000)}"
        tmp.mkdir()
        with (tmp / "data.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = {"name": name, "id": d.name, "source": source, "note": note, "use": use, "added_at": time.time(), **found}
        _write_json(tmp / "dataset.json", meta)
        if d.exists():
            shutil.rmtree(d)
        tmp.rename(d)
        return meta

    def get(self, name: str) -> dict[str, Any]:
        return _read_json(self.dir(name) / "dataset.json") or {}

    def sample(self, name: str, n: int = 3) -> list[dict[str, Any]]:
        out = []
        with (self.dir(name) / "data.jsonl").open(encoding="utf-8") as fh:
            for raw in fh:
                out.append(json.loads(raw))
                if len(out) >= n:
                    break
        return out

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (d / "dataset.json").is_file():
                rows.append(_read_json(d / "dataset.json") or {"id": d.name})
        rows.sort(key=lambda r: -(r.get("added_at") or 0))
        return rows

    def delete(self, name: str) -> None:
        shutil.rmtree(self.dir(name))


def valid_name(name: str) -> bool:
    return bool(re.search(r"[A-Za-z0-9]", name or ""))
