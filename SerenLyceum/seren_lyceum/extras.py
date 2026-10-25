"""
seren_lyceum.extras
════════════════════════════════════════════════════════════════════════
The extras: things a training can switch on that are not the training
itself - an attention kernel, 4-bit loading, packing, a LoRA variant. None
is needed; each is offered when this box has what it needs and the person
has not switched it off on the Config tab.

One shape for all of them, so the ones a person adds are no different from
the ones that ship (24 Oct 2026: "people wanna add their own special
thing"). An extra says:

    id       its name in a job and a config
    name     what the page calls it
    say      a sentence on what it does
    needs    a Python module the TRAINING environment must import (none =
             always there); found by the trainer's own --check
    install  how to get that module, said when it is missing
    model    keyword arguments for loading the model (from_pretrained)
    train    keyword arguments for the training (TRL's SFTConfig)
    lora     keyword arguments for the adapter (PEFT's LoraConfig)
    special  "4bit": the one that needs code, not just arguments

The trainer gets the definitions a job uses copied into its job.yaml, so a
run never depends on the config it was started from. Arguments this
version of TRL or PEFT does not take are dropped with a line in the log,
never a crash at step one.

Add your own in seren-lyceum.yaml:

    extras:
      - id: my-thing
        name: My thing
        say: what it does, in a sentence
        needs: my_module
        install: pip install my-module
        train: {some_sft_option: true}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Extra:
    id: str
    name: str
    say: str = ""
    needs: Optional[str] = None
    install: str = ""
    model: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    lora: dict[str, Any] = field(default_factory=dict)
    special: str = ""
    builtin: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any], builtin: bool = False) -> "Extra":
        def mapping(k: str) -> dict[str, Any]:
            v = d.get(k)
            return dict(v) if isinstance(v, dict) else {}
        eid = str(d.get("id") or "").strip()
        if not eid:
            raise ValueError("an extra needs an id")
        special = str(d.get("special") or "")
        if special not in ("", "4bit"):
            raise ValueError(f"extra {eid!r}: special is 4bit or nothing")
        return cls(id=eid, name=str(d.get("name") or eid), say=str(d.get("say") or ""),
                   needs=str(d["needs"]).strip() or None if d.get("needs") else None,
                   install=str(d.get("install") or ""), model=mapping("model"), train=mapping("train"),
                   lora=mapping("lora"), special=special, builtin=builtin)


BUILTIN = [
    Extra("flash_attention", "Flash attention", "Faster, and much less memory on long examples.",
          needs="flash_attn", install="pip install flash-attn, built for this box's torch (it is picky)",
          model={"attn_implementation": "flash_attention_2"}),
    Extra("load_4bit", "4-bit loading (QLoRA)", "Loads the model in 4 bits: a bigger model in less memory; "
          "a little slower, a little less exact.", needs="bitsandbytes", install="pip install bitsandbytes",
          special="4bit"),
    Extra("liger", "Liger kernels", "Fused kernels that cut memory on long examples by half or more, "
          "and run a little faster.", needs="liger_kernel", install="pip install liger-kernel",
          train={"use_liger_kernel": True}),
    Extra("packing", "Packing", "Packs short examples together into full-length rows: much faster on a "
          "dataset of short examples, no difference on long ones.", train={"packing": True}),
    Extra("neftune", "NEFTune", "A little noise on the inputs while it learns; tends to make chat "
          "answers better, at no cost.", train={"neftune_noise_alpha": 5}),
    Extra("dora", "DoRA", "A LoRA that learns direction and size apart: often better at the same rank, "
          "a little slower.", lora={"use_dora": True}),
    Extra("rslora", "rsLoRA", "Scales the adapter so a high rank keeps learning; worth it at rank 32 "
          "and up.", lora={"use_rslora": True}),
    Extra("optim_8bit", "8-bit optimizer", "Keeps the optimizer's memory in 8 bits: room for a bigger "
          "batch or a longer example.", needs="bitsandbytes", install="pip install bitsandbytes",
          train={"optim": "paged_adamw_8bit"}),
]


class Registry:
    """The built-ins, the config's own, and which are switched off (kept in
    the state folder, so the Config tab can change it without a restart)."""

    def __init__(self, custom: list[dict[str, Any]], state_dir: Path):
        self.state = Path(state_dir) / "extras.json"
        self.items: dict[str, Extra] = {e.id: e for e in BUILTIN}
        self.problems: list[str] = []
        for d in custom or []:
            try:
                e = Extra.from_dict(d if isinstance(d, dict) else {})
                self.items[e.id] = e          # a config's own may replace a built-in on purpose
            except ValueError as ex:
                self.problems.append(str(ex))

    def off(self) -> set[str]:
        try:
            return set(json.loads(self.state.read_text(encoding="utf-8")).get("off") or [])
        except (OSError, ValueError):
            return set()

    def set_on(self, eid: str, on: bool) -> None:
        if eid not in self.items:
            raise KeyError(eid)
        off = self.off()
        (off.discard if on else off.add)(eid)
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"off": sorted(off)}), encoding="utf-8")
        tmp.replace(self.state)

    def modules(self) -> list[str]:
        return sorted({e.needs for e in self.items.values() if e.needs})

    def describe(self, found: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        """Every extra with whether this box has what it needs (found: the
        trainer's --check extras, module -> {version}) and whether it is on."""
        off = self.off()
        out = []
        for e in self.items.values():
            got = (found or {}).get(e.needs) if e.needs else None
            version = (got or {}).get("version") if isinstance(got, dict) else None
            out.append({**e.as_dict(), "on": e.id not in off,
                        "available": True if not e.needs else (bool(version) if found is not None else None),
                        "version": version})
        return out

    def resolve(self, ids: list[str]) -> list[dict[str, Any]]:
        """A job's extras, as the definitions the trainer will apply."""
        out = []
        for eid in ids or []:
            if eid not in self.items:
                raise ValueError(f"no extra called {eid!r}")
            out.append(self.items[eid].as_dict())
        return out
