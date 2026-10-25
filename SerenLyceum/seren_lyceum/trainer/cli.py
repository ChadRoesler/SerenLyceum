"""
seren-lyceum-train - train one job, from its directory.

    seren-lyceum-train /path/to/job          run it: prepare, download, train,
                                             evaluate, export
    seren-lyceum-train --check               what this environment can do, as
                                             one line of JSON (the service's
                                             "is this box ready" question)

The job directory is the whole interface (see seren_lyceum.jobs): read
job.yaml and data/, write status.json, metrics.jsonl, out/, and print the
[SEREN:...] protocol lines to stdout, which the service sends to train.log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _module_version(mod: str) -> Any:
    """The module's version, "present" when it has none to give, None when
    it is absent - or present and broken, which means the same here."""
    try:
        m = __import__(mod)
    except Exception:  # noqa: BLE001
        return None
    return str(getattr(m, "__version__", "") or "present")


def check(modules: list[str] | None = None) -> dict[str, Any]:
    """What this environment has, without raising: torch and the GPU, the
    training libraries, and the modules the extras need (seren_lyceum.extras:
    the service names them). ok means a job can run here."""
    rep: dict[str, Any] = {"python": sys.version.split()[0], "missing": []}
    import os
    if modules is None:
        from ..extras import BUILTIN
        modules = sorted({e.needs for e in BUILTIN if e.needs})
    if os.environ.get("SEREN_LYCEUM_FAKE_TRAIN") == "1":
        # SEREN_LYCEUM_FAKE_EXTRAS names modules a fake box pretends to have
        have = set(filter(None, os.environ.get("SEREN_LYCEUM_FAKE_EXTRAS", "").split(",")))
        return {**rep, "ok": True, "fake": True, "gpu": "a pretend GPU (SEREN_LYCEUM_FAKE_TRAIN=1)",
                "gpu_memory_gb": 0, "extras": {m: {"version": "fake" if m in have else None} for m in modules}}
    try:
        import torch
        rep["torch"] = torch.__version__
        rep["cuda"] = bool(torch.cuda.is_available())
        if rep["cuda"]:
            props = torch.cuda.get_device_properties(0)
            rep["gpu"] = props.name
            rep["gpu_memory_gb"] = round(props.total_memory / 2**30)
            rep["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:  # noqa: BLE001
        rep["missing"].append(f"torch ({type(e).__name__})")
    for mod in ("transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            rep["missing"].append(mod)
    # What the box has that a training may choose to use, or not: none of
    # these is required. Each can be present and still unwanted - a
    # flash_attn wheel built for another torch crashes at the first step,
    # and MsMoEMaker learned bitsandbytes the hard way.
    rep["extras"] = {m: {"version": _module_version(m)} for m in modules}
    rep["ok"] = not rep["missing"]
    if not rep["ok"]:
        rep["why"] = "missing in the training environment: " + ", ".join(rep["missing"])
    return rep


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="seren-lyceum-train",
                                description="Train one SerenLyceum job from its directory.")
    p.add_argument("job_dir", nargs="?", help="the job's directory (job.yaml, data/)")
    p.add_argument("--check", action="store_true", help="report what this environment can do, as JSON")
    p.add_argument("--modules", default=None, help="with --check: the modules to look for, comma-separated")
    a = p.parse_args(argv)
    if a.check:
        print(json.dumps(check([m for m in a.modules.split(",") if m] if a.modules is not None else None)))
        sys.exit(0)
    if not a.job_dir:
        p.error("give a job directory, or --check")
    from .run import run_job
    sys.exit(run_job(Path(a.job_dir)))


if __name__ == "__main__":
    main()
