"""
seren_lyceum.backends.local
════════════════════════════════════════════════════════════════════════
Training on this box. The trainer is forked in the training environment
(local.python: a venv with seren-lyceum[train] and a torch built for this
GPU), detached so it outlives a service restart, its output written straight
into the run directory's train.log. The service never holds the only copy of
anything: after a restart it finds the run by its pid and its folder.

describe() asks the training environment what it has - torch, the GPU, its
memory - by running the trainer's own `--check`, once every ten minutes at
most, so the UI can say "the Spark: NVIDIA GB10, ready" or exactly what is
missing, before anyone presses Start.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from . import pid_alive

CHECK_TTL = 600.0


class LocalBackend:
    name = "local"
    kind = "local"

    def __init__(self, cfg):
        self.cfg = cfg
        self.local = cfg.local
        self._check: Optional[dict[str, Any]] = None
        self._checked_at = 0.0
        self._procs: dict[int, subprocess.Popen] = {}
        self.modules: list[str] = []      # what the extras need; the app sets it from the registry

    # -- what the UI says -----------------------------------------------------
    def check(self, fresh: bool = False) -> dict[str, Any]:
        """The training environment's own report (torch, CUDA, the GPU), or
        why there is none."""
        if not self.local.python:
            return {"ok": False, "why": "this box has no training environment set up (local.python is empty)"}
        if not fresh and self._check is not None and time.time() - self._checked_at < CHECK_TTL:
            return self._check
        try:
            env = {**os.environ, **({"HF_HOME": self.local.hf_home} if self.local.hf_home else {})}
            out = subprocess.run([self.local.python, "-m", "seren_lyceum.trainer.cli", "--check",
                                  "--modules", ",".join(self.modules)],
                                 capture_output=True, text=True, timeout=180, env=env)
            rep = json.loads((out.stdout or "").strip().splitlines()[-1]) if out.stdout.strip() else None
            if rep is None:
                rep = {"ok": False, "why": (out.stderr or "the check printed nothing").strip()[-400:]}
        except FileNotFoundError:
            rep = {"ok": False, "why": f"the training environment is not there: {self.local.python}"}
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            rep = {"ok": False, "why": f"the check did not answer: {type(e).__name__}"}
        self._check, self._checked_at = rep, time.time()
        return rep

    def describe(self) -> dict[str, Any]:
        c = self.check()
        label = self.local.label or "this box"
        gpu = c.get("gpu")
        if c.get("ok") and gpu:
            say = f"Train here, on {label}: {gpu}" + (f", {c['gpu_memory_gb']} GB" if c.get("gpu_memory_gb") else "") + \
                  ". Free, and your data never leaves the house."
        elif c.get("ok"):
            say = f"{label} has a training environment but no GPU it can use; training would be very slow."
        else:
            say = f"{label} cannot train yet: {c.get('why')}"
        return {"name": self.name, "kind": self.kind, "label": label, "ready": bool(c.get("ok") and gpu),
                "say": say, "check": c, "cost": "free", "parallel": self.local.max_parallel}

    # -- running ---------------------------------------------------------------
    def start(self, job_dir: Path, spec: Any) -> dict[str, Any]:
        if not self.local.python:
            raise RuntimeError("this box has no training environment (local.python)")
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        if self.local.llama_cpp:
            env["SEREN_LYCEUM_LLAMA_CPP"] = self.local.llama_cpp
        if self.local.hf_home:
            env["HF_HOME"] = self.local.hf_home
        log = open(job_dir / "train.log", "ab")
        kw: dict[str, Any] = {"cwd": str(job_dir), "stdout": log, "stderr": subprocess.STDOUT,
                              "stdin": subprocess.DEVNULL, "env": env}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008   # DETACHED_PROCESS
        else:
            kw["start_new_session"] = True
        p = subprocess.Popen([self.local.python, "-m", "seren_lyceum.trainer.cli", str(job_dir)], **kw)
        log.close()
        self._procs[p.pid] = p
        return {"backend": self.name, "pid": p.pid, "started_at": time.time(), "host": _host()}

    def poll(self, job_dir: Path, info: dict[str, Any]) -> str:
        pid = info.get("pid")
        p = self._procs.get(pid) if pid else None
        if p is not None:
            if p.poll() is None:
                return "running"
            self._procs.pop(pid, None)
            return "exited"
        if info.get("host") and info["host"] != _host():
            return "lost"          # started from another box's service: not ours to judge
        return "running" if pid_alive(pid) else "exited"

    def stop(self, job_dir: Path, info: dict[str, Any], force: bool = False) -> None:
        (job_dir / "STOP").write_text(str(time.time()), encoding="utf-8")
        if not force:
            return
        pid = info.get("pid")
        if not pid or not pid_alive(pid):
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def _host() -> str:
    import socket
    return socket.gethostname()


def python_here() -> str:
    """The interpreter running this service - what a single-box install
    with [train] in the same venv points local.python at."""
    return sys.executable
