"""
seren_lyceum.backends
════════════════════════════════════════════════════════════════════════
Where a job runs. Two kinds, one shape:

    local   this box: the trainer is forked in the training environment and
            writes its run directory where it stands
    cloud   a rented GPU: the plugin ships the job's directory to a machine
            it starts, runs the SAME command there, keeps pulling the
            trainer's files back so the UI reads them like a local run, brings
            out/ home and stops the machine - the whole round trip, so nobody
            has to know what a pod is

A backend is a class with this surface (duck-typed; see Backend):

    describe()                  -> what the UI says about it: ready or not and
                                   why, the GPU, the cost, in plain words
    start(job_dir, spec)        -> what to remember about the run (backend.json)
    poll(job_dir, info)         -> "running" | "exited" | "lost" (and a cloud
                                   plugin syncs the trainer's files back here)
    stop(job_dir, info, force)  -> ask nicely (the STOP file), or not

Cloud providers are PLUGINS: anything installed that registers an entry point
in the `seren_lyceum.backends` group (name -> class) is offered when the
config has a block for it under `clouds:`. RunPod ships in this package the
same way, as the first one.
"""
from __future__ import annotations

import logging
from importlib import metadata
from pathlib import Path
from typing import Any, Optional, Protocol

log = logging.getLogger(__name__)


class Backend(Protocol):
    name: str
    kind: str                       # "local" | "cloud"

    def describe(self) -> dict[str, Any]: ...
    def start(self, job_dir: Path, spec: Any) -> dict[str, Any]: ...
    def poll(self, job_dir: Path, info: dict[str, Any]) -> str: ...
    def stop(self, job_dir: Path, info: dict[str, Any], force: bool = False) -> None: ...


def _plugins() -> dict[str, Any]:
    found: dict[str, Any] = {}
    try:
        eps = metadata.entry_points(group="seren_lyceum.backends")
    except Exception:  # noqa: BLE001 - a broken install must not take the service down
        return found
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception as e:  # noqa: BLE001
            log.warning("backend plugin %s did not load: %s", ep.name, e)
    return found


def build(cfg) -> dict[str, Backend]:
    """Every backend this box offers: local always (it says when it cannot
    train), and each cloud with a config block and a plugin to drive it."""
    from .local import LocalBackend
    from .runpod import RunPodBackend
    out: dict[str, Backend] = {"local": LocalBackend(cfg)}
    known = {"runpod": RunPodBackend, **_plugins()}
    for c in cfg.clouds:
        if not c.enabled:
            continue
        cls = known.get(c.name)
        if cls is None:
            log.warning("clouds.%s is configured, but no plugin by that name is installed", c.name)
            continue
        out[c.name] = cls(cfg, c.options)
    return out


def pid_alive(pid: Optional[int]) -> bool:
    """Whether a process is alive, on any OS. NOT os.kill(pid, 0) on Windows:
    there 0 is CTRL_C_EVENT, and the 'check' interrupts the process it asks
    about. OpenProcess and GetExitCodeProcess instead."""
    if not pid:
        return False
    import os
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, int(pid))     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259          # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
