"""
seren_lyceum.runner
════════════════════════════════════════════════════════════════════════
The one loop that moves jobs along. Every tick it reads each live job's
folder and does at most one thing to it:

    queued    -> starting   when its backend has room (local: max_parallel)
    starting  -> (the trainer takes status.json over once it is up)
    running   -> failed     when the trainer is gone and never said it was done
    STOP set  -> stopped    when the trainer honoured it; force-killed after
                            STOP_GRACE seconds if it did not

Everything it knows comes from the folder (job.yaml, backend.json,
status.json), so a restarted service picks up where the last one was: a
local run is found by its pid, a cloud run by what its plugin remembered.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .jobs import JobStore

log = logging.getLogger(__name__)
STOP_GRACE = 90.0
LIVE = ("queued", "starting", "running")


class Runner:
    def __init__(self, jobs: JobStore, backends: dict[str, Any], refresh: float = 2.0):
        self.jobs, self.backends, self.refresh = jobs, backends, refresh
        self._task: asyncio.Task | None = None

    def tick(self) -> list[str]:
        """One pass. Returns what it did, for the log and the tests."""
        did: list[str] = []
        rows = [r for r in self.jobs.list() if r["status"].get("state") in LIVE]
        busy: dict[str, int] = {}
        for r in rows:
            if r["status"].get("state") in ("starting", "running"):
                busy[r["backend"]] = busy.get(r["backend"], 0) + 1
        for r in sorted(rows, key=lambda r: r["created_at"] or 0):     # oldest first
            jid, state = r["id"], r["status"].get("state")
            be = self.backends.get(r["backend"])
            if be is None:
                self.jobs.set_status(jid, state="failed", message=f"no backend named {r['backend']!r} on this box")
                did.append(f"{jid}: failed (no backend)")
                continue
            d = self.jobs.dir(jid)
            if state == "queued":
                if self.jobs.stop_requested(jid):
                    self.jobs.set_status(jid, state="stopped", message="stopped before it started", ended_at=time.time())
                    did.append(f"{jid}: stopped while queued")
                    continue
                room = getattr(getattr(be, "local", None), "max_parallel", 1) if be.kind == "local" else 10_000
                if busy.get(r["backend"], 0) >= room:
                    continue
                # "starting" is written BEFORE the fork: from the moment the
                # trainer exists, status.json is its file and only its file
                self.jobs.set_status(jid, state="starting", message="starting the trainer", started_at=time.time())
                try:
                    info = be.start(d, self.jobs.spec(jid))
                except Exception as e:  # noqa: BLE001 - a job that cannot start fails, the loop goes on
                    self.jobs.set_status(jid, state="failed", message=f"could not start: {e}", ended_at=time.time())
                    did.append(f"{jid}: failed to start")
                    continue
                self.jobs.set_backend(jid, **info)
                busy[r["backend"]] = busy.get(r["backend"], 0) + 1
                did.append(f"{jid}: started on {r['backend']}")
                continue
            info = self.jobs.backend(jid)
            seen = be.poll(d, info)
            if seen == "running":
                if self.jobs.stop_requested(jid):
                    asked = (d / "STOP").stat().st_mtime
                    if time.time() - asked > STOP_GRACE:
                        be.stop(d, info, force=True)
                        did.append(f"{jid}: force-stopped")
                continue
            if seen == "lost":
                continue
            # the trainer is gone: did it finish, or did it just stop?
            st = self.jobs.status(jid)
            if st.get("state") in LIVE:
                if self.jobs.stop_requested(jid):
                    self.jobs.set_status(jid, state="stopped", message="stopped", ended_at=time.time())
                    did.append(f"{jid}: stopped")
                else:
                    self.jobs.set_status(jid, state="failed", ended_at=time.time(),
                                         message="the trainer stopped without saying it was done; the log says why")
                    did.append(f"{jid}: failed (trainer gone)")
        return did

    async def run(self) -> None:
        while True:
            try:
                for msg in await asyncio.to_thread(self.tick):
                    log.info("[seren-lyceum] %s", msg)
            except Exception as e:  # noqa: BLE001 - the loop never dies
                log.warning("[seren-lyceum] runner tick failed: %s", e)
            await asyncio.sleep(self.refresh)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
