"""
seren_lyceum.backends.runpod
════════════════════════════════════════════════════════════════════════
RunPod, the first cloud plugin - the way the old engine and Seren trained a
9B for about twenty dollars when the only card in the house had 4 GB.

THE ROUND TRIP it owes the person, so nobody has to know what a pod is:
    1. start a pod with a PyTorch image and the GPU the config names
    2. ship the job's directory up and install seren-lyceum[train] there
    3. run the SAME command a local run runs: seren-lyceum-train <job dir>
    4. keep pulling status.json, metrics.jsonl and the log's tail back, so
       the UI reads the run like a local one
    5. bring out/ home, then STOP the pod (stop, not terminate: the disk
       stays for a re-run until the person lets it go - the old engine's rule)

STATUS: the plugin seam and the config are here; the round trip is the next
piece of work, built and tested against a real account. Until then it says
so in the UI instead of taking a job it cannot run. Configure it with:

    clouds:
      runpod:
        api_key_env: RUNPOD_API_KEY       # or api_key / api_key_keyring
        gpu: "NVIDIA RTX A6000"
        image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class RunPodBackend:
    name = "runpod"
    kind = "cloud"

    def __init__(self, cfg, options: dict[str, Any]):
        self.cfg = cfg
        self.options = options or {}

    def _key(self) -> str:
        from seren_meninges import resolve_token
        o = self.options
        return resolve_token(inline=o.get("api_key") or None, keyring_ref=o.get("api_key_keyring") or None,
                             env_var=o.get("api_key_env") or None)

    def describe(self) -> dict[str, Any]:
        has_key = bool(self._key())
        gpu = self.options.get("gpu") or "a GPU you pick"
        say = (f"Rent {gpu} on RunPod, pay by the minute, and the Lyceum brings the model home and stops the "
               "machine when it is done. Coming next: the round trip is not built yet.")
        if not has_key:
            say += " It also needs your RunPod API key (clouds.runpod.api_key_env)."
        return {"name": self.name, "kind": self.kind, "label": "RunPod", "ready": False, "say": say,
                "cost": "paid by the minute", "parallel": 0, "coming": True}

    def start(self, job_dir: Path, spec: Any) -> dict[str, Any]:
        raise RuntimeError("RunPod training is not built yet; train on this box for now")

    def poll(self, job_dir: Path, info: dict[str, Any]) -> str:
        return "lost"

    def stop(self, job_dir: Path, info: dict[str, Any], force: bool = False) -> None:
        (job_dir / "STOP").write_text("stop", encoding="utf-8")
