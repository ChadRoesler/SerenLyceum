"""
seren_lyceum.jobs
════════════════════════════════════════════════════════════════════════
A job is a directory. Who writes what in it is fixed, so two processes - the
service and the trainer, possibly on a pod three time zones away - never
write the same file:

    job.yaml       the SERVICE: what to train, written once when it is made
    data/          the SERVICE: train.jsonl (and eval.jsonl), copied in
    backend.json   the SERVICE: where it runs (pid, pod id) and when it started
    STOP           the SERVICE: present = please stop (the trainer checks)
    status.json    the TRAINER: where the run is, at a glance
    metrics.jsonl  the TRAINER: one line per logged step (loss, lr, eval)
    train.log      the TRAINER: everything it printed, protocol lines included
    out/           the TRAINER: the adapter, the merged model, the GGUF

A cloud backend ships the service's files up and pulls the trainer's back,
so the UI reads the same folder wherever the run is.

EFFORT is how hard it studies, in words a person picks without knowing what a
learning rate is. Each level is a set of numbers anyone can see and override
under "advanced"; nothing is hidden, it is just not asked first.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

EFFORT = {
    "light": {"say": "A quick lesson. Picks up the shape of your data; good for a first try or a small set.",
              "epochs": 1, "learning_rate": 2e-4, "lora_rank": 8},
    "normal": {"say": "The usual. Learns your data well without memorising it word for word.",
               "epochs": 2, "learning_rate": 2e-4, "lora_rank": 16},
    "thorough": {"say": "Studies longer and holds more. Best with a big, clean dataset; takes the longest.",
                 "epochs": 3, "learning_rate": 1e-4, "lora_rank": 32},
}
DEFAULTS = {"max_length": 4096, "batch_size": 2, "grad_accum": 8, "eval_fraction": 0.05,
            "warmup_ratio": 0.03, "seed": 42,
            # what the box has that a run may use: flash_attention_2 or sdpa
            # (PyTorch's own, always there), and 4-bit loading (QLoRA)
            "attention": "sdpa", "load_4bit": False,
            # how many of the questions it never saw it answers at the end
            "samples": 24,
            # the extras switched on for this run, as definitions (seren_lyceum.extras)
            "extras": []}
QUANTS = {
    "Q4_K_M": "Smallest that still sounds like itself. For tight memory (a Nano).",
    "Q5_K_M": "A good balance: a little bigger, a little sharper.",
    "Q8_0": "Nearly the full model. Big; for a box with room to spare.",
    "none": "No GGUF: keep the merged model and the adapter only.",
}
STATES = ("queued", "starting", "running", "done", "failed", "stopped")
STAGES = ("prepare", "download", "train", "evaluate", "export")
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, n: int = 32) -> str:
    return _SLUG.sub("-", (text or "").lower()).strip("-")[:n] or "job"


@dataclass
class JobSpec:
    """What job.yaml holds: everything the trainer needs and nothing it has
    to ask the service for."""
    name: str
    base_model: str
    dataset: str
    backend: str = "local"
    effort: str = "normal"
    params: dict[str, Any] = field(default_factory=dict)      # resolved numbers
    export: dict[str, Any] = field(default_factory=lambda: {"quant": "Q5_K_M"})
    notes: str = ""
    test_dataset: str = ""        # a dataset to test on instead of a random share held back
    id: str = ""
    created_at: float = 0.0

    def resolve(self, overrides: Optional[dict[str, Any]] = None) -> "JobSpec":
        """The numbers for this effort, with anything the person set by hand
        on top. Unknown keys in overrides are ignored, not passed through."""
        if self.effort not in EFFORT:
            raise ValueError(f"effort is one of {', '.join(EFFORT)}, not {self.effort!r}")
        level = {k: v for k, v in EFFORT[self.effort].items() if k != "say"}
        params = {**DEFAULTS, **level}
        params["lora_alpha"] = params["lora_rank"] * 2
        for k, v in (overrides or {}).items():
            if k == "extras":
                params[k] = list(v or [])
            elif k in params and v not in (None, ""):
                params[k] = type(params[k])(v) if not isinstance(params[k], bool) else bool(v)
        if "lora_rank" in (overrides or {}) and "lora_alpha" not in (overrides or {}):
            params["lora_alpha"] = params["lora_rank"] * 2
        if params["attention"] not in ("sdpa", "flash_attention_2"):
            raise ValueError("attention is sdpa or flash_attention_2")
        self.params = params
        quant = (self.export or {}).get("quant", "Q5_K_M")
        if quant not in QUANTS:
            raise ValueError(f"quant is one of {', '.join(QUANTS)}, not {quant!r}")
        return self


class JobStore:
    """The jobs folder. Every read goes to disk: the trainer may have written
    since, and the folder is the truth."""

    def __init__(self, root: Path):
        self.root = Path(root) / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]{1,80}", job_id or ""):
            raise KeyError(job_id)
        d = self.root / job_id
        if not d.is_dir():
            raise KeyError(job_id)
        return d

    def create(self, spec: JobSpec, dataset_dir: Path, test_dir: Optional[Path] = None) -> str:
        """Make the folder: job.yaml, the data copied in, a queued status. The
        dataset's files are copied, not linked - a run must not change because
        someone replaced the dataset it was made from."""
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(spec.name, 24)}-{uuid.uuid4().hex[:4]}"
        d = self.root / job_id
        (d / "data").mkdir(parents=True)
        (d / "out").mkdir()
        spec.id, spec.created_at = job_id, time.time()
        shutil.copyfile(dataset_dir / "data.jsonl", d / "data" / "train.jsonl")
        if test_dir is not None:
            shutil.copyfile(test_dir / "data.jsonl", d / "data" / "test.jsonl")
        _write_yaml(d / "job.yaml", asdict(spec))
        _write_json(d / "status.json", {"state": "queued", "stage": None, "updated_at": time.time(),
                                        "message": "waiting its turn"})
        return job_id

    def spec(self, job_id: str) -> JobSpec:
        data = yaml.safe_load((self.dir(job_id) / "job.yaml").read_text(encoding="utf-8")) or {}
        return JobSpec(**{k: v for k, v in data.items() if k in JobSpec.__dataclass_fields__})

    def status(self, job_id: str) -> dict[str, Any]:
        return _read_json(self.dir(job_id) / "status.json") or {"state": "queued"}

    def set_status(self, job_id: str, **change: Any) -> dict[str, Any]:
        """The service's own status writes: queued, starting, failed before the
        trainer ever ran, stopped. Once the trainer is running, status.json is
        the trainer's - the service only writes it to say the run is gone."""
        st = {**self.status(job_id), **change, "updated_at": time.time()}
        _write_json(self.dir(job_id) / "status.json", st)
        return st

    def backend(self, job_id: str) -> dict[str, Any]:
        return _read_json(self.dir(job_id) / "backend.json") or {}

    def set_backend(self, job_id: str, **change: Any) -> None:
        _write_json(self.dir(job_id) / "backend.json", {**self.backend(job_id), **change})

    def metrics(self, job_id: str, last: int = 2000) -> list[dict[str, Any]]:
        p = self.dir(job_id) / "metrics.jsonl"
        if not p.is_file():
            return []
        rows = []
        for raw in _tail_lines(p, 4_000_000)[-last:]:
            try:
                rows.append(json.loads(raw))
            except ValueError:
                continue
        return rows

    def log_tail(self, job_id: str, nbytes: int = 24_000) -> list[str]:
        p = self.dir(job_id) / "train.log"
        return _tail_lines(p, nbytes) if p.is_file() else []

    def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        out = self.dir(job_id) / "out"
        rows = []
        for p in sorted(out.rglob("*")) if out.is_dir() else []:
            if p.is_file():
                rows.append({"path": str(p.relative_to(out)).replace("\\", "/"), "bytes": p.stat().st_size})
        return rows

    def request_stop(self, job_id: str) -> None:
        (self.dir(job_id) / "STOP").write_text(str(time.time()), encoding="utf-8")

    def stop_requested(self, job_id: str) -> bool:
        return (self.dir(job_id) / "STOP").exists()

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for d in sorted(self.root.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "job.yaml").is_file():
                continue
            try:
                spec, st = self.spec(d.name), self.status(d.name)
            except Exception:  # noqa: BLE001 - one broken folder must not hide the rest
                continue
            rows.append({"id": d.name, "name": spec.name, "base_model": spec.base_model, "dataset": spec.dataset,
                         "backend": spec.backend, "effort": spec.effort, "created_at": spec.created_at,
                         "status": st})
        return rows

    def delete(self, job_id: str) -> None:
        shutil.rmtree(self.dir(job_id))


# ── small file helpers: atomic writes, tolerant reads ─────────────────────────
def replace_atomic(p: Path, text: str) -> None:
    """Write beside, then swap in. The temp name carries the writer's pid:
    the service and the trainer may write the same folder, and one shared
    'status.json.tmp' let them trample each other's half-written file. On
    Windows a swap over a file a reader holds open is refused, so it is
    tried again for a moment instead of failing the run."""
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for i in range(50):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 49:
                raise
            time.sleep(0.02)


def _write_json(p: Path, data: Any) -> None:
    replace_atomic(p, json.dumps(data, indent=1))


def _write_yaml(p: Path, data: Any) -> None:
    replace_atomic(p, yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def _read_json(p: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tail_lines(p: Path, nbytes: int) -> list[str]:
    """The last nbytes, split on newlines and carriage returns (a progress bar
    writes \\r and no newline: Theatre's carriage-return trap)."""
    try:
        size = p.stat().st_size
        with p.open("rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
                fh.readline()
            raw = fh.read()
    except OSError:
        return []
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return [ln for part in text.split("\n") for ln in part.split("\r") if ln.strip()]
