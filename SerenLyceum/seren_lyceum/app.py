"""
seren_lyceum.app
════════════════════════════════════════════════════════════════════════
The front door. Everything a person does happens here, in plain words:

    GET    /                         service info + version + update status
    GET    /health                   liveness
    GET    /viewer                   the Lyceum itself
    GET    /api/overview             where it can train, how hard, what comes out
    POST   /api/backends/{name}/check  ask a backend again (after setting it up)
    GET    /api/datasets             what it can learn from
    POST   /api/datasets             add one: multipart name + file (+ note)
    GET    /api/datasets/{id}        what was found in it, and a few examples
    DELETE /api/datasets/{id}
    GET    /api/jobs                 every training, newest first
    POST   /api/jobs                 start one
    GET    /api/jobs/{id}            one training: where it is, the numbers,
                                     the log's tail, what it made
    POST   /api/jobs/{id}/stop       stop it (cleanly; forced after a grace)
    DELETE /api/jobs/{id}            forget a finished one (its folder goes)
    GET    /api/jobs/{id}/out/{path} download what it made

The service never trains: it writes a job's folder and the runner forks the
trainer. torch is never imported here.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from seren_meninges import get_version
from seren_meninges.auth import bearer_auth_middleware
from seren_meninges.updates import updates_payload
from seren_meninges.viewer import render_from_dir
from seren_sinew.request_log import RequestLoggingMiddleware

from . import __version__ as _fallback_version
from . import backends as backends_mod
from . import hf
from .links import Links
from .config import ENV_PREFIX, SERVICE, LyceumConfig, load_config
from .datasets import DatasetError, DatasetStore
from .jobs import EFFORT, QUANTS, STAGES, JobSpec, JobStore
from .runner import LIVE, Runner

APP_VERSION = get_version(SERVICE, fallback=_fallback_version)
ACCENT = "#a17056"      # pithos clay: the jar Diogenes lived in, outside the schools
log = logging.getLogger(__name__)

# Base models a person might start from, in words. Anything on Hugging Face
# works too; these are just the ones worth suggesting first. need_gb is a
# rough figure for a LoRA in bf16 - enough to say "fits" or "too big".
SUGGESTED_MODELS = [
    {"id": "Qwen/Qwen3-4B-Instruct-2507", "size": "4B",
     "say": "Small, quick and good at following instructions. What the hippocampus drafts with.", "need_gb": 16},
    {"id": "Qwen/Qwen3-1.7B", "size": "1.7B", "say": "Tiny. Trains in minutes; runs anywhere, even a Nano.",
     "need_gb": 8},
    {"id": "Qwen/Qwen3-8B", "size": "8B", "say": "Bigger and smarter; still fits a single good GPU.", "need_gb": 28},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B", "say": "A well-known all-rounder.", "need_gb": 26},
    {"id": "google/gemma-3-4b-it", "size": "4B",
     "say": "Google's small one. Needs a Hugging Face login that has accepted its licence.", "need_gb": 16},
    {"id": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B",
     "say": "Meta's small one. Needs a Hugging Face login that has accepted its licence.", "need_gb": 14},
]


def create_app(config: Optional[LyceumConfig] = None) -> FastAPI:
    cfg = config or load_config()
    bearer = cfg.server.resolve_bearer()
    root = cfg.storage.root()
    root.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = cfg
        app.state.jobs = JobStore(root)
        app.state.datasets = DatasetStore(root)
        app.state.backends = backends_mod.build(cfg)
        app.state.links = Links()
        from .extras import Registry
        app.state.extras = Registry(cfg.extras, root)
        for p in app.state.extras.problems:
            log.warning("[seren-lyceum] an extra in the config was skipped: %s", p)
        for be in app.state.backends.values():
            if hasattr(be, "modules"):
                be.modules = app.state.extras.modules()
        app.state.runner = Runner(app.state.jobs, app.state.backends, cfg.refresh_seconds)
        app.state.runner.start()
        log.info(f"[seren-lyceum] ready on {cfg.server.host}:{cfg.server.port}; keeping things in {root}")
        yield
        await app.state.runner.stop()

    app = FastAPI(title="SerenLyceum", description="Train your own model: a dataset in, a model out.",
                  version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(bearer_auth_middleware(bearer))
    app.add_middleware(RequestLoggingMiddleware, service_name=SERVICE, env_prefix=ENV_PREFIX)

    try:
        from seren_meninges.updates import UpdateChecker
        app.state.updates = UpdateChecker(SERVICE, enabled=cfg.updates.enabled, index_url=cfg.updates.index_url,
                                          ttl_seconds=cfg.updates.check_interval_hours * 3600.0,
                                          allow_prerelease=cfg.updates.allow_prerelease,
                                          fallback_version=APP_VERSION)
    except Exception as exc:  # noqa: BLE001 - cosmetic; must never stop the boot
        app.state.updates = None
        log.warning("update checking unavailable (%s)", exc)

    viewer_dir = Path(__file__).resolve().parent / "viewer" / "ui"

    @app.get("/")
    async def root_info(request: Request):
        return {"service": "SerenLyceum", "version": APP_VERSION,
                "updates": await updates_payload(getattr(request.app.state, "updates", None),
                                                 distribution=SERVICE, installed=APP_VERSION)}

    @app.get("/health")
    async def health():
        return {"ok": True, "ts": time.time(), "version": APP_VERSION}

    @app.get("/viewer")
    async def viewer():
        return HTMLResponse(render_from_dir(viewer_dir, title="SerenLyceum",
                                            brand="Seren<b>Lyceum</b> · The School",
                                            subtitle=f"v{APP_VERSION} · a dataset in, a model out",
                                            accent=ACCENT))

    # -- what this box offers ------------------------------------------------------
    @app.get("/api/overview")
    async def overview(request: Request):
        import asyncio
        bes = request.app.state.backends
        described = await asyncio.gather(*(asyncio.to_thread(b.describe) for b in bes.values()))
        jobs = request.app.state.jobs.list()
        local_gb = next((d.get("check", {}).get("gpu_memory_gb") for d in described if d["name"] == "local"), None)
        models = [{**m, "fits": (local_gb >= m["need_gb"]) if local_gb else None} for m in SUGGESTED_MODELS]
        for m in models:
            m["needs_login"] = hf.needs_login(m["id"])
        return {"backends": described, "effort": EFFORT, "quants": QUANTS, "stages": STAGES, "models": models,
                "recommended_quant": "Q5_K_M", "hf_login": hf.login(cfg.local.hf_home),
                "hippocampus": {"configured": bool(cfg.hippocampus.url)},
                "extras": request.app.state.extras.describe(
                    next((d.get("check", {}).get("extras") for d in described if d["name"] == "local"), None)),
                "counts": {"jobs": len(jobs), "running": sum(1 for j in jobs if j["status"].get("state") in LIVE),
                           "datasets": len(request.app.state.datasets.list())}}

    @app.post("/api/backends/{name}/check")
    async def recheck(request: Request, name: str):
        import asyncio
        be = request.app.state.backends.get(name)
        if be is None:
            raise HTTPException(404, f"no backend named {name!r}")
        if hasattr(be, "check"):
            await asyncio.to_thread(be.check, True)
        return await asyncio.to_thread(be.describe)

    @app.put("/api/extras/{extra_id}")
    async def set_extra(request: Request, extra_id: str, body: dict[str, Any] = Body(...)):
        """Switch an extra on or off for this Lyceum: off, it is not offered
        in New training."""
        try:
            request.app.state.extras.set_on(extra_id, bool(body.get("on")))
        except KeyError:
            raise HTTPException(404, f"no extra called {extra_id!r}")
        return {"ok": True, "id": extra_id, "on": bool(body.get("on"))}

    @app.post("/api/hf-token")
    async def set_hf_token(body: dict[str, Any] = Body(...)):
        """Save the Hugging Face login where Hugging Face keeps it. The token
        is never sent back."""
        try:
            return {"ok": True, "hf_login": hf.save(str(body.get("token") or ""), cfg.local.hf_home)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    # -- downloads the browser does itself (see links.py) ------------------------
    @app.post("/api/link")
    async def make_link(request: Request, body: dict[str, Any] = Body(...)):
        """A short-lived link to one thing: a file a training made (out), its
        log, its config, its sample answers, or a dataset."""
        import json
        st, kind, ident = request.app.state, str(body.get("kind") or ""), str(body.get("id") or "")
        links: Links = st.links
        try:
            if kind == "dataset":
                d = st.datasets.dir(ident)
                code = links.make(f"{d.name}.jsonl", path=d / "data.jsonl", media="application/x-ndjson")
            elif kind in ("out", "log", "config", "samples"):
                jd = st.jobs.dir(ident)
                if kind == "out":
                    out = (jd / "out").resolve()
                    target = (out / str(body.get("path") or "")).resolve()
                    if out not in target.parents or not target.is_file():
                        raise HTTPException(404, "no such file")
                    code = links.make(target.name, path=target)
                elif kind == "log":
                    code = links.make(f"{ident}.log", path=jd / "train.log", media="text/plain")
                elif kind == "config":
                    data = json.dumps(_config_of(st.jobs.spec(ident)), indent=1).encode()
                    code = links.make(f"{ident}.lyceum.json", data=data, media="application/json")
                else:
                    sp = jd / "out" / "samples.jsonl"
                    if not sp.is_file():
                        raise HTTPException(404, "it has no sample answers yet")
                    code = links.make(f"{ident}-samples.jsonl", path=sp, media="application/x-ndjson")
            else:
                raise HTTPException(400, "kind is one of out, log, config, samples, dataset")
        except KeyError:
            raise HTTPException(404, "no such thing")
        name = links.get(code).filename
        return {"url": f"/viewer/dl/{code}/{name}"}

    @app.get("/viewer/dl/{code}/{name}")
    async def download(request: Request, code: str, name: str):
        from fastapi.responses import Response
        ln = request.app.state.links.get(code)
        if ln is None:
            raise HTTPException(404, "this link has run out; ask the page for a new one")
        if ln.data is not None:
            return Response(ln.data, media_type=ln.media,
                            headers={"Content-Disposition": f'attachment; filename="{ln.filename}"'})
        if ln.path is None or not ln.path.is_file():
            raise HTTPException(404, "the file is gone")
        return FileResponse(ln.path, filename=ln.filename, media_type=ln.media)

    # -- datasets ---------------------------------------------------------------
    @app.get("/api/datasets")
    async def datasets(request: Request):
        return {"datasets": request.app.state.datasets.list()}

    @app.post("/api/datasets")
    async def add_dataset(request: Request, name: str = Form(...), file: UploadFile = File(...),
                          note: str = Form(""), use: str = Form("learn")):
        if not note.strip():
            raise HTTPException(400, "give it a description: what is in it, and where it came from in your words")
        raw = await file.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(400, "the file is not text (UTF-8 JSONL, one example per line)")
        try:
            meta = request.app.state.datasets.add(name, text.splitlines(), source=file.filename or "upload", note=note,
                                                  use=use)
        except DatasetError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "dataset": meta}

    @app.post("/api/datasets/from-hippocampus")
    async def dataset_from_hippocampus(request: Request, body: dict[str, Any] = Body(default={})):
        """The hippocampus's lessons as a dataset: every reviewed drafting
        call, the prompt the small model saw and what landed. {name?, note?};
        the note defaults to what the hippocampus said it holds."""
        import json
        import httpx
        hc = cfg.hippocampus
        if not hc.url:
            raise HTTPException(404, "no hippocampus is configured (hippocampus.url)")
        tok = hc.resolve_bearer()
        try:
            async with httpx.AsyncClient(timeout=hc.timeout_seconds) as c:
                r = await c.get(hc.url.rstrip("/") + "/training-set.jsonl",
                                headers={"Authorization": f"Bearer {tok}"} if tok else {})
        except httpx.HTTPError as e:
            raise HTTPException(502, f"the hippocampus did not answer ({type(e).__name__})")
        if r.status_code >= 400:
            raise HTTPException(502, f"the hippocampus said {r.status_code}: {r.text[:200]}")
        try:
            rep = json.loads(r.headers.get("x-seren-report") or "{}")
        except ValueError:
            rep = {}
        kinds = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in (rep.get("operations_by_kind") or {}).items())
        name = str((body or {}).get("name") or "").strip() or f"consolidator-lessons-{time.strftime('%Y%m%d')}"
        note = str((body or {}).get("note") or "").strip() or (
            f"The hippocampus's lessons, {time.strftime('%d %b %Y')}: {rep.get('examples', '?')} reviewed drafting calls, "
            f"each the prompt the small model saw and what finally landed" + (f" ({kinds})." if kinds else "."))
        try:
            meta = request.app.state.datasets.add(name, r.text.splitlines(), source="the hippocampus", note=note)
        except DatasetError as e:
            raise HTTPException(400, str(e))
        return {"ok": True, "dataset": meta, "report": rep}

    @app.get("/api/datasets/{ds_id}")
    async def dataset(request: Request, ds_id: str):
        try:
            store = request.app.state.datasets
            return {"dataset": store.get(ds_id), "sample": store.sample(ds_id, 3)}
        except KeyError:
            raise HTTPException(404, f"no dataset {ds_id!r}")

    @app.get("/api/datasets/{ds_id}/rows")
    async def dataset_rows(request: Request, ds_id: str, offset: int = 0, limit: int = 25):
        """A page of the whole dataset, for reading it end to end."""
        import json
        try:
            p = request.app.state.datasets.dir(ds_id) / "data.jsonl"
        except KeyError:
            raise HTTPException(404, f"no dataset {ds_id!r}")
        offset, limit = max(0, offset), max(1, min(limit, 200))
        rows, total = [], 0
        with p.open(encoding="utf-8") as fh:
            for i, raw in enumerate(fh):
                total += 1
                if offset <= i < offset + limit:
                    rows.append(json.loads(raw))
        return {"offset": offset, "limit": limit, "total": total, "rows": rows}

    @app.delete("/api/datasets/{ds_id}")
    async def drop_dataset(request: Request, ds_id: str):
        try:
            request.app.state.datasets.delete(ds_id)
        except KeyError:
            raise HTTPException(404, f"no dataset {ds_id!r}")
        return {"ok": True}

    # -- jobs -------------------------------------------------------------------
    @app.get("/api/jobs")
    async def jobs(request: Request):
        return {"jobs": request.app.state.jobs.list()}

    @app.post("/api/jobs")
    async def start_job(request: Request, body: dict[str, Any] = Body(...)):
        st = request.app.state
        try:
            ds_dir = st.datasets.dir(str(body.get("dataset") or ""))
        except KeyError:
            raise HTTPException(400, "pick a dataset to learn from")
        backend = str(body.get("backend") or "local")
        be = st.backends.get(backend)
        if be is None:
            raise HTTPException(400, f"no place to train called {backend!r}")
        if not be.describe().get("ready"):
            raise HTTPException(400, be.describe().get("say") or f"{backend} is not ready")
        base = str(body.get("base_model") or "").strip()
        if not base:
            raise HTTPException(400, "pick a model to start from")
        test_dir = None
        if body.get("test_dataset"):
            try:
                test_dir = st.datasets.dir(str(body["test_dataset"]))
            except KeyError:
                raise HTTPException(400, f"no dataset {body['test_dataset']!r} to test with")
            if test_dir == ds_dir:
                raise HTTPException(400, "testing on what it learns from tells you nothing; pick another dataset to test with")
            kinds = {("text" if s == "text" else "chat") for d in (ds_dir, test_dir)
                     for s in (st.datasets.get(d.name).get("shapes") or {})}
            if len(kinds) > 1 and st.datasets.get(test_dir.name).get("use") != "questions":
                raise HTTPException(400, "one is conversations and the other plain text; test with the same kind it learns")
        name = str(body.get("name") or "").strip() or f"{Path(base).name} on {ds_dir.name}"
        spec = JobSpec(name=name, base_model=base, dataset=ds_dir.name, backend=backend,
                       effort=str(body.get("effort") or "normal"),
                       export={"quant": str(body.get("quant") or "Q5_K_M")}, notes=str(body.get("notes") or ""),
                       test_dataset=test_dir.name if test_dir else "")
        adv = dict(body.get("advanced") or {})
        try:
            # a job carries its extras as definitions, not names: the run must
            # not change if the config's extras do later
            ids = adv.get("extras") or body.get("extras") or []
            adv["extras"] = st.extras.resolve([i if isinstance(i, str) else i.get("id") for i in ids])
            spec.resolve(adv)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e))
        job_id = st.jobs.create(spec, ds_dir, test_dir)
        return {"ok": True, "id": job_id}

    @app.get("/api/jobs/{job_id}/config")
    async def job_config(request: Request, job_id: str):
        """What it would take to run this training again: the walkthrough's
        answers, in the shape New training imports."""
        try:
            return _config_of(request.app.state.jobs.spec(job_id))
        except KeyError:
            raise HTTPException(404, f"no training {job_id!r}")

    @app.get("/api/jobs/{job_id}")
    async def job(request: Request, job_id: str, log_bytes: int = 24_000):
        j = request.app.state.jobs
        try:
            spec = j.spec(job_id)
        except KeyError:
            raise HTTPException(404, f"no training {job_id!r}")
        samples = []
        sp = j.dir(job_id) / "out" / "samples.jsonl"
        if sp.is_file():
            import json
            samples = [json.loads(x) for x in sp.read_text(encoding="utf-8").splitlines() if x.strip()]
        return {"id": job_id, "spec": spec.__dict__, "status": j.status(job_id), "backend": j.backend(job_id),
                "metrics": j.metrics(job_id), "log": j.log_tail(job_id, min(max(log_bytes, 2000), 400_000)),
                "artifacts": j.artifacts(job_id), "samples": samples, "stop_requested": j.stop_requested(job_id)}

    @app.post("/api/jobs/{job_id}/stop")
    async def stop_job(request: Request, job_id: str):
        j = request.app.state.jobs
        try:
            spec, st = j.spec(job_id), j.status(job_id)
        except KeyError:
            raise HTTPException(404, f"no training {job_id!r}")
        if st.get("state") not in LIVE:
            raise HTTPException(409, "it is not running")
        be = request.app.state.backends.get(spec.backend)
        if be is not None and st.get("state") != "queued":
            be.stop(j.dir(job_id), j.backend(job_id), force=False)
        else:
            j.request_stop(job_id)
        return {"ok": True, "note": "asked to stop; it finishes the step it is on"}

    @app.delete("/api/jobs/{job_id}")
    async def forget_job(request: Request, job_id: str):
        j = request.app.state.jobs
        try:
            st = j.status(job_id)
        except KeyError:
            raise HTTPException(404, f"no training {job_id!r}")
        if st.get("state") in LIVE:
            raise HTTPException(409, "stop it first")
        j.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/out/{path:path}")
    async def job_file(request: Request, job_id: str, path: str):
        j = request.app.state.jobs
        try:
            out = (j.dir(job_id) / "out").resolve()
        except KeyError:
            raise HTTPException(404, f"no training {job_id!r}")
        target = (out / path).resolve()
        if out not in target.parents or not target.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(target, filename=target.name)

    return app


def _config_of(spec: JobSpec) -> dict[str, Any]:
    """A training's answers as a config a person can keep, share and import:
    the choices, and every number behind the effort level."""
    return {"kind": "seren-lyceum-config", "version": 1, "name": spec.name, "base_model": spec.base_model,
            "dataset": spec.dataset, "test_dataset": spec.test_dataset, "backend": spec.backend, "effort": spec.effort,
            "quant": (spec.export or {}).get("quant", "Q5_K_M"), "notes": spec.notes,
            "advanced": {**dict(spec.params or {}),
                         "extras": [e.get("id") for e in (spec.params or {}).get("extras") or []]}}
