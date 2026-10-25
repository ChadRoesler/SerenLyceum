"""The whole loop, for real but fake: a dataset uploaded, a job started, the
runner forking the trainer (fake mode: no torch), the trainer writing its
folder, the UI's reads showing it - and stopping, and refusing what cannot run.

A run is a directory: these tests read what the trainer wrote there, the same
files a cloud backend would pull home."""
from __future__ import annotations

import pytest

from seren_lyceum.jobs import EFFORT, JobSpec

from conftest import CHAT, jsonl, wait_for


def _dataset(client, name="chats", rows=CHAT):
    r = client.post("/api/datasets", data={"name": name, "note": "test"},
                    files={"file": ("d.jsonl", jsonl(rows), "application/json")})
    assert r.status_code == 200, r.text
    return r.json()["dataset"]["id"]


def _job(client, **kw):
    body = {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "dataset": "chats", "effort": "light", **kw}
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_the_box_says_it_can_train(client):
    o = client.get("/api/overview").json()
    local = next(b for b in o["backends"] if b["name"] == "local")
    assert local["ready"] and "the test box" in local["say"]
    assert set(o["effort"]) == set(EFFORT) and "Q5_K_M" in o["quants"]
    assert o["models"][0]["id"] == "Qwen/Qwen3-4B-Instruct-2507"


def test_a_training_runs_start_to_finish(client):
    _dataset(client)
    jid = _job(client, name="first lesson", quant="Q4_K_M", advanced={"epochs": 3})
    j = client.get(f"/api/jobs/{jid}").json()
    assert j["spec"]["params"]["epochs"] == 3 and j["spec"]["params"]["lora_rank"] == 8, "light, with epochs by hand"
    assert j["spec"]["export"] == {"quant": "Q4_K_M"}
    done = wait_for(lambda: (s := client.get(f"/api/jobs/{jid}").json())["status"]["state"] in ("done", "failed") and s)
    assert done["status"]["state"] == "done", done["log"][-20:]
    st = done["status"]
    assert st["fake"] is True and st["rows_train"] + st["rows_eval"] == 30 and st["rows_eval"] >= 1
    assert st["stage_index"] == 5 and st["step"] == st["total"]
    assert done["metrics"][-1]["loss"] < done["metrics"][0]["loss"]
    assert any(ln.startswith("[SEREN:PROGRESS] step=") for ln in done["log"])
    assert any(ln.startswith("[SEREN:COMPLETE]") for ln in done["log"])
    assert done["samples"] and done["artifacts"][-1]["path"] in ("adapter/FAKE.txt", "samples.jsonl")
    r = client.get(f"/api/jobs/{jid}/out/samples.jsonl")
    assert r.status_code == 200 and "expected" in r.text
    assert client.get(f"/api/jobs/{jid}/out/../job.yaml").status_code == 404, "only what it made"
    assert client.delete(f"/api/jobs/{jid}").json()["ok"]
    assert client.get("/api/jobs").json()["jobs"] == []


def test_a_queued_training_can_be_stopped(quiet_client):
    client = quiet_client
    _dataset(client)
    runner = client.app.state.runner
    jid = _job(client)
    # stop it before the runner gets to it
    client.app.state.jobs.request_stop(jid)
    runner.tick()
    assert client.get(f"/api/jobs/{jid}").json()["status"]["state"] == "stopped"
    assert client.post(f"/api/jobs/{jid}/stop").status_code == 409


def test_what_cannot_run_is_refused_in_words(client):
    _dataset(client)
    bad = client.post("/api/jobs", json={"base_model": "x", "dataset": "nope"})
    assert bad.status_code == 400 and "dataset" in bad.json()["detail"]
    bad = client.post("/api/jobs", json={"base_model": "", "dataset": "chats"})
    assert "model" in bad.json()["detail"]
    bad = client.post("/api/jobs", json={"base_model": "x", "dataset": "chats", "effort": "insane"})
    assert "effort" in bad.json()["detail"]
    bad = client.post("/api/jobs", json={"base_model": "x", "dataset": "chats", "backend": "runpod"})
    assert bad.status_code == 400, "no runpod block configured here"
    bad = client.post("/api/datasets", data={"name": "x"}, files={"file": ("d", b"\xff\xfe", "x")})
    assert bad.status_code == 400


def test_a_trainer_that_dies_is_failed_by_the_runner(quiet_client):
    client = quiet_client
    _dataset(client)
    jid = _job(client)
    jobs = client.app.state.jobs
    # pretend it started and its process is gone without a word
    jobs.set_backend(jid, backend="local", pid=999_999_9, host=__import__("socket").gethostname())
    jobs.set_status(jid, state="running")
    client.app.state.runner.tick()
    st = jobs.status(jid)
    assert st["state"] == "failed" and "log says why" in st["message"]


def test_effort_resolves_to_numbers_a_person_can_see():
    s = JobSpec(name="n", base_model="m", dataset="d", effort="thorough").resolve({"lora_rank": 64})
    assert s.params["epochs"] == 3 and s.params["lora_rank"] == 64 and s.params["lora_alpha"] == 128
    with pytest.raises(ValueError):
        JobSpec(name="n", base_model="m", dataset="d", export={"quant": "Q2"}).resolve()


def test_the_viewer_is_served(client):
    r = client.get("/viewer")
    assert r.status_code == 200 and "Lyceum" in r.text
