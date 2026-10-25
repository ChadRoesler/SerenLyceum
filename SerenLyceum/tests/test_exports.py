"""What a person takes away and brings back: download links (the browser
fetches them itself, no token on the click), the log, the config that
imports into New training, the whole dataset a page at a time, and the
Hugging Face login that is saved and never shown again."""
from __future__ import annotations

import pytest

from seren_lyceum import hf
from seren_lyceum.jobs import JobSpec
from seren_lyceum.links import Links

from conftest import CHAT, jsonl, wait_for


def _dataset(client, name="chats", note="test chats"):
    return client.post("/api/datasets", data={"name": name, "note": note},
                       files={"file": ("d.jsonl", jsonl(CHAT), "application/json")})


def _done_job(client):
    assert _dataset(client).status_code == 200
    r = client.post("/api/jobs", json={"base_model": "Qwen/Qwen3-4B-Instruct-2507", "dataset": "chats",
                                       "effort": "light", "name": "exported", "advanced": {"epochs": 3}})
    jid = r.json()["id"]
    wait_for(lambda: client.get(f"/api/jobs/{jid}").json()["status"]["state"] in ("done", "failed"))
    return jid


def _fetch(client, **body):
    r = client.post("/api/link", json=body)
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    assert url.startswith("/viewer/dl/")
    return client.get(url)


def test_a_dataset_needs_a_description(client):
    r = _dataset(client, note="  ")
    assert r.status_code == 400 and "description" in r.json()["detail"]


def test_the_whole_dataset_reads_a_page_at_a_time(client):
    assert _dataset(client).status_code == 200
    first = client.get("/api/datasets/chats/rows?offset=0&limit=25").json()
    assert first["total"] == 30 and len(first["rows"]) == 25
    rest = client.get("/api/datasets/chats/rows?offset=25&limit=25").json()
    assert len(rest["rows"]) == 5 and rest["rows"][-1] == CHAT[-1]
    assert client.get("/api/datasets/nope/rows").status_code == 404
    got = _fetch(client, kind="dataset", id="chats")
    assert got.status_code == 200 and got.text.count("\n") == 30


def test_what_a_training_made_downloads_by_link(client):
    jid = _done_job(client)
    j = client.get(f"/api/jobs/{jid}").json()
    path = j["artifacts"][0]["path"]
    got = _fetch(client, kind="out", id=jid, path=path)
    assert got.status_code == 200 and "attachment" in got.headers["content-disposition"]
    assert "[SEREN:COMPLETE]" in _fetch(client, kind="log", id=jid).text
    assert "expected" in _fetch(client, kind="samples", id=jid).text
    assert client.post("/api/link", json={"kind": "out", "id": jid, "path": "../job.yaml"}).status_code == 404
    assert client.post("/api/link", json={"kind": "out", "id": "nope", "path": path}).status_code == 404
    assert client.post("/api/link", json={"kind": "what", "id": jid}).status_code == 400
    assert client.get("/viewer/dl/not-a-code/x.txt").status_code == 404


def test_a_config_exports_and_runs_again(client):
    jid = _done_job(client)
    c = _fetch(client, kind="config", id=jid).json()
    assert c == client.get(f"/api/jobs/{jid}/config").json()
    assert c["kind"] == "seren-lyceum-config" and c["dataset"] == "chats" and c["effort"] == "light"
    assert c["advanced"]["epochs"] == 3 and c["advanced"]["attention"] == "sdpa"
    # what New training does with it: the same answers, posted back
    r = client.post("/api/jobs", json={"base_model": c["base_model"], "dataset": c["dataset"], "effort": c["effort"],
                                       "quant": c["quant"], "advanced": c["advanced"]})
    assert r.status_code == 200, r.text
    again = client.get(f"/api/jobs/{r.json()['id']}").json()["spec"]["params"]
    assert again == c["advanced"]


def test_links_run_out():
    links = Links(ttl=-1)
    code = links.make("x.txt", data=b"x")
    assert links.get(code) is None
    with pytest.raises(ValueError):
        Links().make("x", data=b"x", path=__file__)


def test_attention_is_one_of_two():
    spec = JobSpec(name="a", base_model="m", dataset="d")
    assert spec.resolve({"attention": "flash_attention_2", "load_4bit": True}).params["load_4bit"] is True
    with pytest.raises(ValueError):
        JobSpec(name="a", base_model="m", dataset="d").resolve({"attention": "eager-and-wrong"})


def test_the_login_is_saved_and_never_said(client, cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.local.hf_home = str(tmp_path / "hf")
    assert client.get("/api/overview").json()["hf_login"]["found"] is False
    assert client.post("/api/hf-token", json={"token": "not-a-token"}).status_code == 400
    tok = "hf_" + "a1B2c3D4e5" * 3
    r = client.post("/api/hf-token", json={"token": tok})
    assert r.status_code == 200 and tok not in r.text
    o = client.get("/api/overview")
    assert o.json()["hf_login"]["found"] is True and tok not in o.text
    assert (tmp_path / "hf" / "token").read_text() == tok
    gated = next(m for m in o.json()["models"] if m["id"].startswith("meta-llama/"))
    assert gated["needs_login"] is True


def test_the_environment_login_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_whatever")
    assert hf.login(str(tmp_path))["found"] is True


def test_your_own_questions_are_what_it_is_tested_on(client):
    assert _dataset(client).status_code == 200
    qs = [{"messages": [{"role": "user", "content": f"my question {i}"},
                        {"role": "assistant", "content": f"my answer {i}"}]} for i in range(7)]
    r = client.post("/api/datasets", data={"name": "my-questions", "note": "mine"},
                    files={"file": ("q.jsonl", jsonl(qs), "application/json")})
    assert r.status_code == 200, r.text
    assert client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats",
                                          "test_dataset": "chats"}).status_code == 400, "not on what it learns"
    assert client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats",
                                          "test_dataset": "nope"}).status_code == 400
    r = client.post("/api/jobs", json={"base_model": "Qwen/Qwen3-4B-Instruct-2507", "dataset": "chats",
                                       "test_dataset": "my-questions", "effort": "light"})
    jid = r.json()["id"]
    done = wait_for(lambda: (s := client.get(f"/api/jobs/{jid}").json())["status"]["state"] in ("done", "failed") and s)
    st = done["status"]
    assert st["state"] == "done" and st["rows_train"] == 30 and st["rows_eval"] == 7, "all of it learned, all 7 asked"
    assert [s["asked"] for s in done["samples"]] == [f"my question {i}" for i in range(7)], "every one answered"
    assert client.get(f"/api/jobs/{jid}/config").json()["test_dataset"] == "my-questions"


def test_every_held_back_example_gets_an_answer(client):
    _dataset(client)
    jid = client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats", "effort": "light",
                                         "advanced": {"eval_fraction": 0.2}}).json()["id"]
    done = wait_for(lambda: (s := client.get(f"/api/jobs/{jid}").json())["status"]["state"] in ("done", "failed") and s)
    assert done["status"]["rows_eval"] == 6 and len(done["samples"]) == 6, "not three: all of them"
