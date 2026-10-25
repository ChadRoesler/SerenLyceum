"""The hippocampus's lessons as a dataset: asked for with the hippocampus's
bearer, kept with a description that says what they are, refused when no
hippocampus is configured or it does not answer."""
from __future__ import annotations

import json

import httpx
import pytest

LESSON = {"messages": [{"role": "user", "content": "Fragments: [0] the nuc stays on focal"},
                       {"role": "assistant", "content": '{"operations": [{"kind": "age_out", "aging": [0]}]}'}]}


@pytest.fixture
def hippocampus(monkeypatch):
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.headers.get("authorization") != "Bearer hip-tok":
            return httpx.Response(401, json={"detail": "no"})
        return httpx.Response(200, text="".join(json.dumps(LESSON) + "\n" for _ in range(12)),
                              headers={"x-seren-report": json.dumps({"examples": 12, "operations_by_kind": {"age_out": 12}})})
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return seen


def test_the_lessons_come_in_as_a_dataset(cfg, fake_train, hippocampus):
    from fastapi.testclient import TestClient
    from seren_lyceum.app import create_app
    cfg.hippocampus.url, cfg.hippocampus.bearer_token = "http://nuc:7254", "hip-tok"
    with TestClient(create_app(cfg)) as c:
        assert c.get("/api/overview").json()["hippocampus"] == {"configured": True}
        r = c.post("/api/datasets/from-hippocampus", json={})
        assert r.status_code == 200, r.text
        d = r.json()["dataset"]
        assert d["rows"] == 12 and d["source"] == "the hippocampus" and d["id"].startswith("consolidator-lessons-")
        assert "12 reviewed drafting calls" in d["note"] and "12 age out" in d["note"]
        assert str(hippocampus[0].url) == "http://nuc:7254/training-set.jsonl"
        named = c.post("/api/datasets/from-hippocampus", json={"name": "lessons-v2", "note": "for v2"}).json()["dataset"]
        assert named["id"] == "lessons-v2" and named["note"] == "for v2"


def test_without_a_hippocampus_it_says_so(client):
    assert client.get("/api/overview").json()["hippocampus"] == {"configured": False}
    r = client.post("/api/datasets/from-hippocampus", json={})
    assert r.status_code == 404 and "hippocampus.url" in r.json()["detail"]


def test_a_refusal_from_the_hippocampus_is_said(cfg, fake_train, hippocampus):
    from fastapi.testclient import TestClient
    from seren_lyceum.app import create_app
    cfg.hippocampus.url, cfg.hippocampus.bearer_token = "http://nuc:7254", "wrong"
    with TestClient(create_app(cfg)) as c:
        r = c.post("/api/datasets/from-hippocampus", json={})
        assert r.status_code == 502 and "401" in r.json()["detail"]
