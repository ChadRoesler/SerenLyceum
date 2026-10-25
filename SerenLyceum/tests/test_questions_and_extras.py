"""Question sets and extras (24 Oct 2026).

A question set is what a training is tested on: questions in any shape a
person has them - a line of text, question/answer, messages - the answer
optional. Extras are switched on per Lyceum (Config) and per run (Advanced),
and a run carries their definitions, so it does not change if the config does."""
from __future__ import annotations

import json

from seren_lyceum.datasets import inspect_questions
from seren_lyceum.extras import BUILTIN, Registry

from conftest import CHAT, jsonl, wait_for


def test_questions_come_in_any_shape_and_the_answer_is_optional():
    lines = ["What does the hippocampus do at night?",
             json.dumps({"question": "Who is the assistant?", "answer": "A co-conspirator."}),
             json.dumps({"messages": [{"role": "user", "content": "Is the NUC on focal?"}]}),
             json.dumps({"instruction": "Name the bridge"}),
             json.dumps({"instruction": "Name the jar", "output": "the pithos"}),
             json.dumps({"prompt": "p?", "completion": "c."}),
             json.dumps({"nothing": "here"})]
    rows, found = inspect_questions(lines)
    assert found["rows"] == 6 and found["answered"] == 3 and found["unusable"] == 1
    assert rows[0] == {"messages": [{"role": "user", "content": "What does the hippocampus do at night?"}]}
    assert rows[1]["messages"][-1] == {"role": "assistant", "content": "A co-conspirator."}


def test_a_question_set_is_what_a_training_is_tested_on(client):
    client.post("/api/datasets", data={"name": "chats", "note": "x"}, files={"file": ("d.jsonl", jsonl(CHAT), "x")})
    qs = "\n".join(["first question?", "second question?", json.dumps({"question": "third?", "answer": "three"})])
    r = client.post("/api/datasets", data={"name": "my-questions", "note": "mine", "use": "questions"},
                    files={"file": ("q.txt", qs.encode(), "text/plain")})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["use"] == "questions" and r.json()["dataset"]["answered"] == 1
    jid = client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats", "test_dataset": "my-questions",
                                         "effort": "light"}).json()["id"]
    done = wait_for(lambda: (s := client.get(f"/api/jobs/{jid}").json())["status"]["state"] in ("done", "failed") and s)
    assert done["status"]["state"] == "done"
    s = done["samples"]
    assert [x["asked"] for x in s] == ["first question?", "second question?", "third?"]
    assert [x["expected"] for x in s] == ["", "", "three"], "no answer given is no answer expected"


def test_extras_switch_off_resolve_and_ride_with_the_run(client, tmp_path):
    ov = client.get("/api/overview").json()
    ids = {e["id"] for e in ov["extras"]}
    assert {e.id for e in BUILTIN} <= ids
    packing = next(e for e in ov["extras"] if e["id"] == "packing")
    assert packing["available"] is True and packing["on"] is True, "needs nothing: always there"
    liger = next(e for e in ov["extras"] if e["id"] == "liger")
    assert liger["available"] is False and "pip install" in liger["install"]
    assert client.put("/api/extras/packing", json={"on": False}).json()["on"] is False
    assert next(e for e in client.get("/api/overview").json()["extras"] if e["id"] == "packing")["on"] is False
    assert client.put("/api/extras/nope", json={"on": True}).status_code == 404

    client.post("/api/datasets", data={"name": "chats", "note": "x"}, files={"file": ("d.jsonl", jsonl(CHAT), "x")})
    assert client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats",
                                          "advanced": {"extras": ["nope"]}}).status_code == 400
    jid = client.post("/api/jobs", json={"base_model": "m/x", "dataset": "chats", "effort": "light",
                                         "advanced": {"extras": ["neftune", "liger"]}}).json()["id"]
    done = wait_for(lambda: (s := client.get(f"/api/jobs/{jid}").json())["status"]["state"] in ("done", "failed") and s)
    carried = done["spec"]["params"]["extras"]
    assert [e["id"] for e in carried] == ["neftune", "liger"] and carried[0]["train"] == {"neftune_noise_alpha": 5}
    log = "\n".join(done["log"])
    assert "extra on: NEFTune" in log and "liger_kernel will not load here" in log, "a missing one is said, not a crash"
    assert client.get(f"/api/jobs/{jid}/config").json()["advanced"]["extras"] == ["neftune", "liger"]


def test_your_own_extra_is_one_more(tmp_path):
    reg = Registry([{"id": "mine", "name": "Mine", "train": {"x": 1}}, {"name": "no id"}], tmp_path)
    assert "mine" in reg.items and reg.items["mine"].builtin is False and reg.problems
    assert reg.resolve(["mine"])[0]["train"] == {"x": 1}
