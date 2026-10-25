"""Datasets: every common shape in, chat (or text) out, and what was found
said in words. We just train: nothing about where it came from is asked."""
import json

import pytest

from seren_lyceum.datasets import DatasetError, DatasetStore, inspect_lines


def lines(rows):
    return [json.dumps(r) for r in rows]


def test_every_shape_becomes_chat():
    rows, found = inspect_lines(lines([
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hey"}]},
        {"instruction": "say hi", "input": "", "output": "hi"},
        {"prompt": "2+2", "completion": "4"},
    ]))
    assert found["rows"] == 4 and found["unusable"] == 0
    assert all(r["messages"][-1]["role"] == "assistant" for r in rows)
    assert rows[1]["messages"][0] == {"role": "user", "content": "hi"}
    assert set(found["shapes"]) == {"chat", "sharegpt", "alpaca", "prompt"}


def test_what_cannot_be_used_is_said_with_its_line():
    rows, found = inspect_lines(["not json", json.dumps([1, 2]), json.dumps({"x": 1}),
                                 json.dumps({"messages": [{"role": "user", "content": "no answer"}]}),
                                 json.dumps({"text": "fine"})])
    assert found["rows"] == 1 and found["unusable"] == 4
    whys = {b["line"]: b["why"] for b in found["unusable_shown"]}
    assert whys[1] == "not valid JSON" and "no answer" in whys[4]


def test_a_dataset_is_kept_normalised_and_refused_when_unusable(tmp_path):
    store = DatasetStore(tmp_path)
    meta = store.add("My Chats", lines([{"prompt": "a", "completion": "b"}] * 5), note="mine")
    assert meta["id"] == "my-chats" and meta["rows"] == 5
    assert store.sample("my-chats", 1)[0]["messages"][-1]["content"] == "b"
    assert [d["id"] for d in store.list()] == ["my-chats"]
    with pytest.raises(DatasetError, match="nothing in it"):
        store.add("bad", ["nope"])
    with pytest.raises(DatasetError, match="mixes plain text"):
        store.add("mixed", lines([{"text": "t"}, {"prompt": "a", "completion": "b"}]))
    with pytest.raises(DatasetError, match="name"):
        store.add("  ", lines([{"text": "t"}]))
