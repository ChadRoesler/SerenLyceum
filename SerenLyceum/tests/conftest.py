"""Shared fixtures: a Lyceum on a temp folder, training locally with the
trainer's fake mode (SEREN_LYCEUM_FAKE_TRAIN=1: the stages walked, no torch),
so the whole loop - dataset, job, runner, the forked trainer, its folder - is
exercised for real on any box."""
from __future__ import annotations

import json
import sys
import time

import pytest

from seren_lyceum.config import LocalConfig, LyceumConfig, StorageConfig

CHAT = [{"messages": [{"role": "user", "content": f"question {i}"},
                      {"role": "assistant", "content": f"answer {i}"}]} for i in range(30)]


@pytest.fixture(autouse=True)
def offline_update_checks(monkeypatch):
    """No test may talk to pypi.org."""
    try:
        from seren_meninges.updates import UpdateChecker
    except ImportError:
        return

    async def _no_network(self, distribution):
        raise ConnectionError("network disabled in tests")
    monkeypatch.setattr(UpdateChecker, "_fetch_from_index", _no_network)


@pytest.fixture
def fake_train(monkeypatch):
    monkeypatch.setenv("SEREN_LYCEUM_FAKE_TRAIN", "1")


@pytest.fixture
def cfg(tmp_path):
    return LyceumConfig(storage=StorageConfig(state_dir=str(tmp_path / "lyceum")),
                        local=LocalConfig(python=sys.executable, label="the test box"),
                        refresh_seconds=0.2)


@pytest.fixture
def client(cfg, fake_train):
    from fastapi.testclient import TestClient
    from seren_lyceum.app import create_app
    with TestClient(create_app(cfg)) as c:
        yield c


@pytest.fixture
def quiet_client(cfg, fake_train):
    """A Lyceum whose own loop ticks once at boot and then sleeps: for tests
    that drive the runner by hand and must not race it."""
    from fastapi.testclient import TestClient
    from seren_lyceum.app import create_app
    cfg.refresh_seconds = 3600
    with TestClient(create_app(cfg)) as c:
        yield c


def jsonl(rows) -> bytes:
    return ("\n".join(json.dumps(r) for r in rows) + "\n").encode()


def wait_for(fn, timeout=60.0, every=0.2):
    end = time.time() + timeout
    while time.time() < end:
        got = fn()
        if got:
            return got
        time.sleep(every)
    raise AssertionError("timed out waiting")
