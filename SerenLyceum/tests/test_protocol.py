"""The trainer's lines read and write the way the old engine's did."""
from seren_lyceum import protocol


def test_the_old_engines_lines_read_back():
    assert protocol.parse("[SEREN:PROGRESS] step=40/300 epoch=0.53 loss=1.2345 lr=1.80e-04") == (
        "PROGRESS", {"step": 40, "total": 300, "epoch": 0.53, "loss": 1.2345, "lr": 1.8e-4})
    assert protocol.parse("[SEREN:METRIC] torch_version=2.4.0") == ("METRIC", {"torch_version": "2.4.0"})
    assert protocol.parse("[SEREN:STATUS] setup_complete") == ("STATUS", "setup_complete")
    assert protocol.parse("[SEREN:ERROR] out of memory") == ("ERROR", "out of memory")
    assert protocol.parse("{'loss': 1.2, 'epoch': 0.5}") is None, "ordinary log output is just log"


def test_lines_round_trip():
    ln = protocol.line("PROGRESS", {"step": 7, "total": 70, "epoch": 0.1, "loss": 2.0, "lr": 2e-4})
    assert ln == "[SEREN:PROGRESS] step=7/70 epoch=0.1000 loss=2.0000 lr=2.00e-04"
    kind, got = protocol.parse(ln)
    assert kind == "PROGRESS" and got["step"] == 7 and got["total"] == 70 and got["lr"] == 2e-4
    assert protocol.line("COMPLETE", "out/model-Q5_K_M.gguf") == "[SEREN:COMPLETE] out/model-Q5_K_M.gguf"
