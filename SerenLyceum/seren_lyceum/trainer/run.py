"""
seren_lyceum.trainer.run
════════════════════════════════════════════════════════════════════════
One job, five stages, told as it goes:

    prepare    read the data, hold some back to test with, count it
    download   fetch the base model (once; the Hugging Face cache keeps it)
    train      a LoRA: a small add-on that teaches the model new habits
               without retraining all of it
    evaluate   how it does on the examples it never saw, and a few answers
               written by the new model, so a person can read the change
    export     the adapter, the model with it merged in, and - when this box
               has llama.cpp - a GGUF a llama-server can load

Every stage writes status.json (where the run is), appends metrics.jsonl
(the numbers over time) and prints the [SEREN:...] protocol lines. A STOP
file in the job directory is checked between steps: the run stops cleanly,
says so, and keeps what it has.

What it learns from a conversation is the LAST answer in it (the assistant's
final message), with everything before it as the question. Plain-text data
is learned as it reads.

SEREN_LYCEUM_FAKE_TRAIN=1 walks the same stages with no torch and no model,
in seconds: for the service's tests and for seeing the UI on a box with no
GPU. A fake run says it is fake in its status and its log.
"""
from __future__ import annotations

import inspect
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import yaml

from .. import protocol

STAGES = ("prepare", "download", "train", "evaluate", "export")


class Run:
    """The run directory, from the trainer's side."""

    def __init__(self, d: Path):
        self.d = Path(d)
        self.spec = yaml.safe_load((self.d / "job.yaml").read_text(encoding="utf-8")) or {}
        self.params = self.spec.get("params") or {}
        self.out = self.d / "out"
        self.work = self.d / "work"
        self.out.mkdir(exist_ok=True)
        self.work.mkdir(exist_ok=True)
        self.status: dict[str, Any] = {}
        self.fake = os.environ.get("SEREN_LYCEUM_FAKE_TRAIN") == "1"
        self._t0 = time.time()

    def say(self, kind: str, payload: Any = "") -> None:
        print(protocol.line(kind, payload), flush=True)

    def set(self, **change: Any) -> None:
        self.status.update(change)
        self.status["updated_at"] = time.time()
        from ..jobs import replace_atomic
        replace_atomic(self.d / "status.json", json.dumps(self.status, indent=1))

    def stage(self, name: str, message: str) -> None:
        self.set(stage=name, stage_index=STAGES.index(name), message=message)
        self.say("STATUS", f"{name}_started")
        print(message, flush=True)

    def metric(self, row: dict[str, Any]) -> None:
        with (self.d / "metrics.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time() - self._t0, 1), **row}) + "\n")

    def stopping(self) -> bool:
        return (self.d / "STOP").exists()

    def finish(self, state: str, message: str, **extra: Any) -> None:
        self.set(state=state, message=message, ended_at=time.time(), **extra)


# ── prepare ─────────────────────────────────────────────────────────────────
def prepare(r: Run) -> tuple[list[dict], list[dict], str]:
    rows = [json.loads(ln) for ln in (r.d / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if not rows:
        raise ValueError("the dataset has no examples")
    kind = "text" if "text" in rows[0] else "chat"
    rng = random.Random(int(r.params.get("seed", 42)))
    rng.shuffle(rows)
    test = r.d / "data" / "test.jsonl"
    if test.is_file():
        # the person's own questions (a test dataset, picked in New training):
        # tested on exactly those, and every example of the dataset is learned
        # from - nothing is held back at random (24 Oct 2026: "where do we load
        # or input to check the questions it never saw?")
        held = [json.loads(ln) for ln in test.read_text(encoding="utf-8").splitlines() if ln.strip()]
        train = rows
        print(f"testing on your own {len(held)} questions", flush=True)
    else:
        frac = float(r.params.get("eval_fraction", 0.05))
        n_eval = min(max(1, round(len(rows) * frac)), 200) if len(rows) >= 20 and frac > 0 else 0
        held, train = rows[:n_eval], rows[n_eval:]
    for name, part in (("train.jsonl", train), ("eval.jsonl", held)):
        with (r.work / name).open("w", encoding="utf-8") as fh:
            for row in part:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    r.set(rows_train=len(train), rows_eval=len(held), data_kind=kind)
    r.say("METRIC", {"rows_train": len(train), "rows_eval": len(held)})
    print(f"{len(train)} examples to learn from, {len(held)} held back to test with.", flush=True)
    return train, held, kind


# ── download ────────────────────────────────────────────────────────────────
def download(r: Run, base: str) -> str:
    if Path(base).is_dir():
        print(f"using the model at {base}", flush=True)
        return base
    if r.fake:
        return base
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=base, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                           "tokenizer*", "*.jinja", "*.py"])
    r.say("METRIC", {"base_model_path": path.replace(" ", "_")})
    return path


# ── train ───────────────────────────────────────────────────────────────────
def _kwargs_for(cls, wanted: dict[str, Any]) -> dict[str, Any]:
    """Only the arguments this version of the class takes: TRL and
    transformers rename things between releases (max_seq_length became
    max_length; tokenizer became processing_class), and a run must not die
    of a rename."""
    if hasattr(cls, "__dataclass_fields__"):
        names = set(cls.__dataclass_fields__)
    else:
        names = set(inspect.signature(cls).parameters)
    return {k: v for k, v in wanted.items() if k in names}


def _extras_plan(r: Run) -> tuple[dict, dict, dict]:
    """The run's extras (seren_lyceum.extras) as keyword arguments for the
    model, the training and the adapter. One whose module will not import
    here is said and skipped."""
    model_kw: dict = {}
    train_kw: dict = {}
    lora_kw: dict = {}
    for e in r.params.get("extras") or []:
        if e.get("needs"):
            try:
                __import__(e["needs"])
            except Exception as ex:  # noqa: BLE001
                r.say("STATUS", f"extra_unavailable:{e.get('id')}")
                print(f"extra {e.get('name') or e.get('id')}: {e['needs']} will not load here "
                      f"({type(ex).__name__}); going on without it", flush=True)
                continue
        if e.get("special") == "4bit":
            r.params["load_4bit"] = True
        if (e.get("model") or {}).get("attn_implementation") == "flash_attention_2":
            r.params["attention"] = "flash_attention_2"
        model_kw.update({k: v for k, v in (e.get("model") or {}).items() if k != "attn_implementation"})
        train_kw.update(e.get("train") or {})
        lora_kw.update(e.get("lora") or {})
        print(f"extra on: {e.get('name') or e.get('id')}", flush=True)
        r.say("METRIC", {"extra": e.get("id")})
    return model_kw, train_kw, lora_kw


def _fit(cls, wanted: dict, what: str) -> dict:
    """_kwargs_for, saying what this version does not take."""
    kept = _kwargs_for(cls, wanted)
    for k in sorted(set(wanted) - set(kept)):
        print(f"{what}: this version has no {k!r}; skipped", flush=True)
    return kept


def train(r: Run, model_path: str, train_rows: list[dict], eval_rows: list[dict], kind: str):
    model_kw, train_kw, lora_kw = _extras_plan(r)
    if r.fake:
        return _fake_train(r, len(train_rows))
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    p = r.params
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    load = {"device_map": {"": 0}} if cuda else {}
    dt = torch.bfloat16 if bf16 else (torch.float16 if cuda else torch.float32)
    load.update(_extras(r, cuda, dt))
    load.update(model_kw)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dt, **load)
    except TypeError:                                  # transformers before `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dt, **load)
    if r.params.get("load_4bit") and "quantization_config" in load:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    def shape(row: dict) -> dict:
        if kind == "text":
            return {"text": row["text"]}
        msgs = row["messages"]
        return {"prompt": msgs[:-1], "completion": [msgs[-1]]}
    train_ds = Dataset.from_list([shape(x) for x in train_rows])
    # a question with no answer given can be asked, not scored: the loss on
    # held-back examples counts only the ones with an answer
    scored = [x for x in eval_rows if kind == "text" or (x["messages"] and x["messages"][-1]["role"] == "assistant")]
    eval_ds = Dataset.from_list([shape(x) for x in scored]) if scored else None

    steps_per_epoch = max(1, len(train_rows) // (int(p["batch_size"]) * int(p["grad_accum"])))
    total = steps_per_epoch * int(p["epochs"])
    wanted = {
        "output_dir": str(r.work / "checkpoints"), "num_train_epochs": float(p["epochs"]),
        "learning_rate": float(p["learning_rate"]), "per_device_train_batch_size": int(p["batch_size"]),
        "per_device_eval_batch_size": int(p["batch_size"]), "gradient_accumulation_steps": int(p["grad_accum"]),
        "lr_scheduler_type": "cosine", "warmup_ratio": float(p.get("warmup_ratio", 0.03)),
        "logging_steps": 1 if total < 200 else 5, "save_strategy": "no", "report_to": "none",
        "eval_strategy": "steps" if eval_ds is not None else "no", "eval_steps": max(10, total // 8),
        "bf16": bf16, "fp16": cuda and not bf16, "seed": int(p.get("seed", 42)),
        "gradient_checkpointing": True, "max_length": int(p["max_length"]),
        "max_seq_length": int(p["max_length"]), "completion_only_loss": kind != "text",
        "dataset_text_field": "text", "disable_tqdm": True,
    }
    if train_kw:
        wanted.update(_fit(SFTConfig, train_kw, "extra"))
    cfg = SFTConfig(**_kwargs_for(SFTConfig, wanted))
    lora = LoraConfig(r=int(p["lora_rank"]), lora_alpha=int(p["lora_alpha"]), lora_dropout=0.05,
                      target_modules="all-linear", task_type="CAUSAL_LM", **_fit(LoraConfig, lora_kw, "extra"))

    class Progress(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            r.set(step=0, total=state.max_steps, message="learning")
            r.say("STATUS", "train_running")

        def on_log(self, args, state, control, logs=None, **kw):
            logs = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            row = {"step": state.global_step, "total": state.max_steps, "epoch": round(state.epoch or 0, 3), **logs}
            r.metric(row)
            done = max(1, state.global_step)
            per = (time.time() - started) / done
            change = {"step": state.global_step, "total": state.max_steps, "epoch": row["epoch"],
                      "eta_seconds": round(per * max(0, state.max_steps - state.global_step))}
            for k in ("loss", "eval_loss", "learning_rate"):
                if k in logs:
                    change["lr" if k == "learning_rate" else k] = logs[k]
            r.set(**change)
            if "loss" in logs:
                r.say("PROGRESS", {"step": state.global_step, "total": state.max_steps, "epoch": row["epoch"],
                                   "loss": logs["loss"], "lr": logs.get("learning_rate", 0.0)})
            if "eval_loss" in logs:
                r.say("METRIC", {"eval_loss": logs["eval_loss"]})

        def on_step_end(self, args, state, control, **kw):
            if r.stopping():
                control.should_training_stop = True
            return control

    started = time.time()
    trainer_kw_model_path = model_path
    trainer = SFTTrainer(**_kwargs_for(SFTTrainer, {
        "model": model, "args": cfg, "train_dataset": train_ds, "eval_dataset": eval_ds,
        "processing_class": tok, "tokenizer": tok, "peft_config": lora, "callbacks": [Progress()]}))
    trainer.train()
    trainer._seren_tok = tok
    trainer._seren_base = trainer_kw_model_path
    return trainer


def _extras(r: Run, cuda: bool, dt) -> dict:
    """The box's extras this run asked for, each checked before it is used:
    an extra that is missing or will not import is said and passed by, never
    a crash at step one."""
    out: dict = {}
    want_attn = r.params.get("attention", "sdpa")
    if want_attn == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            out["attn_implementation"] = "flash_attention_2"
            r.say("METRIC", {"attention": "flash_attention_2"})
        except Exception as e:  # noqa: BLE001
            r.say("STATUS", "flash_attn_unavailable")
            print(f"flash attention was asked for but will not load here ({type(e).__name__}); using sdpa", flush=True)
            out["attn_implementation"] = "sdpa"
    else:
        out["attn_implementation"] = "sdpa"
    if r.params.get("load_4bit"):
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            out["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=dt, bnb_4bit_use_double_quant=True)
            r.say("METRIC", {"load": "4bit_nf4"})
        except Exception as e:  # noqa: BLE001
            r.say("STATUS", "bitsandbytes_unavailable")
            print(f"4-bit loading was asked for but bitsandbytes will not load here ({type(e).__name__}); "
                  "loading in full precision", flush=True)
    return out


def _fake_train(r: Run, n: int) -> None:
    total = 12
    r.set(step=0, total=total, message="learning (fake run: no model is being trained)")
    loss = 2.4
    for step in range(1, total + 1):
        if r.stopping():
            return None
        time.sleep(0.05)
        loss *= 0.93
        r.metric({"step": step, "total": total, "epoch": round(step / total, 3), "loss": round(loss, 4),
                  "learning_rate": 2e-4})
        r.set(step=step, loss=round(loss, 4), eta_seconds=0)
        r.say("PROGRESS", {"step": step, "total": total, "epoch": step / total, "loss": loss, "lr": 2e-4})
    return None


# ── evaluate ────────────────────────────────────────────────────────────────
def evaluate(r: Run, trainer, eval_rows: list[dict], kind: str) -> None:
    if r.fake or trainer is None:
        r.set(eval_loss=0.9)
        r.say("METRIC", {"eval_loss": 0.9})
        # the real questions, so the page shows what a real run would; only
        # the answers are pretend
        n = int(r.params.get("samples", 24))
        _write_samples(r, [{"asked": _asked(row), "expected": _expected(row), "got": "(fake run: no model answered)"}
                           for row in eval_rows[:n]] or
                          [{"asked": "a held-back question", "expected": "its answer", "got": "(fake run: no model answered)"}])
        return
    # Testing is how a person reads the change, not part of making it: a
    # failure here is said and the run goes on to save what it learned.
    if eval_rows:
        try:
            m = trainer.evaluate()
            if "eval_loss" in m:
                r.set(eval_loss=m["eval_loss"])
                r.say("METRIC", {"eval_loss": m["eval_loss"]})
                r.metric({"step": trainer.state.global_step, "eval_loss": m["eval_loss"]})
        except Exception as e:  # noqa: BLE001
            r.say("STATUS", "eval_loss_skipped")
            print(f"could not measure the held-back examples ({type(e).__name__}: {e}); going on", flush=True)
    if kind == "text" or not eval_rows:
        return
    try:
        _samples(r, trainer, eval_rows)
    except Exception as e:  # noqa: BLE001
        r.say("STATUS", "samples_skipped")
        print(f"could not write sample answers ({type(e).__name__}: {e}); going on", flush=True)


def _samples(r: Run, trainer, eval_rows: list[dict]) -> None:
    import torch
    tok = trainer._seren_tok
    model = trainer.model
    model.eval()
    samples = []
    # every held-back example gets its answer, up to `samples` (24 by default:
    # each is a generation, and a run should not spend an hour on its own
    # test). Only three were answered before 24 Oct 2026.
    todo = eval_rows[:int(r.params.get("samples", 24))]
    for i, row in enumerate(todo):
        r.set(message=f"answering the questions it never saw: {i + 1} of {len(todo)}")
        msgs = row["messages"]
        # return_dict: transformers 5 hands back a BatchEncoding either way,
        # and asking for the dict is what both old and new versions agree on
        # enable_thinking=False: a thinking model (Qwen3) answers instead of
        # spending the sample's whole budget thinking; templates without the
        # switch ignore it
        ask = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs
        enc = tok.apply_chat_template(ask, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      enable_thinking=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            got = model.generate(**enc, max_new_tokens=768, do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        samples.append({"asked": _asked(row), "expected": _expected(row),
                        "got": tok.decode(got[0][n:], skip_special_tokens=True)})
        _write_samples(r, samples)          # as they come, so the page shows them arriving


def _asked(row: dict) -> str:
    users = [m["content"] for m in row.get("messages") or [] if m["role"] == "user"]
    return users[-1] if users else (row.get("text") or "")[:400]


def _expected(row: dict) -> str:
    msgs = row.get("messages") or []
    return msgs[-1]["content"] if msgs and msgs[-1]["role"] == "assistant" else ""


def _write_samples(r: Run, samples: list[dict]) -> None:
    with (r.out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    r.set(samples=len(samples))


# ── export ──────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "model").lower()).strip("-")[:40] or "model"


def export(r: Run, trainer) -> list[str]:
    made: list[str] = []
    quant = (r.spec.get("export") or {}).get("quant", "Q5_K_M")
    name = _slug(r.spec.get("name", "model"))
    if r.fake or trainer is None:
        (r.out / "adapter").mkdir(exist_ok=True)
        (r.out / "adapter" / "FAKE.txt").write_text("a fake run made no model\n", encoding="utf-8")
        return ["adapter/FAKE.txt"]
    tok = trainer._seren_tok
    trainer.model.save_pretrained(str(r.out / "adapter"))
    tok.save_pretrained(str(r.out / "adapter"))
    made.append("adapter/")
    r.say("STATUS", "adapter_saved")
    if r.params.get("load_4bit"):
        # merging into a 4-bit model loses what 4-bit lost: merge the adapter
        # into a full-precision copy of the base instead
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        del trainer.model
        torch.cuda.empty_cache()
        try:
            base = AutoModelForCausalLM.from_pretrained(trainer._seren_base, dtype=torch.bfloat16)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(trainer._seren_base, torch_dtype=torch.bfloat16)
        merged = PeftModel.from_pretrained(base, str(r.out / "adapter")).merge_and_unload()
    else:
        merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(r.out / "merged"), safe_serialization=True)
    tok.save_pretrained(str(r.out / "merged"))
    made.append("merged/")
    r.say("STATUS", "merged_saved")
    if quant == "none":
        return made
    llama = os.environ.get("SEREN_LYCEUM_LLAMA_CPP") or (r.spec.get("export") or {}).get("llama_cpp") or ""
    # The GGUF is a convenience on top of the result, not the result: a run
    # whose merged model is saved is done, and a GGUF that could not be made
    # is said as a warning with the reason, never as a failed run.
    try:
        gguf = _gguf(r, Path(llama) if llama else None, name, quant)
    except Exception as e:  # noqa: BLE001
        why = str(e)
        if isinstance(e, subprocess.CalledProcessError):
            why = f"llama.cpp's {Path(str(e.cmd[1] if len(e.cmd) > 1 else e.cmd[0])).name} stopped with an error (the log has it)"
        r.say("STATUS", "gguf_failed")
        r.set(warning=f"The GGUF could not be made: {why}. The merged model is in out/merged.")
        print(f"the GGUF could not be made: {e}", flush=True)
        gguf = None
    if gguf:
        made.append(gguf)
    return made


def _gguf(r: Run, llama: Optional[Path], name: str, quant: str) -> Optional[str]:
    """merged/ -> a GGUF at the chosen quant, with llama.cpp's own tools. No
    llama.cpp, no GGUF: the merged model is still there, and the log says so."""
    convert = llama / "convert_hf_to_gguf.py" if llama else None
    if not convert or not convert.is_file():
        r.say("STATUS", "gguf_skipped")
        print("No llama.cpp on this box (local.llama_cpp): the merged model is in out/merged, "
              "and the GGUF can be made wherever llama.cpp lives.", flush=True)
        return None
    f16 = r.out / f"{name}-F16.gguf"
    subprocess.run([sys.executable, str(convert), str(r.out / "merged"), "--outfile", str(f16), "--outtype", "f16"],
                   check=True)
    qbin = next((p for p in (llama / "build" / "bin" / "llama-quantize", llama / "llama-quantize",
                             Path(shutil.which("llama-quantize") or "")) if p and p.is_file()), None)
    if qbin is None:
        r.say("STATUS", "quantize_skipped")
        print("llama-quantize was not found; the GGUF is left at F16.", flush=True)
        return f16.name
    out = r.out / f"{name}-{quant}.gguf"
    subprocess.run([str(qbin), str(f16), str(out), quant], check=True)
    f16.unlink(missing_ok=True)
    return out.name


# ── the whole run ───────────────────────────────────────────────────────────
def run_job(d: Path) -> int:
    r = Run(d)
    r.set(state="running", started_at=time.time(), fake=r.fake, pid=os.getpid())
    if r.fake:
        print("FAKE RUN (SEREN_LYCEUM_FAKE_TRAIN=1): the stages are walked, no model is trained.", flush=True)
    trainer = None
    try:
        r.stage("prepare", "reading your data and holding some back to test with")
        train_rows, eval_rows, kind = prepare(r)
        r.stage("download", f"getting the base model ({r.spec.get('base_model')})")
        path = download(r, r.spec["base_model"])
        r.stage("train", "learning")
        trainer = train(r, path, train_rows, eval_rows, kind)
        if r.stopping():
            r.say("STATUS", "stopped")
            r.finish("stopped", "stopped when asked; nothing was exported")
            return 0
        r.stage("evaluate", "testing it on the examples it never saw")
        evaluate(r, trainer, eval_rows, kind)
        r.stage("export", "saving the result")
        made = export(r, trainer)
        main = next((m for m in made if m.endswith(".gguf")), made[-1] if made else "")
        r.say("COMPLETE", f"out/{main}")
        r.finish("done", "done", artifacts=made, stage_index=len(STAGES))
        return 0
    except Exception as e:  # noqa: BLE001 - every failure is said, in the status and the log
        traceback.print_exc()
        r.say("ERROR", f"{type(e).__name__}: {e}")
        r.finish("failed", f"{type(e).__name__}: {e}")
        return 1
