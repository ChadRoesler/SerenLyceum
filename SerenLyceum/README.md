# SerenLyceum

**Train your own model without reading fifty papers first.** A dataset in, a
model out, on this box or on a rented GPU, told in plain words the whole way.

Named for Aristotle's school - the one Diogenes heckled from the big clay jar
(a *pithos*) he lived in outside it, lantern in hand. The jar is the colour.

---

## The three rules everything else falls out of

**1. We just train.** The Lyceum does not choose your model, does not make or
vet your data, and has no opinion on where a corpus came from or what a
teacher model's licence says. That is your business. Not our monkey, not our
circus.

**2. A run is a directory.** A training is a folder: `job.yaml` and `data/`
(written by the service), `status.json`, `metrics.jsonl`, `train.log` and
`out/` (written by the trainer), and a `STOP` file when you ask it to stop.
That folder is the only thing the service and the trainer share, so a restart
loses nothing, and a cloud run is the same folder shipped up and pulled home.

**3. Fork, never import.** Training is a command:

```bash
seren-lyceum-train /path/to/a/job
```

It runs in its own environment (`seren-lyceum[train]` plus a torch built for
your GPU). The service forks exactly that command; you can type it yourself;
a cloud plugin runs it on a pod. The service never imports torch, so it runs
on any box, and the automated path and the hand-run path cannot drift apart.

---

## What a person does

Open `/viewer` and **New training** asks five questions, each answered by
picking a card:

1. **What should it learn?** A dataset: one example per line (JSONL).
   Conversations (`messages`), ShareGPT (`conversations`), instruction/output,
   prompt/completion and plain `text` all work. It says what it found - how
   many examples, the lines it could not use and why.
2. **Start from which model?** A few suggestions with what they are good for
   and whether they fit this box, or any Hugging Face name.
3. **Where should it train?** This box (free, your data never leaves the
   house) or a cloud.
4. **How hard should it study?** Light, Normal or Thorough, each a sentence.
   The numbers behind them are under *Advanced*, visible and overridable.
5. **What should come out?** A GGUF at a quant (Q4_K_M, Q5_K_M, Q8_0), or the
   merged model and adapter only.

**Trainings** then shows each run: the five stages (reading your data,
getting the model, learning, testing itself, saving the result), the learning
curve, how long is left, a few answers the new model wrote to examples it
never saw beside the answers in your data, and what it made - your model
first, one file to download.

## How it trains

A **LoRA** (a small add-on that teaches the model new habits without
retraining all of it), with Hugging Face TRL: the last answer in each
conversation is what it learns, everything before it is the question. Some
examples are held back to test with. When it is done it saves the adapter,
merges it into the base model, and - with llama.cpp on the box - converts to
GGUF and quantizes. A GGUF that cannot be made is a warning on a finished run,
not a failure: the merged model is the result.

The trainer prints the same `[SEREN:STATUS|PROGRESS|METRIC|ERROR|COMPLETE]`
lines the original Seren engine's RunPod trainer printed.

## Where it trains

- **local** - this box. Set `local.python` to the training environment.
- **cloud** - plugins, registered under the `seren_lyceum.backends` entry-point
  group and configured under `clouds:`. **RunPod** ships first; its round trip
  (start a pod, ship the job, run the same command, pull results home, stop
  the pod) is the next piece of work, and the UI lists it as coming until then.

## Install

```bash
pip install seren-lyceum                       # the service
python3 -m venv ~/seren-lyceum/train           # the training environment
~/seren-lyceum/train/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
~/seren-lyceum/train/bin/pip install "seren-lyceum[train]"
```

Copy `seren-lyceum.yaml.sample` to `~/seren-lyceum/seren-lyceum.yaml`, set
`local.python`, then:

```bash
python -m seren_lyceum --config ~/seren-lyceum/seren-lyceum.yaml
```

and open <http://127.0.0.1:7428/viewer>.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests run the whole loop for real - upload, job, runner, the forked
trainer, its folder - with the trainer's fake mode (`SEREN_LYCEUM_FAKE_TRAIN=1`:
the stages walked, no torch), so they pass on any box. Fake mode is also how
to see the UI on a box with no GPU; a fake run says so everywhere.

## License

AGPL-3.0-or-later.
