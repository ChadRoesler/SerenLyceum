"""SerenLyceum - where a model goes to school.

Named for Aristotle's school, the one Diogenes heckled from his jar outside.
The jar is the colour (pithos terracotta); the school is the job: a dataset
in, a model out, on this box or on a rented GPU, without anyone having to read
fifty papers or sit through a single AI-buzzword video first.

Three rules everything else falls out of:

  * WE JUST TRAIN. The Lyceum does not choose the model, does not make or vet
    the data, and has no opinion on where a corpus came from. Not our monkey,
    not our circus.
  * A RUN IS A DIRECTORY. A job is a folder - job.yaml, the dataset, the log,
    status.json, metrics.jsonl, out/ - and that folder is the only thing the
    service and the trainer share. A cloud run is the same folder, shipped to
    a pod and pulled back. Theatre can watch it like any other stage.
  * FORK, NEVER IMPORT. Training is `seren-lyceum-train <job dir>`, the same
    command a person could type, run in its own environment. torch never
    enters the service; the front door stays light enough for any box.
"""
from __future__ import annotations

# Version flows from the git tag via setuptools-scm (written to _version.py at
# build time, read here). The fallback only fires in a bare source checkout that
# was never built.
try:
    from ._version import version as __version__
except Exception:  # noqa: BLE001 - source checkout without a build
    __version__ = "0.0.0+unknown"
