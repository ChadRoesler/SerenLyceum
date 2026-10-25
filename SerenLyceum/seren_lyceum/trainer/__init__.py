"""The trainer: `seren-lyceum-train <job dir>`.

Runs in the TRAINING environment (seren-lyceum[train] plus a torch built for
the GPU), never in the service. The service forks it; a person can type it;
a cloud plugin runs it on a pod. All three are the same command, so the
automated path and the hand-run path cannot drift apart.

Nothing here imports torch at module scope: `--check` must answer even when
torch is the thing that is missing.
"""
