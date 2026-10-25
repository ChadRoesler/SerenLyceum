"""
seren_lyceum.config
════════════════════════════════════════════════════════════════════════
Typed configuration, the family's way: dataclasses with lenient from_dict,
the shared ServerConfig / TlsConfig from SerenMeninges.

Resolution order (later wins):
    1. Defaults (this file)
    2. seren-lyceum.yaml (path from --config, $SEREN_LYCEUM_CONFIG, or
       ./seren-lyceum.yaml)
    3. Environment variables (SEREN_LYCEUM_*)

The only thing a box MUST say is where its training environment is
(local.python), and only if it trains locally: the service itself runs
anywhere, and a box with no GPU can still send every job to the cloud.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from seren_meninges import ServerConfig, TlsConfig, resolve_token
from seren_meninges.config import apply_env_overrides, read_yaml

log = logging.getLogger(__name__)

# Family convention: memory 7420, margin 7421, loci 7422, corpus-callosum 7423,
# hippocampus 7424, workbench 7425, symposium 7426, theatre 7427, lyceum 7428,
# probe 7430, observatory 7777.
DEFAULT_PORT = 7428
SERVICE = "seren-lyceum"
ENV_PREFIX = "SEREN_LYCEUM"


@dataclass
class UpdatesConfig:
    """"Is there a newer seren-lyceum" checking. Cosmetic, opt-outable."""
    enabled: bool = True
    check_interval_hours: float = 6.0
    index_url: str = "https://pypi.org/pypi/{distribution}/json"
    allow_prerelease: bool = False

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "UpdatesConfig":
        d = d if isinstance(d, dict) else {}
        default = cls()
        try:
            hours = float(d.get("check_interval_hours", default.check_interval_hours))
        except (TypeError, ValueError):
            hours = default.check_interval_hours
        return cls(enabled=bool(d.get("enabled", True)),
                   check_interval_hours=hours if hours > 0 else default.check_interval_hours,
                   index_url=str(d.get("index_url") or default.index_url),
                   allow_prerelease=bool(d.get("allow_prerelease", False)))


@dataclass
class StorageConfig:
    """Where the Lyceum keeps what it keeps: jobs (a directory each),
    datasets (what was uploaded, with what was found in it), and the base
    models it has downloaded. Empty means the default, ~/.seren-lyceum."""
    state_dir: str = ""

    def root(self) -> Path:
        raw = self.state_dir.strip() or "~/.seren-lyceum"
        return Path(os.path.expanduser(raw)).resolve()

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "StorageConfig":
        d = d if isinstance(d, dict) else {}
        return cls(state_dir=str(d.get("state_dir") or "").strip())


@dataclass
class LocalConfig:
    """Training on THIS box. `python` is the training environment's
    interpreter - the venv with seren-lyceum[train] and a torch built for this
    GPU. Empty means this box does not train locally (the backend says so in
    the UI instead of failing a job). `gguf` is a llama.cpp checkout or
    install, for turning a finished model into a GGUF file; without it a job
    still ends with the merged model and the adapter."""
    python: str = ""
    llama_cpp: str = ""
    hf_home: str = ""                # Hugging Face's home (models, the login); empty = $HF_HOME
    label: str = ""                  # how the UI names this box ("the Spark")
    max_parallel: int = 1            # jobs at once on this box

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "LocalConfig":
        d = d if isinstance(d, dict) else {}
        try:
            par = max(1, int(d.get("max_parallel", 1)))
        except (TypeError, ValueError):
            par = 1
        return cls(python=str(d.get("python") or "").strip(),
                   llama_cpp=str(d.get("llama_cpp") or "").strip(),
                   hf_home=str(d.get("hf_home") or "").strip(),
                   label=str(d.get("label") or "").strip(),
                   max_parallel=par)


@dataclass
class HippocampusLink:
    """A sibling hippocampus (seren-hippocampus), for its lessons: the
    consolidator's training set, made from the reviews, imported here as a
    dataset ("From the hippocampus"). The bearer is the one the hippocampus
    was installed with; inline, an env var, or a keyring ref, like every
    server block. Optional: without it the button is not there."""
    url: str = ""
    bearer_token: str = field(default="", repr=False)
    bearer_token_env: str = ""
    bearer_token_keyring: str = ""
    timeout_seconds: float = 180.0       # the lessons are built when asked, from every review

    def resolve_bearer(self) -> str:
        return resolve_token(inline=self.bearer_token or None, keyring_ref=self.bearer_token_keyring or None,
                             env_var=self.bearer_token_env or None) or ""

    @classmethod
    def from_dict(cls, d: Optional[dict[str, Any]]) -> "HippocampusLink":
        d = d if isinstance(d, dict) else {}
        try:
            t = float(d.get("timeout_seconds", 180.0))
        except (TypeError, ValueError):
            t = 180.0
        return cls(url=str(d.get("url") or "").strip(), bearer_token=str(d.get("bearer_token") or ""),
                   bearer_token_env=str(d.get("bearer_token_env") or ""),
                   bearer_token_keyring=str(d.get("bearer_token_keyring") or ""),
                   timeout_seconds=t if t > 0 else 180.0)


@dataclass
class CloudConfig:
    """A cloud provider's block, passed to its plugin as-is: the plugin owns
    its own keys (api key pointer, GPU type, image, region). Kept as a mapping
    so a new plugin needs no change here."""
    name: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class LyceumConfig:
    """The whole service."""
    server: ServerConfig = field(default_factory=lambda: ServerConfig(port=DEFAULT_PORT))
    tls: TlsConfig = field(default_factory=TlsConfig)
    updates: UpdatesConfig = field(default_factory=UpdatesConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    clouds: list[CloudConfig] = field(default_factory=list)
    hippocampus: HippocampusLink = field(default_factory=HippocampusLink)
    # a person's own extras, the same shape as the built-ins (seren_lyceum.extras)
    extras: list[dict[str, Any]] = field(default_factory=list)
    refresh_seconds: float = 2.0     # how often a running job's folder is re-read

    def cloud(self, name: str) -> Optional[CloudConfig]:
        return next((c for c in self.clouds if c.name == name), None)


def _clouds_from(raw: Any) -> list[CloudConfig]:
    out = []
    if isinstance(raw, dict):
        for name, opts in raw.items():
            opts = opts if isinstance(opts, dict) else {}
            out.append(CloudConfig(name=str(name), enabled=bool(opts.get("enabled", True)),
                                   options={k: v for k, v in opts.items() if k != "enabled"}))
    return out


def default_config_path() -> str:
    return os.environ.get(f"{ENV_PREFIX}_CONFIG") or "seren-lyceum.yaml"


def load_config(path: Optional[str] = None) -> LyceumConfig:
    data = read_yaml(path or default_config_path())
    try:
        refresh = float(data.get("refresh_seconds", 2.0))
    except (TypeError, ValueError):
        refresh = 2.0
    cfg = LyceumConfig(
        server=ServerConfig.from_dict(data.get("server"), default_port=DEFAULT_PORT),
        tls=TlsConfig.from_dict(data.get("tls")),
        updates=UpdatesConfig.from_dict(data.get("updates")),
        storage=StorageConfig.from_dict(data.get("storage")),
        local=LocalConfig.from_dict(data.get("local")),
        clouds=_clouds_from(data.get("clouds")),
        hippocampus=HippocampusLink.from_dict(data.get("hippocampus")),
        extras=[e for e in (data.get("extras") or []) if isinstance(e, dict)] if isinstance(data.get("extras"), list) else [],
        refresh_seconds=refresh if refresh > 0 else 2.0,
    )
    apply_env_overrides(cfg.server, prefix=ENV_PREFIX)
    if v := os.environ.get(f"{ENV_PREFIX}_STATE_DIR"):
        cfg.storage.state_dir = v
    if v := os.environ.get(f"{ENV_PREFIX}_LOCAL_PYTHON"):
        cfg.local.python = v
    return cfg
