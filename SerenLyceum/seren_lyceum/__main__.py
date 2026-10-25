"""
Entry point: python -m seren_lyceum [--config path]   (also the `seren-lyceum` script)

Boots the service with uvicorn on the resolved config. The trainer is a
different command (seren-lyceum-train) and a different environment.
"""
from __future__ import annotations

import argparse
import sys

import uvicorn
from seren_meninges.exposure import enforce_server

from .app import create_app
from .config import ENV_PREFIX, SERVICE, load_config


def _force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _maybe_inject_truststore(cfg) -> None:
    if not cfg.tls.trust_system_store:
        return
    try:
        import truststore
    except ImportError:
        print("[seren-lyceum] tls.trust_system_store is ON but 'truststore' isn't installed: "
              "pip install 'seren-lyceum[corp]' (continuing with certifi defaults).")
        return
    truststore.inject_into_ssl()


def main() -> None:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(prog="seren-lyceum",
                                     description="SerenLyceum - train your own model: a dataset in, a model out.")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to seren-lyceum.yaml (default: $SEREN_LYCEUM_CONFIG or ./seren-lyceum.yaml).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    # FIRST: an open bind with no token is refused here, with the three ways out printed.
    enforce_server(cfg.server, service=SERVICE, env_prefix=ENV_PREFIX)
    _maybe_inject_truststore(cfg)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port, log_level="info")


if __name__ == "__main__":
    main()
