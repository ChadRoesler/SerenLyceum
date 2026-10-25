"""
seren_lyceum.links
════════════════════════════════════════════════════════════════════════
Downloads the browser can do itself. A file behind the bearer cannot be an
ordinary link (a browser sends no Authorization header with a click), and
fetching it with the token into memory first made a 1 GB model look like a
dead click for a minute. So the page asks for a link - with the token - and
gets back a short-lived one under /viewer/ (public, like the page itself,
which Meninges serves without a token): an unguessable code that names one
file for a few minutes. The browser then downloads it the normal way, with
its own progress bar, as big as it is.

A link names a file on disk, or a few bytes made on the spot (a config, the
sample answers): nothing else is reachable through it.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

TTL = 600.0


@dataclass
class _Link:
    filename: str
    path: Optional[Path]
    data: Optional[bytes]
    media: str
    expires: float


class Links:
    def __init__(self, ttl: float = TTL):
        self.ttl = ttl
        self._links: dict[str, _Link] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        now = time.time()
        for code in [c for c, ln in self._links.items() if ln.expires < now]:
            self._links.pop(code, None)

    def make(self, filename: str, *, path: Optional[Path] = None, data: Optional[bytes] = None,
             media: str = "application/octet-stream") -> str:
        """A link's code, for a file (path) or for bytes (data)."""
        if (path is None) == (data is None):
            raise ValueError("a link names a file or bytes, not both and not neither")
        code = secrets.token_urlsafe(24)
        with self._lock:
            self._sweep()
            self._links[code] = _Link(filename=filename, path=path, data=data, media=media,
                                      expires=time.time() + self.ttl)
        return code

    def get(self, code: str) -> Optional[_Link]:
        with self._lock:
            self._sweep()
            return self._links.get(code)
