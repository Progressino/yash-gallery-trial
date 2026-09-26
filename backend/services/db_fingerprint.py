"""Cheap change detection for SQLite files (main DB + WAL).

Any committed write bumps the WAL (or, after a checkpoint, the main file) size or
mtime, so a matching fingerprint means cached reads derived from these files are
still current. Costs one ``os.stat`` per file — safe to call per request.
"""
from __future__ import annotations

import os


def db_fingerprint(*paths: str) -> tuple:
    parts: list = []
    for path in paths:
        parts.append(path)
        for p in (path, f"{path}-wal"):
            try:
                st = os.stat(p)
            except OSError:
                parts.append((0, 0))
                continue
            # An empty WAL is created/deleted as connections open and close — it
            # holds no data, so it must not look like a change.
            if p.endswith("-wal") and st.st_size == 0:
                parts.append((0, 0))
            else:
                parts.append((st.st_mtime_ns, st.st_size))
    return tuple(parts)
