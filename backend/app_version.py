"""Build metadata for /api/health and the UI version badge."""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_build_info() -> dict:
    version = (os.environ.get("APP_VERSION") or "1.0.0").strip() or "1.0.0"
    sha = (os.environ.get("APP_GIT_SHA") or "").strip()
    if not sha:
        try:
            sha = (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=os.path.dirname(os.path.dirname(__file__)),
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
        except Exception:
            sha = "dev"
    built_at = (os.environ.get("APP_BUILT_AT") or "").strip()
    label = sha if sha and sha != "dev" else version
    return {
        "version": version,
        "git_sha": sha,
        "built_at": built_at,
        "label": label,
    }
