#!/usr/bin/env python3
"""Post-deploy: precompute default Intelligence bundle (Tier-3 + gap-fill, memory-safe)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_srv = Path(__file__).resolve().parents[2]
if str(_srv) not in sys.path:
    sys.path.insert(0, str(_srv))


def main() -> int:
    import os

    import backend.main as main_mod
    from backend.routers.data import (
        _default_intelligence_date_window,
        global_intelligence_bundle_ready,
        precompute_tier3_gapfill_intelligence_bundles,
    )

    os.environ.setdefault("INTELLIGENCE_PRECOMPUTE_MODE", "tier3_gapfill")

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        print(json.dumps({"ok": False, "error": "warm_cache_disk_load_failed"}))
        return 1

    main_mod._warm_cache = data
    main_mod._warm_cache_generation = max(int(getattr(main_mod, "_warm_cache_generation", 0) or 0), 2)
    main_mod._warm_cache_ready.set()

    from backend.routers.data import _load_intelligence_bundle_cache_from_disk

    _load_intelligence_bundle_cache_from_disk()

    precompute_tier3_gapfill_intelligence_bundles()

    from backend.services.intelligence_artifacts import (
        KIND_HOT,
        load_artifact,
        prebuild_standard_artifacts,
        standard_intelligence_windows,
    )

    prebuild_standard_artifacts()

    start_date, end_date = _default_intelligence_date_window()
    core_ready = global_intelligence_bundle_ready(
        start_date, end_date, limit=10, basis="gross", include_extras=False
    )
    extras_ready = global_intelligence_bundle_ready(
        start_date, end_date, limit=10, basis="gross", include_extras=True
    )
    artifact_windows = {}
    day_count = 0
    from backend.services.intelligence_artifact_store import _by_date_root

    for s, e in standard_intelligence_windows():
        hot, _ = load_artifact(s, e, KIND_HOT, allow_stale=False)
        artifact_windows[f"{s}_{e}"] = hot is not None
    by_date = _by_date_root()
    if os.path.isdir(by_date):
        day_count = len([f for f in os.listdir(by_date) if f.endswith(".parquet")])

    out = {
        "ok": core_ready,
        "window": {"start_date": start_date, "end_date": end_date},
        "core_bundle_ready": core_ready,
        "extras_bundle_ready": extras_ready,
        "artifact_windows": artifact_windows,
        "day_parquet_count": day_count,
    }
    print(json.dumps(out, indent=2))
    return 0 if core_ready else 2


if __name__ == "__main__":
    sys.exit(main())
