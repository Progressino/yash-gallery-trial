#!/usr/bin/env python3
"""Rebuild warm sales_df from platform parquets and clear Tier-3 range cache (prod repair)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SRV = Path("/srv")
if _SRV.is_dir() and str(_SRV) not in sys.path:
    sys.path.insert(0, str(_SRV))

from backend.scripts.rebuild_sales_disk import main as rebuild_main  # noqa: E402
from backend.services.daily_store import clear_tier3_range_cache  # noqa: E402


def main() -> int:
    clear_tier3_range_cache()
    print("tier3 range cache cleared", flush=True)
    return rebuild_main()


if __name__ == "__main__":
    raise SystemExit(main())
