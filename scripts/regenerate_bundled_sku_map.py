#!/usr/bin/env python3
"""Rebuild backend/data/yash_sku_mapping_master.json from the bundled .xlsx (run from repo root)."""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.services.sku_mapping import clear_bundled_sku_mapping_cache, parse_sku_mapping  # noqa: E402

def main() -> None:
    xlsx = _ROOT / "backend" / "data" / "yash_sku_mapping_master.xlsx"
    out = _ROOT / "backend" / "data" / "yash_sku_mapping_master.json"
    if not xlsx.is_file():
        raise SystemExit(f"Missing {xlsx}")
    m = parse_sku_mapping(xlsx.read_bytes())
    out.write_text(json.dumps(m, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    clear_bundled_sku_mapping_cache()
    print(f"Wrote {len(m):,} keys → {out}")

if __name__ == "__main__":
    main()
