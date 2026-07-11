#!/usr/bin/env python3
"""Ingest Combo Sku Map.xlsx into warm-cache (+ optional bundled JSON)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.services.combo_sku_map import (  # noqa: E402
    bundled_combo_sku_map_path,
    combo_bom_to_jsonable,
    combo_keys_as_identity_sku_mapping,
    combo_sku_map_disk_path,
    load_combo_sku_map_from_disk,
    merge_combo_sku_map,
    parse_combo_sku_map,
    persist_combo_sku_map_globally,
)
from backend.services.sku_mapping import (  # noqa: E402
    load_sku_mapping_from_disk,
    merge_sku_mapping_upload,
    persist_sku_mapping_globally,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "xlsx",
        nargs="?",
        default=str(Path.home() / "Downloads" / "Combo Sku Map.xlsx"),
    )
    ap.add_argument(
        "--also-bundled",
        action="store_true",
        help="Write backend/data/combo_sku_map.json",
    )
    args = ap.parse_args()
    path = Path(args.xlsx)
    if not path.is_file():
        print(f"missing file: {path}", file=sys.stderr)
        return 1

    bom = parse_combo_sku_map(path.read_bytes())
    if not bom:
        print("no combo rows parsed", file=sys.stderr)
        return 2

    merged = merge_combo_sku_map(load_combo_sku_map_from_disk(), bom)
    persist_combo_sku_map_globally(merged)

    stubs = combo_keys_as_identity_sku_mapping(merged)
    persist_sku_mapping_globally(
        merge_sku_mapping_upload(load_sku_mapping_from_disk(), stubs)
    )

    if args.also_bundled:
        bp = bundled_combo_sku_map_path()
        bp.parent.mkdir(parents=True, exist_ok=True)
        bp.write_text(
            json.dumps(combo_bom_to_jsonable(merged), ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"bundled → {bp}")

    print(f"combo_sku_map: {len(merged):,} listings → {combo_sku_map_disk_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
