"""SKU replacement map resolution + inv history/matrix consistency tests."""
from __future__ import annotations

import io

import pandas as pd
import pytest


def test_resolve_breaks_reverse_pair_cycles():
    from backend.services.sku_mapping import (
        follow_sku_replacement,
        resolve_sku_replacement_map,
    )

    # Opposite Replace-SKU files: CMB↔MB
    raw = {
        "1172YKCMB5018-M": "1172YKMB5018-M",
        "1172YKMB5018-M": "1172YKCMB5018-M",
        "1172YKCMB5018-S": "1172YKMB5018-S",
        "1172YKMB5018-S": "1172YKCMB5018-S",
    }
    resolved = resolve_sku_replacement_map(raw)
    # Both ends must land on the same terminal
    a = follow_sku_replacement("1172YKCMB5018-M", resolved)
    b = follow_sku_replacement("1172YKMB5018-M", resolved)
    assert a == b
    assert a in ("1172YKCMB5018-M", "1172YKMB5018-M")
    c = follow_sku_replacement("1172YKCMB5018-S", resolved)
    d = follow_sku_replacement("1172YKMB5018-S", resolved)
    assert c == d


def test_resolve_collapses_chains():
    from backend.services.sku_mapping import resolve_sku_replacement_map

    raw = {"A-OLD": "A-MID", "A-MID": "A-NEW", "B": "B"}  # identity kept as registration
    r = resolve_sku_replacement_map(raw)
    assert r["A-OLD"] == "A-NEW"
    assert r["A-MID"] == "A-NEW"
    assert r.get("B") == "B"


def test_parse_replace_sheets_no_reverse_after_merge():
    from backend.services.sku_mapping import (
        merge_sku_mapping_upload,
        parse_sku_mapping,
    )

    p1 = open("/Users/samraisinghani/Downloads/Replace sku 4-8-26 1.xlsx", "rb").read()
    p2 = open("/Users/samraisinghani/Downloads/Replace sku 4-8-26 2.xlsx", "rb").read()
    m1 = parse_sku_mapping(p1)
    m2 = parse_sku_mapping(p2)
    assert len(m1) > 100 and len(m2) > 50
    merged = merge_sku_mapping_upload(m1, m2)
    # No mutual reverse pairs remain (A→B and B→A); identity A→A is fine
    for k, v in list(merged.items()):
        if k == v:
            continue
        if merged.get(v) == k:
            pytest.fail(f"reverse pair remains {k}<->{v}")
        follow_ok(merged, k)


def follow_ok(mp, k, lim=20):
    from backend.services.sku_mapping import follow_sku_replacement

    t = follow_sku_replacement(k, mp)
    assert t
    return True


def test_recanonicalize_sums_reverse_pair_stock():
    from backend.services.daily_inventory_history import recanonicalize_inventory_history_skus
    from backend.services.sku_mapping import resolve_sku_replacement_map

    mp = resolve_sku_replacement_map(
        {
            "1172YKCMB5018-M": "1172YKMB5018-M",
            "1172YKMB5018-M": "1172YKCMB5018-M",
        }
    )
    df = pd.DataFrame(
        {
            "OMS_SKU": ["1172YKCMB5018-M", "1172YKMB5018-M"],
            "Date": ["2026-08-03", "2026-08-03"],
            "Qty": [10.0, 15.0],
            "Source": ["uploaded", "uploaded"],
            "Channel": ["oms", "oms"],
        }
    )
    out = recanonicalize_inventory_history_skus(df, mp)
    assert len(out) == 1
    assert float(out["Qty"].iloc[0]) == 25.0


def test_wide_matrix_applies_mapping():
    from backend.services.daily_inventory_history import inventory_history_wide_matrix

    df = pd.DataFrame(
        {
            "OMS_SKU": ["OLD-A", "NEW-A", "OLD-A", "NEW-A"],
            "Date": ["2026-08-01", "2026-08-01", "2026-08-02", "2026-08-02"],
            "Qty": [3.0, 7.0, 1.0, 4.0],
            "Source": ["uploaded"] * 4,
            "Channel": ["oms"] * 4,
        }
    )
    wide = inventory_history_wide_matrix(
        df,
        limit=50,
        days=2,
        end_date="2026-08-02",
        channel="oms",
        sku_mapping={"OLD-A": "NEW-A"},
    )
    assert wide["loaded"] is True
    skus = [r["sku"] for r in wide["rows"]]
    assert skus == ["NEW-A"]
    # totals: day1 10, day2 5
    assert wide["date_totals"][-1] == pytest.approx(5.0)
    assert wide["rows"][0]["qtys"][-1] == pytest.approx(5.0)


def test_matrix_reference_consolidation_preserves_units():
    """User matrix + replace sheets: unit total conserved after replace consolidate."""
    import re
    from backend.services.sku_mapping import (
        merge_sku_mapping_upload,
        parse_sku_mapping,
    )
    from backend.services.inventory import _inventory_alias_oms_key

    p1 = open("/Users/samraisinghani/Downloads/Replace sku 4-8-26 1.xlsx", "rb").read()
    p2 = open("/Users/samraisinghani/Downloads/Replace sku 4-8-26 2.xlsx", "rb").read()
    mp = merge_sku_mapping_upload(parse_sku_mapping(p1), parse_sku_mapping(p2))
    m = pd.read_csv("/Users/samraisinghani/Downloads/inventory-matrix-oms-2026-08-03.csv")
    date_cols = [c for c in m.columns if re.match(r"\d{1,2}-\d{1,2}-\d{2}$", str(c))]
    dc = date_cols[-1]  # 3-8-26
    skus = m[~m["SKU"].astype(str).str.lower().str.contains("total")].copy()
    skus[dc] = pd.to_numeric(skus[dc], errors="coerce").fillna(0)
    before = float(skus[dc].sum())
    skus["canon"] = skus["SKU"].map(lambda s: _inventory_alias_oms_key(s, mp))
    after = float(skus.groupby("canon")[dc].sum().sum())
    assert after == pytest.approx(before, rel=0, abs=0.01)
    # reverse-pair families must not remain as two positive stock rows
    for a, b in [
        ("1172YKCMB5018-M", "1172YKMB5018-M"),
        ("1172YKCMB5018-S", "1172YKMB5018-S"),
    ]:
        ca = _inventory_alias_oms_key(a, mp)
        cb = _inventory_alias_oms_key(b, mp)
        assert ca == cb
