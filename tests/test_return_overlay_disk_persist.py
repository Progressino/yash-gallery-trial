"""Return overlay disk merge + shrink guards."""

import pandas as pd
import pytest

from backend.services.po_return_import import (
    _merge_overlay_import_locked,
    apply_return_overlay_import,
    load_return_overlay_df_from_disk,
    load_return_overlay_meta_from_disk,
    persist_return_overlay_df_to_disk,
    return_overlay_meta_disk_mismatch_warning,
)


def _overlay(skus_units: list[tuple[str, int]], source: str) -> pd.DataFrame:
    rows = []
    for sku, units in skus_units:
        rows.append(
            {
                "OMS_SKU": sku,
                "Return_Platform": "flipkart",
                "Return_Date": "2026-06-01",
                "Return_Units": units,
                "Source_File": source,
            }
        )
    return pd.DataFrame(rows)


def test_disk_merge_accumulates_across_imports(monkeypatch, tmp_path):
    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))
    import backend.main as main_mod

    monkeypatch.setattr(main_mod, "_DISK_CACHE_DIR", str(warm))
    main_mod._warm_cache = {}

    from backend.session import AppSession

    sess = AppSession()
    first = _overlay([("A", 5), ("B", 3)], "file-a.csv")
    second = _overlay([("C", 7)], "file-b.csv")

    apply_return_overlay_import(sess, first, replace=False, filename="file-a.csv")
    apply_return_overlay_import(sess, second, replace=False, filename="file-b.csv")

    disk = load_return_overlay_df_from_disk()
    assert int(disk["Return_Units"].sum()) == 15
    meta = load_return_overlay_meta_from_disk()
    assert int(meta.get("return_overlay_units") or 0) == 15
    assert len(meta.get("return_overlay_sources") or []) == 2


def test_persist_refuses_shrink_without_allow(monkeypatch, tmp_path):
    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))

    big = _overlay([("A", 100), ("B", 50)], "big.csv")
    small = _overlay([("A", 1)], "small.csv")
    assert persist_return_overlay_df_to_disk(big) is True
    assert persist_return_overlay_df_to_disk(small) is False
    disk = load_return_overlay_df_from_disk()
    assert int(disk["Return_Units"].sum()) == 150


def test_merge_locked_reads_existing_disk(monkeypatch, tmp_path):
    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))

    existing = _overlay([("X", 4)], "existing.csv")
    persist_return_overlay_df_to_disk(existing)
    incoming = _overlay([("Y", 6)], "incoming.csv")
    merged, sources = _merge_overlay_import_locked(
        incoming,
        file_key="incoming.csv",
        replace=False,
        uploaded_at="2026-06-25T12:00:00Z",
    )
    assert int(merged["Return_Units"].sum()) == 10
    assert len(sources) == 2


def test_meta_disk_mismatch_warning(monkeypatch, tmp_path):
    warm = tmp_path / "warm"
    warm.mkdir(parents=True)
    monkeypatch.setenv("WARM_CACHE_DIR", str(warm))

    persist_return_overlay_df_to_disk(_overlay([("A", 10)], "a.csv"))
    meta_path = warm / "return_overlay_meta.json"
    meta_path.write_text(
        '{"return_overlay_units": 1000, "return_overlay_skus": 50, "return_overlay_sources": []}',
        encoding="utf-8",
    )
    warn = return_overlay_meta_disk_mismatch_warning()
    assert warn is not None
    assert "1,000" in warn or "1000" in warn
