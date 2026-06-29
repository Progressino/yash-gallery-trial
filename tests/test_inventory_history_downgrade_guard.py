"""Daily inventory history must never be downgraded on disk/warm-cache by stale PO sessions."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_is_newer_than,
    inventory_history_view_end_date,
    overlay_inventory_variant_from_history,
)


def _hist(sku: str, end: str, days: int = 30) -> pd.DataFrame:
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    return pd.DataFrame({"OMS_SKU": [sku] * days, "Date": dates, "Qty": [5.0] * days})


def test_inventory_history_is_newer_than_prefers_later_max_date():
    old = _hist("A", "2026-05-30", 3)
    new = _hist("A", "2026-06-26", 30)
    assert inventory_history_is_newer_than(new, old)
    assert not inventory_history_is_newer_than(old, new)


def test_inventory_history_is_newer_than_wins_with_fewer_rows():
    """June upload trimmed to 30d must beat May blob with more rows."""
    may = _hist("A", "2026-05-30", 30)
    june = _hist("A", "2026-06-26", 10)
    assert len(may) > len(june)
    assert inventory_history_is_newer_than(june, may)
    assert not inventory_history_is_newer_than(may, june)


def test_recanonicalize_inventory_history_skus_maps_aliases():
    from backend.services.daily_inventory_history import recanonicalize_inventory_history_skus

    df = pd.DataFrame(
        {
            "OMS_SKU": ["OLD-SKU", "OLD-SKU"],
            "Date": pd.to_datetime(["2026-06-25", "2026-06-26"]),
            "Qty": [3.0, 5.0],
        }
    )
    mapping = {"OLD-SKU": "NEW-SKU"}
    out = recanonicalize_inventory_history_skus(df, mapping)
    assert set(out["OMS_SKU"].astype(str)) == {"NEW-SKU"}
    assert len(out) == 2


def test_existing_po_frame_is_newer_than_prefers_generation():
    from backend.services.existing_po import existing_po_frame_is_newer_than

    old = pd.DataFrame({"OMS_SKU": ["A"] * 500, "PO_Pipeline_Total": [1.0] * 500})
    new = pd.DataFrame({"OMS_SKU": ["A"], "PO_Pipeline_Total": [1.0]})
    assert existing_po_frame_is_newer_than(
        new,
        old,
        incoming_meta={"existing_po_generation": 3, "existing_po_uploaded_at": "2026-06-26T10:00:00"},
        existing_meta={"existing_po_generation": 2, "existing_po_uploaded_at": "2026-05-01T10:00:00"},
    )
    assert not existing_po_frame_is_newer_than(
        old,
        new,
        incoming_meta={"existing_po_generation": 2},
        existing_meta={"existing_po_generation": 3},
    )


def test_view_end_date_anchors_on_matrix_when_behind_today():
    df = _hist("A", "2026-05-30", 3)
    assert inventory_history_view_end_date(df) == "2026-05-30"


def test_overlay_updates_total_inventory_only():
    hist = _hist("SKU-A", "2026-06-26", 5)
    inv = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "OMS_Inventory": [10.0], "Total_Inventory": [25.0]},
    )
    out, meta = overlay_inventory_variant_from_history(
        inv,
        hist,
        snapshot_date="",
        reference_date="2026-06-26",
    )
    assert meta["applied"] is True
    assert float(out.loc[0, "Total_Inventory"]) == 5.0
    assert float(out.loc[0, "OMS_Inventory"]) == 10.0


def test_authoritative_cap_date_never_exceeds_upload_end():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import inventory_history_authoritative_cap_date
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-26", 5)
    sess.inventory_snapshot_date = "2026-06-26"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: None
    try:
        cap = inventory_history_authoritative_cap_date(sess)
        assert str(cap.date()) == "2026-06-26"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_authoritative_cap_includes_daily_snapshot_beyond_wide_matrix():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-27", 5)
    sess.inventory_snapshot_date = "2026-06-27"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-06-26",
        "daily_inventory_history_max_date": "2026-06-26",
        "daily_inventory_history_rows": 5000,
        "daily_inventory_history_skus": 2000,
    }
    try:
        cap = dih.inventory_history_authoritative_cap_date(sess)
        assert str(cap.date()) == "2026-06-27"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_snapshot_extends_history_beyond_wide_matrix_end():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import (
        refresh_inventory_history_rollforward,
    )

    sess = AppSession()
    hist = _hist("SKU-A", "2026-06-25", 5)
    sess.daily_inventory_history_df = hist
    sess.daily_inventory_history_matrix_max_date = "2026-06-25"
    sess.inventory_snapshot_date = "2026-06-27"
    sess.inventory_df_variant = __import__("pandas").DataFrame(
        {"OMS_SKU": ["SKU-A"], "OMS_Inventory": [42.0], "Amazon_Inventory": [0.0]}
    )
    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert result.get("ok") is True
    assert result.get("snapshot_appended") is True
    out = sess.daily_inventory_history_df
    max_d = str(__import__("pandas").to_datetime(out["Date"]).max().date())
    assert max_d == "2026-06-27"
    snap_row = out[
        (__import__("pandas").to_datetime(out["Date"]).dt.normalize() == __import__("pandas").Timestamp("2026-06-27"))
        & (out["OMS_SKU"].astype(str) == "SKU-A")
    ]
    assert not snap_row.empty
    assert float(snap_row.iloc[0]["Qty"]) == 42.0


def test_wide_matrix_includes_date_totals():
    from backend.services.daily_inventory_history import inventory_history_wide_matrix

    df = __import__("pandas").DataFrame(
        {
            "OMS_SKU": ["A", "A", "B", "B"],
            "Date": ["2026-06-24", "2026-06-25", "2026-06-24", "2026-06-25"],
            "Qty": [10.0, 20.0, 5.0, 15.0],
        }
    )
    wide = inventory_history_wide_matrix(df, days=30)
    assert wide["date_totals"] == [15.0, 35.0]


def test_meta_bundle_max_date_follows_dataframe_not_upload_cap():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-05-30", 10)
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-06-27",
        "daily_inventory_history_rows": 5000,
        "daily_inventory_history_skus": 2000,
    }
    try:
        meta = dih.daily_inventory_history_meta_bundle(sess)
        assert meta["daily_inventory_history_max_date"] == "2026-05-30"
        assert meta["daily_inventory_history_matrix_max_date"] == "2026-06-27"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_inventory_sheet_end_date_from_filename():
    from backend.services.daily_inventory_history import inventory_sheet_end_date_from_filename

    assert inventory_sheet_end_date_from_filename("Daily Inventory History 1-May To 25-Jun-26.xlsx") == "2026-06-25"
    assert inventory_sheet_end_date_from_filename("history through 30-apr-2026.xlsx") == "2026-04-30"
    assert inventory_sheet_end_date_from_filename("no-date.xlsx") == ""


def test_matrix_cap_uses_filename_when_meta_stale():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-05-30",
        "daily_inventory_history_max_date": "2026-05-30",
        "daily_inventory_history_filename": "Daily Inventory History 1-May To 25-Jun-26.xlsx",
        "daily_inventory_history_rows": 280000,
        "daily_inventory_history_skus": 9000,
    }
    try:
        cap = dih.inventory_history_matrix_cap_date(sess)
        assert str(cap.date()) == "2026-06-25"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_matrix_cap_prefers_session_over_stale_disk_placeholder():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-05-30",
        "daily_inventory_history_max_date": "2026-05-30",
        "daily_inventory_history_rows": 1,
        "daily_inventory_history_skus": 1,
    }
    try:
        cap = dih.inventory_history_matrix_cap_date(sess)
        assert str(cap.date()) == "2026-06-27"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_matrix_cap_infers_from_session_df_without_matrix_max_date():
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    sess = AppSession()
    dates = pd.date_range("2026-06-01", "2026-06-27", freq="D")
    rows = []
    for i in range(60):
        sku = f"BULK-SKU-{i}"
        for d in dates:
            rows.append({"OMS_SKU": sku, "Date": d, "Qty": 5})
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    orig = dih.read_daily_inventory_history_disk_meta
    dih.read_daily_inventory_history_disk_meta = lambda: {
        "daily_inventory_history_matrix_max_date": "2026-05-30",
        "daily_inventory_history_rows": 1,
        "daily_inventory_history_skus": 1,
    }
    try:
        cap = dih.inventory_history_matrix_cap_date(sess)
        assert str(cap.date()) == "2026-06-27"
    finally:
        dih.read_daily_inventory_history_disk_meta = orig


def test_reconcile_disk_meta_overwrites_placeholder(monkeypatch, tmp_path):
    from backend.session import AppSession
    import backend.services.daily_inventory_history as dih

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    (tmp_path / "daily_inventory_history_meta.json").write_text(
        '{"daily_inventory_history_rows":1,"daily_inventory_history_skus":1,'
        '"daily_inventory_history_matrix_max_date":"2026-05-30"}',
        encoding="utf-8",
    )
    sess = AppSession()
    dates = pd.date_range("2026-06-01", "2026-06-27", freq="D")
    rows = []
    for i in range(80):
        for d in dates:
            rows.append({"OMS_SKU": f"SKU-{i}", "Date": d, "Qty": 3})
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    sess.daily_inventory_history_uploaded_at = "2026-06-28T10:00:00"
    assert dih.reconcile_daily_inventory_meta_if_session_newer(sess) is True
    meta = dih.read_daily_inventory_history_disk_meta()
    assert int(meta["daily_inventory_history_skus"]) == 80
    assert meta["daily_inventory_history_matrix_max_date"] == "2026-06-27"


def test_authoritative_read_keeps_full_matrix_when_disk_meta_stale(monkeypatch):
    from backend.session import AppSession
    from backend.services.po_session_hydrate import ensure_inventory_history_authoritative_for_read

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("1488YKWHITE-XS-S", "2026-06-27", 30)
    sess.daily_inventory_history_matrix_max_date = "2026-06-27"
    sess.sku_mapping = {"1488YKWHITE-XS-S": "1488YKWHITE-XS-S"}

    monkeypatch.setattr(
        "backend.services.po_session_hydrate.ensure_po_sidecars_hydrated",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.read_daily_inventory_history_disk_meta",
        lambda: {
            "daily_inventory_history_matrix_max_date": "2026-05-30",
            "daily_inventory_history_max_date": "2026-05-30",
            "daily_inventory_history_rows": 1,
            "daily_inventory_history_skus": 1,
        },
    )

    out = ensure_inventory_history_authoritative_for_read(sess)
    sub = out[out["OMS_SKU"] == "1488YKWHITE-XS-S"]
    assert len(sub) == 30
    assert str(sub["Date"].max().date()) == "2026-06-27"


def test_snapshot_append_does_not_inflate_sku_count():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("MATRIX-SKU", "2026-06-26", 10)
    sess.daily_inventory_history_matrix_max_date = "2026-06-26"
    sess.inventory_snapshot_date = "2026-06-26"
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": [f"EXTRA-{i}" for i in range(500)] + ["MATRIX-SKU"],
            "Total_Inventory": [1] * 500 + [12],
        }
    )
    out = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert out.get("ok") is True
    skus = sess.daily_inventory_history_df["OMS_SKU"].astype(str).nunique()
    assert skus == 1


def test_ensure_inventory_history_trims_beyond_cap(monkeypatch):
    from backend.session import AppSession
    from backend.services.po_session_hydrate import ensure_inventory_history_authoritative_for_read

    sess = AppSession()
    sess.daily_inventory_history_df = _hist("A", "2026-06-27", 3)
    sess.inventory_snapshot_date = "2026-06-26"

    monkeypatch.setattr(
        "backend.services.po_session_hydrate.ensure_po_sidecars_hydrated",
        lambda _s: None,
    )
    monkeypatch.setattr(
        "backend.services.daily_inventory_history.read_daily_inventory_history_disk_meta",
        lambda: {
            "daily_inventory_history_max_date": "2026-06-26",
            "daily_inventory_history_matrix_max_date": "2026-06-26",
            "daily_inventory_history_rows": 5000,
            "daily_inventory_history_skus": 2000,
        },
    )

    out = ensure_inventory_history_authoritative_for_read(sess)
    mx = out["Date"].max().normalize()
    assert str(mx.date()) == "2026-06-26"


def test_merge_prefers_snapshot_over_derived():
    from backend.services.daily_inventory_history import merge_inventory_history

    existing = pd.DataFrame(
        {
            "OMS_SKU": ["A"],
            "Date": [pd.Timestamp("2026-06-26")],
            "Qty": [62.0],
            "Source": "derived",
        }
    )
    incoming = pd.DataFrame(
        {
            "OMS_SKU": ["A"],
            "Date": [pd.Timestamp("2026-06-26")],
            "Qty": [38.0],
            "Source": "snapshot",
        }
    )
    out = merge_inventory_history(existing, incoming)
    assert len(out) == 1
    assert float(out.iloc[0]["Qty"]) == 38.0
    assert out.iloc[0]["Source"] == "snapshot"


def test_snapshot_upload_prunes_derived_gap_and_skips_rollforward():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import (
        merge_inventory_history,
        refresh_inventory_history_rollforward,
    )

    sess = AppSession()
    dates = pd.date_range("2026-06-20", "2026-06-25", freq="D")
    hist = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"] * len(dates),
            "Date": dates,
            "Qty": [10.0, 20, 30, 40, 50, 60],
            "Source": "uploaded",
        }
    )
    filler = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"] * 3,
            "Date": pd.date_range("2026-06-26", "2026-06-28", freq="D"),
            "Qty": [62.0, 62.0, 62.0],
            "Source": "derived",
        }
    )
    sess.daily_inventory_history_df = merge_inventory_history(hist, filler)
    sess.daily_inventory_history_filename = "Inventory through 25 Jun 2026.xlsx"
    sess.daily_inventory_history_wide_end_date = "2026-06-25"
    sess.inventory_snapshot_date = "2026-06-29"
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "Total_Inventory": [99]},
    )

    out = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert out.get("ok") is True

    sub = sess.daily_inventory_history_df
    sub = sub[sub["OMS_SKU"].astype(str) == "SKU-A"]
    day_qty = {
        str(pd.Timestamp(d).date()): float(q)
        for d, q in zip(sub["Date"], sub["Qty"])
    }
    assert day_qty.get("2026-06-26") != 62.0
    assert day_qty.get("2026-06-27") != 62.0
    assert day_qty.get("2026-06-28") != 62.0
    assert day_qty.get("2026-06-29") == 99.0
    assert "2026-06-29" in (sess.daily_inventory_history_snapshot_dates or [])


def test_inventory_history_is_newer_prefers_later_max_not_row_count():
    may = _hist("A", "2026-05-30", 40)
    june = _hist("A", "2026-06-29", 10)
    assert inventory_history_is_newer_than(june, may)
    assert not inventory_history_is_newer_than(may, june)


def test_reconcile_disk_integrity_repairs_inflated_meta(tmp_path, monkeypatch):
    import json

    from backend.services import daily_inventory_history as dih

    hist = _hist("SKU", "2026-05-30", 5)
    cache = tmp_path / "warm_cache"
    cache.mkdir()
    hist.to_parquet(cache / "daily_inventory_history_df.parquet", index=False)
    (cache / "daily_inventory_history_meta.json").write_text(
        json.dumps(
            {
                "daily_inventory_history_max_date": "2026-06-29",
                "daily_inventory_history_rows": 999999,
                "daily_inventory_history_skus": 9999,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dih, "_warm_cache_dir", lambda: cache)
    out = dih.reconcile_inventory_history_disk_integrity(repair=True)
    assert out.get("repaired") is True
    meta = json.loads((cache / "daily_inventory_history_meta.json").read_text())
    assert meta["daily_inventory_history_max_date"] == "2026-05-30"
    assert meta["daily_inventory_history_rows"] == 5


def test_disk_reconcile_rollforward_preserves_pre_window_days(tmp_path, monkeypatch):
    """Appending a June snapshot must not drop May rows during disk reconcile."""
    from backend.session import AppSession
    from backend.services import daily_inventory_history as dih

    cache = tmp_path / "warm_cache"
    cache.mkdir()
    may = _hist("SKU-A", "2026-05-30", 30)
    may["Source"] = "uploaded"
    may.to_parquet(cache / "daily_inventory_history_df.parquet", index=False)

    monkeypatch.setattr(dih, "_warm_cache_dir", lambda: cache)

    sess = AppSession()
    sess.daily_inventory_history_df = may.copy()
    sess.inventory_snapshot_date = "2026-06-29"
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "Total_Inventory": [77.0]},
    )

    out = dih.refresh_inventory_history_rollforward(
        sess,
        include_snapshot=True,
        max_history_days=dih.INVENTORY_HISTORY_DISK_RECONCILE_DAYS,
    )
    assert out.get("ok") is True
    days = sorted(
        pd.to_datetime(sess.daily_inventory_history_df["Date"], errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .unique()
    )
    assert days[0] == "2026-05-01"
    assert days[-1] == "2026-06-29"
    assert len(days) == 31


def test_inventory_history_view_uses_data_days_not_calendar_gap():
    """May matrix + June snapshot must show ~30 May columns, not one June day."""
    from backend.services.daily_inventory_history import (
        filter_inventory_history_view,
        inventory_history_wide_matrix,
    )

    may = _hist("SKU-A", "2026-05-30", 30)
    may["Source"] = "uploaded"
    june = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Date": pd.Timestamp("2026-06-29"),
            "Qty": [99.0],
            "Source": "snapshot",
        }
    )
    df = pd.concat([may, june], ignore_index=True)

    view = filter_inventory_history_view(df, days=30)
    days = sorted(pd.to_datetime(view["Date"]).dt.strftime("%Y-%m-%d").unique())
    assert len(days) == 30
    assert days[0] == "2026-05-02"
    assert days[-1] == "2026-06-29"

    wide = inventory_history_wide_matrix(df, days=30, limit=10)
    assert len(wide["dates"]) == 30
    assert wide["dates"][0] == "2026-05-02"
    assert wide["dates"][-1] == "2026-06-29"


def test_reconcile_restores_parquet_from_pipeline_when_behind_wide_end(tmp_path, monkeypatch):
    import json

    from backend.services import daily_inventory_history as dih

    cache = tmp_path / "warm_cache"
    pipeline = cache / "pipeline"
    pipeline.mkdir(parents=True)
    june = _hist("SKU-A", "2026-06-25", 20)
    june["Source"] = "uploaded"
    june.to_parquet(pipeline / "inventory_history_snapshot.parquet", index=False)

    may = _hist("SKU-A", "2026-05-30", 20)
    may["Source"] = "uploaded"
    may.to_parquet(cache / "daily_inventory_history_df.parquet", index=False)
    (cache / "daily_inventory_history_meta.json").write_text(
        json.dumps(
            {
                "daily_inventory_history_wide_end_date": "2026-06-25",
                "daily_inventory_history_matrix_max_date": "2026-06-29",
                "daily_inventory_history_max_date": "2026-06-29",
                "daily_inventory_history_rows": 200000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dih, "_warm_cache_dir", lambda: cache)
    out = dih.reconcile_inventory_history_disk_integrity(repair=True)
    assert out.get("repaired") is True
    assert any("restored" in a for a in out.get("actions", []))
    df = pd.read_parquet(cache / "daily_inventory_history_df.parquet")
    assert str(pd.to_datetime(df["Date"]).max().date()) == "2026-06-25"


def test_refresh_fills_sales_gap_between_wide_end_and_snapshot():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward

    sess = AppSession()
    hist = _hist("SKU-A", "2026-06-25", 5)
    hist["Source"] = "uploaded"
    sess.daily_inventory_history_df = hist
    sess.daily_inventory_history_wide_end_date = "2026-06-25"
    sess.inventory_snapshot_date = "2026-06-29"
    sess.inventory_df_variant = pd.DataFrame(
        {"OMS_SKU": ["SKU-A"], "Total_Inventory": [42.0]},
    )
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"] * 3,
            "TxnDate": pd.date_range("2026-06-26", "2026-06-28", freq="D"),
            "Units_Effective": [1.0, 1.0, 1.0],
        }
    )
    sess.sales_df = sales
    out = refresh_inventory_history_rollforward(sess, include_snapshot=True, sales_df=sales)
    assert out.get("ok") is True
    days = sorted(pd.to_datetime(sess.daily_inventory_history_df["Date"]).dt.strftime("%Y-%m-%d").unique())
    assert "2026-06-26" in days
    assert "2026-06-27" in days
    assert "2026-06-28" in days
    assert days[-1] == "2026-06-29"


def test_refresh_fills_gap_when_snapshot_already_present():
    from backend.session import AppSession
    from backend.services.daily_inventory_history import (
        merge_inventory_history,
        refresh_inventory_history_rollforward,
    )

    sess = AppSession()
    hist = _hist("SKU-A", "2026-06-25", 3)
    hist["Source"] = "uploaded"
    snap = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A"],
            "Date": pd.Timestamp("2026-06-29"),
            "Qty": [99.0],
            "Source": "snapshot",
        }
    )
    sess.daily_inventory_history_df = merge_inventory_history(hist, snap)
    sess.daily_inventory_history_wide_end_date = "2026-06-25"
    sess.inventory_snapshot_date = "2026-06-29"
    sess.inventory_df_variant = pd.DataFrame({"OMS_SKU": ["SKU-A"], "Total_Inventory": [99.0]})
    sales = pd.DataFrame(
        {
            "Sku": ["SKU-A"] * 3,
            "TxnDate": pd.date_range("2026-06-26", "2026-06-28", freq="D"),
            "Units_Effective": [1.0, 1.0, 1.0],
        }
    )
    out = refresh_inventory_history_rollforward(sess, include_snapshot=True, sales_df=sales)
    assert out.get("ok") is True
    days = sorted(pd.to_datetime(sess.daily_inventory_history_df["Date"]).dt.strftime("%Y-%m-%d").unique())
    assert all(d in days for d in ["2026-06-26", "2026-06-27", "2026-06-28", "2026-06-29"])


def test_restore_inventory_history_merges_github_backup(tmp_path, monkeypatch):
    from backend.services import daily_inventory_history as dih

    cache = tmp_path / "warm_cache"
    gh = tmp_path / "github_cache" / "2026-05-29T07-15-45"
    cache.mkdir(parents=True)
    gh.mkdir(parents=True)

    may = _hist("SKU-A", "2026-05-30", 20)
    may["Source"] = "uploaded"
    may.to_parquet(gh / "daily_inventory_history_df.parquet", index=False)

    june = _hist("SKU-A", "2026-06-29", 1)
    june["Source"] = "snapshot"
    june.to_parquet(cache / "daily_inventory_history_df.parquet", index=False)

    monkeypatch.setattr(dih, "_warm_cache_dir", lambda: cache)

    restored = dih.restore_inventory_history_from_best_disk_backups(june)
    assert restored is not None
    days = sorted(
        pd.to_datetime(restored["Date"], errors="coerce").dt.strftime("%Y-%m-%d").unique()
    )
    assert days[0] == "2026-05-11"
    assert days[-1] == "2026-06-29"
    assert len(days) == 21
