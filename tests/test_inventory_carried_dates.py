"""Days without a snapshot/uploaded file must be flagged as carried in Inv History."""
import pandas as pd

from backend.services.daily_inventory_history import (
    extend_history_gaps_with_sales,
    extend_history_with_sales,
    inventory_history_wide_matrix,
    merge_inventory_history_preserving_channels,
    non_uploaded_inventory_dates,
)


def test_amazon_matrix_uses_amazon_source_not_oms_for_gaps():
    """Amazon FBA uploaded/carried must follow amazon-channel Source, not OMS."""
    rows = []
    for d, oms_src, amz_src, amz_qty in [
        ("2026-08-10", "snapshot", "snapshot", 100),
        ("2026-08-11", "snapshot", "derived", 95),  # OMS uploaded; Amazon carried
        ("2026-08-12", "snapshot", "snapshot", 0),  # Amazon authentic zero census
    ]:
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": 200,
                "Source": oms_src,
                "Channel": "oms",
            }
        )
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": amz_qty,
                "Source": amz_src,
                "Channel": "amazon",
            }
        )
    df = pd.DataFrame(rows)
    wide = inventory_history_wide_matrix(df, days=3, end_date="2026-08-12", channel="amazon")
    assert "2026-08-11" in (wide.get("gap_dates") or [])
    assert "2026-08-10" not in (wide.get("gap_dates") or [])
    assert "2026-08-12" not in (wide.get("gap_dates") or []), wide
    assert wide["date_totals"] == [100.0, 95.0, 0.0]


def test_overlay_restores_amazon_when_oms_snapshot_already_present(tmp_path, monkeypatch):
    from backend.services import daily_inventory_history as dih
    from backend.session import AppSession

    monkeypatch.setattr(dih, "_warm_cache_dir", lambda: tmp_path)
    snap_dir = tmp_path / "inventory_day_snapshots"
    snap_dir.mkdir(parents=True)
    variant = pd.DataFrame(
        {
            "OMS_SKU": ["SKU-A", "SKU-B"],
            "OMS_Inventory": [10.0, 5.0],
            "Amazon_Inventory": [7.0, 3.0],
            "Total_Inventory": [17.0, 8.0],
        }
    )
    variant.to_parquet(snap_dir / "2026-08-24.parquet", index=False)

    # History already has OMS snapshot for the day, but no amazon channel rows.
    hist = pd.DataFrame(
        [
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp("2026-08-24"),
                "Qty": 10.0,
                "Source": "snapshot",
                "Channel": "oms",
            },
            {
                "OMS_SKU": "SKU-B",
                "Date": pd.Timestamp("2026-08-24"),
                "Qty": 5.0,
                "Source": "snapshot",
                "Channel": "oms",
            },
        ]
    )
    sess = AppSession()
    sess.daily_inventory_history_df = hist
    restored = dih.overlay_persisted_inventory_snapshots(sess)
    assert restored >= 1
    out = sess.daily_inventory_history_df
    amz = out[out["Channel"].astype(str).str.lower() == "amazon"]
    assert not amz.empty
    assert float(amz["Qty"].sum()) == 10.0
    wide = inventory_history_wide_matrix(out, days=1, end_date="2026-08-24", channel="amazon")
    assert "2026-08-24" in (wide.get("uploaded_dates") or [])
    assert "2026-08-24" not in (wide.get("gap_dates") or [])


def test_uploaded_days_not_carried_even_when_snapshots_exist():
    """Wide-matrix ``uploaded`` days must not show as CARRIED once RAR snapshots exist."""
    rows = []
    for d, src, qty in [
        ("2026-06-26", "uploaded", 26),
        ("2026-06-27", "uploaded", 22),
        ("2026-06-28", "uploaded", 12),
        ("2026-06-29", "derived", 12),
        ("2026-07-01", "snapshot", 8),
    ]:
        rows.append(
            {
                "OMS_SKU": "165YK251MUSTRAD-XXL",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms" if src == "snapshot" else "",
            }
        )
    df = pd.DataFrame(rows)
    dates = list(pd.date_range("2026-06-26", "2026-07-01"))
    carried = non_uploaded_inventory_dates(df, dates)
    assert "2026-06-29" in carried
    assert "2026-06-26" not in carried
    assert "2026-06-27" not in carried
    assert "2026-06-28" not in carried
    assert "2026-07-01" not in carried


def test_matrix_marks_derived_only_days_as_gap_dates():
    rows = []
    for d, src, qty in [
        ("2026-07-04", "snapshot", 100),
        ("2026-07-05", "derived", 95),
        ("2026-07-06", "snapshot", 90),
    ]:
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms",
            }
        )
        rows.append(
            {
                "OMS_SKU": "SKU-A",
                "Date": pd.Timestamp(d),
                "Qty": 0,
                "Source": src if src == "snapshot" else "derived",
                "Channel": "amazon",
            }
        )
    df = pd.DataFrame(rows)
    wide = inventory_history_wide_matrix(df, days=3, end_date="2026-07-06", channel="combined")
    assert "2026-07-05" in (wide.get("gap_dates") or [])
    assert "2026-07-04" not in (wide.get("gap_dates") or [])
    assert "2026-07-06" not in (wide.get("gap_dates") or [])
    assert "2026-07-05" not in (wide.get("uploaded_dates") or [])


def test_snapshot_rollforward_preserves_uploaded_matrix_days():
    """Daily RAR snapshot must not wipe prior wide-matrix ``uploaded`` census days."""
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward
    from backend.session import AppSession

    rows = []
    for d, qty in [
        ("2026-07-01", 100),
        ("2026-07-02", 98),
        ("2026-07-03", 95),
        ("2026-07-04", 90),
        ("2026-07-05", 88),
    ]:
        rows.append(
            {
                "OMS_SKU": "S1",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": "uploaded",
                "Channel": "oms",
            }
        )
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    sess.inventory_snapshot_date = "2026-07-06"
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["S1"],
            "OMS_Inventory": [80],
            "Total_Inventory": [80],
            "Amazon_Inventory": [0],
        }
    )
    sess.sales_df = pd.DataFrame(
        columns=["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source"]
    )
    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert result.get("ok")
    out = sess.daily_inventory_history_df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    oms = out[out["Channel"].astype(str).str.lower().eq("oms")].copy()
    oms["d"] = oms["Date"].dt.strftime("%Y-%m-%d")
    src = oms.groupby("d")["Source"].first().astype(str).str.lower()
    for d in ["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-04", "2026-07-05"]:
        assert src.get(d) == "uploaded", (d, src.to_dict())
    assert "2026-07-06" in set(oms["d"])
    assert (oms.loc[oms["d"] == "2026-07-06", "Source"].astype(str).str.lower() == "snapshot").any()


def test_oms_overlay_preserves_amazon_census():
    existing = pd.DataFrame(
        {
            "OMS_SKU": ["A", "A", "A", "A"],
            "Date": pd.to_datetime(
                ["2026-06-28", "2026-06-28", "2026-06-29", "2026-06-29"]
            ),
            "Qty": [10.0, 40.0, 9.0, 40.0],
            "Source": ["derived", "snapshot", "derived", "snapshot"],
            "Channel": ["oms", "amazon", "oms", "amazon"],
        }
    )
    incoming = pd.DataFrame(
        {
            "OMS_SKU": ["A", "A"],
            "Date": pd.to_datetime(["2026-06-28", "2026-06-29"]),
            "Qty": [12.0, 3.0],
            "Source": ["uploaded", "uploaded"],
            "Channel": ["", ""],
        }
    )
    out = merge_inventory_history_preserving_channels(existing, incoming)
    sub = out[out["OMS_SKU"] == "A"].copy()
    sub["Date"] = pd.to_datetime(sub["Date"]).dt.normalize()
    jun28 = sub[sub["Date"] == pd.Timestamp("2026-06-28")]
    jun29 = sub[sub["Date"] == pd.Timestamp("2026-06-29")]
    assert float(jun28.loc[jun28["Channel"] == "oms", "Qty"].iloc[0]) == 12.0
    assert float(jun28.loc[jun28["Channel"] == "amazon", "Qty"].iloc[0]) == 40.0
    assert float(jun29.loc[jun29["Channel"] == "oms", "Qty"].iloc[0]) == 3.0
    assert float(jun29.loc[jun29["Channel"] == "amazon", "Qty"].iloc[0]) == 40.0
    assert str(jun29.loc[jun29["Channel"] == "oms", "Source"].iloc[0]) == "snapshot"


def test_empty_snapshot_date_does_not_overwrite_last_uploaded_day(monkeypatch):
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward
    from backend.session import AppSession

    monkeypatch.setattr(
        "backend.services.daily_inventory_history.today_ist_timestamp",
        lambda: pd.Timestamp("2026-07-06"),
    )
    rows = []
    for d, qty in [("2026-07-04", 100), ("2026-07-05", 90)]:
        rows.append(
            {
                "OMS_SKU": "S1",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": "uploaded",
                "Channel": "oms",
            }
        )
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    sess.inventory_snapshot_date = ""
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["S1"],
            "OMS_Inventory": [80],
            "Total_Inventory": [80],
            "Amazon_Inventory": [0],
        }
    )
    sess.sales_df = pd.DataFrame(
        columns=["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source"]
    )
    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert result.get("ok")
    out = sess.daily_inventory_history_df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    oms = out[out["Channel"].astype(str).str.lower().eq("oms")].copy()
    oms["d"] = oms["Date"].dt.strftime("%Y-%m-%d")
    jul5 = oms.loc[oms["d"] == "2026-07-05"]
    assert float(jul5["Qty"].iloc[0]) == 90.0
    assert str(jul5["Source"].iloc[0]).lower() == "uploaded"
    jul6 = oms.loc[oms["d"] == "2026-07-06"]
    assert float(jul6["Qty"].iloc[0]) == 80.0
    assert str(jul6["Source"].iloc[0]).lower() == "snapshot"


def test_overlay_restores_carried_days_from_persisted_snapshots():
    from backend.services.daily_inventory_history import (
        overlay_persisted_inventory_snapshots,
        non_uploaded_inventory_dates,
    )
    from backend.session import AppSession

    rows = []
    for d, src, qty in [
        ("2026-08-08", "snapshot", 1382),
        ("2026-08-09", "derived", 1382),
        ("2026-08-10", "derived", 1382),
        ("2026-08-13", "snapshot", 1337),
    ]:
        rows.append(
            {
                "OMS_SKU": "DPT21MULTI-M",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms",
            }
        )
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    n = overlay_persisted_inventory_snapshots(
        sess,
        snapshots=[
            {
                "snapshot_date": "2026-08-08",
                "uploaded_at": "2026-08-08T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["DPT21MULTI-M"], "OMS_Inventory": [1382]}),
            },
            {
                "snapshot_date": "2026-08-08",
                "uploaded_at": "2026-08-09T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["DPT21MULTI-M"], "OMS_Inventory": [1370]}),
            },
            {
                "snapshot_date": "2026-08-08",
                "uploaded_at": "2026-08-10T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["DPT21MULTI-M"], "OMS_Inventory": [1360]}),
            },
            {
                "snapshot_date": "2026-08-13",
                "uploaded_at": "2026-08-13T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["DPT21MULTI-M"], "OMS_Inventory": [1337]}),
            },
        ],
    )
    assert n == 2
    out = sess.daily_inventory_history_df.copy()
    out["d"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    by = out.set_index("d")
    assert str(by.loc["2026-08-09", "Source"]).lower() == "snapshot"
    assert float(by.loc["2026-08-09", "Qty"]) == 1370
    assert str(by.loc["2026-08-10", "Source"]).lower() == "snapshot"
    assert float(by.loc["2026-08-10", "Qty"]) == 1360
    carried = non_uploaded_inventory_dates(
        out, list(pd.date_range("2026-08-08", "2026-08-13"))
    )
    assert "2026-08-09" not in carried
    assert "2026-08-10" not in carried
    assert "2026-08-11" in carried
    assert "2026-08-12" in carried


def test_mid_window_snapshot_does_not_drop_later_census():
    from backend.services.daily_inventory_history import refresh_inventory_history_rollforward
    from backend.session import AppSession

    rows = []
    for d, src, qty in [
        ("2026-08-08", "snapshot", 100),
        ("2026-08-09", "derived", 100),
        ("2026-08-13", "snapshot", 80),
    ]:
        rows.append(
            {
                "OMS_SKU": "S1",
                "Date": pd.Timestamp(d),
                "Qty": qty,
                "Source": src,
                "Channel": "oms",
            }
        )
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(rows)
    sess.inventory_snapshot_date = "2026-08-11"
    sess.inventory_df_variant = pd.DataFrame(
        {
            "OMS_SKU": ["S1"],
            "OMS_Inventory": [90],
            "Total_Inventory": [90],
            "Amazon_Inventory": [0],
        }
    )
    sess.sales_df = pd.DataFrame(
        columns=["Sku", "TxnDate", "Transaction Type", "Quantity", "Units_Effective", "Source"]
    )
    result = refresh_inventory_history_rollforward(sess, include_snapshot=True)
    assert result.get("ok")
    out = sess.daily_inventory_history_df.copy()
    out["d"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    by = out.groupby("d", as_index=True).agg(Qty=("Qty", "sum"), Source=("Source", "first"))
    assert str(by.loc["2026-08-11", "Source"]).lower() == "snapshot"
    assert float(by.loc["2026-08-11", "Qty"]) == 90
    assert "2026-08-13" in by.index
    assert str(by.loc["2026-08-13", "Source"]).lower() == "snapshot"
    assert float(by.loc["2026-08-13", "Qty"]) == 80


def test_overlay_from_disk_day_archive(tmp_path, monkeypatch):
    from backend.services import daily_inventory_history as dih
    from backend.session import AppSession

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        [
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-08"),
                "Qty": 10,
                "Source": "snapshot",
                "Channel": "oms",
            },
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-09"),
                "Qty": 10,
                "Source": "derived",
                "Channel": "oms",
            },
        ]
    )
    variant = pd.DataFrame({"OMS_SKU": ["S1-M"], "OMS_Inventory": [7]})
    path = dih.archive_inventory_day_snapshot(variant, "2026-08-09")
    assert path
    n = dih.overlay_persisted_inventory_snapshots(sess)
    assert n == 1
    out = sess.daily_inventory_history_df
    row = out[pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d") == "2026-08-09"].iloc[0]
    assert str(row["Source"]).lower() == "snapshot"
    assert float(row["Qty"]) == 7.0


def test_overlay_day_archives_once_does_not_merge_backups(tmp_path, monkeypatch):
    from backend.services import daily_inventory_history as dih
    from backend.session import AppSession

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    dih._MATRIX_DAY_OVERLAY_MTIME = None
    called = {"restore": 0}

    def _boom(*_a, **_k):
        called["restore"] += 1
        raise AssertionError("bak merge must not run on matrix read overlay")

    monkeypatch.setattr(dih, "restore_inventory_history_from_best_disk_backups", _boom)
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        [
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-09"),
                "Qty": 10,
                "Source": "derived",
                "Channel": "oms",
            }
        ]
    )
    n = dih.overlay_day_archives_for_read_once(sess)
    assert n == 0
    assert called["restore"] == 0
    n2 = dih.overlay_day_archives_for_read_once(sess)
    assert n2 == 0


def test_overlay_day_archives_applies_gap_file_once(tmp_path, monkeypatch):
    from backend.services import daily_inventory_history as dih
    from backend.session import AppSession
    import backend.main as _main

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_main, "_warm_cache", {}, raising=False)
    dih._MATRIX_DAY_OVERLAY_MTIME = None
    hist_path = tmp_path / "daily_inventory_history_df.parquet"
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        [
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-08"),
                "Qty": 10,
                "Source": "snapshot",
                "Channel": "oms",
            },
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-09"),
                "Qty": 10,
                "Source": "derived",
                "Channel": "oms",
            },
        ]
    )
    sess.daily_inventory_history_df.to_parquet(hist_path, index=False)
    dih.archive_inventory_day_snapshot(
        pd.DataFrame({"OMS_SKU": ["S1-M"], "OMS_Inventory": [7]}),
        "2026-08-09",
    )
    n = dih.overlay_day_archives_for_read_once(sess)
    assert n == 1
    row = sess.daily_inventory_history_df
    day = row[pd.to_datetime(row["Date"]).dt.strftime("%Y-%m-%d") == "2026-08-09"].iloc[0]
    assert str(day["Source"]).lower() == "snapshot"
    assert float(day["Qty"]) == 7.0
    n2 = dih.overlay_day_archives_for_read_once(sess)
    assert n2 == 0
    disk = pd.read_parquet(hist_path)
    persisted = disk[pd.to_datetime(disk["Date"]).dt.strftime("%Y-%m-%d") == "2026-08-09"].iloc[0]
    assert str(persisted["Source"]).lower() == "snapshot"
    assert float(persisted["Qty"]) == 7.0


def test_overlay_day_archives_applies_snapshot_older_than_40_days(tmp_path, monkeypatch):
    """July day files must overlay even when history max is mid-August."""
    from backend.services import daily_inventory_history as dih
    from backend.session import AppSession
    import backend.main as _main

    monkeypatch.setenv("WARM_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(_main, "_warm_cache", {}, raising=False)
    dih._MATRIX_DAY_OVERLAY_MTIME = None
    hist_path = tmp_path / "daily_inventory_history_df.parquet"
    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        [
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-07-02"),
                "Qty": 10,
                "Source": "derived",
                "Channel": "oms",
            },
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-08-17"),
                "Qty": 8,
                "Source": "snapshot",
                "Channel": "oms",
            },
        ]
    )
    sess.daily_inventory_history_df.to_parquet(hist_path, index=False)
    dih.archive_inventory_day_snapshot(
        pd.DataFrame({"OMS_SKU": ["S1-M"], "OMS_Inventory": [22]}),
        "2026-07-02",
    )
    n = dih.overlay_day_archives_for_read_once(sess)
    assert n == 1
    row = sess.daily_inventory_history_df
    day = row[pd.to_datetime(row["Date"]).dt.strftime("%Y-%m-%d") == "2026-07-02"].iloc[0]
    assert str(day["Source"]).lower() == "snapshot"
    assert float(day["Qty"]) == 22.0
    disk = pd.read_parquet(hist_path)
    persisted = disk[pd.to_datetime(disk["Date"]).dt.strftime("%Y-%m-%d") == "2026-07-02"].iloc[0]
    assert str(persisted["Source"]).lower() == "snapshot"
    assert float(persisted["Qty"]) == 22.0


def test_overlay_extends_history_with_snapshots_after_hist_max():
    """July/August day files after a June-capped matrix must be merged in."""
    from backend.services.daily_inventory_history import overlay_persisted_inventory_snapshots
    from backend.session import AppSession

    sess = AppSession()
    sess.daily_inventory_history_df = pd.DataFrame(
        [
            {
                "OMS_SKU": "S1-M",
                "Date": pd.Timestamp("2026-06-30"),
                "Qty": 10,
                "Source": "uploaded",
                "Channel": "oms",
            }
        ]
    )
    n = overlay_persisted_inventory_snapshots(
        sess,
        snapshots=[
            {
                "snapshot_date": "2026-07-15",
                "uploaded_at": "2026-07-15T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["S1-M"], "OMS_Inventory": [22]}),
            },
            {
                "snapshot_date": "2026-08-10",
                "uploaded_at": "2026-08-10T06:00:00Z",
                "df": pd.DataFrame({"OMS_SKU": ["S1-M"], "OMS_Inventory": [18]}),
            },
        ],
    )
    assert n == 2
    out = sess.daily_inventory_history_df.copy()
    out["d"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    by = out.set_index("d")
    assert float(by.loc["2026-07-15", "Qty"]) == 22.0
    assert str(by.loc["2026-07-15", "Source"]).lower() == "snapshot"
    assert float(by.loc["2026-08-10", "Qty"]) == 18.0
    assert str(by.loc["2026-08-10", "Source"]).lower() == "snapshot"


def test_matrix_gap_days_use_sales_adjusted_qty():
    hist = pd.DataFrame(
        [
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-02"), "Qty": 10, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-06"), "Qty": 5, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-B", "Date": pd.Timestamp("2026-07-02"), "Qty": 5, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-B", "Date": pd.Timestamp("2026-07-06"), "Qty": 4, "Source": "snapshot", "Channel": "oms"},
        ]
    )
    sales = pd.DataFrame(
        [
            {"Sku": "SKU-A", "TxnDate": "2026-07-03 23:59:00", "Units_Effective": 2},
            {"Sku": "SKU-A", "TxnDate": "2026-07-05 00:01:00", "Units_Effective": 3},
            {"Sku": "SKU-B", "TxnDate": "2026-07-04 12:00:00", "Units_Effective": 1},
        ]
    )

    extended = extend_history_gaps_with_sales(hist, sales, cap_date=pd.Timestamp("2026-07-06"))
    out = inventory_history_wide_matrix(
        extended,
        days=5,
        end_date="2026-07-06",
        channel="oms",
    )

    assert out["dates"] == [
        "2026-07-02",
        "2026-07-03",
        "2026-07-04",
        "2026-07-05",
        "2026-07-06",
    ]
    by_sku = {row["sku"]: row["qtys"] for row in out["rows"]}
    assert by_sku["SKU-A"] == [10.0, 8.0, 8.0, 5.0, 5.0]
    assert by_sku["SKU-B"] == [5.0, 5.0, 4.0, 4.0, 4.0]
    assert out["date_totals"] == [15.0, 13.0, 12.0, 9.0, 9.0]
    assert out["gap_dates"] == ["2026-07-03", "2026-07-04", "2026-07-05"]


def test_matrix_no_duplicate_carry_forward():
    hist = pd.DataFrame(
        [
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-02"), "Qty": 10, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-06"), "Qty": 3, "Source": "snapshot", "Channel": "oms"},
        ]
    )
    sales = pd.DataFrame(
        [
            {"Sku": "SKU-A", "TxnDate": "2026-07-03", "Units_Effective": 2},
            {"Sku": "SKU-A", "TxnDate": "2026-07-04", "Units_Effective": 1},
            {"Sku": "SKU-A", "TxnDate": "2026-07-06", "Units_Effective": 4},
        ]
    )

    extended = extend_history_gaps_with_sales(hist, sales, cap_date=pd.Timestamp("2026-07-06"))
    out = inventory_history_wide_matrix(
        extended,
        days=5,
        end_date="2026-07-06",
        channel="oms",
    )
    qtys = {row["sku"]: row["qtys"] for row in out["rows"]}["SKU-A"]

    assert qtys == [10.0, 8.0, 7.0, 7.0, 3.0]
    assert qtys[1] == qtys[0] - 2
    assert qtys[2] == qtys[1] - 1
    assert qtys[3] == qtys[2]
    assert qtys[4] == qtys[3] - 4


def test_matrix_no_sales_keeps_flat_ffill():
    hist = pd.DataFrame(
        [
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-02"), "Qty": 10, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-06"), "Qty": 10, "Source": "snapshot", "Channel": "oms"},
        ]
    )
    out = inventory_history_wide_matrix(
        hist,
        days=5,
        end_date="2026-07-06",
        sales_df=pd.DataFrame(),
        channel="oms",
    )
    qtys = {row["sku"]: row["qtys"] for row in out["rows"]}["SKU-A"]
    assert qtys == [10.0, 10.0, 10.0, 10.0, 10.0]


def test_matrix_date_boundary_inclusion_with_sales():
    hist = pd.DataFrame(
        [
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-01"), "Qty": 20, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-08-16"), "Qty": 12, "Source": "snapshot", "Channel": "oms"},
        ]
    )
    sales = pd.DataFrame(
        [
            {"Sku": "SKU-A", "TxnDate": "2026-07-02 00:00:00", "Units_Effective": 5},
            {"Sku": "SKU-A", "TxnDate": "2026-08-16 23:59:59", "Units_Effective": 3},
            {"Sku": "SKU-A", "TxnDate": "2026-08-17 00:00:01", "Units_Effective": 7},
        ]
    )
    extended = extend_history_gaps_with_sales(hist, sales, cap_date=pd.Timestamp("2026-08-16"))
    out = inventory_history_wide_matrix(
        extended,
        days=46,
        end_date="2026-08-16",
        channel="oms",
    )
    qtys = {row["sku"]: row["qtys"] for row in out["rows"]}["SKU-A"]
    assert out["dates"][0] == "2026-07-02"
    assert out["dates"][-1] == "2026-08-16"
    assert qtys[0] == 15.0
    assert qtys[-1] == 12.0


def test_matrix_multiple_skus_independent_carry():
    hist = pd.DataFrame(
        [
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-02"), "Qty": 10, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-A", "Date": pd.Timestamp("2026-07-05"), "Qty": 6, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-B", "Date": pd.Timestamp("2026-07-02"), "Qty": 6, "Source": "snapshot", "Channel": "oms"},
            {"OMS_SKU": "SKU-B", "Date": pd.Timestamp("2026-07-05"), "Qty": 6, "Source": "snapshot", "Channel": "oms"},
        ]
    )
    sales = pd.DataFrame(
        [
            {"Sku": "SKU-A", "TxnDate": "2026-07-03", "Units_Effective": 2},
            {"Sku": "SKU-A", "TxnDate": "2026-07-04", "Units_Effective": 2},
        ]
    )
    extended = extend_history_gaps_with_sales(hist, sales, cap_date=pd.Timestamp("2026-07-05"))
    out = inventory_history_wide_matrix(
        extended,
        days=4,
        end_date="2026-07-05",
        channel="oms",
    )
    by_sku = {row["sku"]: row["qtys"] for row in out["rows"]}
    assert by_sku["SKU-A"] == [10.0, 8.0, 6.0, 6.0]
    assert by_sku["SKU-B"] == [6.0, 6.0, 6.0, 6.0]
