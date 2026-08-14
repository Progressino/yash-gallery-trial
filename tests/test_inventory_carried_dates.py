"""Days without a snapshot/uploaded file must be flagged as carried in Inv History."""
import pandas as pd

from backend.services.daily_inventory_history import (
    inventory_history_wide_matrix,
    merge_inventory_history_preserving_channels,
    non_uploaded_inventory_dates,
)


def test_non_uploaded_dates_are_days_without_snapshot():
    rows = []
    for d, src, qty in [
        ("2026-07-04", "snapshot", 10),
        ("2026-07-05", "derived", 9),  # no file
        ("2026-07-06", "snapshot", 8),
        ("2026-07-07", "derived", 7),  # no file
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
    df = pd.DataFrame(rows)
    dates = list(pd.date_range("2026-07-04", "2026-07-07"))
    carried = non_uploaded_inventory_dates(df, dates)
    assert carried == ["2026-07-05", "2026-07-07"]


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