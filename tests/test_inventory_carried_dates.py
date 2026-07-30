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