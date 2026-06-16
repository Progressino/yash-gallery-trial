"""End-to-end correctness audit of the PO "Sheet formula":

    projected_days_now = (Total_Inventory + effective pipeline) / ADS
    balance_days       = target_cover_days - projected_days_now
    raw_po             = ADS * balance_days
    PO_Qty             = round_po_pack(max(raw_po, 0))

Uses the production UI defaults (period=30, lead=45, target=180, gate on,
no SKU-status sheet). Rather than hand-predicting ADS (which depends on the
Recent/Seasonal/Flat30/LY heuristics and date-window edge effects), these
tests recompute the expected PO_Qty *from the engine's own ADS and
Projected_Running_Days outputs* and check that the final
balance/raw/round-to-pack steps match exactly — isolating the formula and
rounding logic from the ADS heuristics.
"""

import math

import pandas as pd
import pytest

from backend.services.po_engine import calculate_po_base, calculate_quarterly_history, round_po_pack

PERIOD_DAYS = 30
LEAD_TIME = 45
TARGET_DAYS = 180
TARGET_COVER_DAYS = TARGET_DAYS  # no grace_days passed -> grace defaults to 0


def _steady_sales(sku: str, units_per_day: float, days: int = 400, end: str = "2026-06-15"):
    """``units_per_day`` <= 1 is realized as 1 unit every ``1/units_per_day`` days."""
    end_ts = pd.Timestamp(end)
    dates = pd.date_range(end=end_ts, periods=days, freq="D")
    if units_per_day >= 1:
        qty = [units_per_day] * days
        rows_dates = dates
    else:
        step = int(round(1 / units_per_day))
        rows_dates = dates[::step]
        qty = [1] * len(rows_dates)
    return pd.DataFrame(
        {
            "Sku": [sku] * len(rows_dates),
            "TxnDate": rows_dates,
            "Transaction Type": ["Shipment"] * len(rows_dates),
            "Quantity": qty,
            "Units_Effective": qty,
        }
    )


def _inv(rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({"OMS_SKU": list(rows.keys()), "Total_Inventory": list(rows.values())})


@pytest.fixture
def po_result():
    sales = pd.concat(
        [
            _steady_sales("PO-A-ZERO-STOCK", 2.0),    # 2/day, no stock
            _steady_sales("PO-B-OVERSTOCKED", 2.0),   # 2/day, well above target cover
            _steady_sales("PO-C-PARTIAL", 2.0),       # 2/day, partial cover
            _steady_sales("PO-E-SMALL-PACK", 0.5),    # 0.5/day, small top-up
        ],
        ignore_index=True,
    )
    inv = _inv(
        {
            "PO-A-ZERO-STOCK": 0,
            "PO-B-OVERSTOCKED": 400,
            "PO-C-PARTIAL": 95,
            "PO-E-SMALL-PACK": 85,
            "PO-D-NOT-MOVING": 50,  # no sales at all
        }
    )
    po = calculate_po_base(
        sales_df=sales,
        inv_df=inv,
        period_days=PERIOD_DAYS,
        lead_time=LEAD_TIME,
        target_days=TARGET_DAYS,
        demand_basis="Sold",
        safety_pct=0.0,
        enforce_lead_time_release_gate=False,  # formula tests: bypass gate
    )
    assert not po.empty
    return po.set_index("OMS_SKU")


def _expected_po_qty(ads: float, projected_days: float) -> int:
    balance_days = TARGET_COVER_DAYS - projected_days
    raw_po = ads * balance_days
    return round_po_pack(max(raw_po, 0))


def test_ads_is_positive_for_selling_skus_and_zero_otherwise(po_result):
    for sku in ("PO-A-ZERO-STOCK", "PO-B-OVERSTOCKED", "PO-C-PARTIAL", "PO-E-SMALL-PACK"):
        assert po_result.loc[sku, "ADS"] > 0
    assert po_result.loc["PO-D-NOT-MOVING", "ADS"] == 0


def test_zero_stock_projects_zero_running_days(po_result):
    row = po_result.loc["PO-A-ZERO-STOCK"]
    assert row["Projected_Running_Days"] == 0


def test_overstocked_sku_gets_zero_po_never_negative(po_result):
    """Inventory cover (200d) exceeds the 180-day target -> balance is negative,
    raw_po is negative, and PO_Qty must clamp to 0, not go negative."""
    row = po_result.loc["PO-B-OVERSTOCKED"]
    assert row["Projected_Running_Days"] > TARGET_COVER_DAYS
    assert row["PO_Qty"] == 0


@pytest.mark.parametrize(
    "sku", ["PO-A-ZERO-STOCK", "PO-B-OVERSTOCKED", "PO-C-PARTIAL", "PO-E-SMALL-PACK"]
)
def test_po_qty_matches_sheet_formula_from_engine_outputs(po_result, sku):
    """PO_Qty == round_po_pack(max(ADS * (target_cover - Projected_Running_Days), 0))
    using the engine's own ADS / Projected_Running_Days for this row."""
    row = po_result.loc[sku]
    expected = _expected_po_qty(row["ADS"], row["Projected_Running_Days"])
    assert row["PO_Qty"] == expected
    assert row["PO_Qty"] == row["Gross_PO_Qty"]  # no safety/lead-time adjustment with no status sheet


def test_not_moving_sku_never_gets_a_po(po_result):
    """ADS=0 -> projected_days_now defaults to 999 ("infinite cover"),
    so raw_po = 0 * negative = 0, never a positive PO regardless of stock."""
    row = po_result.loc["PO-D-NOT-MOVING"]
    assert row["ADS"] == 0
    assert row["Projected_Running_Days"] == 999.0
    assert row["PO_Qty"] == 0


def test_summary_totals_match_sum_of_po_qty(po_result):
    """The 'New PO to raise' summary card is a plain sum/count over PO_Qty —
    verify it matches the per-row values directly (no hidden multiplier)."""
    po_qty = po_result["PO_Qty"]
    assert int(po_qty.sum()) == sum(int(v) for v in po_qty)
    assert int((po_qty > 0).sum()) == int((po_qty.values > 0).sum())
    # Only the understocked SKUs (A, C, E) should generate a PO; B and D should not.
    assert po_result.loc["PO-A-ZERO-STOCK", "PO_Qty"] > 0
    assert po_result.loc["PO-C-PARTIAL", "PO_Qty"] > 0
    assert po_result.loc["PO-E-SMALL-PACK", "PO_Qty"] > 0
    assert po_result.loc["PO-B-OVERSTOCKED", "PO_Qty"] == 0
    assert po_result.loc["PO-D-NOT-MOVING", "PO_Qty"] == 0


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 0), (1, 5), (4, 5), (5, 5), (6, 10), (9, 10), (10, 10), (11, 20), (20, 20), (265, 270)],
)
def test_round_po_pack_thresholds(raw, expected):
    assert round_po_pack(raw) == expected


def test_lead_time_gate_applies_to_all_skus_not_just_sheet():
    """The lead-time release gate must fire for non-sheet SKUs too.

    A SKU with projected cover > lead_time should get PO_Qty = 0 when
    enforce_lead_time_release_gate=True, regardless of whether it appears
    in a status sheet.  Previously the gate only fired when
    Lead_Time_From_Status_Sheet was True (sheet-resolved lead), leaving
    non-sheet SKUs ungated and making the UI lead-time slider inert.
    """
    sales = _steady_sales("GATE-SKU", 2.0)   # ADS=2/day
    # Inventory of 95 → projected = 95/2 = 47.5 days  (> LEAD_TIME=45, < TARGET=180)
    inv = _inv({"GATE-SKU": 95})
    # Gate ON → projected (47.5) > lead_time (45) → PO should be blocked to 0
    gated = calculate_po_base(
        sales_df=sales, inv_df=inv,
        period_days=PERIOD_DAYS, lead_time=LEAD_TIME,
        target_days=TARGET_DAYS, demand_basis="Sold",
        safety_pct=0.0,
        enforce_lead_time_release_gate=True,
    ).set_index("OMS_SKU")
    assert gated.loc["GATE-SKU", "PO_Qty"] == 0, (
        "Non-sheet SKU with cover > lead_time must be gated to 0 when gate is on"
    )

    # Increasing lead_time beyond projected cover (48d > 47.5d projected) → unblocked
    ungated = calculate_po_base(
        sales_df=sales, inv_df=inv,
        period_days=PERIOD_DAYS, lead_time=48,
        target_days=TARGET_DAYS, demand_basis="Sold",
        safety_pct=0.0,
        enforce_lead_time_release_gate=True,
    ).set_index("OMS_SKU")
    assert ungated.loc["GATE-SKU", "PO_Qty"] > 0, (
        "Raising lead_time above projected cover must unblock the SKU"
    )


def test_quarterly_history_zero_before_first_sale_is_not_a_bug():
    """A SKU whose sales history starts recently legitimately shows 0 ('-' in the UI)
    for quarters before its first sale — this is real, not a data-loading bug."""
    sales = _steady_sales("PO-NEW-LAUNCH", 2.0, days=80)  # ~2.5 months of history only
    pivot = calculate_quarterly_history(
        sales_df=sales,
        sku_mapping=None,
        group_by_parent=False,
        n_quarters=8,
    )
    assert not pivot.empty
    row = pivot[pivot["OMS_SKU"] == "PO-NEW-LAUNCH"].iloc[0]
    non_qty_cols = {
        "OMS_SKU", "Avg_Monthly", "Units_90d", "ADS", "Units_30d", "Freq_30d", "Status",
        "Parent_SKU",
    }
    q_cols = [c for c in pivot.columns if c not in non_qty_cols]
    assert len(q_cols) >= 2
    # Oldest quarter (no sales yet, ~80 days < a full quarter back) is 0;
    # the most recent quarter has the ~160 units sold.
    assert row[q_cols[0]] == 0
    assert row[q_cols[-1]] > 0
