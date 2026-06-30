"""Catalog-wide ADS period-cap validation — every SKU must obey sold÷period ceiling."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest

from backend.services.po_engine import calculate_po_base

PO_FRESH_BODY = {
    "period_days": 30,
    "lead_time": 60,
    "target_days": 180,
    "grace_days": 0,
    "demand_basis": "Sold",
    "group_by_parent": False,
    "safety_pct": 0,
    "use_seasonality": False,
    "use_ly_fallback": False,
    "enforce_two_size_minimum": True,
    "enforce_lead_time_release_gate": False,
    "use_shared_cache": False,
}


def _expected_ads_row(
    row: pd.Series,
    *,
    period_days: int,
    use_ly_fallback: bool,
    seasonal_weight: float = 0.5,
) -> float:
    """Mirror engine ADS layering + period burst cap (rounded to 3 dp)."""
    sold = float(row.get("Sold_Units") or 0)
    recent = float(row.get("Recent_ADS") or 0)
    ly = float(row.get("LY_ADS") or 0)
    flat = float(row.get("Flat30_ADS") or 0)
    if use_ly_fallback and ly > 0:
        prim = recent * seasonal_weight + ly * (1.0 - seasonal_weight)
    else:
        prim = recent
    period_rate = sold / float(period_days) if period_days else 0.0
    ceil = max(flat, period_rate)
    if sold >= 6 and prim > ceil:
        prim = ceil
    return round(max(prim, flat), 3)


def _formula_mismatches(
    po_df: pd.DataFrame,
    *,
    period_days: int = 30,
    use_ly_fallback: bool = False,
    tol: float = 0.002,
) -> list[str]:
    violations: list[str] = []
    for idx, row in po_df.iterrows():
        exp = _expected_ads_row(row, period_days=period_days, use_ly_fallback=use_ly_fallback)
        got = round(float(row.get("ADS") or 0), 3)
        if abs(got - exp) > tol:
            sku = row.get("OMS_SKU", idx)
            violations.append(
                f"{sku}: ADS got={got} expected={exp} "
                f"(sold={row.get('Sold_Units')} recent={row.get('Recent_ADS')} "
                f"eff={row.get('Eff_Days')} flat={row.get('Flat30_ADS')} "
                f"season={row.get('Seasonal_Month_ADS')})"
            )
    return violations


@pytest.fixture(scope="module")
def warm_cache_po_df():
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        pytest.skip("warm cache not on disk")
    main_mod._warm_cache = data
    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        pytest.skip("warm cache copy failed")
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)
    if sess.sales_df.empty or sess.inventory_df_variant.empty:
        pytest.skip("warm cache missing sales or inventory")
    result = execute_po_calculate(
        sess,
        PO_FRESH_BODY,
        session_id=f"ads-audit-{uuid.uuid4().hex[:8]}",
    )
    if not result.get("ok"):
        pytest.skip(result.get("message") or "PO calculate failed")
    po_df = getattr(sess, "po_calculate_result_df", None)
    if po_df is None or po_df.empty:
        pytest.skip("empty PO result")
    return po_df


def test_catalog_ads_formula_every_row(warm_cache_po_df):
    """Every SKU row: engine ADS matches recomputed cap + floor formula."""
    violations = _formula_mismatches(
        warm_cache_po_df,
        period_days=PO_FRESH_BODY["period_days"],
        use_ly_fallback=PO_FRESH_BODY["use_ly_fallback"],
    )
    assert not violations, (
        f"{len(violations)} formula mismatches (first 20):\n" + "\n".join(violations[:20])
    )


def test_catalog_no_uncapped_burst_recent(warm_cache_po_df):
    """No SKU keeps ADS at inflated Recent when sold÷30 cap should apply."""
    po_df = warm_cache_po_df
    sold = pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0)
    recent = pd.to_numeric(po_df["Recent_ADS"], errors="coerce").fillna(0)
    ads = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0)
    flat = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0)
    season = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0)
    period_rate = sold / float(PO_FRESH_BODY["period_days"])
    stuck = (
        (sold >= 6)
        & (recent > period_rate + 0.05)
        & (np.abs(ads - recent) < 0.003)
        & (season < ads - 0.01)
        & (flat < ads - 0.01)
    )
    assert not stuck.any(), (
        f"{int(stuck.sum())} SKUs stuck at uncapped Recent_ADS: "
        f"{po_df.loc[stuck, 'OMS_SKU'].astype(str).head(10).tolist()}"
    )


def test_catalog_short_eff_days_respect_floors(warm_cache_po_df):
    """Eff_Days < period: ADS never exceeds max(sold÷30, Flat30, Seasonal)."""
    po_df = warm_cache_po_df
    sold = pd.to_numeric(po_df["Sold_Units"], errors="coerce").fillna(0)
    ads = pd.to_numeric(po_df["ADS"], errors="coerce").fillna(0)
    flat = pd.to_numeric(po_df["Flat30_ADS"], errors="coerce").fillna(0)
    season = pd.to_numeric(po_df["Seasonal_Month_ADS"], errors="coerce").fillna(0)
    eff = pd.to_numeric(po_df["Eff_Days"], errors="coerce").fillna(0)
    period = PO_FRESH_BODY["period_days"]
    short = (sold >= 6) & (eff < period)
    max_allowed = np.maximum(np.maximum(sold / float(period), flat), season)
    over = short & (ads > max_allowed + 0.004)
    assert not over.any(), (
        f"{int(over.sum())} short-span SKUs above allowed ADS: "
        f"{po_df.loc[over, 'OMS_SKU'].astype(str).head(10).tolist()}"
    )


def test_1050ykblue_l_six_units_in_thirty_days(warm_cache_po_df):
    row = warm_cache_po_df[warm_cache_po_df["OMS_SKU"].astype(str) == "1050YKBLUE-L"]
    if row.empty:
        pytest.skip("1050YKBLUE-L not in warm-cache PO result")
    r = row.iloc[0]
    assert int(r["Sold_Units"]) == 6
    assert float(r["Recent_ADS"]) == pytest.approx(1.0, abs=0.05)
    assert float(r["ADS"]) == pytest.approx(0.213, abs=0.02)
    assert float(r["ADS"]) < float(r["Recent_ADS"]) - 0.5


def test_sparse_burst_synthetic_cases():
    """Synthetic edge cases for sold÷30 cap (not tied to warm cache)."""
    rows = []
    for d in pd.to_datetime(
        ["2026-05-01", "2026-05-05", "2026-05-10", "2026-05-18", "2026-05-22", "2026-05-28"]
    ):
        rows.append(
            {
                "Sku": "CAP-SPARSE-6",
                "TxnDate": d,
                "Transaction Type": "Shipment",
                "Quantity": 1,
                "Units_Effective": 1,
                "Source": "Amazon",
            }
        )
    po = calculate_po_base(
        sales_df=pd.DataFrame(rows),
        inv_df=pd.DataFrame({"OMS_SKU": ["CAP-SPARSE-6"], "Total_Inventory": [11]}),
        period_days=30,
        lead_time=60,
        target_days=180,
        safety_pct=0.0,
        use_ly_fallback=False,
    )
    r = po.iloc[0]
    assert int(r["Sold_Units"]) == 6
    assert float(r["ADS"]) == pytest.approx(0.2, abs=0.02)

    days = pd.date_range("2026-05-01", periods=24, freq="D")
    rows2 = [
        {
            "Sku": "CAP-DAILY-96",
            "TxnDate": d,
            "Transaction Type": "Shipment",
            "Quantity": 4,
            "Units_Effective": 4,
            "Source": "Amazon",
        }
        for d in days
    ]
    po2 = calculate_po_base(
        sales_df=pd.DataFrame(rows2),
        inv_df=pd.DataFrame({"OMS_SKU": ["CAP-DAILY-96"], "Total_Inventory": [40]}),
        period_days=30,
        lead_time=60,
        target_days=180,
        safety_pct=0.0,
        use_ly_fallback=False,
    )
    r2 = po2.iloc[0]
    assert int(r2["Sold_Units"]) == 96
    assert float(r2["Recent_ADS"]) == pytest.approx(4.0, abs=0.05)
    assert float(r2["ADS"]) == pytest.approx(3.2, abs=0.05)


def test_apply_period_burst_cap_used_by_both_main_and_fanout_paths():
    """Regression: bundled-fan-out ADS recompute must obey the same period-burst
    cap as the main pass — previously the fan-out block skipped the cap entirely,
    letting sibling-size rows (e.g. when a bundled parent fans demand to in-stock
    children) report uncapped, inflated Recent_ADS and over-order PO quantity."""
    from backend.services.po_engine import _apply_period_burst_cap

    # 6 sold over a short Eff_Days window (burst) → must cap to sold/period_days.
    capped = _apply_period_burst_cap(
        prim_ads=[3.0], sold_units=[6], flat_ads=[0.1], period_days=30
    )
    assert capped[0] == pytest.approx(0.2, abs=1e-6)

    # Below the sold>=6 threshold → no cap applied.
    uncapped = _apply_period_burst_cap(
        prim_ads=[3.0], sold_units=[5], flat_ads=[0.1], period_days=30
    )
    assert uncapped[0] == pytest.approx(3.0, abs=1e-6)

    # Flat30 floor still wins when it exceeds the period rate.
    floored = _apply_period_burst_cap(
        prim_ads=[3.0], sold_units=[6], flat_ads=[0.5], period_days=30
    )
    assert floored[0] == pytest.approx(0.5, abs=1e-6)


def test_catalog_ads_with_ly_fallback_enabled():
    """Full catalog with LY fallback ON — every row must match formula."""
    import backend.main as main_mod
    from backend.session import AppSession
    from backend.services.existing_po import ensure_existing_po_hydrated
    from backend.services.po_calculate_run import execute_po_calculate

    ok, data = main_mod._load_warm_cache_from_disk(ignore_age=True)
    if not ok or not data:
        pytest.skip("warm cache not on disk")
    main_mod._warm_cache = data
    sess = AppSession()
    if not main_mod._copy_warm_cache_to_session(sess):
        pytest.skip("warm cache copy failed")
    main_mod.restore_po_sidecars_from_warm(sess)
    ensure_existing_po_hydrated(sess)
    body = {**PO_FRESH_BODY, "use_ly_fallback": True}
    result = execute_po_calculate(sess, body, session_id=f"ads-ly-{uuid.uuid4().hex[:8]}")
    if not result.get("ok"):
        pytest.skip(result.get("message") or "PO failed")
    po_df = sess.po_calculate_result_df
    violations = _formula_mismatches(po_df, period_days=30, use_ly_fallback=True, tol=0.003)
    assert not violations, "\n".join(violations[:20])
