"""PO row audit — cross-check engine output vs Tier-3 sales and inventory history."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .po_engine import canonical_oms_key
from .po_shared_cache import _CALC_PARAM_DEFAULTS, _CALC_PARAM_KEYS, _parquet_path, lookup_shared_cache


def _body_from_query(qp: dict) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for k in _CALC_PARAM_KEYS:
        if k not in qp:
            continue
        raw = qp.get(k)
        if raw is None or raw == "":
            continue
        if k in (
            "use_seasonality",
            "group_by_parent",
            "enforce_two_size_minimum",
            "enforce_lead_time_release_gate",
            "auto_import_yesterday_ledger",
            "use_ly_fallback",
        ):
            body[k] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        elif k in (
            "period_days",
            "lead_time",
            "target_days",
            "min_denominator",
            "grace_days",
            "urgent_all_sizes_days",
            "raise_ledger_lookback_days",
        ):
            try:
                body[k] = int(raw)
            except (TypeError, ValueError):
                pass
        elif k in ("seasonal_weight", "safety_pct"):
            try:
                body[k] = float(raw)
            except (TypeError, ValueError):
                pass
        else:
            body[k] = raw
    if qp.get("planning_date"):
        body["planning_date"] = str(qp.get("planning_date"))[:10]
    return body


def _tier3_sku_totals(sess, sku: str, start_date: str, end_date: str) -> dict[str, Any]:
    from ..routers.data import _load_tier3_frames_for_platforms
    from .daily_store import platforms_with_uploads_in_range
    from .helpers import get_parent_sku

    s, e = start_date[:10], end_date[:10]
    uploaded = platforms_with_uploads_in_range(s, e)
    if not uploaded:
        return {"sold_units": 0, "return_units": 0, "net_units": 0, "platforms": {}}

    sku_map = getattr(sess, "sku_mapping", None) or None
    target = canonical_oms_key(sku, sku_map)
    parent = get_parent_sku(target)

    frames = _load_tier3_frames_for_platforms(sorted(uploaded), s, e, dedup=False, columns_only=True)
    if not frames:
        frames = _load_tier3_frames_for_platforms(sorted(uploaded), s, e, dedup=False, columns_only=False)

    specs = (
        ("amazon", "SKU", "Transaction_Type"),
        ("myntra", "OMS_SKU", "TxnType"),
        ("meesho", "OMS_SKU", "TxnType"),
        ("flipkart", "OMS_SKU", "TxnType"),
        ("snapdeal", "OMS_SKU", "TxnType"),
    )
    sold = ret = 0
    plat_breakdown: dict[str, dict[str, int]] = {}
    for plat, sku_col, txn_col in specs:
        if plat not in frames:
            continue
        df = frames.get(plat)
        if df is None or df.empty or sku_col not in df.columns:
            continue
        work = df.copy()
        work["_canon"] = work[sku_col].astype(str).map(lambda v: canonical_oms_key(v, sku_map))
        if parent and parent != target:
            work["_parent"] = work["_canon"].map(get_parent_sku)
            sub = work[(work["_canon"] == target) | (work["_parent"] == parent)]
        else:
            sub = work[work["_canon"] == target]
        if sub.empty:
            continue
        qty = pd.to_numeric(sub.get("Quantity"), errors="coerce").fillna(0)
        if txn_col in sub.columns:
            txn = sub[txn_col].astype(str).str.strip().str.lower()
            ship = txn.str.contains("ship", na=False) & ~txn.str.contains("refund|return", na=False)
            rtn = txn.str.contains("refund|return", na=False)
            p_sold = int(qty[ship].sum())
            p_ret = int(qty[rtn].abs().sum())
        else:
            p_sold = int(qty.clip(lower=0).sum())
            p_ret = 0
        sold += p_sold
        ret += p_ret
        plat_breakdown[plat] = {"sold": p_sold, "returns": p_ret}

    return {
        "sold_units": sold,
        "return_units": ret,
        "net_units": sold - ret,
        "platforms": plat_breakdown,
        "window": {"start": s, "end": e},
    }


def build_po_sku_audit(sess, sku: str, query_params: dict) -> dict[str, Any]:
    """Return PO engine row + Tier-3 cross-check for one SKU."""
    sku = str(sku or "").strip()
    if not sku:
        return {"ok": False, "message": "sku is required"}

    body = _body_from_query(query_params)
    for k, v in _CALC_PARAM_DEFAULTS.items():
        body.setdefault(k, v)

    planning = str(body.get("planning_date") or "")[:10]
    if len(planning) != 10:
        planning = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        body["planning_date"] = planning

    period = int(body.get("period_days") or 30)
    start = (pd.Timestamp(planning) - pd.Timedelta(days=max(1, period) - 1)).strftime("%Y-%m-%d")
    end = planning

    po_row: dict[str, Any] | None = None
    cache_meta: dict[str, Any] | None = None
    meta = lookup_shared_cache(sess, body)
    if meta:
        cache_meta = {
            "cache_key": meta.get("cache_key"),
            "created_at_ist": meta.get("created_at_ist"),
            "total_rows": meta.get("total_rows"),
            "sales_through": meta.get("sales_through"),
        }
        try:
            df = pd.read_parquet(_parquet_path(str(meta["cache_key"])))
            sku_map = getattr(sess, "sku_mapping", None) or None
            target = canonical_oms_key(sku, sku_map)
            if "OMS_SKU" in df.columns:
                work = df.copy()
                work["_k"] = work["OMS_SKU"].astype(str).map(lambda v: canonical_oms_key(v, sku_map))
                hit = work[work["_k"] == target]
                if hit.empty:
                    from .helpers import get_parent_sku

                    parent = get_parent_sku(target)
                    if parent:
                        hit = work[work["_k"].map(get_parent_sku) == parent]
                if not hit.empty:
                    po_row = hit.iloc[0].to_dict()
        except Exception:
            po_row = None

    if po_row is None:
        spill = getattr(sess, "po_calculate_result_df", None)
        if spill is not None and hasattr(spill, "empty") and not spill.empty and "OMS_SKU" in spill.columns:
            sku_map = getattr(sess, "sku_mapping", None) or None
            target = canonical_oms_key(sku, sku_map)
            work = spill.copy()
            work["_k"] = work["OMS_SKU"].astype(str).map(lambda v: canonical_oms_key(v, sku_map))
            hit = work[work["_k"] == target]
            if not hit.empty:
                po_row = hit.iloc[0].to_dict()

    tier3 = _tier3_sku_totals(sess, sku, start, end)

    checks: list[dict[str, Any]] = []
    if po_row:
        eng_sold = int(pd.to_numeric(po_row.get("Sold_Units"), errors="coerce") or 0)
        eng_ret = int(pd.to_numeric(po_row.get("Return_Units"), errors="coerce") or 0)
        t3_sold = int(tier3.get("sold_units") or 0)
        t3_ret = int(tier3.get("return_units") or 0)
        checks.append(
            {
                "field": "Sold_Units",
                "engine": eng_sold,
                "tier3_window": t3_sold,
                "delta": eng_sold - t3_sold,
                "ok": eng_sold == t3_sold,
                "note": f"Tier-3 shipments {start} → {end} (PO period_days={period})",
            }
        )
        checks.append(
            {
                "field": "Return_Units",
                "engine": eng_ret,
                "tier3_window": t3_ret,
                "delta": eng_ret - t3_ret,
                "ok": eng_ret == t3_ret,
                "note": "Includes return-sheet overlay in PO when uploaded",
            }
        )

    return {
        "ok": True,
        "sku": sku,
        "planning_date": planning,
        "period_days": period,
        "ads_window": {"start": start, "end": end},
        "po_row": po_row,
        "shared_cache": cache_meta,
        "tier3": tier3,
        "checks": checks,
        "message": (
            "PO row loaded from shared cache — compare Sold/Return vs Tier-3; "
            "open Eff-Days drawer for inventory history."
            if po_row
            else "No PO row for this SKU/settings — run PO calculate first or check SKU spelling."
        ),
    }
