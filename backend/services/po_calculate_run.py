"""Background PO calculate — keeps POST /api/po/calculate under proxy timeouts."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _dedupe_column_names(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    out: list[str] = []
    for c in df.columns:
        base = str(c)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    df = df.copy()
    df.columns = out
    return df


def execute_po_calculate(
    sess,
    body: dict,
    *,
    session_id: Optional[str] = None,
    sync_sidecars: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    """Run full PO engine math and return the same payload as the legacy sync endpoint."""
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.po_engine import calculate_po_base

    group_by_parent = bool(body.get("group_by_parent", False))
    inv_df = sess.inventory_df_parent if group_by_parent else sess.inventory_df_variant

    from ..services.po_raise_import import hydrate_session_ledger_from_db

    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)
    hydrate_session_ledger_from_db(sess, body.get("planning_date"), lookback_days=lookback)

    ledger_auto_import = None
    if body.get("auto_import_yesterday_ledger", True) and session_id:
        from ..services.po_raise_archive import try_auto_import_recent_ledgers

        ledger_auto_import = try_auto_import_recent_ledgers(
            sess,
            session_id,
            body.get("planning_date"),
            group_by_parent=group_by_parent,
            lookback_days=lookback,
        )
        if ledger_auto_import and ledger_auto_import.get("ok") and sync_sidecars:
            try:
                sync_sidecars()
            except Exception:
                logger.exception("sync_sidecars after ledger auto-import failed")

    _ledger = getattr(sess, "po_raise_ledger_df", None)
    try:
        po_df = calculate_po_base(
            sales_df=sess.sales_df,
            inv_df=inv_df,
            period_days=int(body.get("period_days", 90)),
            lead_time=int(body.get("lead_time", 30)),
            target_days=int(body.get("target_days", 210)),
            demand_basis=str(body.get("demand_basis", "Sold")),
            min_denominator=int(body.get("min_denominator", 7)),
            grace_days=int(body.get("grace_days", 0)),
            safety_pct=float(body.get("safety_pct", 0.0)),
            use_seasonality=bool(body.get("use_seasonality", False)),
            seasonal_weight=float(body.get("seasonal_weight", 0.5)),
            sku_mapping=sess.sku_mapping or None,
            group_by_parent=group_by_parent,
            existing_po_df=sess.existing_po_df if not sess.existing_po_df.empty else None,
            sku_status_df=sess.sku_status_lead_df if not sess.sku_status_lead_df.empty else None,
            enforce_two_size_minimum=bool(body.get("enforce_two_size_minimum", False)),
            enforce_lead_time_release_gate=bool(body.get("enforce_lead_time_release_gate", True)),
            inventory_history_df=(
                sess.daily_inventory_history_df
                if not sess.daily_inventory_history_df.empty
                else None
            ),
            po_raise_ledger_df=(_ledger if _ledger is not None and not _ledger.empty else None),
            planning_date=body.get("planning_date"),
            raise_ledger_lookback_days=lookback,
            raise_view_date=body.get("raise_view_date"),
            po_return_overlay_df=(
                getattr(sess, "po_return_overlay_df", None)
                if getattr(sess, "po_return_overlay_df", None) is not None
                and not getattr(sess, "po_return_overlay_df", pd.DataFrame()).empty
                else None
            ),
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df is None or po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    po_df = po_df.copy()
    po_df = _dedupe_column_names(po_df)
    for c in ["Suggest_Close_SKU", "PO_Block_Reason", "SKU_Sheet_Status"]:
        if c in po_df.columns:
            po_df[c] = po_df[c].fillna("").astype(str)
    num_cols = po_df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        po_df[num_cols] = (
            po_df[num_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .round(3)
        )
    for c in ["Days_Left", "Projected_Running_Days"]:
        if c in po_df.columns:
            s = pd.to_numeric(po_df[c], errors="coerce")
            po_df[c] = s.where(np.isfinite(s), 999.0).fillna(999.0).round(1)

    sess.po_calculate_result_df = po_df

    sales_through = None
    try:
        st = pd.to_datetime(sess.sales_df["TxnDate"], errors="coerce").max()
        if pd.notna(st):
            sales_through = str(pd.Timestamp(st).date())
    except Exception:
        pass

    planning_out = None
    pd_raw = body.get("planning_date")
    if pd_raw and str(pd_raw).strip():
        try:
            planning_out = str(pd.Timestamp(pd.to_datetime(pd_raw).normalize()).date())
        except Exception:
            planning_out = str(pd_raw).strip()

    _ledger_after = getattr(sess, "po_raise_ledger_df", None)
    ledger_n = (
        int(len(_ledger_after))
        if _ledger_after is not None and not getattr(_ledger_after, "empty", True)
        else 0
    )
    auto_msg = None
    if ledger_auto_import and ledger_auto_import.get("ok"):
        auto_msg = ledger_auto_import.get("message")

    return {
        "ok": True,
        "columns": list(po_df.columns),
        "total_rows": int(len(po_df)),
        "sales_through": sales_through,
        "planning_date": planning_out,
        "raise_ledger_rows": ledger_n,
        "ledger_auto_import": auto_msg,
    }


def background_po_calculate(session_id: str, body: dict) -> None:
    import threading

    from ..session import store

    sess = store.get(session_id)
    if sess is None:
        return

    def _sync() -> None:
        try:
            import backend.main as _main

            _main.merge_po_optional_sheets_into_warm_cache(sess)
        except Exception:
            logger.exception("merge_po_optional_sheets_into_warm_cache failed")
        try:
            from ..db.forecast_session_pg import persist_session_bundle

            persist_session_bundle(session_id, sess)
        except Exception:
            logger.exception("PostgreSQL persist after PO calculate failed")

    try:
        _inv_n = int(len(getattr(sess, "daily_inventory_history_df", pd.DataFrame())))
        if _inv_n > 500_000:
            sess.po_calculate_message = (
                f"Calculating PO (trimmed inventory window, {_inv_n:,} baseline rows)…"
            )
        result = execute_po_calculate(sess, body, session_id=session_id, sync_sidecars=None)
        sess.po_calculate_result = result
        if result.get("ok"):
            sess.po_calculate_status = "done"
            n = int(result.get("total_rows") or 0)
            sess.po_calculate_message = f"PO calculation complete ({n:,} rows)."
            threading.Thread(
                target=_sync,
                daemon=True,
                name=f"po-save-{session_id[:8]}",
            ).start()
        else:
            sess.po_calculate_status = "error"
            sess.po_calculate_message = result.get("message") or "PO calculation failed."
    except Exception as e:
        logger.exception("background_po_calculate failed")
        sess.po_calculate_status = "error"
        sess.po_calculate_message = str(e)
        sess.po_calculate_result = {"ok": False, "message": str(e)}
