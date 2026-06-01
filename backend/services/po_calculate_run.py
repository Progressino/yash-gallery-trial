"""Background PO calculate — keeps POST /api/po/calculate under proxy timeouts."""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _set_po_calculate_progress(
    sess,
    session_id: Optional[str],
    pct: int,
    message: str,
) -> None:
    p = max(0, min(100, int(pct)))
    sess.po_calculate_progress = p
    sess.po_calculate_message = message
    if session_id:
        from .po_calculate_jobs import set_po_job

        set_po_job(session_id, status="running", ok=True, progress=p, message=message)


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
    _set_po_calculate_progress(sess, session_id, 5, "Validating sales and inventory…")
    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.po_engine import calculate_po_base

    group_by_parent = bool(body.get("group_by_parent", False))
    inv_df = sess.inventory_df_parent if group_by_parent else sess.inventory_df_variant

    from ..services.po_raise_import import hydrate_session_ledger_from_db

    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)
    _set_po_calculate_progress(sess, session_id, 12, "Loading raise ledger…")
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

    # ── Inventory-history memory management ───────────────────────────────────
    # Two-stage trim so the heavy calculate_po_base call starts lean.
    #
    # Stage 1 — session migration: sessions restored from a warm-cache or PG
    # snapshot saved BEFORE the upload-time trim was added may carry multi-year
    # baselines (30M+ rows).  Trim those in-place once so every future request
    # for this session is fast.  The trimmed version is saved back to PG/warm-
    # cache by the post-calculate _sync() thread.
    #
    # Stage 2 — calc-time pre-trim: keep only rows needed for this run. Cap depth at
    # min(period_days + 14, DAILY_INV_MAX_DAYS) so we never scan more inventory history
    # than the server retains (default 30 days).
    import gc as _gc

    _period = int(body.get("period_days", 90))
    _raw_ih = getattr(sess, "daily_inventory_history_df", None)
    _inv_history_for_calc: pd.DataFrame | None = None

    if _raw_ih is not None and not _raw_ih.empty:
        from .daily_inventory_upload_run import _MAX_HISTORY_DAYS, _trim_history_to_recent

        _set_po_calculate_progress(sess, session_id, 22, "Preparing daily inventory history…")
        # Stage 1: migrate oversized sessions (warm-cache restore with old data).
        if len(_raw_ih) > (_MAX_HISTORY_DAYS + 10) * 500:
            # threshold ~ _MAX_HISTORY_DAYS days × 500 SKUs (very conservative)
            _trimmed_sess, _trim_note = _trim_history_to_recent(_raw_ih, _MAX_HISTORY_DAYS)
            if len(_trimmed_sess) < len(_raw_ih):
                logger.info(
                    "PO calc: session inventory history over-sized (%s rows) — "
                    "trimming to %d days in-place (%s rows). %s",
                    f"{len(_raw_ih):,}",
                    _MAX_HISTORY_DAYS,
                    f"{len(_trimmed_sess):,}",
                    _trim_note,
                )
                sess.daily_inventory_history_df = _trimmed_sess
                _raw_ih = _trimmed_sess
            del _trimmed_sess, _trim_note

        # Stage 2: calc-time pre-trim (bounded by DAILY_INV_MAX_DAYS / _MAX_HISTORY_DAYS).
        _inv_dates = pd.to_datetime(_raw_ih["Date"], errors="coerce")
        _max_inv = _inv_dates.max()
        if pd.notna(_max_inv):
            _depth_days = min(_period + 14, _MAX_HISTORY_DAYS)
            _pretrim_cutoff = pd.Timestamp(_max_inv).normalize() - pd.Timedelta(
                days=_depth_days
            )
            _mask = _inv_dates >= _pretrim_cutoff
            _inv_history_for_calc = _raw_ih[_mask].reset_index(drop=True)
            if len(_inv_history_for_calc) < len(_raw_ih):
                logger.info(
                    "PO calc pre-trim: %s → %s inventory-history rows (period=%d days, depth_cap=%d)",
                    f"{len(_raw_ih):,}",
                    f"{len(_inv_history_for_calc):,}",
                    _period,
                    _depth_days,
                )
            del _mask
        else:
            _inv_history_for_calc = _raw_ih
        del _inv_dates
        _gc.collect()

    _set_po_calculate_progress(
        sess,
        session_id,
        30,
        "Running PO calculation engine (this step may take 1–3 minutes)…",
    )
    try:
        po_df = calculate_po_base(
            sales_df=sess.sales_df,
            inv_df=inv_df,
            period_days=_period,
            lead_time=int(body.get("lead_time", 30)),
            target_days=int(body.get("target_days", 135)),
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
            enforce_lead_time_release_gate=bool(body.get("enforce_lead_time_release_gate", False)),
            inventory_history_df=_inv_history_for_calc,
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
            urgent_all_sizes_days=int(body.get("urgent_all_sizes_days", 45)),
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}

    if po_df is None or po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    _set_po_calculate_progress(sess, session_id, 85, "Formatting PO results…")
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

    cols = list(po_df.columns)
    return {
        "ok": True,
        "columns": cols,
        "total_rows": int(len(po_df)),
        "sales_through": sales_through,
        "planning_date": planning_out,
        "raise_ledger_rows": ledger_n,
        "ledger_auto_import": auto_msg,
    }


def background_po_calculate(session_id: str, body: dict) -> None:
    import threading

    from ..session import store
    from .po_calculate_jobs import set_po_job

    sess = store.get(session_id)
    if sess is None:
        set_po_job(session_id, status="error", ok=False, message="Session not found.")
        return

    sess.po_calculate_progress = 2
    try:
        from .po_result_spill import clear_spill

        clear_spill(session_id)
    except Exception:
        pass
    set_po_job(
        session_id,
        status="running",
        ok=True,
        progress=2,
        message="Calculating PO recommendations…",
    )

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
            msg = f"Calculating PO (trimmed inventory window, {_inv_n:,} baseline rows)…"
            _set_po_calculate_progress(sess, session_id, 8, msg)
        result = execute_po_calculate(sess, body, session_id=session_id, sync_sidecars=None)
        sess.po_calculate_result = result
        if result.get("ok"):
            n = int(result.get("total_rows") or 0)
            msg = f"PO calculation complete ({n:,} rows)."
            cols: list[str] = []
            try:
                from .po_result_spill import spill_df

                po_df = getattr(sess, "po_calculate_result_df", None)
                if po_df is not None and not getattr(po_df, "empty", True):
                    cols = list(po_df.columns)
                    spill_df(session_id, po_df)
                    if sess.po_calculate_result:
                        sess.po_calculate_result["columns"] = cols
            except Exception:
                logger.exception("spill_po_result_df after calculate")

            sess.po_calculate_status = "done"
            _set_po_calculate_progress(sess, session_id, 100, msg)
            set_po_job(
                session_id,
                status="done",
                ok=True,
                progress=100,
                message=msg,
                total_rows=n,
                columns=cols or result.get("columns"),
                sales_through=result.get("sales_through"),
                planning_date=result.get("planning_date"),
                raise_ledger_rows=result.get("raise_ledger_rows"),
                ledger_auto_import=result.get("ledger_auto_import"),
            )

            def _warm_quarterly_bg() -> None:
                try:
                    from .po_quarterly_warmup import warmup_quarterly_cache

                    warmup_quarterly_cache(
                        sess,
                        group_by_parent=bool(body.get("group_by_parent", False)),
                        n_quarters=8,
                    )
                except Exception:
                    logger.exception("quarterly warmup after PO calculate failed")

            threading.Thread(
                target=_warm_quarterly_bg,
                daemon=True,
                name=f"po-qtr-{session_id[:8]}",
            ).start()
            threading.Thread(
                target=_sync,
                daemon=True,
                name=f"po-save-{session_id[:8]}",
            ).start()
        else:
            msg = result.get("message") or "PO calculation failed."
            sess.po_calculate_status = "error"
            sess.po_calculate_progress = 0
            sess.po_calculate_message = msg
            set_po_job(session_id, status="error", ok=False, progress=0, message=msg)
    except Exception as e:
        logger.exception("background_po_calculate failed")
        msg = str(e)
        sess.po_calculate_status = "error"
        sess.po_calculate_progress = 0
        sess.po_calculate_message = msg
        sess.po_calculate_result = {"ok": False, "message": msg}
        set_po_job(session_id, status="error", ok=False, progress=0, message=msg)
