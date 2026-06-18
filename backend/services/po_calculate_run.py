"""Background PO calculate — keeps POST /api/po/calculate under proxy timeouts."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _trim_sales_for_po_memory(
    sales_df: pd.DataFrame,
    *,
    period_days: int,
    use_seasonality: bool,
    use_ly_fallback: bool,
) -> pd.DataFrame:
    """Local Mac: pass a recent sales window into the engine to avoid OOM on full history."""
    if sales_df is None or getattr(sales_df, "empty", True):
        return sales_df
    try:
        from backend.main import warm_cache_po_session_only

        if not warm_cache_po_session_only():
            return sales_df
    except Exception:
        return sales_df
    if "TxnDate" not in sales_df.columns:
        return sales_df
    horizon = max(int(period_days), 90)
    if use_seasonality or use_ly_fallback:
        horizon = max(horizon, 400)
    dates = pd.to_datetime(sales_df["TxnDate"], errors="coerce")
    end = dates.max()
    if pd.isna(end):
        return sales_df
    start = pd.Timestamp(end).normalize() - pd.Timedelta(days=horizon)
    mask = dates >= start
    if not bool(mask.any()):
        return sales_df
    trimmed = sales_df.loc[mask]
    if len(trimmed) < len(sales_df):
        logger.info(
            "PO local memory trim: %s → %s sales rows (horizon=%dd)",
            f"{len(sales_df):,}",
            f"{len(trimmed):,}",
            horizon,
        )
    return trimmed.reset_index(drop=True)


def _po_return_overlay_for_calc(sess) -> pd.DataFrame | None:
    from .po_return_import import aggregate_return_overlay_for_use

    ov = aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None))
    if ov is None or getattr(ov, "empty", True):
        return None
    return ov


def _build_platform_sales_df(sess, *, frame_overrides: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Combine platform raw DataFrames into unified sales rows for ADS calculation."""
    from .flipkart import flipkart_to_sales_rows
    from .meesho import meesho_to_sales_rows
    from .myntra import myntra_to_sales_rows
    from .po_engine import _mtr_to_sales_df_local
    from .snapdeal import snapdeal_to_sales_rows

    overrides = frame_overrides or {}
    sku_map = getattr(sess, "sku_mapping", None) or {}
    parts: list[pd.DataFrame] = []

    mtr = overrides.get("mtr_df", getattr(sess, "mtr_df", None))
    if mtr is not None and not mtr.empty:
        try:
            parts.append(_mtr_to_sales_df_local(mtr, sku_map))
        except Exception:
            logger.exception("_build_platform_sales_df: mtr_df conversion failed")

    for key, attr, converter in [
        ("myntra_df", "myntra_df", lambda df: myntra_to_sales_rows(df)),
        ("meesho_df", "meesho_df", lambda df: meesho_to_sales_rows(df, sku_map)),
        ("flipkart_df", "flipkart_df", lambda df: flipkart_to_sales_rows(df)),
        ("snapdeal_df", "snapdeal_df", lambda df: snapdeal_to_sales_rows(df)),
    ]:
        raw = overrides.get(key, getattr(sess, attr, None))
        if raw is not None and not raw.empty:
            try:
                parts.append(converter(raw))
            except Exception:
                logger.exception("_build_platform_sales_df: %s conversion failed", attr)

    non_empty = [p for p in parts if p is not None and not p.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


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
    skip_hydrate: bool = False,
) -> dict[str, Any]:
    """Run full PO engine math and return the same payload as the legacy sync endpoint."""
    from .po_session_hydrate import (
        effective_sku_status_df_for_engine,
        hydrate_po_session_for_calculate,
    )
    from .po_stage_timer import PoStageTimer

    stage_timer = PoStageTimer()

    if not skip_hydrate:
        hydrate_po_session_for_calculate(sess)
    _set_po_calculate_progress(sess, session_id, 5, "Validating sales and inventory…")

    if sess.sales_df.empty:
        return {"ok": False, "message": "Build Sales first (upload platforms, then POST /api/upload/build-sales)."}
    if sess.inventory_df_variant.empty:
        return {"ok": False, "message": "Upload Inventory first."}

    from ..services.po_engine import calculate_po_base

    group_by_parent = bool(body.get("group_by_parent", False))
    inv_df = sess.inventory_df_parent if group_by_parent else sess.inventory_df_variant

    _period = int(body.get("period_days", 30))

    # ── Stage 1: sales load (Tier-3 overlay + ADS window trim) ────────────────
    from .tier3_session_merge import build_po_ads_platform_sales

    _set_po_calculate_progress(sess, session_id, 12, "Loading Tier-3 daily sales for ADS window…")
    _platform_sales = build_po_ads_platform_sales(
        sess,
        planning_date=body.get("planning_date"),
        period_days=_period,
        use_seasonality=bool(body.get("use_seasonality", False)),
        use_ly_fallback=bool(body.get("use_ly_fallback", True)),
    )
    if _platform_sales.empty:
        _platform_sales = _build_platform_sales_df(sess)
    _ads_source = _platform_sales if not _platform_sales.empty else sess.sales_df
    _ads_source = _trim_sales_for_po_memory(
        _ads_source,
        period_days=_period,
        use_seasonality=bool(body.get("use_seasonality", False)),
        use_ly_fallback=bool(body.get("use_ly_fallback", True)),
    )
    logger.info(
        "PO ADS source: %s (%d rows)",
        "platform" if not _platform_sales.empty else "OMS",
        len(_ads_source),
    )
    stage_timer.mark("sales load")

    # ── Stage 2: inventory sidecars (ledger, existing PO sheet, daily history) ─
    from ..services.po_raise_import import hydrate_session_ledger_from_db

    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)
    _set_po_calculate_progress(sess, session_id, 12, "Loading raise ledger…")
    hydrate_session_ledger_from_db(sess, body.get("planning_date"), lookback_days=lookback, authoritative=True)

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

    # Persist ledger to warm cache without blocking the PO engine thread.
    try:
        import backend.main as _main

        threading.Thread(
            target=lambda: _main.merge_po_optional_sheets_into_warm_cache(sess),
            daemon=True,
            name="po-ledger-warm",
        ).start()
    except Exception:
        logger.exception("merge_po_optional_sheets_into_warm_cache after ledger load failed")

    _ledger = getattr(sess, "po_raise_ledger_df", None)

    _set_po_calculate_progress(sess, session_id, 18, "Loading existing PO sheet…")
    try:
        from .existing_po import ensure_existing_po_hydrated

        ensure_existing_po_hydrated(sess)
    except Exception:
        logger.exception("ensure_existing_po_hydrated before calculate failed")

    _existing_po = (
        sess.existing_po_df if getattr(sess, "existing_po_df", None) is not None and not sess.existing_po_df.empty else None
    )

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

    _raw_ih = getattr(sess, "daily_inventory_history_df", None)
    _inv_history_for_calc: pd.DataFrame | None = None

    _rows_dropped = 0
    if _raw_ih is not None and not _raw_ih.empty:
        from .daily_inventory_upload_run import (
            _MAX_HISTORY_DAYS,
            _series_as_dates,
            _trim_history_to_recent,
        )

        _ih_rows = len(_raw_ih)
        _set_po_calculate_progress(
            sess,
            session_id,
            22,
            f"Preparing daily inventory history ({_ih_rows:,} rows)…",
        )
        # Stage 1: migrate oversized sessions (warm-cache restore with old data).
        if _ih_rows > (_MAX_HISTORY_DAYS + 10) * 500:
            _set_po_calculate_progress(
                sess,
                session_id,
                24,
                f"Trimming oversized inventory history ({_ih_rows:,} rows)…",
            )
            _trimmed_sess, _trim_note = _trim_history_to_recent(_raw_ih, _MAX_HISTORY_DAYS)
            if len(_trimmed_sess) < len(_raw_ih):
                _rows_dropped += len(_raw_ih) - len(_trimmed_sess)
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
        _inv_dates = _series_as_dates(_raw_ih["Date"])
        _max_inv = _inv_dates.max()
        if pd.notna(_max_inv):
            _depth_days = min(_period + 14, _MAX_HISTORY_DAYS)
            _pretrim_cutoff = pd.Timestamp(_max_inv).normalize() - pd.Timedelta(
                days=_depth_days
            )
            _min_inv = _inv_dates.min()
            if pd.notna(_min_inv) and _min_inv >= _pretrim_cutoff:
                _inv_history_for_calc = _raw_ih
            else:
                _mask = _inv_dates >= _pretrim_cutoff
                _inv_history_for_calc = _raw_ih.loc[_mask].reset_index(drop=True)
                if len(_inv_history_for_calc) < len(_raw_ih):
                    _rows_dropped += len(_raw_ih) - len(_inv_history_for_calc)
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
        if _rows_dropped >= 100_000:
            _set_po_calculate_progress(sess, session_id, 27, "Releasing trimmed inventory memory…")
            _gc.collect()

    stage_timer.mark("inventory")

    _set_po_calculate_progress(
        sess,
        session_id,
        30,
        "Running PO calculation engine…",
    )
    _hb_stop = threading.Event()

    def _calc_heartbeat() -> None:
        pct = 32
        while not _hb_stop.wait(20):
            pct = min(pct + 3, 78)
            _set_po_calculate_progress(
                sess,
                session_id,
                pct,
                f"Running PO engine… ({pct}%)",
            )

    _hb_thread = threading.Thread(target=_calc_heartbeat, daemon=True, name="po-calc-hb")
    _hb_thread.start()

    try:
        _lt_raw = body.get("lead_time")
        _lead_time = 0 if _lt_raw in (None, "") else int(_lt_raw or 0)
        po_df = calculate_po_base(
            sales_df=_ads_source,
            inv_df=inv_df,
            period_days=_period,
            lead_time=_lead_time,
            target_days=int(body.get("target_days", 135)),
            demand_basis=str(body.get("demand_basis", "Sold")),
            min_denominator=int(body.get("min_denominator", 7)),
            grace_days=int(body.get("grace_days", 0)),
            safety_pct=float(body.get("safety_pct", 0.0)),
            use_seasonality=bool(body.get("use_seasonality", False)),
            seasonal_weight=float(body.get("seasonal_weight", 0.5)),
            sku_mapping=sess.sku_mapping or None,
            group_by_parent=group_by_parent,
            existing_po_df=_existing_po,
            sku_status_df=effective_sku_status_df_for_engine(sess),
            enforce_two_size_minimum=bool(body.get("enforce_two_size_minimum", False)),
            # App PO mode always uses Excel Qty rule (lead-time gate). Do not honour
            # ``False`` from legacy API defaults — ``body.get(..., True)`` would still
            # return False when the key is present.
            enforce_lead_time_release_gate=True,
            inventory_history_df=_inv_history_for_calc,
            po_raise_ledger_df=(_ledger if _ledger is not None and not _ledger.empty else None),
            planning_date=body.get("planning_date"),
            raise_ledger_lookback_days=lookback,
            raise_view_date=body.get("raise_view_date"),
            po_return_overlay_df=_po_return_overlay_for_calc(sess),
            urgent_all_sizes_days=int(body.get("urgent_all_sizes_days", 45)),
            use_ly_fallback=bool(body.get("use_ly_fallback", True)),
            stage_timer=stage_timer,
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}
    finally:
        _hb_stop.set()

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

    stage_timer.mark("result format")
    stage_timings = stage_timer.log_summary()

    sess.po_calculate_result_df = po_df
    sess.po_calculate_existing_po_generation = int(getattr(sess, "existing_po_generation", 0) or 0)

    try:
        from .daily_store import get_summary
        from .tier3_session_merge import mark_tier3_sync_applied

        summary = get_summary() or {}
        if any(int((summary.get(p) or {}).get("file_count") or 0) > 0 for p in summary):
            mark_tier3_sync_applied(sess)
    except Exception:
        logger.exception("mark_tier3_sync_applied after PO calc failed")

    sales_through = None
    try:
        from .tier3_session_merge import effective_sales_through

        sales_through = effective_sales_through(sess, _ads_source) or None
    except Exception:
        pass
    if not sales_through:
        try:
            st = pd.to_datetime(_ads_source["TxnDate"], errors="coerce").max()
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

    def _num_col(name: str) -> pd.Series:
        if name not in po_df.columns:
            return pd.Series(0.0, index=po_df.index)
        return pd.to_numeric(po_df[name], errors="coerce").fillna(0)

    _po_qty = _num_col("PO_Qty")
    _pipe = _num_col("PO_Pipeline_Total")
    _ordered = _num_col("PO_Qty_Ordered")
    _ep_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    from .po_shared_cache import PO_MERGE_LOGIC_VERSION

    return {
        "ok": True,
        "columns": cols,
        "total_rows": int(len(po_df)),
        "po_merge_version": PO_MERGE_LOGIC_VERSION,
        "sales_through": sales_through,
        "planning_date": planning_out,
        "raise_ledger_rows": ledger_n,
        "ledger_auto_import": auto_msg,
        "stage_timings": stage_timings,
        "summary": {
            "new_po_qty_sum": int(_po_qty.sum()),
            "new_po_sku_count": int((_po_qty > 0).sum()),
            "pipeline_qty_sum": int(_pipe.sum()),
            "pipeline_sku_count": int((_pipe > 0).sum()),
            "sheet_po_ordered_sum": int(_ordered.sum()),
            "existing_po_applied": _ep_gen > 0 and int((_pipe > 0).sum()) > 0,
            "existing_po_generation": _ep_gen,
            "existing_po_filename": str(getattr(sess, "existing_po_filename", "") or ""),
        },
    }


def background_po_calculate(session_id: str, body: dict) -> None:
    import threading

    from ..session import store
    from .po_calculate_jobs import set_po_job

    sess = store.get(session_id)
    if sess is None:
        set_po_job(session_id, status="error", ok=False, message="Session not found.")
        return

    set_po_job(
        session_id,
        status="running",
        ok=True,
        progress=2,
        message="Calculating PO recommendations…",
    )
    sess.po_calculate_status = "running"
    sess.po_calculate_progress = 2
    sess.po_calculate_message = "Calculating PO recommendations…"

    try:
        from .po_result_spill import clear_spill

        clear_spill(session_id)
    except Exception:
        pass

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
        result = execute_po_calculate(
            sess,
            body,
            session_id=session_id,
            sync_sidecars=None,
            skip_hydrate=True,
        )
        sess.po_calculate_result = result
        if result.get("ok"):
            n = int(result.get("total_rows") or 0)
            msg = f"PO calculation complete ({n:,} rows)."
            cols: list[str] = []
            po_df = getattr(sess, "po_calculate_result_df", None)
            _persist_start = time.perf_counter()
            try:
                from .po_result_spill import spill_df

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

            try:
                from .po_shared_cache import save_shared_cache

                save_shared_cache(sess, body, po_df, result)
            except Exception:
                logger.exception("save_shared_cache after PO calculate failed")

            _persist_sec = time.perf_counter() - _persist_start
            from .po_stage_timer import po_stage_timing_enabled

            if po_stage_timing_enabled() and _persist_sec >= 0.05:
                logging.getLogger("perf.po").info(
                    "PO stages: persist cache/spill  %6.3fs", _persist_sec
                )

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
