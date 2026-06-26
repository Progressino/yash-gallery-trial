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
    job_id: Optional[str],
    pct: int,
    message: str,
) -> None:
    p = max(0, min(100, int(pct)))
    sess.po_calculate_progress = p
    sess.po_calculate_message = message
    if job_id:
        from .po_calculate_jobs import set_po_job

        set_po_job(job_id, status="running", ok=True, progress=p, message=message)


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
    job_id: Optional[str] = None,
    sync_sidecars: Optional[Callable[[], None]] = None,
    skip_hydrate: bool = False,
) -> dict[str, Any]:
    """Run full PO engine math and return the same payload as the legacy sync endpoint."""
    from ..services.po_engine import calculate_po_base
    from .po_pipeline import PoInputSnapshot, prepare_po_snapshot
    from .po_stage_timer import PoStageTimer

    stage_timer = PoStageTimer()
    progress_id = job_id or session_id

    def _progress(pct: int, message: str) -> None:
        _set_po_calculate_progress(sess, progress_id, pct, message)

    lookback = max(int(body.get("raise_ledger_lookback_days") or 14), 14)
    group_by_parent = bool(body.get("group_by_parent", False))

    try:
        import backend.main as _main

        threading.Thread(
            target=lambda: _main.merge_po_optional_sheets_into_warm_cache(sess),
            daemon=True,
            name="po-ledger-warm",
        ).start()
    except Exception:
        logger.exception("merge_po_optional_sheets_into_warm_cache after ledger load failed")

    _progress(4, "Building PO input snapshot…")
    snap_or_err = prepare_po_snapshot(
        sess,
        body,
        skip_hydrate=skip_hydrate,
        progress_cb=_progress,
        enforce_gate=True,
        session_id=session_id,
        sync_sidecars=sync_sidecars,
    )
    if isinstance(snap_or_err, dict):
        return snap_or_err
    snap: PoInputSnapshot = snap_or_err

    _period = int(body.get("period_days", 30))
    _ads_source = snap.sales_df
    _ads_label = snap.ads_label
    inv_df = snap.inv_df
    sku_mapping = snap.sku_mapping
    _existing_po = snap.existing_po_df
    _ledger = snap.po_raise_ledger_df
    _inv_history_for_calc = snap.inventory_history_df

    stage_timer.mark("inventory")
    stage_timer.mark("sales load")

    _progress(30, "Running PO calculation engine…")
    _hb_stop = threading.Event()

    def _calc_heartbeat() -> None:
        pct = 32
        while not _hb_stop.wait(20):
            pct = min(pct + 3, 78)
            _set_po_calculate_progress(
                sess,
                progress_id,
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
            sku_mapping=sku_mapping,
            group_by_parent=group_by_parent,
            existing_po_df=_existing_po,
            sku_status_df=snap.sku_status_df,
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
            po_return_overlay_df=snap.po_return_overlay_df,
            urgent_all_sizes_days=int(body.get("urgent_all_sizes_days", 45)),
            use_ly_fallback=bool(body.get("use_ly_fallback", True)),
            stage_timer=stage_timer,
        )
    except Exception as e:
        return {"ok": False, "message": f"PO calculation error: {e}"}
    finally:
        _hb_stop.set()

    if po_df is not None and not po_df.empty:
        try:
            from .po_quarterly_warmup import attach_quarterly_columns_to_po_df

            po_df = attach_quarterly_columns_to_po_df(
                po_df, sess, group_by_parent=group_by_parent, n_quarters=8
            )
        except Exception:
            logger.exception("attach_quarterly_columns_to_po_df failed (non-fatal)")

    if po_df is None or po_df.empty:
        return {"ok": False, "message": "PO result is empty."}

    _set_po_calculate_progress(sess, progress_id, 85, "Formatting PO results…")
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
        "ads_source": _ads_label,
        "snapshot_id": snap.snapshot_id,
        "pipeline_warnings": snap.warnings,
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


def background_po_calculate(job_id: str, session_id: str, body: dict) -> None:
    import threading

    from ..session import store
    from .po_calculate_jobs import set_po_job
    from .po_session_hydrate import hydrate_po_session_for_calculate
    from .shared_frames import session_inventory_variant, session_sales_df

    sess = store.get(session_id)
    if sess is None:
        set_po_job(job_id, status="error", ok=False, message="Session not found.")
        return

    set_po_job(
        job_id,
        status="running",
        ok=True,
        progress=2,
        message="Preparing PO calculation…",
    )
    sess.po_calculate_status = "running"
    sess.po_calculate_progress = 2
    sess.po_calculate_message = "Preparing PO calculation…"
    _po_wall_start = time.perf_counter()

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
        _set_po_calculate_progress(sess, job_id, 4, "Hydrating sales and inventory…")

        if body.get("use_shared_cache"):
            from .po_shared_cache import apply_shared_cache_to_session

            cached = apply_shared_cache_to_session(sess, session_id, body, job_id=job_id)
            if cached:
                sess.po_calculate_existing_po_generation = int(
                    getattr(sess, "existing_po_generation", 0) or 0
                )
                return

        try:
            import backend.main as _main

            _main.try_attach_shared_frames_fast(sess)
        except Exception:
            pass
        hydrate_po_session_for_calculate(sess)

        from .po_pipeline import check_calculate_gate

        gate = check_calculate_gate(sess)
        if not gate.get("calculate_allowed"):
            msg = gate["blockers"][0] if gate.get("blockers") else "PO inputs not ready."
            sess.po_calculate_status = "error"
            sess.po_calculate_progress = 0
            sess.po_calculate_message = msg
            sess.po_calculate_result = {
                "ok": False,
                "message": msg,
                "pipeline_blockers": gate.get("blockers") or [],
                "pipeline_warnings": gate.get("warnings") or [],
            }
            set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
            return

        sales_df = session_sales_df(sess)
        inv_df = session_inventory_variant(sess)
        if sales_df.empty:
            try:
                from .po_inputs import load_po_inputs

                pg_inputs = load_po_inputs(sess, body)
                if not pg_inputs.sales_df.empty:
                    sales_df = pg_inputs.sales_df
                if not pg_inputs.inventory_df_variant.empty:
                    inv_df = pg_inputs.inventory_df_variant
            except Exception:
                pass

        if sales_df.empty:
            msg = "Build Sales first (upload platforms, then POST /api/upload/build-sales)."
            sess.po_calculate_status = "error"
            sess.po_calculate_progress = 0
            sess.po_calculate_message = msg
            sess.po_calculate_result = {"ok": False, "message": msg}
            set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
            return
        if inv_df.empty:
            msg = "Upload Inventory first."
            sess.po_calculate_status = "error"
            sess.po_calculate_progress = 0
            sess.po_calculate_message = msg
            sess.po_calculate_result = {"ok": False, "message": msg}
            set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
            return

        _inv_n = int(len(getattr(sess, "daily_inventory_history_df", pd.DataFrame())))
        if _inv_n > 500_000:
            msg = f"Calculating PO (trimmed inventory window, {_inv_n:,} baseline rows)…"
            _set_po_calculate_progress(sess, job_id, 8, msg)
        result = execute_po_calculate(
            sess,
            body,
            session_id=session_id,
            job_id=job_id,
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
            _set_po_calculate_progress(sess, job_id, 100, msg)
            set_po_job(
                job_id,
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
                name=f"po-qtr-{job_id[:8]}",
            ).start()
            threading.Thread(
                target=_sync,
                daemon=True,
                name=f"po-save-{job_id[:8]}",
            ).start()
            try:
                from .perf_metrics import record_po_calculate

                record_po_calculate(
                    time.perf_counter() - _po_wall_start,
                    ok=True,
                    total_rows=n,
                    stage_timings=result.get("stage_timings"),
                    ads_source=str(result.get("ads_source") or ""),
                )
            except Exception:
                pass
        else:
            msg = result.get("message") or "PO calculation failed."
            sess.po_calculate_status = "error"
            sess.po_calculate_progress = 0
            sess.po_calculate_message = msg
            set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
            try:
                from .perf_metrics import record_po_calculate

                record_po_calculate(
                    time.perf_counter() - _po_wall_start,
                    ok=False,
                    stage_timings=result.get("stage_timings"),
                )
            except Exception:
                pass
    except Exception as e:
        logger.exception("background_po_calculate failed")
        msg = str(e)
        sess.po_calculate_status = "error"
        sess.po_calculate_progress = 0
        sess.po_calculate_message = msg
        sess.po_calculate_result = {"ok": False, "message": msg}
        set_po_job(job_id, status="error", ok=False, progress=0, message=msg)
        try:
            from .perf_metrics import record_po_calculate

            record_po_calculate(time.perf_counter() - _po_wall_start, ok=False)
        except Exception:
            pass
