"""
Upload router — all file ingestion endpoints.
"""
import asyncio
import gc
import io
import os
import re
import zipfile
import shutil
import subprocess
import tempfile
from collections import defaultdict
from typing import Any, Callable, List, Optional, Set, Tuple

import logging
import time
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from ..concurrency import HEAVY_EXECUTOR, run_heavy

from ..session import store, resume_auto_data_restore, AppSession
from ..models.schemas import UploadResponse
from ..services.sku_mapping import parse_sku_mapping
from ..services.mtr import load_mtr_from_zip, load_mtr_from_extracted_files, parse_mtr_csv
from ..services.myntra import load_myntra_from_zip
from ..services.meesho import (
    load_meesho_from_zip,
    looks_like_meesho_order_export,
    parse_meesho_order_export_xlsx,
)
from ..services.flipkart import load_flipkart_from_zip
from ..services.snapdeal import load_snapdeal_from_zip
from ..services.inventory import load_inventory_consolidated
from ..services.sales import build_sales_df, list_sku_mapping_gaps
from ..services.existing_po import parse_existing_po
from ..services.github_cache import save_cache_to_drive
from ..services.daily_store import save_daily_file, merge_platform_data as _merge_platform_data
from ..services.helpers import apply_dsr_segment_from_upload_filename


router = APIRouter()

_log = logging.getLogger(__name__)


def _sess_return_overlay(sess):
    """Optional PO return-sheet overlay (OMS_SKU + Return_Units) merged into unified sales."""
    ov = getattr(sess, "po_return_overlay_df", None)
    if ov is None or getattr(ov, "empty", True):
        return None
    return ov


def _session_data_changed(sess) -> None:
    """Undo pause_auto_data_restore after uploads / builds so Tier-3 merge and warm cache work again."""
    resume_auto_data_restore(sess)


def _upload_quality_from_merge(
    *,
    parsed_rows: int,
    pre_total: int,
    post_total: int,
    saved_rows: Optional[int] = None,
) -> tuple[int, int, list[str]]:
    parsed = max(0, int(parsed_rows))
    kept = max(0, int(post_total) - int(pre_total))
    dropped = max(0, parsed - kept)
    reasons: list[str] = []
    if saved_rows is not None and int(saved_rows) < parsed:
        reasons.append(f"{parsed - int(saved_rows)} rows blocked during daily-store save")
    if dropped > 0:
        reasons.append(f"{dropped} rows dropped as duplicate/overlap during merge")
    return kept, dropped, reasons


def _collect_validation_warnings(skipped: List[str]) -> list[str]:
    """
    Pull actionable data-quality warnings from parser skipped/partial messages so the
    user can fix source files and re-upload.
    """
    if not skipped:
        return []
    keys = (
        "no date", "date col", "invalid date", "all dates invalid",
        "no sku", "sku col", "no status", "status", "parse error",
        "no data", "unrecognised", "empty file", "partial",
    )
    out: list[str] = []
    for s in skipped:
        t = str(s).strip()
        if not t:
            continue
        tl = t.lower()
        if any(k in tl for k in keys):
            out.append(t)
    return out[:5]

_RAR_MAGIC = b"Rar!\x1a\x07"


def _extract_rar_files(rar_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    Extract all files from a RAR archive (incl. RAR5). Returns [(relative path, bytes), ...].

    Tries, in order: **bsdtar** (libarchive), **unar**, **7zz** / **7z**. Many Linux images
    have no ``bsdtar``, or libarchive without RAR — then Meesho CSVs inside a RAR never
    reach the dashboard. ``unar`` / p7zip are common Homebrew / apt packages.
    """
    tmpdir = tempfile.mkdtemp(prefix="rar_daily_")
    rar_path = os.path.join(tmpdir, "upload.rar")
    extract_root = os.path.join(tmpdir, "_out")

    def _collect(out_root: str) -> list[tuple[str, bytes]]:
        result: list[tuple[str, bytes]] = []
        if not os.path.isdir(out_root):
            return result
        for walk_root, _dirs, files in os.walk(out_root):
            for fname in files:
                if fname == "upload.rar":
                    continue
                full = os.path.join(walk_root, fname)
                try:
                    rel = os.path.relpath(full, out_root).replace("\\", "/")
                except ValueError:
                    continue
                with open(full, "rb") as fh:
                    result.append((rel, fh.read()))
        return result

    def _reset_out() -> None:
        shutil.rmtree(extract_root, ignore_errors=True)
        os.makedirs(extract_root, exist_ok=True)

    try:
        with open(rar_path, "wb") as f:
            f.write(rar_bytes)

        os.makedirs(extract_root, exist_ok=True)
        errors: list[str] = []

        bsdtar = shutil.which("bsdtar")
        if bsdtar:
            try:
                r = subprocess.run(
                    [bsdtar, "xf", rar_path, "-C", extract_root],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if r.returncode == 0 and got:
                    _log.info("RAR extract: bsdtar OK (%d files)", len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                if r.returncode != 0:
                    errors.append(f"bsdtar exit {r.returncode}: {err[:320]}")
                else:
                    errors.append("bsdtar: 0 files extracted")
            except subprocess.TimeoutExpired:
                errors.append("bsdtar: timeout")
            except Exception as e:
                errors.append(f"bsdtar: {e}")
            _reset_out()

        unar = shutil.which("unar")
        if unar:
            try:
                r = subprocess.run(
                    [unar, "-o", extract_root, "-f", rar_path],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if got:
                    _log.info("RAR extract: unar OK (%d files)", len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                errors.append(f"unar exit {r.returncode}: {err[:320]}")
            except subprocess.TimeoutExpired:
                errors.append("unar: timeout")
            except Exception as e:
                errors.append(f"unar: {e}")
            _reset_out()

        for bin_name in ("7zz", "7z"):
            seven = shutil.which(bin_name)
            if not seven:
                continue
            try:
                r = subprocess.run(
                    [seven, "x", "-y", f"-o{extract_root}", rar_path],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                got = _collect(extract_root)
                if got:
                    _log.info("RAR extract: %s OK (%d files)", bin_name, len(got))
                    return got
                err = (r.stderr or r.stdout or "").strip()
                errors.append(f"{bin_name} exit {r.returncode}: {err[:320]}")
            except subprocess.TimeoutExpired:
                errors.append(f"{bin_name}: timeout")
            except Exception as e:
                errors.append(f"{bin_name}: {e}")
            _reset_out()

        hint = (
            "Install one of: bsdtar (libarchive with RAR), unar (`brew install unar`), "
            "or p7zip (`apt install p7zip-full` / `brew install p7zip`). "
            "Or extract the RAR locally and upload the Meesho `Orders_*.csv` files as a ZIP."
        )
        raise ValueError(
            "Cannot extract RAR — " + (" | ".join(errors) if errors else "no files produced") + ". " + hint
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _rebuild_sales_sync(sess) -> tuple[bool, str]:
    """Rebuild combined sales_df from platform frames. Caller should hold ``_daily_restore_lock``."""
    try:
        from ..services.platform_session_window import trim_session_platform_frames

        trim_session_platform_frames(sess)
        sess.sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
            return_overlay_df=_sess_return_overlay(sess),
        )
        sess._quarterly_cache.clear()
        _session_data_changed(sess)
        rows = len(sess.sales_df)
        return True, f"Sales rebuilt ({rows:,} rows)."
    except Exception as e:
        _log.exception("sales rebuild failed")
        return False, str(e)


def _store_daily_auto_ingest_result(sess, payload: dict) -> None:
    """Persist last ingest outcome for coverage polling and completion UI."""
    sess.daily_auto_ingest_result = {
        "ok": bool(payload.get("ok")),
        "message": str(payload.get("message") or ""),
        "detected_platforms": list(payload.get("detected_platforms") or []),
        "warnings": list(payload.get("warnings") or []),
        "processed_files": int(payload.get("processed_files") or 0),
        "detected_files": int(payload.get("detected_files") or 0),
        "unknown_files": int(payload.get("unknown_files") or 0),
    }


def _run_daily_auto_sales_rebuild(session_id: str) -> None:
    """Background task: rebuild sales after Tier-3 ingest returned HTTP 200."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_message = "Rebuilding combined sales…"
    try:
        with sess._daily_restore_lock:
            ok, msg = _rebuild_sales_sync(sess)
        if ok:
            sess.sales_rebuild_status = "done"
            sess.sales_rebuild_message = msg
            _auto_save_cache(sess)
        else:
            sess.sales_rebuild_status = "error"
            sess.sales_rebuild_message = msg
    except Exception as e:
        sess.sales_rebuild_status = "error"
        sess.sales_rebuild_message = str(e)
        _log.exception("daily-auto background sales rebuild failed")
    finally:
        if getattr(sess, "daily_auto_ingest_status", "") == "done":
            sess.daily_auto_ingest_status = "idle"
            sess.daily_auto_ingest_message = ""
            sess.daily_auto_ingest_started = 0.0


@router.post("/daily-auto/reset-stuck")
async def reset_stuck_daily_upload(request: Request):
    """Clear a session stuck in daily ingest / sales rebuild (UI “Clear stuck upload”)."""
    sess = _get_session(request)
    cleared = _clear_stuck_daily_ingest(sess, force=True)
    return {
        "ok": True,
        "cleared": cleared,
        "daily_auto_ingest_status": getattr(sess, "daily_auto_ingest_status", "idle"),
        "message": (
            "Upload status reset — you can upload again. "
            "If files were already parsed, use Load Cache on the dashboard."
            if cleared
            else "No stuck upload on this session."
        ),
    }


def _queue_daily_auto_ingest(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Schedule ingest on the heavy executor (keeps asyncio responsive)."""
    HEAVY_EXECUTOR.submit(_run_daily_auto_ingest_pipeline, session_id, file_parts)


def _run_daily_auto_ingest_pipeline(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Parse Tier-3 upload off the request thread, then queue the usual sales rebuild."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    sess.daily_auto_ingest_status = "running"
    sess.daily_auto_ingest_started = time.time()
    sess.daily_auto_ingest_message = "Parsing daily files and merging into session…"
    try:
        payload = _process_daily_auto_sync(sess, file_parts, rebuild_sales=False)
    except Exception as e:
        _log.exception("daily-auto background ingest")
        err_payload = {
            "ok": False,
            "message": str(e),
            "detected_platforms": [],
            "warnings": [str(e)],
            "processed_files": len(file_parts),
            "detected_files": 0,
            "unknown_files": len(file_parts),
        }
        _store_daily_auto_ingest_result(sess, err_payload)
        sess.daily_auto_ingest_status = "error"
        sess.daily_auto_ingest_message = str(e)
        sess.sales_rebuild_status = "error"
        sess.sales_rebuild_message = f"Ingest failed: {e}"
        return

    _store_daily_auto_ingest_result(sess, payload)
    if not payload.get("ok"):
        sess.daily_auto_ingest_status = "error"
        sess.daily_auto_ingest_message = str(payload.get("message") or "Ingest failed")
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""
        return

    sess.daily_auto_ingest_status = "done"
    sess.daily_auto_ingest_message = str(payload.get("message") or "Ingest finished.")
    sess.daily_auto_ingest_started = 0.0

    if payload.get("sales_rebuild") == "pending":
        sess.sales_rebuild_status = "running"
        try:
            from ..db.forecast_session_pg import persist_session_bundle_thread_safe

            persist_session_bundle_thread_safe(session_id, sess)
        except Exception:
            _log.exception("PostgreSQL persist after Tier-3 background ingest")
        _run_daily_auto_sales_rebuild(session_id)
    else:
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""


def _auto_save_cache(sess) -> None:
    """Run in background after build-sales: silently push session to GitHub Releases."""
    session_data = {
        "sales_df":             sess.sales_df,
        "mtr_df":               sess.mtr_df,
        "meesho_df":            sess.meesho_df,
        "myntra_df":            sess.myntra_df,
        "flipkart_df":          sess.flipkart_df,
        "snapdeal_df":          sess.snapdeal_df,
        "sku_mapping":          sess.sku_mapping,
        "inventory_df_variant": sess.inventory_df_variant,
        "inventory_df_parent":  sess.inventory_df_parent,
        "existing_po_df":        sess.existing_po_df,
        "po_return_overlay_df": getattr(sess, "po_return_overlay_df", pd.DataFrame()),
    }
    ok, msg = save_cache_to_drive(session_data)
    if ok:
        _log.info("Auto-save cache succeeded: %s", msg)
    else:
        _log.warning("Auto-save cache skipped/failed: %s", msg)

    sid = getattr(sess, "_persist_sid", None)
    if sid:
        try:
            from ..db.forecast_session_pg import persist_session_bundle

            if persist_session_bundle(sid, sess):
                _log.info("PostgreSQL session snapshot saved (%s…)", sid[:8])
        except Exception:
            _log.exception("PostgreSQL session snapshot failed")


def _get_session(request: Request):
    sess = request.state.session
    if sess is None:
        raise HTTPException(status_code=500, detail="Session not initialised")
    return sess


def _resolve_upload_session(session_id: str) -> AppSession | None:
    """Session for background chunk finalize — must not silently skip ingest."""
    if not session_id:
        return None
    sess = store._sessions.get(session_id)
    if sess is not None:
        sess.last_accessed = time.time()
        return sess
    _log.warning(
        "Upload session %s… missing from RAM — attaching empty session for background ingest",
        session_id[:8],
    )
    sess = AppSession()
    sess.last_accessed = time.time()
    store._sessions[session_id] = sess
    return sess


def _clear_stuck_daily_ingest(sess: AppSession, *, force: bool = False) -> bool:
    """Reset a session stuck in daily_auto_ingest_status=running."""
    if getattr(sess, "daily_auto_ingest_status", "idle") != "running":
        return False
    started = float(getattr(sess, "daily_auto_ingest_started", 0) or 0)
    age = time.time() - started if started > 0 else 999999
    stuck_sec = int(os.environ.get("DAILY_INGEST_STUCK_SEC", "1800"))
    if not force and age < stuck_sec:
        return False
    sess.daily_auto_ingest_status = "error"
    sess.daily_auto_ingest_message = (
        "Previous daily upload did not finish (timed out or server was busy). "
        "Your files may still be saved on the server — try Load Cache, or upload again."
    )
    _store_daily_auto_ingest_result(
        sess,
        {
            "ok": False,
            "message": sess.daily_auto_ingest_message,
            "detected_platforms": [],
            "warnings": [sess.daily_auto_ingest_message],
            "processed_files": 0,
            "detected_files": 0,
            "unknown_files": 0,
        },
    )
    if getattr(sess, "sales_rebuild_status", "idle") == "running":
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""
    return True


async def _session_lock_apply(sess, fn: Callable[[], Any]) -> Any:
    """
    Run ``fn`` while holding ``sess._daily_restore_lock`` on a worker thread.

    Tier-3 daily-auto ingest already mutates the same DataFrames under this lock.
    Tier-1 uploads used to run on the asyncio thread without the lock, which could
    interleave with Tier-3 or with a second tab and corrupt *this session's* frames.
    Different browser sessions still run in parallel (separate locks).

    Using ``asyncio.to_thread`` avoids blocking the event loop while waiting for
    the lock or while running CPU-heavy parse/merge work.
    """
    return await asyncio.to_thread(_session_lock_apply_sync, sess, fn)


def _session_lock_apply_sync(sess, fn: Callable[[], Any]) -> Any:
    with sess._daily_restore_lock:
        return fn()


# ── SKU Mapping ───────────────────────────────────────────────

@router.post("/sku-mapping", response_model=UploadResponse)
async def upload_sku_mapping(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()

        def work():
            mapping = parse_sku_mapping(file_bytes)
            sess.sku_mapping = mapping

            had_platform = (
                not sess.mtr_df.empty
                or not sess.myntra_df.empty
                or not sess.meesho_df.empty
                or not sess.flipkart_df.empty
                or not sess.snapdeal_df.empty
            )
            if had_platform:
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=mapping,
                    return_overlay_df=_sess_return_overlay(sess),
                )

            gaps = list_sku_mapping_gaps(sess.sales_df, mapping)
            msg = f"SKU mapping loaded: {len(mapping):,} entries"
            if had_platform:
                msg += f"; sales rebuilt ({len(sess.sales_df):,} rows)"
            if gaps:
                msg += (
                    f". Warning: {len(gaps)} SKU(s) in sales are not in this map "
                    f"(as seller key or OMS value) — add them to the master or fix typos."
                )

            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=msg,
                sku_count=len(mapping),
                unmapped_skus=gaps or None,
            ), had_platform

        resp, had_platform = await _session_lock_apply(sess, work)
        if had_platform:
            background_tasks.add_task(_auto_save_cache, sess)
        return resp
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse SKU mapping: {e}")


# ── Amazon MTR ────────────────────────────────────────────────

@router.post("/mtr", response_model=UploadResponse)
async def upload_mtr(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    raw = await file.read()
    orig_name = file.filename
    try:

        def work():
            fn = (orig_name or "").lower()
            if raw[:6] == _RAR_MAGIC or fn.endswith(".rar"):
                inner = _extract_rar_files(raw)
                df, csv_count, skipped = load_mtr_from_extracted_files(inner)
            else:
                df, csv_count, skipped = load_mtr_from_zip(raw)

            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No valid CSV files found. Issues: {'; '.join(skipped[:5])}",
                )

            pre_total = len(sess.mtr_df)
            _fd, saved_rows = save_daily_file("amazon", orig_name or "mtr-upload.zip", df)
            sess.mtr_df = _merge_platform_data(
                sess.mtr_df, df, "amazon", source_filename=orig_name or None,
            )
            total = len(sess.mtr_df)
            kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
            )
            val_warn = _collect_validation_warnings(skipped)
            years = sorted(sess.mtr_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=(
                    f"Amazon MTR: added {kept_rows:,} rows ({csv_count} files). "
                    f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                    + (
                        f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                        if dropped_rows > 0 else ""
                    )
                    + (
                        f" Validation issues: {' | '.join(val_warn[:3])}. "
                        "Please fix file columns/values and re-upload."
                        if val_warn else ""
                    )
                ),
                rows=total,
                parsed_rows=len(df),
                kept_rows=kept_rows,
                dropped_rows=dropped_rows,
                dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None,
                years=years,
            )

        resp = await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse MTR archive (ZIP/RAR): {e}")
    finally:
        del raw
        gc.collect()
    if resp.ok:
        background_tasks.add_task(_auto_save_cache, sess)
    return resp


# ── Myntra ────────────────────────────────────────────────────

@router.post("/myntra", response_model=UploadResponse)
async def upload_myntra(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    zip_bytes = await file.read()
    orig_fn = file.filename
    try:

        def work():
            if not sess.sku_mapping:
                return UploadResponse(ok=False, message="Upload SKU Mapping first.")
            df, csv_count, skipped = load_myntra_from_zip(
                zip_bytes, sess.sku_mapping, orig_fn or None,
            )

            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )

            pre_total = len(sess.myntra_df)
            _fd, saved_rows = save_daily_file("myntra", orig_fn or "myntra-upload.zip", df)
            sess.myntra_df = _merge_platform_data(
                sess.myntra_df, df, "myntra", source_filename=orig_fn or None,
            )
            total = len(sess.myntra_df)
            years = sorted(sess.myntra_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            _session_data_changed(sess)
            quality_line = next((s for s in skipped if str(s).startswith("IMPORT_QUALITY:")), "")
            parsed_rows = None
            kept_rows = int(len(df))
            dropped_rows = None
            dropped_reasons: list[str] = []
            if quality_line:
                import re as _re
                m_parsed = _re.search(r"parsed=(\d+)", quality_line)
                m_kept = _re.search(r"kept=(\d+)", quality_line)
                m_drop = _re.search(r"dropped=(\d+)", quality_line)
                if m_parsed:
                    parsed_rows = int(m_parsed.group(1))
                if m_kept:
                    kept_rows = int(m_kept.group(1))
                if m_drop:
                    dropped_rows = int(m_drop.group(1))
                if ";" in quality_line:
                    tail = quality_line.split(";", 1)[1].strip()
                    dropped_reasons = [x.strip() for x in tail.split(";") if x.strip()]
            merge_kept, merge_dropped, merge_reasons = _upload_quality_from_merge(
                parsed_rows=(parsed_rows if parsed_rows is not None else len(df)),
                pre_total=pre_total,
                post_total=total,
                saved_rows=saved_rows,
            )
            kept_rows = min(int(kept_rows), int(merge_kept))
            dropped_rows = max(int(dropped_rows or 0), int(merge_dropped))
            for rr in merge_reasons:
                if rr not in dropped_reasons:
                    dropped_reasons.append(rr)
            val_warn = _collect_validation_warnings(skipped)

            extra_warn = ""
            if dropped_rows and dropped_rows > 0:
                reason_txt = f" ({'; '.join(dropped_reasons)})" if dropped_reasons else ""
                extra_warn = (
                    f" Warning: {dropped_rows:,} rows were dropped during import{reason_txt}. "
                    "Please fix source rows before relying on dashboard totals."
                )

            return UploadResponse(
                ok=True,
                message=(
                    f"Myntra: added {kept_rows:,} rows ({csv_count} CSVs). "
                    f"Parsed: {(parsed_rows if parsed_rows is not None else len(df)):,}, "
                    f"Kept: {kept_rows:,}. Total: {total:,} rows."
                    f"{extra_warn}"
                    + (
                        f" Validation issues: {' | '.join(val_warn[:3])}. "
                        "Please fix file columns/values and re-upload."
                        if val_warn else ""
                    )
                ),
                rows=total,
                parsed_rows=parsed_rows,
                kept_rows=kept_rows,
                dropped_rows=dropped_rows,
                dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None,
                years=years,
            )

        resp = await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Myntra ZIP: {e}")
    finally:
        del zip_bytes
        gc.collect()
    if resp.ok:
        background_tasks.add_task(_auto_save_cache, sess)
    return resp


# ── Meesho ────────────────────────────────────────────────────

@router.post("/meesho", response_model=UploadResponse)
async def upload_meesho(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    raw_bytes = await file.read()
    fname = (file.filename or "").lower()
    display_name = file.filename
    try:

        def work():
            # Unified sales Excel export (TxnDate, Sku, Sku.1 listing fallback) — must run before PK/ZIP logic
            if fname.endswith((".xlsx", ".xls")):
                df, msg = parse_meesho_order_export_xlsx(raw_bytes)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(
                        df, display_name or None, "Meesho",
                    )
                    pre_total = len(sess.meesho_df)
                    _fd, saved_rows = save_daily_file("meesho", display_name or "meesho-order.xlsx", df)
                    sess.meesho_df = _merge_platform_data(
                        sess.meesho_df, df, "meesho", source_filename=display_name or None,
                    )
                    sess.sales_df = build_sales_df(
                        mtr_df=sess.mtr_df,
                        myntra_df=sess.myntra_df,
                        meesho_df=sess.meesho_df,
                        flipkart_df=sess.flipkart_df,
                        snapdeal_df=sess.snapdeal_df,
                        sku_mapping=sess.sku_mapping,
                        return_overlay_df=_sess_return_overlay(sess),
                    )
                    total = len(sess.meesho_df)
                    years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
                    skus = int((sess.meesho_df["SKU"].astype(str).str.strip() != "").sum())
                    kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                        parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
                    )
                    val_warn: list[str] = []
                    _session_data_changed(sess)
                    return UploadResponse(
                        ok=True,
                        message=(
                            f"Meesho order export (Excel): added {kept_rows:,} rows ({skus:,} with SKU). "
                            f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                            + (
                                f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                                if dropped_rows > 0 else ""
                            )
                            + (
                                f" Validation issues: {' | '.join(val_warn[:3])}. "
                                "Please fix file columns/values and re-upload."
                                if val_warn else ""
                            )
                        ),
                        rows=total,
                        parsed_rows=len(df),
                        kept_rows=kept_rows,
                        dropped_rows=dropped_rows,
                        dropped_reasons=dropped_reasons or None,
                        validation_warnings=val_warn or None,
                        years=years,
                    )
                return UploadResponse(
                    ok=False,
                    message=(
                        f"Excel is not a recognised Meesho sales export ({msg}). "
                        "Expected columns TxnDate, Transaction Type, Sku, Quantity (and optional Sku.1 listing). "
                        "Or upload supplier Order CSV / monthly ZIP."
                    ),
                )

            # Auto-detect: CSV order report vs ZIP (TCS / monthly archive)
            is_csv = fname.endswith(".csv") or (not fname.endswith(".zip") and raw_bytes[:3] != b"PK\x03")

            if is_csv:
                from ..services.meesho import parse_meesho_csv
                df, msg = parse_meesho_csv(raw_bytes)
                if df.empty:
                    return UploadResponse(ok=False, message=f"Meesho CSV parse error: {msg}")
                df = apply_dsr_segment_from_upload_filename(df, display_name or None, "Meesho")
                pre_total = len(sess.meesho_df)
                _fd, saved_rows = save_daily_file("meesho", display_name or "meesho-orders.csv", df)
                sess.meesho_df = _merge_platform_data(
                    sess.meesho_df, df, "meesho", source_filename=display_name or None,
                )
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                    return_overlay_df=_sess_return_overlay(sess),
                )
                total = len(sess.meesho_df)
                years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
                skus = int((sess.meesho_df["SKU"].astype(str).str.strip() != "").sum()) if "SKU" in sess.meesho_df.columns else 0
                kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                    parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
                )
                val_warn: list[str] = []
                _session_data_changed(sess)
                return UploadResponse(
                    ok=True,
                    message=(
                        f"Meesho Order CSV: added {kept_rows:,} rows ({skus:,} with SKU). "
                        f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                        + (
                            f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                            if dropped_rows > 0 else ""
                        )
                        + (
                            f" Validation issues: {' | '.join(val_warn[:3])}. "
                            "Please fix file columns/values and re-upload."
                            if val_warn else ""
                        )
                    ),
                    rows=total,
                    parsed_rows=len(df),
                    kept_rows=kept_rows,
                    dropped_rows=dropped_rows,
                    dropped_reasons=dropped_reasons or None,
                    validation_warnings=val_warn or None,
                    years=years,
                )
            df, zip_count, skipped = load_meesho_from_zip(
                raw_bytes, source_filename=display_name or None,
            )
            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )
            pre_total = len(sess.meesho_df)
            _fd, saved_rows = save_daily_file("meesho", display_name or "meesho-upload.zip", df)
            sess.meesho_df = _merge_platform_data(
                sess.meesho_df, df, "meesho", source_filename=display_name or None,
            )
            sess.sales_df = build_sales_df(
                mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping,
                return_overlay_df=_sess_return_overlay(sess),
            )
            total = len(sess.meesho_df)
            years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
            )
            val_warn = _collect_validation_warnings(skipped)
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=(
                    f"Meesho: added {kept_rows:,} rows ({zip_count} monthly ZIPs). "
                    f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                    + (
                        f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                        if dropped_rows > 0 else ""
                    )
                    + (
                        f" Validation issues: {' | '.join(val_warn[:3])}. "
                        "Please fix file columns/values and re-upload."
                        if val_warn else ""
                    )
                ),
                rows=total,
                parsed_rows=len(df),
                kept_rows=kept_rows,
                dropped_rows=dropped_rows,
                dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None,
                years=years,
            )

        resp = await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Meesho file: {e}")
    finally:
        del raw_bytes
        gc.collect()
    if resp.ok:
        background_tasks.add_task(_auto_save_cache, sess)
    return resp


# ── Flipkart ──────────────────────────────────────────────────

@router.post("/flipkart", response_model=UploadResponse)
async def upload_flipkart(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    tmp_path: Optional[str] = None
    orig_fn = file.filename
    resp: Optional[UploadResponse] = None
    try:
        # Stream to disk so multi-year Tier-1 ZIPs are not held wholly in RAM (reduces OOM → 502).
        _fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(_fd)
        chunk_size = 8 * 1024 * 1024
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)

        def work():
            if not sess.sku_mapping:
                return UploadResponse(ok=False, message="Upload SKU Mapping first.")
            df, xlsx_count, skipped = load_flipkart_from_zip(
                tmp_path, sess.sku_mapping, source_filename=orig_fn or None,
            )
            gc.collect()

            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )

            pre_total = len(sess.flipkart_df)
            _fd2, saved_rows = save_daily_file("flipkart", orig_fn or "flipkart-upload.zip", df)
            sess.flipkart_df = _merge_platform_data(
                sess.flipkart_df, df, "flipkart", source_filename=orig_fn or None,
            )
            total = len(sess.flipkart_df)
            kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
            )
            val_warn = _collect_validation_warnings(skipped)
            years = sorted(sess.flipkart_df["Date"].dt.year.dropna().unique().astype(int).tolist())
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=(
                    f"Flipkart: added {kept_rows:,} rows ({xlsx_count} files). "
                    f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                    + (
                        f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                        if dropped_rows > 0 else ""
                    )
                    + (
                        f" Validation issues: {' | '.join(val_warn[:3])}. "
                        "Please fix file columns/values and re-upload."
                        if val_warn else ""
                    )
                ),
                rows=total,
                parsed_rows=len(df),
                kept_rows=kept_rows,
                dropped_rows=dropped_rows,
                dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None,
                years=years,
            )

        resp = await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Flipkart ZIP: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if resp and resp.ok:
        background_tasks.add_task(_auto_save_cache, sess)
    return resp


# ── Snapdeal ──────────────────────────────────────────────────

@router.post("/snapdeal", response_model=UploadResponse)
async def upload_snapdeal(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    sess = _get_session(request)
    zip_bytes = await file.read()
    display = file.filename or "upload"
    snap_fn = file.filename
    try:

        def work():
            df, file_count, skipped, parse_info = load_snapdeal_from_zip(zip_bytes, sess.sku_mapping, display)

            # Store parse diagnostics (merge with existing)
            sess.snapdeal_parse_info.update(parse_info)

            if df.empty:
                return UploadResponse(
                    ok=False,
                    message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )

            pre_total = len(sess.snapdeal_df)
            _fd, saved_rows = save_daily_file("snapdeal", snap_fn or "snapdeal-upload.zip", df)

            # Snapdeal may not have OrderId — dedup on all columns
            if sess.snapdeal_df.empty:
                sess.snapdeal_df = df
            else:
                sess.snapdeal_df = pd.concat([sess.snapdeal_df, df], ignore_index=True).drop_duplicates()
            post_total = len(sess.snapdeal_df)
            kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                parsed_rows=len(df), pre_total=pre_total, post_total=post_total, saved_rows=saved_rows,
            )
            val_warn = _collect_validation_warnings(skipped)

            # Rebuild sales_df if SKU mapping is loaded
            if sess.sku_mapping:
                from ..services.sales import build_sales_df
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df,
                    myntra_df=sess.myntra_df,
                    meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df,
                    snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping,
                    return_overlay_df=_sess_return_overlay(sess),
                )
                sess._quarterly_cache.clear()

            years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=(
                    f"Snapdeal loaded: added {kept_rows:,} rows from {file_count} file(s). "
                    f"Parsed: {len(df):,}, Kept: {kept_rows:,}."
                    + (
                        f" Warning: {dropped_rows:,} rows dropped ({'; '.join(dropped_reasons[:2])})."
                        if dropped_rows > 0 else ""
                    )
                    + (
                        f" Validation issues: {' | '.join(val_warn[:3])}. "
                        "Please fix file columns/values and re-upload."
                        if val_warn else ""
                    )
                    + (f" Warnings: {'; '.join(skipped[:3])}" if skipped else "")
                ),
                rows=post_total,
                parsed_rows=len(df),
                kept_rows=kept_rows,
                dropped_rows=dropped_rows,
                dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None,
                years=years,
            )

        resp = await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse Snapdeal ZIP: {e}")
    finally:
        del zip_bytes
        gc.collect()
    if resp.ok:
        background_tasks.add_task(_auto_save_cache, sess)
    return resp


# ── Inventory ─────────────────────────────────────────────────

def _detect_inventory_type(filename: str, content_bytes: bytes) -> str:
    """
    Guess inventory file type from filename and first ~2 KB of content.
    Returns: 'oms', 'flipkart', 'myntra', 'amazon', 'rar'
    """
    fn = filename.lower()

    # RAR by magic bytes or extension
    if content_bytes[:6] == _RAR_MAGIC or fn.endswith(".rar"):
        return "rar"

    # Filename hints
    if "myntra" in fn:
        return "myntra"
    if "flipkart" in fn or fn.startswith("fk"):
        return "flipkart"
    if "amazon" in fn or fn.startswith("amz"):
        return "amazon"

    # Content-based detection
    try:
        # XLSX files are ZIP containers and won't decode to useful text.
        # Probe headers via pandas to classify Flipkart/Myntra/OMS uploads.
        if filename.lower().endswith((".xlsx", ".xls")) or content_bytes[:4] == b"PK\x03\x04":
            try:
                xdf = pd.read_excel(io.BytesIO(content_bytes), nrows=5)
                cols = [str(c).strip().lower() for c in xdf.columns]
                joined = " | ".join(cols)
                if "seller sku code" in joined or ("style id" in joined and "inventory count" in joined):
                    return "myntra"
                if "live on website" in joined or ("sku" in joined and "available to promise" in joined):
                    return "flipkart"
                if "item skucode" in joined or "buffer stock" in joined:
                    return "oms"
            except Exception:
                pass

        text = content_bytes[:2000].decode("utf-8", errors="ignore").lower()
        if "item skucode" in text or "buffer stock" in text:
            return "oms"
        if "combo sku code" in text and "combo qty stock" in text:
            return "oms"
        if "seller sku code" in text or ("style id" in text and "inventory count" in text):
            return "myntra"
        if "live on website" in text:
            return "flipkart"
        if "msku" in text and "ending warehouse balance" in text:
            return "amazon"
        if "merchant sku" in text and "shipped" in text:
            return "amazon"
    except Exception:
        pass

    return "oms"  # safe default


def _classify_inventory_file_parts(
    file_parts: list[tuple[str, bytes]],
) -> tuple[list[bytes], list[bytes] | None, list[bytes] | None, bytes | None, list[str]]:
    oms_bytes_list: list[bytes] = []
    fk_bytes_list: list[bytes] = []
    myntra_bytes_list: list[bytes] = []
    amz_bytes = None
    detected: list[str] = []
    for fname, raw in file_parts:
        inv_type = _detect_inventory_type(fname, raw)
        if inv_type == "rar":
            amz_bytes = raw
            detected.append(f"RAR archive ({fname})")
        elif inv_type == "flipkart":
            fk_bytes_list.append(raw)
            detected.append(f"Flipkart ({fname})")
        elif inv_type == "myntra":
            myntra_bytes_list.append(raw)
            detected.append(f"Myntra ({fname})")
        elif inv_type == "amazon":
            amz_bytes = raw
            detected.append(f"Amazon ({fname})")
        else:
            oms_bytes_list.append(raw)
            detected.append(f"OMS ({fname})")
    fk_bytes = fk_bytes_list or None
    myntra_bytes = myntra_bytes_list if myntra_bytes_list else None
    return oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes, detected


async def _run_inventory_auto_from_parts(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Parse assembled inventory file bytes on a worker thread."""
    sess = store._sessions.get(session_id)
    if sess is None:
        return
    try:
        oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes, detected = _classify_inventory_file_parts(file_parts)
    except Exception as e:
        _log.exception("inventory-auto classify files")
        sess.inventory_upload_status = "error"
        sess.inventory_upload_message = str(e)
        sess.inventory_upload_result = {"ok": False, "message": str(e)}
        return

    if not any([oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes]):
        sess.inventory_upload_status = "error"
        sess.inventory_upload_message = "No files provided."
        sess.inventory_upload_result = {"ok": False, "message": "No files provided."}
        return

    def work():
        df_variant, debug = load_inventory_consolidated(
            oms_bytes_list or None, fk_bytes, myntra_bytes, amz_bytes, sess.sku_mapping,
            group_by_parent=False, return_debug=True,
        )
        try:
            from ..services.helpers import get_parent_sku
            df_parent = df_variant.copy()
            inv_cols = [
                c for c in df_parent.columns
                if c.endswith("_Inventory") or c.endswith("_Live") or c.endswith("_InTransit")
                   or c in ("Buffer_Stock", "Marketplace_Total", "Total_Inventory")
            ]
            df_parent["Parent_SKU"] = df_parent["OMS_SKU"].apply(get_parent_sku)
            df_parent = (
                df_parent.groupby("Parent_SKU")[inv_cols].sum()
                .reset_index()
                .rename(columns={"Parent_SKU": "OMS_SKU"})
            )
        except Exception:
            df_parent = df_variant
        sess.inventory_df_variant = df_variant
        sess.inventory_df_parent = df_parent
        sess.inventory_debug = debug
        _session_data_changed(sess)
        parts = [f"{len(df_variant):,} total SKUs"]
        for src, info in debug.items():
            parts.append(f"{src}: {info}")
        payload = {
            "ok": True,
            "message": " | ".join(parts),
            "rows": len(df_variant),
            "debug": debug,
            "detected": detected,
            "v": "inv-v9",
        }
        sess.inventory_upload_result = payload
        sess.inventory_upload_status = "done"
        sess.inventory_upload_message = payload["message"]
        return payload

    try:
        await asyncio.to_thread(_session_lock_apply_sync, sess, work)
        _auto_save_cache(sess)
    except Exception as e:
        _log.exception("inventory-auto parse")
        sess.inventory_upload_status = "error"
        sess.inventory_upload_message = f"Parse error: {e}"
        sess.inventory_upload_result = {"ok": False, "message": f"Parse error: {e}"}


_INVENTORY_AUTO_DIRECT_MAX_FILES = int(os.environ.get("INVENTORY_AUTO_DIRECT_MAX_FILES", "3"))


@router.post("/inventory-auto")
async def upload_inventory_auto(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Drop any mix of inventory files — type auto-detected from filename/content.
    Accepts: OMS inventory CSV/XLSX, Flipkart CSV, Myntra CSV, Amazon RAR/CSV.
    """
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    if not files:
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return JSONResponse(
            content={
                "ok": False,
                "message": "An inventory upload is still processing. Wait for it to finish, then try again.",
            }
        )

    if len(files) > _INVENTORY_AUTO_DIRECT_MAX_FILES:
        return JSONResponse(
            content={
                "ok": False,
                "message": (
                    f"Too many files ({len(files)}) for a single upload. "
                    "Refresh the page — the app will upload in smaller chunks automatically."
                ),
                "require_chunked": True,
            }
        )

    file_parts: list[tuple[str, bytes]] = []
    try:
        for f in files:
            file_parts.append((f.filename or "upload", await f.read()))
    except Exception as e:
        _log.exception("inventory-auto read files in request")
        return JSONResponse(content={"ok": False, "message": str(e)})

    sid = getattr(request.state, "session_id", None)
    if sid:

        def mark_async_start():
            sess.inventory_upload_status = "running"
            sess.inventory_upload_message = "Parsing inventory files and merging snapshot…"
            sess.inventory_upload_result = {}

        await _session_lock_apply(sess, mark_async_start)
        background_tasks.add_task(_run_inventory_auto_from_parts, sid, file_parts)
        return JSONResponse(
            content={
                "ok": True,
                "ingest_async": True,
                "message": (
                    "Upload accepted — parsing inventory on the server. "
                    "Status updates below; you can stay on this page."
                ),
            }
        )

    oms_bytes_list: list[bytes] = []
    fk_bytes_list: list[bytes] = []
    myntra_bytes_list: list[bytes] = []
    amz_bytes = None
    detected: list[str] = []
    for file in files:
        raw = await file.read()
        fname = file.filename or ""
        inv_type = _detect_inventory_type(fname, raw)
        if inv_type == "rar":
            amz_bytes = raw
            detected.append(f"RAR archive ({fname})")
        elif inv_type == "flipkart":
            fk_bytes_list.append(raw)
            detected.append(f"Flipkart ({fname})")
        elif inv_type == "myntra":
            myntra_bytes_list.append(raw)
            detected.append(f"Myntra ({fname})")
        elif inv_type == "amazon":
            amz_bytes = raw
            detected.append(f"Amazon ({fname})")
        else:
            oms_bytes_list.append(raw)
            detected.append(f"OMS ({fname})")

    fk_bytes = fk_bytes_list or None
    myntra_bytes = myntra_bytes_list if myntra_bytes_list else None
    if not any([oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes]):
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    try:
        def work():
            df_variant, debug = load_inventory_consolidated(
                oms_bytes_list or None, fk_bytes, myntra_bytes, amz_bytes, sess.sku_mapping,
                group_by_parent=False, return_debug=True,
            )
            try:
                from ..services.helpers import get_parent_sku
                df_parent = df_variant.copy()
                inv_cols = [
                    c for c in df_parent.columns
                    if c.endswith("_Inventory") or c.endswith("_Live") or c.endswith("_InTransit")
                       or c in ("Buffer_Stock", "Marketplace_Total", "Total_Inventory")
                ]
                df_parent["Parent_SKU"] = df_parent["OMS_SKU"].apply(get_parent_sku)
                df_parent = (
                    df_parent.groupby("Parent_SKU")[inv_cols].sum()
                    .reset_index()
                    .rename(columns={"Parent_SKU": "OMS_SKU"})
                )
            except Exception:
                df_parent = df_variant
            sess.inventory_df_variant = df_variant
            sess.inventory_df_parent = df_parent
            sess.inventory_debug = debug
            _session_data_changed(sess)
            parts = [f"{len(df_variant):,} total SKUs"]
            for src, info in debug.items():
                parts.append(f"{src}: {info}")
            return {
                "ok": True,
                "message": " | ".join(parts),
                "rows": len(df_variant),
                "debug": debug,
                "detected": detected,
                "v": "inv-v8",
            }

        data = await _session_lock_apply(sess, work)
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    background_tasks.add_task(_auto_save_cache, sess)
    return JSONResponse(content=data)


@router.post("/inventory")
async def upload_inventory(
    request: Request,
    background_tasks: BackgroundTasks,
    oms_file:    List[UploadFile] = File(default=[]),
    fk_file:     Optional[UploadFile] = File(None),
    myntra_file: Optional[UploadFile] = File(None),
    amz_file:    Optional[UploadFile] = File(None),
):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    oms_b_list = [await f.read() for f in oms_file] if oms_file else []
    fk_b     = await fk_file.read()    if fk_file     else None
    myntra_b = await myntra_file.read() if myntra_file else None
    amz_b    = await amz_file.read()   if amz_file    else None

    if not any([oms_b_list, fk_b, myntra_b, amz_b]):
        return JSONResponse(content={"ok": False, "message": "No files provided."})

    try:

        def work():
            df_variant, debug = load_inventory_consolidated(
                oms_b_list or None, fk_b, myntra_b, amz_b, sess.sku_mapping,
                group_by_parent=False, return_debug=True,
            )
            df_parent = load_inventory_consolidated(
                oms_b_list or None, fk_b, myntra_b, amz_b, sess.sku_mapping, group_by_parent=True
            )
            sess.inventory_df_variant = df_variant
            sess.inventory_df_parent = df_parent
            _session_data_changed(sess)
            parts = [f"{len(df_variant):,} total SKUs"]
            for src, info in debug.items():
                parts.append(f"{src}: {info}")
            return {
                "ok": True,
                "message": " | ".join(parts),
                "rows": len(df_variant),
                "debug": debug,
            }

        data = await _session_lock_apply(sess, work)
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    background_tasks.add_task(_auto_save_cache, sess)
    return JSONResponse(content=data)


# ── Amazon B2C (single CSV) ────────────────────────────────────

@router.post("/amazon-b2c", response_model=UploadResponse)
async def upload_amazon_b2c(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    csv_bytes = await file.read()
    orig_fn = file.filename
    try:

        def work():
            df, msg = parse_mtr_csv(csv_bytes, orig_fn or "b2c.csv")

            if df.empty:
                return UploadResponse(ok=False, message=f"B2C parse failed: {msg}")

            sess.mtr_df = _merge_platform_data(
                sess.mtr_df, df, "amazon", source_filename=orig_fn or None,
            )
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=f"Amazon B2C loaded: {len(df):,} rows. {msg if msg != 'OK' else ''}".strip(),
                rows=len(df),
            )

        return await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse B2C CSV: {e}")
    finally:
        del csv_bytes
        gc.collect()


# ── Amazon B2B (single CSV) ────────────────────────────────────

@router.post("/amazon-b2b", response_model=UploadResponse)
async def upload_amazon_b2b(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    csv_bytes = await file.read()
    orig_fn = file.filename
    try:

        def work():
            df, msg = parse_mtr_csv(csv_bytes, orig_fn or "b2b.csv")

            if df.empty:
                return UploadResponse(ok=False, message=f"B2B parse failed: {msg}")

            sess.mtr_df = _merge_platform_data(
                sess.mtr_df, df, "amazon", source_filename=orig_fn or None,
            )
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=f"Amazon B2B loaded: {len(df):,} rows. {msg if msg != 'OK' else ''}".strip(),
                rows=len(df),
            )

        return await _session_lock_apply(sess, work)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse B2B CSV: {e}")
    finally:
        del csv_bytes
        gc.collect()


# ── Existing PO Sheet ──────────────────────────────────────────

@router.post("/existing-po", response_model=UploadResponse)
async def upload_existing_po(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()
        orig_fn = file.filename or "existing_po.xlsx"

        def work():
            df = parse_existing_po(file_bytes, orig_fn)
            sess.existing_po_df = df
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=f"Existing PO loaded: {len(df):,} SKUs with pipeline quantities.",
                rows=len(df),
            )

        return await _session_lock_apply(sess, work)
    except Exception as e:
        _log.warning("existing-po parse failed: %s", e, exc_info=True)
        return UploadResponse(ok=False, message=f"Failed to parse Existing PO: {e}")


@router.post("/cogs", response_model=UploadResponse)
async def upload_cogs(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    try:
        from ..services.finance import parse_cogs_sheet
        file_bytes = await file.read()
        orig_fn = file.filename or "cogs.xlsx"

        def work():
            df = parse_cogs_sheet(file_bytes, orig_fn)
            sess.cogs_df = df
            _session_data_changed(sess)
            return UploadResponse(ok=True, message=f"COGS sheet loaded: {len(df):,} SKUs.", rows=len(df))

        return await _session_lock_apply(sess, work)
    except ValueError as e:
        return UploadResponse(ok=False, message=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse COGS sheet: {e}")


# ── Daily Orders — Auto-detect (drop all files, we figure it out) ─

def _zip_member_paths(file_bytes: bytes) -> tuple[list[str], str]:
    """Non-directory ZIP members (no __MACOSX); returns (paths, space-joined lowercase)."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [
                n for n in zf.namelist()
                if "__MACOSX" not in n and not n.endswith("/")
            ]
        return names, " ".join(n.lower() for n in names)
    except Exception:
        return [], ""


def _zip_is_meesho_monthly(file_bytes: bytes, outer_fn: str) -> bool:
    if "meesho" in outer_fn.lower():
        return True
    _names, joined = _zip_member_paths(file_bytes)
    return "tcs_sales" in joined or "forwardreports" in joined


def _zip_is_amazon_mtr_master(file_bytes: bytes) -> bool:
    try:
        probe, n_csv, _sk = load_mtr_from_zip(file_bytes)
        return not probe.empty and n_csv > 0
    except Exception:
        return False


def _detect_platform_zip(file_bytes: bytes, fn: str) -> str:
    """
    Distinguish Meesho monthly ZIP vs Amazon MTR master ZIP (nested CSV/ZIPs).
    Used only when the **entire** archive is passed to ``_handle_one`` (see daily-auto ZIP routing).
    """
    if "meesho" in fn:
        return "meesho"
    _names, joined = _zip_member_paths(file_bytes)
    if "tcs_sales" in joined or "forwardreports" in joined:
        return "meesho"
    try:
        probe, n_csv, _sk = load_mtr_from_zip(file_bytes)
        if not probe.empty and n_csv > 0:
            return "amazon_mtr_zip"
    except Exception:
        pass
    return "meesho"


def _zip_is_flipkart_spreadsheet_bundle(file_bytes: bytes, outer_fn: str) -> bool:
    """ZIP of XLSX/XLSB only (typical Flipkart seller export) — no CSV members."""
    if "flipkart" in outer_fn.lower():
        return True
    names, _joined = _zip_member_paths(file_bytes)
    if not names:
        return False
    has_sheet = any(n.lower().endswith((".xlsx", ".xls", ".xlsb")) for n in names)
    has_csv = any(n.lower().endswith(".csv") for n in names)
    return has_sheet and not has_csv


def _detect_platform(filename: str, file_bytes: bytes) -> str:
    """
    Guess platform from path/filename + file contents.
    Returns: 'amazon_b2c', 'amazon_b2b', 'amazon_mtr_zip', 'myntra', 'meesho',
    'meesho_csv', 'meesho_order_xlsx', 'flipkart', 'snapdeal', 'unknown'
    """
    fn = (filename or "").lower().replace("\\", "/")

    if "snapdeal" in fn:
        return "snapdeal"

    if fn.endswith(".zip"):
        return _detect_platform_zip(file_bytes, fn)

    if fn.endswith(".xlsb"):
        return "flipkart"

    if fn.endswith((".xlsx", ".xls")):
        try:
            peek = pd.read_excel(io.BytesIO(file_bytes), nrows=40)
            if looks_like_meesho_order_export(peek):
                src = next((c for c in peek.columns if str(c).strip().lower() == "source"), None)
                if src is None or peek[src].astype(str).str.strip().str.lower().eq("meesho").any():
                    return "meesho_order_xlsx"
        except Exception:
            pass
        return "flipkart"

    header_bytes = file_bytes[:3000]
    # CSV — filename hints first
    if "myntra" in fn or "ppmp" in fn or "seller_orders" in fn or "seller orders" in fn or "my ppmp" in fn:
        return "myntra"
    if "b2b" in fn:
        return "amazon_b2b"
    if "b2c" in fn or "mtr" in fn or "merchant" in fn or "tax report" in fn:
        return "amazon_b2c"
    if "meesho" in fn:
        return "meesho_csv"
    # Meesho supplier panel CSV: Orders_YYYY-MM-DD_... (strict — avoid *orders_report* false positives)
    if fn.endswith(".csv") and "seller_orders" not in fn and re.search(
        r"(?:^|/)orders_\d{4}-\d{2}-\d{2}", fn
    ):
        try:
            head = file_bytes[:4000].decode("utf-8", errors="ignore").lower()
            if "reason for credit entry" in head or (
                "sub order no" in head and "customer state" in head
            ):
                return "meesho_csv"
        except Exception:
            pass
    # "Amz ..." or "amz ..." filenames → Amazon FBA
    if fn.startswith("amz ") or fn.startswith("amz_"):
        return "amazon_b2c"
    # Purely numeric filename (e.g. 937788020504.csv) → Amazon FBA/MTR export
    if re.match(r"^\d+\.csv$", fn):
        return "amazon_b2c"

    # Content-based detection (first 3 KB)
    try:
        text = header_bytes[:3000].decode("utf-8", errors="ignore").lower()
        if "buyer name" in text or "customer bill to gstid" in text:
            return "amazon_b2b"
        # Meesho daily CSV report ("Orders_..." filenames from Meesho Supplier Panel)
        if "reason for credit entry" in text or (
                "sub order no" in text and "customer state" in text):
            return "meesho_csv"
        # Amazon FBA Shipment Report
        if "customer shipment date" in text and "merchant sku" in text:
            return "amazon_b2c"
        # Amazon Order Report format (dash-separated headers)
        if "amazon-order-id" in text or "purchase-date" in text or "merchant-order-id" in text:
            return "amazon_b2c"
        if "order_created_date" in text or "product_mrp" in text or "sub_order_no" in text:
            return "myntra"
        # Myntra Seller Orders Report (space-separated headers)
        if "sub order id" in text and ("myntra" in text or "sku id" in text):
            return "myntra"
        if "sub_order_num" in text and "meesho" in text:
            return "meesho_csv"
        if "shipment date" in text or "invoice date" in text or "transaction type" in text:
            return "amazon_b2c"
        if "sales report" in text or "buyer invoice date" in text:
            return "flipkart"
    except Exception:
        pass

    return "unknown"


_DeferredSlice = Tuple[str, pd.DataFrame, str]


def _merge_slice_into_session(
    sess,
    platform: str,
    df_slice: pd.DataFrame,
    *,
    source_filename: Optional[str] = None,
) -> None:
    """Merge one parsed upload slice into session — avoids reloading full Tier-3 SQLite."""
    if df_slice is None or df_slice.empty:
        return
    if platform == "amazon":
        sess.mtr_df = _merge_platform_data(
            sess.mtr_df, df_slice, "amazon", source_filename=source_filename,
        )
    elif platform == "myntra":
        sess.myntra_df = _merge_platform_data(
            sess.myntra_df, df_slice, "myntra", source_filename=source_filename,
        )
    elif platform == "meesho":
        sess.meesho_df = _merge_platform_data(
            sess.meesho_df, df_slice, "meesho", source_filename=source_filename,
        )
    elif platform == "flipkart":
        sess.flipkart_df = _merge_platform_data(
            sess.flipkart_df, df_slice, "flipkart", source_filename=source_filename,
        )
    else:
        return
    sess.daily_restored = False


def _flush_deferred_session_slices(sess, queue: List[_DeferredSlice]) -> None:
    """Batch deferred per-file slices into one merge per platform (ZIP/RAR)."""
    if not queue:
        return
    by_plat: dict[str, list[_DeferredSlice]] = defaultdict(list)
    for plat, df, fname in queue:
        if df is not None and not df.empty:
            by_plat[plat].append((plat, df, fname))
    queue.clear()
    for plat, pairs in by_plat.items():
        frames = [df for _, df, _ in pairs]
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        src = pairs[0][2]
        _merge_slice_into_session(sess, plat, combined, source_filename=src)


def _process_daily_auto_sync(
    sess,
    file_parts: list[tuple[str, bytes]],
    *,
    rebuild_sales: bool = True,
) -> dict:
    """
    RAR/ZIP extract, parsers, daily SQLite — CPU-heavy.
    ``build_sales_df`` can run here or in a background task (see ``upload_daily_auto``).
    Runs in ``run_in_threadpool`` so the asyncio loop can still answer /auth/* and /api/health.
    Serialized with ``sess._daily_restore_lock`` alongside Tier-3 restore in ``data.py``.
    """
    with sess._daily_restore_lock:
            detected: list[str] = []
            warnings: list[str] = []

            def _apply_parsed_slice(
                p: str,
                df: pd.DataFrame,
                src_fname: str,
                defer_queue: Optional[List[_DeferredSlice]] = None,
            ) -> None:
                if defer_queue is not None:
                    defer_queue.append((p, df, src_fname))
                else:
                    _merge_slice_into_session(sess, p, df, source_filename=src_fname)

            def _handle_one(
                fname: str,
                raw: bytes,
                defer_queue: Optional[List[_DeferredSlice]] = None,
            ) -> None:
                """Process a single (non-RAR) file and mutate detected/warnings."""
                platform = _detect_platform(fname, raw)
                try:
                    if platform == "amazon_mtr_zip":
                        df_mtr, _n, sk_mtr = load_mtr_from_zip(raw)
                        if not df_mtr.empty:
                            save_daily_file("amazon", fname, df_mtr)
                            _apply_parsed_slice("amazon", df_mtr, fname, defer_queue)
                            detected.append(f"Amazon MTR ({fname})")
                            if sk_mtr:
                                warnings.append(f"{fname}: {'; '.join(sk_mtr[:2])}")
                        else:
                            warnings.append(f"{fname}: No Amazon MTR CSVs — {'; '.join(sk_mtr[:3])}")

                    elif platform == "amazon_b2c":
                        df, msg = parse_mtr_csv(raw, fname)
                        if not df.empty:
                            save_daily_file("amazon", fname, df)
                            _apply_parsed_slice("amazon", df, fname, defer_queue)
                            detected.append(f"Amazon ({fname})")
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")
                        else:
                            warnings.append(f"{fname}: {msg}")

                    elif platform == "amazon_b2b":
                        df, msg = parse_mtr_csv(raw, fname)
                        if not df.empty:
                            save_daily_file("amazon", fname, df)
                            _apply_parsed_slice("amazon", df, fname, defer_queue)
                            detected.append(f"Amazon B2B ({fname})")
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")
                        else:
                            warnings.append(f"{fname}: {msg}")

                    elif platform == "myntra":
                        from ..services.myntra import _parse_myntra_csv
                        df, msg = _parse_myntra_csv(raw, fname, sess.sku_mapping)
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Myntra")
                            save_daily_file("myntra", fname, df)
                            _apply_parsed_slice("myntra", df, fname, defer_queue)
                            detected.append(f"Myntra ({fname})")
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")
                        else:
                            warnings.append(f"{fname}: {msg}")

                    elif platform == "meesho":
                        df, _count, _skipped = load_meesho_from_zip(raw, source_filename=fname)
                        if not df.empty:
                            save_daily_file("meesho", fname, df)
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            detected.append(f"Meesho ({fname})")
                            if _skipped:
                                warnings.append(f"{fname}: {'; '.join(_skipped[:2])}")
                        else:
                            warnings.append(f"{fname}: No data extracted")

                    elif platform == "meesho_csv":
                        from ..services.meesho import parse_meesho_csv
                        df, msg = parse_meesho_csv(raw)
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                            save_daily_file("meesho", fname, df)
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            detected.append(f"Meesho ({fname})")
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")
                        else:
                            warnings.append(f"{fname}: {msg}")

                    elif platform == "meesho_order_xlsx":
                        df, msg = parse_meesho_order_export_xlsx(raw)
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                            save_daily_file("meesho", fname, df)
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            detected.append(f"Meesho order export ({fname})")
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")
                        else:
                            warnings.append(f"{fname}: {msg}")

                    elif platform == "snapdeal":
                        df_sd, _fc, skipped_sd, parse_info = load_snapdeal_from_zip(
                            raw, sess.sku_mapping or {}, fname,
                        )
                        sess.snapdeal_parse_info.update(parse_info)
                        if df_sd.empty:
                            warnings.append(f"{fname}: Snapdeal — {'; '.join(skipped_sd[:3])}")
                        else:
                            if sess.snapdeal_df.empty:
                                sess.snapdeal_df = df_sd
                            else:
                                sess.snapdeal_df = pd.concat(
                                    [sess.snapdeal_df, df_sd], ignore_index=True,
                                ).drop_duplicates()
                            sess.daily_restored = False
                            detected.append(f"Snapdeal ({fname})")

                    elif platform == "flipkart":
                        from ..services.flipkart import (
                            _parse_flipkart_xlsx, _parse_flipkart_orders_sheet,
                            _parse_flipkart_earn_more, _parse_flipkart_xlsb,
                        )
                        if fname.lower().endswith(".xlsb"):
                            df = _parse_flipkart_xlsb(raw, fname, sess.sku_mapping)
                            if df.empty:
                                warnings.append(f"{fname}: No data extracted from Flipkart XLSB file")
                                return
                        else:
                            try:
                                xl_sheets = pd.ExcelFile(io.BytesIO(raw)).sheet_names
                            except Exception:
                                xl_sheets = []
                            if "Sales Report" in xl_sheets:
                                df = _parse_flipkart_xlsx(raw, fname, sess.sku_mapping)
                            elif "Orders" in xl_sheets:
                                df = _parse_flipkart_orders_sheet(raw, fname, sess.sku_mapping)
                            elif "earn_more_report" in xl_sheets:
                                df = _parse_flipkart_earn_more(raw, fname, sess.sku_mapping)
                            else:
                                df = _parse_flipkart_xlsx(raw, fname, sess.sku_mapping)
                                if df.empty:
                                    warnings.append(f"{fname}: Skipped — no Sales Report, Orders, or earn_more_report sheet (sheets: {', '.join(xl_sheets[:4])})")
                                    return
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Flipkart")
                            save_daily_file("flipkart", fname, df)
                            _apply_parsed_slice("flipkart", df, fname, defer_queue)
                            detected.append(f"Flipkart ({fname})")
                        else:
                            warnings.append(f"{fname}: No data extracted from Flipkart file")

                    else:
                        warnings.append(f"{fname}: Could not detect platform (unknown format)")

                except Exception as e:
                    warnings.append(f"{fname}: {e}")

            for fname, raw in file_parts:
                try:
                    fl = fname.lower()
                    # RAR archive — extract and process each file inside
                    if raw[:6] == _RAR_MAGIC or fl.endswith(".rar"):
                        try:
                            inner_files = _extract_rar_files(raw)
                            if not inner_files:
                                warnings.append(f"{fname}: RAR archive extracted to zero files — check archive is not corrupt.")
                            defer_rar: List[_DeferredSlice] = []
                            for inner_name, inner_bytes in inner_files:
                                if not inner_bytes:
                                    continue
                                norm = inner_name.replace("\\", "/").strip()
                                base = norm.rsplit("/", 1)[-1] if norm else ""
                                if not base or base.endswith("/"):
                                    continue
                                _handle_one(inner_name, inner_bytes, defer_rar)
                            _flush_deferred_session_slices(sess, defer_rar)
                        except Exception as e:
                            warnings.append(f"{fname} (RAR extract): {e}")
                    elif fl.endswith(".zip"):
                        if "snapdeal" in fl:
                            _handle_one(fname, raw)
                        elif _zip_is_meesho_monthly(raw, fname):
                            _handle_one(fname, raw)
                        elif _zip_is_amazon_mtr_master(raw):
                            _handle_one(fname, raw)
                        elif _zip_is_flipkart_spreadsheet_bundle(raw, fname):
                            try:
                                df_fk, n_fc, skipped_fk = load_flipkart_from_zip(
                                    raw,
                                    sess.sku_mapping or {},
                                    source_filename=fname,
                                )
                                if not df_fk.empty:
                                    save_daily_file("flipkart", fname, df_fk)
                                    _merge_slice_into_session(
                                        sess, "flipkart", df_fk, source_filename=fname,
                                    )
                                    detected.append(
                                        f"Flipkart ZIP ({fname}, {n_fc} file(s))",
                                    )
                                    if skipped_fk:
                                        warnings.append(
                                            f"{fname}: {'; '.join(skipped_fk[:2])}",
                                        )
                                else:
                                    defer_z: List[_DeferredSlice] = []
                                    try:
                                        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                                            for info in zf.infolist():
                                                if info.is_dir():
                                                    continue
                                                n = info.filename
                                                if "__MACOSX" in n:
                                                    continue
                                                base = n.rsplit("/", 1)[-1]
                                                if base.startswith("."):
                                                    continue
                                                _handle_one(n, zf.read(n), defer_z)
                                        _flush_deferred_session_slices(sess, defer_z)
                                    except Exception as e:
                                        warnings.append(f"{fname} (ZIP expand): {e}")
                                    fk_hits = [d for d in detected if "Flipkart" in d]
                                    if not fk_hits:
                                        hint = (
                                            "; ".join(skipped_fk[:3])
                                            if skipped_fk
                                            else "no parsable sheets"
                                        )
                                        warnings.append(
                                            f"{fname}: Flipkart ZIP — no rows ({hint})",
                                        )
                            except Exception as e:
                                warnings.append(f"{fname} (Flipkart ZIP): {e}")
                        else:
                            defer_z: List[_DeferredSlice] = []
                            try:
                                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                                    for info in zf.infolist():
                                        if info.is_dir():
                                            continue
                                        n = info.filename
                                        if "__MACOSX" in n:
                                            continue
                                        base = n.rsplit("/", 1)[-1]
                                        if base.startswith("."):
                                            continue
                                        _handle_one(n, zf.read(n), defer_z)
                                _flush_deferred_session_slices(sess, defer_z)
                            except Exception as e:
                                warnings.append(f"{fname} (ZIP): {e}")
                    else:
                        _handle_one(fname, raw)
                finally:
                    del raw
                    gc.collect()

            if not detected:
                warn_str = "; ".join(warnings) if warnings else "No valid files found."
                return {
                    "ok": False,
                    "message": warn_str,
                    "detected_platforms": [],
                    "warnings": warnings,
                    "processed_files": len(file_parts),
                    "detected_files": 0,
                    "unknown_files": len(file_parts),
                }

            # Track loaded platforms; rebuild sales now or defer to background (daily-auto HTTP path).
            sess.daily_sales_sources = list(set(sess.daily_sales_sources + detected))
            sales_rebuild = "inline"
            if rebuild_sales:
                ok_rb, rb_msg = _rebuild_sales_sync(sess)
                if not ok_rb:
                    warnings.append(f"Sales rebuild warning: {rb_msg}")
            else:
                sales_rebuild = "pending"
                _session_data_changed(sess)

            msg_parts = [f"Loaded {len(detected)} file(s): {', '.join(d.split('(')[0].strip() for d in detected)}."]
            if rebuild_sales and not sess.sales_df.empty:
                msg_parts.append(f"Sales rebuilt ({len(sess.sales_df):,} rows).")
            elif not rebuild_sales:
                msg_parts.append("Sales rebuild started in background.")
            if warnings:
                msg_parts.append(f"Warnings: {'; '.join(warnings)}")
            return {
                "ok": True,
                "message": " ".join(msg_parts),
                "detected_platforms": detected,
                "warnings": warnings,
                "processed_files": len(file_parts),
                "detected_files": len(detected),
                "unknown_files": max(0, len(file_parts) - len(detected)),
                "sales_rebuild": sales_rebuild,
            }


# Batches above this count must use POST /upload/chunk/* (UploadFile streams close after HTTP 200).
_DAILY_AUTO_DIRECT_MAX_FILES = int(os.environ.get("DAILY_AUTO_DIRECT_MAX_FILES", "3"))


@router.post("/daily-auto")
async def upload_daily_auto(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Drop any mix of daily report files — platform is auto-detected per file.
    Also accepts a RAR archive — each file inside is extracted and processed.
    Appends to existing session data and rebuilds sales_df automatically.

    Large / RAR / multi-file uploads run ingest in a FastAPI **background task** so the
    HTTP response returns immediately (avoids nginx/Cloudflare 502 while work continues).
    ``build_sales_df`` still runs after ingest (same as the historical async sales rebuild).
    """
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", None)
    n_files = len(files)

    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        if not _clear_stuck_daily_ingest(sess, force=False):
            return JSONResponse(
                content={
                    "ok": False,
                    "stuck": True,
                    "message": (
                        "A daily upload is still processing. Wait for it to finish, "
                        "or use “Clear stuck upload”, then try again."
                    ),
                    "detected_platforms": [],
                    "warnings": [],
                    "processed_files": n_files,
                    "detected_files": 0,
                    "unknown_files": 0,
                }
            )

    if n_files > _DAILY_AUTO_DIRECT_MAX_FILES:
        return JSONResponse(
            content={
                "ok": False,
                "message": (
                    f"Too many files ({n_files}) for a single upload. "
                    "Refresh the page — the app will upload in smaller chunks automatically."
                ),
                "require_chunked": True,
                "detected_platforms": [],
                "warnings": [],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": n_files,
            }
        )

    file_parts: list[tuple[str, bytes]] = []
    try:
        for fobj in files:
            file_parts.append((fobj.filename or "upload", await fobj.read()))
    except Exception as e:
        _log.exception("daily-auto read files in request")
        return JSONResponse(
            content={
                "ok": False,
                "message": str(e),
                "detected_platforms": [],
                "warnings": [str(e)],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": n_files,
            }
        )

    if not sid:
        payload = await run_heavy(
            _process_daily_auto_sync, sess, file_parts, rebuild_sales=True,
        )
        return JSONResponse(content=payload)

    msg = (
        "Upload accepted — parsing daily files on the server. "
        "Status updates below; you can stay on this page."
    )

    def mark_async_start():
        sess.daily_auto_ingest_status = "running"
        sess.daily_auto_ingest_started = time.time()
        sess.daily_auto_ingest_message = "Parsing daily files and merging into session…"
        sess.daily_auto_ingest_result = {}
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""

    await _session_lock_apply(sess, mark_async_start)
    background_tasks.add_task(_queue_daily_auto_ingest, sid, file_parts)
    return JSONResponse(
        content={
            "ok": True,
            "ingest_async": True,
            "message": msg,
            "detected_platforms": [],
            "warnings": [],
            "processed_files": n_files,
            "detected_files": 0,
            "unknown_files": n_files,
            "sales_rebuild": "pending",
        }
    )

# ── Daily Orders (multi-platform CSVs) ────────────────────────

@router.post("/daily")
async def upload_daily(
    request: Request,
    amz_b2c:  Optional[UploadFile] = File(None),
    amz_b2b:  Optional[UploadFile] = File(None),
    myntra:   Optional[UploadFile] = File(None),
    meesho:   Optional[UploadFile] = File(None),
    flipkart: Optional[UploadFile] = File(None),
):
    """
    Upload daily order report CSVs from one or more platforms.
    Each is parsed and appended to existing session data.
    """
    sess = _get_session(request)

    b2c_raw = await amz_b2c.read() if amz_b2c else None
    b2c_fn = amz_b2c.filename if amz_b2c else None
    b2b_raw = await amz_b2b.read() if amz_b2b else None
    b2b_fn = amz_b2b.filename if amz_b2b else None
    myn_raw = await myntra.read() if myntra else None
    myn_fn = myntra.filename if myntra else None
    msh_raw = await meesho.read() if meesho else None
    msh_fn = meesho.filename if meesho else None
    flk_raw = await flipkart.read() if flipkart else None
    flk_fn = flipkart.filename if flipkart else None

    n_files = sum(1 for x in (amz_b2c, amz_b2b, myntra, meesho, flipkart) if x is not None)

    def work():
        detected: list[str] = []
        warnings: list[str] = []

        for raw, fn, label in (
            (b2c_raw, b2c_fn, "Amazon B2C"),
            (b2b_raw, b2b_fn, "Amazon B2B"),
        ):
            if raw is None:
                continue
            try:
                df, msg = parse_mtr_csv(raw, fn or f"{label}.csv")
                if not df.empty:
                    sess.mtr_df = _merge_platform_data(
                        sess.mtr_df, df, "amazon", source_filename=fn or None,
                    )
                    detected.append(label)
                    if msg != "OK":
                        warnings.append(f"{label}: {msg}")
                else:
                    warnings.append(f"{label}: {msg}")
            except Exception as e:
                warnings.append(f"{label}: {e}")

        if myn_raw is not None:
            try:
                from ..services.myntra import _parse_myntra_csv
                df, msg = _parse_myntra_csv(myn_raw, myn_fn or "myntra.csv", sess.sku_mapping)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(
                        df, myn_fn or None, "Myntra",
                    )
                    sess.myntra_df = _merge_platform_data(
                        sess.myntra_df, df, "myntra", source_filename=myn_fn or None,
                    )
                    detected.append("Myntra")
                    if msg != "OK":
                        warnings.append(f"Myntra: {msg}")
                else:
                    warnings.append(f"Myntra: {msg}")
            except Exception as e:
                warnings.append(f"Myntra: {e}")

        if msh_raw is not None:
            try:
                from ..services.meesho import load_meesho_from_zip
                df, _count, _skipped = load_meesho_from_zip(
                    msh_raw, source_filename=msh_fn or None,
                )
                if not df.empty:
                    sess.meesho_df = _merge_platform_data(
                        sess.meesho_df, df, "meesho", source_filename=msh_fn or None,
                    )
                    detected.append("Meesho")
                    if _skipped:
                        warnings.append(f"Meesho: {'; '.join(_skipped[:3])}")
                else:
                    warnings.append(f"Meesho: No data. {'; '.join(_skipped[:3])}")
            except Exception as e:
                warnings.append(f"Meesho: {e}")

        if flk_raw is not None:
            try:
                from ..services.flipkart import _parse_flipkart_xlsx
                df = _parse_flipkart_xlsx(flk_raw, flk_fn or "flipkart.xlsx", sess.sku_mapping)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(
                        df, flk_fn or None, "Flipkart",
                    )
                    sess.flipkart_df = _merge_platform_data(
                        sess.flipkart_df, df, "flipkart", source_filename=flk_fn or None,
                    )
                    detected.append("Flipkart")
                else:
                    warnings.append("Flipkart: No data extracted")
            except Exception as e:
                warnings.append(f"Flipkart: {e}")

        gc.collect()

        if not detected:
            warn_str = "; ".join(warnings) if warnings else "No valid files provided."
            return {
                "ok": False,
                "message": warn_str,
                "detected_platforms": [],
                "warnings": warnings,
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": n_files,
            }

        _session_data_changed(sess)
        msg_parts = [f"Daily data loaded — {', '.join(detected)}."]
        if warnings:
            msg_parts.append(f"Warnings: {'; '.join(warnings)}")
        return {
            "ok": True,
            "message": " ".join(msg_parts),
            "detected_platforms": detected,
            "warnings": warnings,
            "processed_files": n_files,
            "detected_files": len(detected),
            "unknown_files": max(0, n_files - len(detected)),
        }

    content = await _session_lock_apply(sess, work)
    return JSONResponse(content=content)


# ── Build Sales (merge all platforms into unified sales_df) ───

@router.post("/build-sales")
async def build_sales(request: Request, background_tasks: BackgroundTasks):
    sess = _get_session(request)
    if not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})

    def work():
        sales_df = build_sales_df(
            mtr_df=sess.mtr_df,
            myntra_df=sess.myntra_df,
            meesho_df=sess.meesho_df,
            flipkart_df=sess.flipkart_df,
            snapdeal_df=sess.snapdeal_df,
            sku_mapping=sess.sku_mapping,
            return_overlay_df=_sess_return_overlay(sess),
        )
        sess.sales_df = sales_df
        sess._quarterly_cache.clear()  # invalidate quarterly cache on new sales data
        _session_data_changed(sess)
        gaps = list_sku_mapping_gaps(sales_df, sess.sku_mapping)
        msg = f"Sales built: {len(sales_df):,} rows. Saving to cache in background…"
        if gaps:
            msg += f" Warning: {len(gaps)} SKU(s) not found as map key or OMS value — see Upload → SKU Mapping list."
        return {
            "ok": True,
            "message": msg,
            "rows": len(sales_df),
            "unmapped_skus": gaps or None,
        }

    try:
        out = await _session_lock_apply(sess, work)
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Build error: {e}"})
    background_tasks.add_task(_auto_save_cache, sess)
    return JSONResponse(content=out)


# ── Clear platform data ────────────────────────────────────────

_PLATFORM_CLEAR = {
    "mtr":      lambda sess, pd: setattr(sess, "mtr_df",      pd.DataFrame()),
    "myntra":   lambda sess, pd: setattr(sess, "myntra_df",   pd.DataFrame()),
    "meesho":   lambda sess, pd: setattr(sess, "meesho_df",   pd.DataFrame()),
    "flipkart": lambda sess, pd: setattr(sess, "flipkart_df", pd.DataFrame()),
    "snapdeal": lambda sess, pd: setattr(sess, "snapdeal_df", pd.DataFrame()),
    "sales":    lambda sess, pd: setattr(sess, "sales_df",    pd.DataFrame()),
}

@router.delete("/clear/{platform}")
async def clear_platform(platform: str, request: Request):
    """Clear a specific platform's data from the session."""
    sess = _get_session(request)
    cleaner = _PLATFORM_CLEAR.get(platform)
    if not cleaner:
        return JSONResponse(content={"ok": False, "message": f"Unknown platform: {platform}"}, status_code=400)
    import pandas as pd

    def work():
        cleaner(sess, pd)
        if platform != "sales":
            sess.sales_df = pd.DataFrame()   # invalidate combined sales
            sess._quarterly_cache.clear()
        return {"ok": True, "message": f"{platform} data cleared."}

    body = await _session_lock_apply(sess, work)
    return JSONResponse(content=body)


# ── Chunked upload (large daily / inventory batches) ───────────────

from ..services.chunk_upload_store import chunk_store, CHUNK_SIZE_BYTES


class ChunkInitFileIn(BaseModel):
    name: str
    size: int = Field(ge=0, le=500 * 1024 * 1024)


class ChunkInitIn(BaseModel):
    target: str
    files: list[ChunkInitFileIn]


class ChunkCompleteIn(BaseModel):
    upload_id: str


async def _accept_daily_auto_file_parts(
    sess,
    sid: str,
    file_parts: list[tuple[str, bytes]],
    background_tasks: BackgroundTasks,
) -> dict:
    n_files = len(file_parts)
    if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
        if not _clear_stuck_daily_ingest(sess, force=False):
            return {
                "ok": False,
                "stuck": True,
                "message": (
                    "A daily upload is still processing. Wait for it to finish, "
                    "or use “Clear stuck upload”, then try again."
                ),
                "detected_platforms": [],
                "warnings": [],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": 0,
            }

    def mark_async_start():
        sess.daily_auto_ingest_status = "running"
        sess.daily_auto_ingest_started = time.time()
        sess.daily_auto_ingest_message = "Parsing daily files and merging into session…"
        sess.daily_auto_ingest_result = {}
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""

    await _session_lock_apply(sess, mark_async_start)
    background_tasks.add_task(_queue_daily_auto_ingest, sid, file_parts)
    return {
        "ok": True,
        "ingest_async": True,
        "chunked": True,
        "parsing_pending": True,
        "message": (
            "Upload complete — parsing daily files on the server. "
            "Status updates below; you can stay on this page."
        ),
        "detected_platforms": [],
        "warnings": [],
        "processed_files": n_files,
        "detected_files": 0,
        "unknown_files": 0,
        "sales_rebuild": "pending",
    }


async def _accept_inventory_auto_file_parts(
    sess,
    sid: str,
    file_parts: list[tuple[str, bytes]],
    background_tasks: BackgroundTasks,
) -> dict:
    if getattr(sess, "inventory_upload_status", "idle") == "running":
        return {
            "ok": False,
            "message": "An inventory upload is still processing. Wait for it to finish, then try again.",
        }

    def mark_async_start():
        sess.inventory_upload_status = "running"
        sess.inventory_upload_message = "Parsing inventory files and merging snapshot…"
        sess.inventory_upload_result = {}

    await _session_lock_apply(sess, mark_async_start)
    background_tasks.add_task(_run_inventory_auto_from_parts, sid, file_parts)
    return {
        "ok": True,
        "ingest_async": True,
        "chunked": True,
        "message": (
            "Upload complete — parsing inventory on the server. "
            "Status updates below; you can stay on this page."
        ),
    }


@router.post("/chunk/init")
async def chunk_upload_init(request: Request, body: ChunkInitIn):
    """Start a chunked upload session; client uploads 4 MB parts via POST /chunk."""
    sid = getattr(request.state, "session_id", None)
    if not sid:
        return JSONResponse(content={"ok": False, "message": "Session required."}, status_code=400)
    sess = _get_session(request)
    if body.target not in ("daily-auto", "inventory-auto"):
        return JSONResponse(content={"ok": False, "message": "Invalid target."}, status_code=400)
    if body.target == "inventory-auto" and not sess.sku_mapping:
        return JSONResponse(content={"ok": False, "message": "Upload SKU Mapping first."})
    if not body.files:
        return JSONResponse(content={"ok": False, "message": "No files listed."}, status_code=400)
    try:
        upload_id, chunk_size = chunk_store.create(
            sid,
            target=body.target,  # type: ignore[arg-type]
            files=[(f.name, f.size) for f in body.files],
        )
    except ValueError as e:
        return JSONResponse(content={"ok": False, "message": str(e)}, status_code=400)
    return JSONResponse(
        content={
            "ok": True,
            "upload_id": upload_id,
            "chunk_size": chunk_size,
            "file_count": len(body.files),
        }
    )


@router.post("/chunk")
async def chunk_upload_part(
    request: Request,
    upload_id: str = Form(...),
    file_index: int = Form(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    chunk: UploadFile = File(...),
):
    """Upload one chunk (typically 4 MB) for a file in an active chunked session."""
    sid = getattr(request.state, "session_id", None)
    if not sid:
        return JSONResponse(content={"ok": False, "message": "Session required."}, status_code=400)
    raw = await chunk.read()
    try:
        progress = chunk_store.write_chunk(
            sid,
            upload_id,
            file_index=file_index,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            data=raw,
        )
    except FileNotFoundError:
        return JSONResponse(content={"ok": False, "message": "Unknown or expired upload."}, status_code=404)
    except ValueError as e:
        return JSONResponse(content={"ok": False, "message": str(e)}, status_code=400)
    return JSONResponse(content={"ok": True, **progress})


def _set_chunk_finalize_error(sess, target: str, message: str) -> None:
    if target == "inventory-auto":
        sess.inventory_upload_status = "error"
        sess.inventory_upload_message = message
        sess.inventory_upload_result = {"ok": False, "message": message}
    else:
        sess.daily_auto_ingest_status = "error"
        sess.daily_auto_ingest_message = message
        _store_daily_auto_ingest_result(
            sess,
            {
                "ok": False,
                "message": message,
                "detected_platforms": [],
                "warnings": [message],
                "processed_files": 0,
                "detected_files": 0,
                "unknown_files": 0,
            },
        )


def _finalize_chunk_upload_worker(session_id: str, upload_id: str) -> None:
    """Assemble chunk files on disk, then run Tier-3 ingest (heavy worker thread)."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    target = "daily-auto"
    try:
        target = chunk_store.get_target(session_id, upload_id)
        _target, file_parts = chunk_store.assemble(session_id, upload_id)
        target = _target
    except Exception as e:
        _log.exception("chunk finalize assemble upload_id=%s", upload_id)
        _set_chunk_finalize_error(sess, target, str(e))
        return

    if target == "daily-auto":
        sess.daily_auto_ingest_message = "Parsing daily files and merging into session…"
        _run_daily_auto_ingest_pipeline(session_id, file_parts)
    else:
        sess.inventory_upload_message = "Parsing inventory files and merging snapshot…"
        try:
            asyncio.run(_run_inventory_auto_from_parts(session_id, file_parts))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_run_inventory_auto_from_parts(session_id, file_parts))
            finally:
                loop.close()


def _finalize_chunk_upload(session_id: str, upload_id: str) -> None:
    """Queue chunk finalize on the heavy executor (do not block the asyncio event loop)."""
    HEAVY_EXECUTOR.submit(_finalize_chunk_upload_worker, session_id, upload_id)


@router.post("/chunk/complete")
async def chunk_upload_complete(
    request: Request,
    background_tasks: BackgroundTasks,
    body: ChunkCompleteIn,
):
    """Verify all chunks, return immediately, assemble + ingest in background."""
    sid = getattr(request.state, "session_id", None)
    if not sid:
        return JSONResponse(content={"ok": False, "message": "Session required."}, status_code=400)
    sess = _get_session(request)
    try:
        info = chunk_store.verify_ready(sid, body.upload_id)
    except FileNotFoundError:
        return JSONResponse(content={"ok": False, "message": "Unknown or expired upload."}, status_code=404)
    except ValueError as e:
        return JSONResponse(content={"ok": False, "message": str(e)}, status_code=400)

    target = info["target"]
    n_files = int(info["file_count"])

    if target == "daily-auto":
        if getattr(sess, "daily_auto_ingest_status", "idle") == "running":
            if not _clear_stuck_daily_ingest(sess, force=False):
                return JSONResponse(
                    content={
                        "ok": False,
                        "stuck": True,
                        "message": (
                            "A daily upload is still processing. Wait for it to finish, "
                            "or use “Clear stuck upload” below, then try again."
                        ),
                        "detected_platforms": [],
                        "warnings": [],
                        "processed_files": n_files,
                        "detected_files": 0,
                        "unknown_files": 0,
                    }
                )
            _log.info("Cleared stuck daily ingest before new chunk upload session=%s", sid[:8])
    elif getattr(sess, "inventory_upload_status", "idle") == "running":
        return JSONResponse(
            content={
                "ok": False,
                "message": "An inventory upload is still processing. Wait for it to finish, then try again.",
            }
        )

    def mark_assembling():
        if target == "daily-auto":
            sess.daily_auto_ingest_status = "running"
            sess.daily_auto_ingest_started = time.time()
            sess.daily_auto_ingest_message = "Assembling uploaded files…"
            sess.daily_auto_ingest_result = {}
            sess.sales_rebuild_status = "idle"
            sess.sales_rebuild_message = ""
        else:
            sess.inventory_upload_status = "running"
            sess.inventory_upload_message = "Assembling uploaded files…"
            sess.inventory_upload_result = {}

    await _session_lock_apply(sess, mark_assembling)
    background_tasks.add_task(_finalize_chunk_upload, sid, body.upload_id)

    if target == "daily-auto":
        return JSONResponse(
            content={
                "ok": True,
                "ingest_async": True,
                "chunked": True,
                "parsing_pending": True,
                "message": (
                    "All chunks received — parsing on the server. "
                    "Status updates below; you can stay on this page."
                ),
                "detected_platforms": [],
                "warnings": [],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": 0,
                "sales_rebuild": "pending",
            }
        )
    return JSONResponse(
        content={
            "ok": True,
            "ingest_async": True,
            "chunked": True,
            "message": (
                "All chunks received — assembling inventory on the server. "
                "Status updates below; you can stay on this page."
            ),
        }
    )


@router.delete("/chunk/{upload_id}")
async def chunk_upload_abort(request: Request, upload_id: str):
    sid = getattr(request.state, "session_id", None)
    if sid:
        chunk_store.abort(sid, upload_id)
    return JSONResponse(content={"ok": True})
