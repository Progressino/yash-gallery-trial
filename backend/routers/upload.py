"""
Upload router — all file ingestion endpoints.
"""
import asyncio
import gc
import threading
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
from ..concurrency import (
    AUX_EXECUTOR,
    HEAVY_EXECUTOR,
    DAILY_UPLOAD_EXECUTOR,
    INVENTORY_EXECUTOR,
    _UPLOAD_MEMORY_LOCK,
    run_heavy,
)

from ..session import store, resume_auto_data_restore, AppSession
from ..models.schemas import UploadResponse
from ..services.sku_mapping import (
    merge_sku_mapping_upload,
    parse_sku_mapping,
    persist_sku_mapping_globally,
    resolve_sku_mapping_base,
)
from ..services.mtr import load_mtr_from_zip, load_mtr_from_extracted_files, parse_mtr_csv
from ..services.myntra import load_myntra_from_zip
from ..services.meesho import (
    load_meesho_from_zip,
    looks_like_meesho_order_export,
    parse_meesho_order_export_xlsx,
)
from ..services.flipkart import load_flipkart_from_zip
from ..services.snapdeal import load_snapdeal_from_zip
from ..services.inventory import (
    apply_inventory_snapshot_metadata,
    backup_inventory_before_upload,
    inventory_missing_marketplace_warnings,
    load_inventory_consolidated,
    oms_loaded_in_debug,
    refresh_inventory_api_cache,
    restore_inventory_upload_backup,
    upload_bundle_expects_oms,
)
from ..services.sales import build_sales_df, list_sku_mapping_gaps
from ..services.existing_po import parse_existing_po
from ..services.github_cache import save_cache_to_drive
from ..services.daily_store import save_daily_file, merge_platform_data as _merge_platform_data
from ..services.helpers import apply_dsr_segment_from_upload_filename


router = APIRouter()

_log = logging.getLogger(__name__)


def _sess_return_overlay(sess):
    """Optional PO return-sheet overlay (OMS_SKU + Return_Units) merged into unified sales."""
    from ..services.po_return_import import aggregate_return_overlay_for_use

    ov = aggregate_return_overlay_for_use(getattr(sess, "po_return_overlay_df", None))
    if ov is None or getattr(ov, "empty", True):
        return None
    return ov


def _sales_overlay_build_kwargs(sess) -> dict:
    """Pass return overlay + as-of date into build_sales_df."""
    kw: dict = {}
    ov = _sess_return_overlay(sess)
    if ov is not None:
        kw["return_overlay_df"] = ov
    as_of = getattr(sess, "return_overlay_as_of", None)
    if as_of and str(as_of).strip():
        kw["return_overlay_as_of"] = str(as_of).strip()[:10]
    return kw


def _session_data_changed(sess) -> None:
    """Undo pause_auto_data_restore after uploads / builds so Tier-3 merge and warm cache work again."""
    resume_auto_data_restore(sess)


def _finalize_sales_data_refresh(sess: AppSession) -> None:
    """After sales_df / platform frames change — bust PO + Intelligence caches and signal the UI."""
    sess.sales_data_revision = int(getattr(sess, "sales_data_revision", 0) or 0) + 1
    sess.daily_restored = True
    try:
        from ..services.po_shared_cache import invalidate_po_after_sales_or_returns_change

        invalidate_po_after_sales_or_returns_change(sess)
    except Exception:
        _log.exception("PO/intelligence cache invalidation after sales refresh failed")
    try:
        from ..services.tier3_session_merge import mark_tier3_sync_applied

        mark_tier3_sync_applied(sess)
    except Exception:
        _log.exception("tier3 sync token apply after sales refresh failed")


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


def _daily_auto_fast_ingest_enabled() -> bool:
    """Skip merging each file into huge in-memory platform frames during ingest (SQLite only)."""
    v = (os.environ.get("DAILY_AUTO_FAST_INGEST") or "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _daily_rebuild_sqlite_months() -> int:
    try:
        return max(1, int((os.environ.get("DAILY_REBUILD_SQLITE_MONTHS") or "6").strip()))
    except ValueError:
        return 6


def _daily_rebuild_max_files() -> int:
    try:
        return max(5, int((os.environ.get("DAILY_REBUILD_MAX_FILES") or "80").strip()))
    except ValueError:
        return 80


def _track_daily_auto_platform(sess: AppSession, platform: str) -> None:
    touched = getattr(sess, "_daily_auto_platforms_touched", None)
    if touched is None:
        touched = set()
        sess._daily_auto_platforms_touched = touched
    touched.add(platform)


def _buffer_daily_auto_parsed(sess: AppSession, platform: str, df: pd.DataFrame) -> None:
    """Keep parsed rows from this upload in RAM for incremental sales rebuild (small)."""
    if df is None or df.empty:
        return
    buf = getattr(sess, "_daily_auto_parsed_buffers", None)
    if not isinstance(buf, dict):
        buf = {}
        sess._daily_auto_parsed_buffers = buf
    buf.setdefault(platform, []).append(df.copy())


def _combine_buffered_platform_df(sess: AppSession, platform: str) -> pd.DataFrame:
    from ..services.daily_store import _dedup_platform_df, merge_platform_data

    chunks = (getattr(sess, "_daily_auto_parsed_buffers", None) or {}).get(platform) or []
    if not chunks:
        return pd.DataFrame()
    combined = chunks[0]
    for chunk in chunks[1:]:
        combined = merge_platform_data(combined, chunk, platform)
    return _dedup_platform_df(combined, platform)


def _incremental_sales_rebuild_from_buffers(
    sess: AppSession,
    platforms_touched: set[str] | list[str],
) -> tuple[bool, str]:
    """
    Build sales only from this upload's parsed buffers (~thousands of rows),
    patch into existing sales_df — no 6-month SQLite reload, no full rebuild.
    """
    from ..services.platform_session_window import platform_frames_trimmed_for_sales_build
    from ..services.sales import (
        build_sales_df,
        patch_sales_df_after_daily_upload,
        sales_date_window_from_platform_dfs,
    )

    plats = set(platforms_touched)
    buffers = getattr(sess, "_daily_auto_parsed_buffers", None) or {}
    plat_frames: dict[str, pd.DataFrame] = {}
    for plat in plats:
        combined = _combine_buffered_platform_df(sess, plat)
        if not combined.empty:
            plat_frames[plat] = combined

    if not plat_frames:
        return False, "No parsed data in upload buffers."

    d0, d1 = sales_date_window_from_platform_dfs(plat_frames)
    if d0 is None or d1 is None:
        return False, "Could not determine upload date window."

    mtr = plat_frames.get("amazon", pd.DataFrame())
    myntra = plat_frames.get("myntra", pd.DataFrame())
    meesho = plat_frames.get("meesho", pd.DataFrame())
    flipkart = plat_frames.get("flipkart", pd.DataFrame())
    snapdeal = plat_frames.get("snapdeal", pd.DataFrame())

    fresh = build_sales_df(
        mtr_df=mtr,
        myntra_df=myntra,
        meesho_df=meesho,
        flipkart_df=flipkart,
        snapdeal_df=snapdeal,
        sku_mapping=sess.sku_mapping,
        **_sales_overlay_build_kwargs(sess),
    )

    existing = getattr(sess, "sales_df", None)
    if existing is None:
        existing = pd.DataFrame()
    sess.sales_df = patch_sales_df_after_daily_upload(
        existing, fresh, plats, d0, d1,
    )

    attrs = {
        "amazon": "mtr_df",
        "myntra": "myntra_df",
        "meesho": "meesho_df",
        "flipkart": "flipkart_df",
        "snapdeal": "snapdeal_df",
    }
    for plat, attr in attrs.items():
        if plat not in plats:
            continue
        combined = plat_frames.get(plat)
        if combined is None or combined.empty:
            continue
        cur = getattr(sess, attr)
        setattr(sess, attr, _merge_platform_data(cur, combined, plat, source_filename=None))

    sess._quarterly_cache.clear()
    _session_data_changed(sess)
    _finalize_sales_data_refresh(sess)
    sess._daily_auto_parsed_buffers = {}
    rows = len(sess.sales_df)
    return True, f"Sales updated ({rows:,} rows) from upload."


def _replace_touched_platforms_from_sqlite(
    sess: AppSession,
    platforms: set[str] | list[str],
) -> None:
    """Reload only uploaded marketplaces from Tier-3 (bounded window) — avoids 1M-row concat."""
    from ..services.daily_store import load_platform_data, merge_platform_data

    months = _daily_rebuild_sqlite_months()
    max_files = _daily_rebuild_max_files()
    attrs = {
        "amazon": "mtr_df",
        "myntra": "myntra_df",
        "meesho": "meesho_df",
        "flipkart": "flipkart_df",
        "snapdeal": "snapdeal_df",
    }
    for plat in platforms:
        attr = attrs.get(plat)
        if not attr:
            continue
        recent = load_platform_data(
            plat,
            months=months,
            dedup=False,
            max_files=max_files,
        )
        if recent.empty:
            continue
        cur = getattr(sess, attr)
        setattr(sess, attr, merge_platform_data(cur, recent, plat))
    sess.daily_restored = True


def _schedule_sales_cache_save(sess: AppSession) -> None:
    """GitHub/Postgres cache save can take minutes — never block sales rebuild on it."""

    def _run() -> None:
        try:
            _auto_save_cache(sess)
        except Exception:
            _log.exception("background sales cache save failed")

    threading.Thread(target=_run, name="sales-cache-save", daemon=True).start()


def _rebuild_sales_sync(
    sess,
    *,
    refresh_sqlite: bool = False,
    platforms_touched: set[str] | None = None,
) -> tuple[bool, str]:
    """Rebuild combined sales_df from platform frames. Caller should hold ``_daily_restore_lock``."""
    try:
        from ..services.platform_session_window import platform_frames_trimmed_for_sales_build

        if platforms_touched and _daily_auto_fast_ingest_enabled():
            buffers = getattr(sess, "_daily_auto_parsed_buffers", None) or {}
            if buffers and any(buffers.get(p) for p in platforms_touched):
                ok_inc, msg_inc = _incremental_sales_rebuild_from_buffers(
                    sess, platforms_touched,
                )
                if ok_inc:
                    return ok_inc, msg_inc
            _replace_touched_platforms_from_sqlite(sess, platforms_touched)
        elif refresh_sqlite:
            _sync_session_platforms_from_sqlite(sess)
        trimmed = platform_frames_trimmed_for_sales_build(sess)
        sess.sales_df = build_sales_df(
            mtr_df=trimmed["mtr_df"],
            myntra_df=trimmed["myntra_df"],
            meesho_df=trimmed["meesho_df"],
            flipkart_df=trimmed["flipkart_df"],
            snapdeal_df=trimmed["snapdeal_df"],
            sku_mapping=sess.sku_mapping,
            **_sales_overlay_build_kwargs(sess),
        )
        sess._quarterly_cache.clear()
        _session_data_changed(sess)
        _finalize_sales_data_refresh(sess)
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
        "expanded_files": int(payload.get("expanded_files") or 0),
        "saved_files": int(payload.get("saved_files") or 0),
        "file_results": list(payload.get("file_results") or []),
    }


def _sync_session_platforms_from_sqlite(sess, *, months: int = 4) -> None:
    """Merge recent Tier-3 SQLite uploads into session before sales rebuild."""
    from ..services.daily_store import load_platform_data

    for plat, attr in (
        ("amazon", "mtr_df"),
        ("myntra", "myntra_df"),
        ("meesho", "meesho_df"),
        ("flipkart", "flipkart_df"),
        ("snapdeal", "snapdeal_df"),
    ):
        df = load_platform_data(plat, months=months)
        if df is not None and not df.empty:
            cur = getattr(sess, attr)
            setattr(sess, attr, _merge_platform_data(cur, df, plat))
    sess.daily_restored = True


def _run_sales_rebuild_worker(
    session_id: str,
    *,
    refresh_sqlite: bool = False,
    platforms_touched: set[str] | None = None,
) -> None:
    """Background sales rebuild (daily-auto or manual ↻ Rebuild)."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    touched = platforms_touched or getattr(sess, "_daily_auto_platforms_touched", None) or None
    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_started = time.time()
    if touched and _daily_auto_fast_ingest_enabled():
        plats = ", ".join(sorted(touched))
        if getattr(sess, "_daily_auto_parsed_buffers", None):
            sess.sales_rebuild_message = f"Updating sales ({plats})…"
        else:
            sess.sales_rebuild_message = f"Rebuilding sales ({plats})…"
    else:
        sess.sales_rebuild_message = "Rebuilding combined sales…"
    try:
        with sess._daily_restore_lock:
            ok, msg = _rebuild_sales_sync(
                sess,
                refresh_sqlite=refresh_sqlite,
                platforms_touched=touched if touched else None,
            )
        if ok:
            sess.sales_rebuild_status = "done"
            sess.sales_rebuild_message = msg
            _schedule_sales_cache_save(sess)
            try:
                import backend.main as _main

                _main.publish_warm_cache_from_session(sess)
            except Exception:
                _log.exception("warm cache publish after sales rebuild")
            # Re-invalidate Intelligence bundle AFTER the rebuild so the next
            # precompute uses the freshly rebuilt sales_df, not the pre-rebuild data.
            try:
                from ..routers.data import _invalidate_intelligence_bundle_cache
                _invalidate_intelligence_bundle_cache()
            except Exception:
                _log.exception("Intelligence bundle re-invalidation after sales rebuild")
            try:
                from ..db.forecast_session_pg import persist_session_bundle_thread_safe

                persist_session_bundle_thread_safe(session_id, sess)
            except Exception:
                _log.exception("PostgreSQL persist after sales rebuild")
            try:
                from ..db.forecast_sales_materializations import refresh_from_sales_df

                if not sess.sales_df.empty:
                    refresh_from_sales_df(sess.sales_df)
            except Exception:
                _log.exception("SKU sales materialization refresh after sales rebuild")
        else:
            sess.sales_rebuild_status = "error"
            sess.sales_rebuild_message = msg
    except Exception as e:
        sess.sales_rebuild_status = "error"
        sess.sales_rebuild_message = str(e)
        _log.exception("background sales rebuild failed")
    finally:
        sess.sales_rebuild_started = 0.0
        if getattr(sess, "daily_auto_ingest_status", "") == "done":
            sess.daily_auto_ingest_status = "idle"
            sess.daily_auto_ingest_message = ""
            sess.daily_auto_ingest_started = 0.0


def _clear_stuck_sales_rebuild(sess: AppSession, *, force: bool = False) -> bool:
    """Reset sales_rebuild when stuck in running (orphan or timed out)."""
    if getattr(sess, "sales_rebuild_status", "idle") != "running":
        return False
    started = float(getattr(sess, "sales_rebuild_started", 0) or 0)
    age = time.time() - started if started > 0 else 999999
    stuck_sec = int(os.environ.get("SALES_REBUILD_STUCK_SEC", "600"))
    if not force and age < stuck_sec:
        return False
    sess.sales_rebuild_status = "idle"
    sess.sales_rebuild_message = (
        "Previous sales rebuild did not finish (server was busy). "
        "Data may still be updating — refresh in a minute."
    )
    sess.sales_rebuild_started = 0.0
    return True


def _clear_stuck_session_restore(sess: AppSession, *, force: bool = False) -> bool:
    """Reset session_restore when stuck in running (orphan worker thread or crash)."""
    if getattr(sess, "session_restore_status", "idle") != "running":
        return False
    started = float(getattr(sess, "session_restore_started", 0) or 0)
    age = time.monotonic() - started if started > 0 else 999999
    # Restores can legitimately take several minutes on a large GitHub cache.
    stuck_sec = int(os.environ.get("SESSION_RESTORE_STUCK_SEC", "900"))
    if not force and age < stuck_sec:
        return False
    sess.session_restore_status = "error"
    sess.session_restore_message = (
        "Previous session restore did not finish (server was busy or restarted). "
        "Retrying automatically…"
    )
    sess.session_restore_started = 0.0
    return True


def _clear_stuck_returns_import(sess: AppSession, *, force: bool = False) -> bool:
    """Reset returns_import when stuck in running (orphan worker or crash)."""
    if getattr(sess, "returns_import_status", "idle") != "running":
        return False
    started = float(getattr(sess, "returns_import_started", 0) or 0)
    age = time.monotonic() - started if started > 0 else 999999
    stuck_sec = int(os.environ.get("RETURNS_IMPORT_STUCK_SEC", "900"))
    if not force and age < stuck_sec:
        return False
    sess.returns_import_status = "error"
    sess.returns_import_message = (
        "Return import stopped responding (server may have restarted). "
        "Refresh and upload again, or check Upload → Returns status."
    )
    sess.returns_import_started = 0.0
    return True


def clear_stale_background_jobs(sess: AppSession) -> None:
    """Clear orphan/timed-out ingest, sales-rebuild, restore, and tier-1 bulk flags (safe on every light poll)."""
    _clear_stuck_daily_ingest(sess, force=False)
    _clear_stuck_returns_import(sess, force=False)
    _clear_stuck_sales_rebuild(sess, force=False)
    _clear_stuck_session_restore(sess, force=False)
    _clear_stuck_tier1_bulk(sess, force=False)


def _run_daily_auto_sales_rebuild(session_id: str) -> None:
    """Background task: rebuild sales after Tier-3 ingest (bounded SQLite, no full re-merge)."""
    sess = _resolve_upload_session(session_id)
    touched = getattr(sess, "_daily_auto_platforms_touched", None) if sess else None
    # Always sync from SQLite after a daily upload so the session reflects the
    # freshly-saved files even when warm-cache data is already loaded (otherwise
    # the incremental buffer path may miss data that was in a prior warm cache
    # but not included in this ingest session's parsed buffers).
    refresh_sqlite = True
    _run_sales_rebuild_worker(
        session_id,
        refresh_sqlite=refresh_sqlite,
        platforms_touched=None if refresh_sqlite else (touched if touched else None),
    )


def _run_returns_import_followup(session_id: str) -> None:
    """After return overlay import: invalidate PO caches; patch sales + persist in background."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return

    try:
        from ..services.po_shared_cache import invalidate_po_after_sales_or_returns_change

        invalidate_po_after_sales_or_returns_change(sess)
    except Exception:
        _log.exception("PO cache invalidation after return import")

    def _background_sales_patch_and_persist() -> None:
        s = _resolve_upload_session(session_id)
        if s is None:
            return
        sales_df = getattr(s, "sales_df", None)
        if sales_df is not None and not getattr(sales_df, "empty", True):
            try:
                from ..services.sales import patch_sales_df_return_overlay

                as_of = getattr(s, "return_overlay_as_of", None)
                s.sales_df = patch_sales_df_return_overlay(
                    sales_df,
                    _sess_return_overlay(s),
                    dict(s.sku_mapping or {}),
                    return_overlay_as_of=str(as_of).strip()[:10] if as_of else None,
                )
                s._quarterly_cache.clear()
                _session_data_changed(s)
                _finalize_sales_data_refresh(s)
            except Exception:
                _log.exception("background return overlay sales patch failed")
        try:
            import backend.main as _main

            _main.merge_po_optional_sheets_into_warm_cache(s)
        except Exception:
            _log.exception("warm cache sidecar merge after return import (background)")
        try:
            _schedule_sales_cache_save(s)
        except Exception:
            _log.exception("background sales cache save after return import failed")
        try:
            from ..db.forecast_session_pg import (
                persist_session_bundle_thread_safe,
                pg_session_persist_enabled,
            )

            if pg_session_persist_enabled():
                persist_session_bundle_thread_safe(session_id, s)
        except Exception:
            _log.exception("PostgreSQL persist after return import (background)")

    threading.Thread(
        target=_background_sales_patch_and_persist,
        name="return-import-followup",
        daemon=True,
    ).start()


@router.post("/daily-auto/reset-stuck")
async def reset_stuck_daily_upload(request: Request):
    """Clear a session stuck in daily ingest / sales rebuild (UI “Clear stuck upload”)."""
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
    cleared = _clear_stuck_daily_ingest(sess, force=True)
    _clear_stuck_sales_rebuild(sess, force=True)
    # Abort any pending chunk uploads so a fresh upload starts cleanly.
    if sid:
        try:
            chunk_store.abort_all(sid)
        except Exception:
            pass
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


@router.get("/daily-auto/verify")
def verify_daily_upload(request: Request, date: str = ""):
    """
    Confirm Tier-3 persistence and session sales for a calendar day (YYYY-MM-DD).
    Use after upload to verify data before opening Intelligence.
    """
    from ..services.daily_store import get_upload_report_day_coverage, get_summary, list_uploads

    day = str(date or "").strip()[:10]
    if not day or len(day) < 10:
        return {"ok": False, "message": "Provide date=YYYY-MM-DD (e.g. 2026-06-02)."}

    coverage = get_upload_report_day_coverage()
    tier3_platforms = sorted(p for p, days in coverage.items() if day in days)
    tier3_summary = get_summary()
    recent = [
        u
        for u in list_uploads()
        if day in (str(u.get("date_from") or "")[:10], str(u.get("date_to") or "")[:10], str(u.get("file_date") or "")[:10])
        or (
            str(u.get("date_from") or "")[:10] <= day <= str(u.get("date_to") or "")[:10]
            if u.get("date_from") and u.get("date_to")
            else False
        )
    ]
    sess = _get_session(request)
    ingest = getattr(sess, "daily_auto_ingest_result", None) or {}
    sales_df = getattr(sess, "sales_df", None)
    sales_rows = len(sales_df) if sales_df is not None and not sales_df.empty else 0
    sales_ready = sales_rows > 0
    session_sales_range: dict[str, str | None] = {"min": None, "max": None}
    if sales_ready and sales_df is not None and "TxnDate" in sales_df.columns:
        try:
            import pandas as pd
            from ..services.sales import txn_reporting_naive_ist

            t = txn_reporting_naive_ist(sales_df["TxnDate"]).dropna()
            if not t.empty:
                session_sales_range = {
                    "min": str(t.min().normalize())[:10],
                    "max": str(t.max().normalize())[:10],
                }
        except Exception:
            pass
    dashboard_ready = sales_ready and any(
        not getattr(sess, attr).empty
        for attr in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df")
        if getattr(sess, attr, None) is not None and hasattr(getattr(sess, attr), "empty")
    )
    ok = bool(tier3_platforms)
    # If no files cover the exact date but nearby files exist (e.g. a range report
    # uploaded for Jun 18–21 when verifying Jun 22), treat as OK and surface which
    # platforms are actually present in the database so the user is not alarmed.
    nearby_platforms: list[str] = []
    nearby_range_str = ""
    if not ok and recent:
        nearby_platforms = sorted({str(u.get("platform") or "").lower() for u in recent if u.get("platform")})
        date_froms = [str(u.get("date_from") or "")[:10] for u in recent if u.get("date_from")]
        date_tos = [str(u.get("date_to") or "")[:10] for u in recent if u.get("date_to")]
        if date_froms and date_tos:
            nearby_range_str = f"{min(date_froms)} → {max(date_tos)}"
        if nearby_platforms:
            ok = True  # Files exist, just not specifically for this exact calendar date

    hint = ""
    if ok and not tier3_platforms and nearby_platforms:
        plat_str = ", ".join(nearby_platforms)
        hint = (
            f" Uploads found for {plat_str}"
            + (f" covering {nearby_range_str}" if nearby_range_str else "")
            + f" (no file dated exactly {day} — this is normal for multi-day reports)."
        )
    elif not ok and sales_ready:
        hint = (
            " Session has sales data but no Tier-3 daily uploads found. "
            f"Upload files that cover {day} to persist them."
        )
    elif ok and not sales_ready:
        hint = " Tier-3 saved — wait for sales rebuild or tap ↻ Rebuild on this page."
    return {
        "ok": ok,
        "date": day,
        "tier3_platforms": tier3_platforms or nearby_platforms,
        "tier3_upload_count": len(recent),
        "tier3_summary": tier3_summary,
        "recent_uploads": recent[:12],
        "session_sales_rows": sales_rows,
        "session_sales_range": session_sales_range,
        "sales_ready": sales_ready,
        "dashboard_ready": dashboard_ready,
        "daily_auto_ingest_status": getattr(sess, "daily_auto_ingest_status", "idle"),
        "sales_rebuild_status": getattr(sess, "sales_rebuild_status", "idle"),
        "last_ingest_message": str(ingest.get("message") or ""),
        "message": (
            (
                f"Tier-3 has {len(tier3_platforms)} platform(s) for {day}: {', '.join(tier3_platforms)}."
                if tier3_platforms
                else f"Uploads saved for {', '.join(nearby_platforms) or 'none'}."
            )
            + (f" Session sales: {sales_rows:,} rows." if sales_ready else " Sales not rebuilt yet — wait or tap ↻ Rebuild.")
            + hint
        ),
    }


def _queue_daily_auto_ingest(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Schedule ingest on the dedicated upload executor (never blocks behind warm-cache)."""
    DAILY_UPLOAD_EXECUTOR.submit(_run_daily_auto_ingest_pipeline, session_id, file_parts)


def _memory_lock_wait_sec() -> int:
    try:
        return max(30, int((os.environ.get("DAILY_INGEST_LOCK_WAIT_SEC") or "90").strip()))
    except ValueError:
        return 90


def _acquire_ingest_memory_lock(sess, *, fnames: str) -> bool:
    """
    Fast Tier-3 ingest (SQLite + small buffers) does not need the global memory lock.
    Legacy full in-memory merge waits briefly, then proceeds so uploads are not stuck
    behind a multi-minute warm-cache load.
    """
    if _daily_auto_fast_ingest_enabled():
        if _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
            return True
        _log.info(
            "daily-auto fast ingest: proceeding without memory lock (session=%s)",
            getattr(sess, "_persist_sid", "")[:8],
        )
        sess.daily_auto_ingest_message = (
            f"Parsing files… ({fnames}) — server finishing cache load in background"
        )
        return False
    if _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
        return True
    wait_sec = _memory_lock_wait_sec()
    sess.daily_auto_ingest_message = (
        f"Queued: waiting for server ({wait_sec}s max)… ({fnames})"
    )
    _log.info("daily-auto ingest waiting for memory lock (session=%s)", getattr(sess, "_persist_sid", "")[:8])
    if _UPLOAD_MEMORY_LOCK.acquire(timeout=wait_sec):
        return True
    _log.warning("daily-auto ingest timeout waiting for memory lock — running anyway")
    sess.daily_auto_ingest_message = f"Parsing files… ({fnames})"
    return False


def _run_daily_auto_ingest_pipeline(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Parse Tier-3 upload off the request thread, then rebuild sales in the same worker."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    route_notes: list[str] = []
    try:
        from ..services.upload_file_sniff import partition_files_by_upload_target

        buckets, route_notes = partition_files_by_upload_target(file_parts, "daily_sales")
        file_parts = list(buckets.get("daily_sales") or [])
        inv_parts = list(buckets.get("snapshot_inventory") or [])
        ret_parts = list(buckets.get("returns") or [])
        if inv_parts:
            INVENTORY_EXECUTOR.submit(_run_inventory_auto_from_parts, session_id, inv_parts)
        if ret_parts:
            file_parts = list(file_parts) + ret_parts
        if route_notes:
            note = "Auto-routed: " + "; ".join(route_notes)
            _log.info("daily-auto auto-route session=%s %s", session_id[:8], note)
            sess.daily_auto_ingest_message = note
    except Exception:
        _log.exception("daily-auto worker upload partition failed")
        file_parts = file_parts
    if not file_parts:
        if route_notes:
            msg = "Auto-routed to the correct upload handler — " + "; ".join(route_notes)
            _store_daily_auto_ingest_result(
                sess,
                {
                    "ok": True,
                    "message": msg,
                    "auto_routed": route_notes,
                    "detected_platforms": [],
                    "warnings": [],
                    "processed_files": 0,
                    "detected_files": 0,
                    "unknown_files": 0,
                },
            )
            sess.daily_auto_ingest_status = "done"
            sess.daily_auto_ingest_message = msg
            sess.sales_rebuild_status = "idle"
            sess.sales_rebuild_message = ""
        return
    sess.daily_auto_ingest_status = "running"
    sess.daily_auto_ingest_started = time.time()
    n = len(file_parts)
    fnames = ", ".join(name for name, _ in file_parts[:3])
    lock_held = _acquire_ingest_memory_lock(sess, fnames=fnames)
    if lock_held:
        sess.daily_auto_ingest_message = f"Parsing {n} file{'s' if n != 1 else ''}… ({fnames})"
    try:
        payload = _process_daily_auto_sync(sess, file_parts, rebuild_sales=False)
    except Exception as e:
        _log.exception("daily-auto background ingest")
        err_payload = {
            "ok": False,
            "message": str(e),
            "detected_platforms": [],
            "warnings": [str(e)],
            "processed_files": n,
            "detected_files": 0,
            "unknown_files": n,
        }
        _store_daily_auto_ingest_result(sess, err_payload)
        sess.daily_auto_ingest_status = "error"
        sess.daily_auto_ingest_message = str(e)
        sess.sales_rebuild_status = "error"
        sess.sales_rebuild_message = f"Ingest failed: {e}"
        return
    finally:
        if lock_held:
            _UPLOAD_MEMORY_LOCK.release()

    _store_daily_auto_ingest_result(sess, payload)
    if not payload.get("ok"):
        # Use "done" (not "error") so the frontend's waitForDailyAutoIngest returns
        # normally and captureGenericAlert shows the import-completeness box with
        # per-file details and warnings.  The result payload has ok=False so the
        # frontend still shows an error toast via the `failed` flag.
        sess.daily_auto_ingest_status = "done"
        sess.daily_auto_ingest_message = str(payload.get("message") or "No files detected.")
        sess.daily_auto_ingest_started = 0.0
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""
        return

    sess.daily_auto_ingest_status = "done"
    sess.daily_auto_ingest_message = str(payload.get("message") or "Ingest finished.")
    sess.daily_auto_ingest_started = 0.0

    if payload.get("sales_rebuild") == "pending":
        _n_sales = len(getattr(sess, "sales_df", None) or [])
        sess.sales_rebuild_status = "running"
        sess.sales_rebuild_started = time.time()
        sess.sales_rebuild_message = (
            f"Updating dashboard sales ({_n_sales:,} rows)…"
            if _n_sales else "Updating dashboard sales…"
        )
        try:
            from ..db.forecast_session_pg import persist_session_bundle_thread_safe

            persist_session_bundle_thread_safe(session_id, sess)
        except Exception:
            _log.exception("PostgreSQL persist after Tier-3 background ingest")
        AUX_EXECUTOR.submit(_run_daily_auto_sales_rebuild, session_id)
    else:
        sess.sales_rebuild_status = "idle"
        sess.sales_rebuild_message = ""


def _auto_save_sku_mapping_cache(sess) -> None:
    """Persist SKU map to disk, GitHub, and PostgreSQL without requiring platform uploads."""
    mapping = getattr(sess, "sku_mapping", None) or {}
    if not mapping:
        return
    persist_sku_mapping_globally(mapping)
    try:
        from ..services.github_cache import save_cache_to_drive

        ok, msg = save_cache_to_drive({"sku_mapping": mapping})
        if ok:
            _log.info("SKU mapping GitHub save: %s", msg)
        else:
            _log.warning("SKU mapping GitHub save skipped: %s", msg)
    except Exception:
        _log.exception("SKU mapping GitHub save failed")
    sid = getattr(sess, "_persist_sid", None)
    if sid:
        try:
            from ..db.forecast_session_pg import persist_session_bundle

            if persist_session_bundle(sid, sess):
                _log.info("PostgreSQL session snapshot saved after SKU upload (%s…)", sid[:8])
        except Exception:
            _log.exception("PostgreSQL session snapshot failed after SKU upload")


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

    try:
        import backend.main as _main

        _main.publish_warm_cache_from_session(sess)
    except Exception:
        _log.exception("warm cache publish after auto-save failed")

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
    if sess is not None:
        return sess
    sid = getattr(request.state, "session_id", None) or request.cookies.get("session_id")
    if sid:
        recovered = _resolve_upload_session(sid)
        if recovered is not None:
            request.state.session = recovered
            request.state.session_id = sid
            setattr(recovered, "_persist_sid", sid)
            return recovered
    raise HTTPException(
        status_code=503,
        detail="Session is still loading — refresh the page and try again in a few seconds.",
    )


def _resolve_upload_session(session_id: str) -> AppSession | None:
    """Session for background inventory ingest — reuse PG/warm state, never a blank frame."""
    if not session_id:
        return None
    sess = store._sessions.get(session_id)
    if sess is not None:
        try:
            import backend.main as _main

            if _main.session_needs_operational_data(sess):
                from ..db.forecast_session_pg import load_session_from_pg, pg_session_persist_enabled

                if pg_session_persist_enabled():
                    loaded = load_session_from_pg(session_id)
                    if loaded is not None and not _main.session_needs_operational_data(loaded):
                        loaded.last_accessed = time.time()
                        store._sessions[session_id] = loaded
                        _log.info(
                            "Upload session %s… replaced shell with PostgreSQL restore",
                            session_id[:8],
                        )
                        return loaded
        except Exception:
            _log.exception("hydrate upload shell session %s", session_id[:8])
        sess.last_accessed = time.time()
        return sess
    try:
        from ..db.forecast_session_pg import load_session_from_pg, pg_session_persist_enabled

        if pg_session_persist_enabled():
            loaded = load_session_from_pg(session_id)
            if loaded is not None:
                loaded.last_accessed = time.time()
                store._sessions[session_id] = loaded
                _log.info(
                    "Upload session %s… restored from PostgreSQL for background ingest",
                    session_id[:8],
                )
                return loaded
    except Exception:
        _log.exception("PostgreSQL restore for upload session %s", session_id[:8])
    _log.warning(
        "Upload session %s… missing from RAM — creating session and syncing warm cache",
        session_id[:8],
    )
    sess = AppSession()
    sess.last_accessed = time.time()
    store._sessions[session_id] = sess
    try:
        from ..services.inventory import sync_inventory_snapshot_from_warm

        sync_inventory_snapshot_from_warm(sess)
    except Exception:
        pass
    return sess


def _clear_stuck_daily_ingest(sess: AppSession, *, force: bool = False) -> bool:
    """Reset a session stuck in daily_auto_ingest_status=running."""
    if getattr(sess, "daily_auto_ingest_status", "idle") != "running":
        return False
    started = float(getattr(sess, "daily_auto_ingest_started", 0) or 0)
    age = time.time() - started if started > 0 else 999999
    # Auto-clear after 3 minutes by default — large RAR archives parse in < 2 min.
    # Override with DAILY_INGEST_STUCK_SEC env var if needed.
    stuck_sec = int(os.environ.get("DAILY_INGEST_STUCK_SEC", "180"))
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


def _mark_daily_auto_ingest_running(sess: AppSession, message: str) -> None:
    """Set ingest status without waiting on ``_daily_restore_lock``.

    ``chunk/complete`` and ``/daily-auto`` used to call ``_session_lock_apply`` here,
    which blocked behind an in-flight ingest until Cloudflare returned 502 (~100s).
    """
    sess.daily_auto_ingest_status = "running"
    sess.daily_auto_ingest_started = time.time()
    sess.daily_auto_ingest_message = message
    sess.daily_auto_ingest_result = {}
    sess.sales_rebuild_status = "idle"
    sess.sales_rebuild_message = ""


def _mark_inventory_upload_running(sess: AppSession, message: str, *, progress: int = 2) -> None:
    """Set inventory ingest status without waiting on ``_daily_restore_lock``.

    ``/inventory-auto`` and chunked finalize used ``_session_lock_apply`` here, which
    blocked behind a long RAR parse until the gateway returned 502 (~100s).
    """
    if getattr(sess, "inventory_upload_status", "idle") != "running":
        backup_inventory_before_upload(sess)
    sess.inventory_upload_status = "running"
    sess.inventory_upload_started = time.time()
    _set_inventory_upload_progress(sess, progress, message)
    sess.inventory_upload_result = {}


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


def _tier1_bulk_busy(sess: AppSession) -> bool:
    return getattr(sess, "tier1_bulk_status", "idle") == "running"


def _mark_tier1_bulk_running(sess: AppSession, platform: str, message: str) -> None:
    sess.tier1_bulk_status = "running"
    sess.tier1_bulk_platform = platform
    sess.tier1_bulk_message = message
    sess.tier1_bulk_started = time.time()
    sess.tier1_bulk_result = {}


def _finish_tier1_bulk(sess: AppSession, resp: UploadResponse) -> None:
    sess.tier1_bulk_started = 0.0
    if resp.ok:
        sess.tier1_bulk_status = "done"
        sess.tier1_bulk_message = resp.message
        try:
            sess.tier1_bulk_result = resp.model_dump()
        except Exception:
            sess.tier1_bulk_result = {"ok": True, "message": resp.message, "rows": resp.rows}
    else:
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = resp.message
        sess.tier1_bulk_result = {"ok": False, "message": resp.message}


def _tier1_rows(sess: AppSession, attr: str) -> int:
    cur = getattr(sess, attr, None)
    if cur is None or not hasattr(cur, "__len__"):
        return 0
    return len(cur)


def _accept_tier1_bulk_async(
    sess: AppSession,
    session_id: str,
    *,
    platform: str,
    label: str,
    filename: str,
    rows_attr: str,
    submit,
) -> UploadResponse:
    if _tier1_bulk_busy(sess):
        return UploadResponse(
            ok=False,
            message="Another bulk upload is still processing — wait for the row count to update.",
        )
    _mark_tier1_bulk_running(sess, platform, f"Queued {label}: {filename or 'upload'}…")
    submit(session_id)
    return UploadResponse(
        ok=True,
        message=(
            f"{label} upload accepted — parsing in background. "
            "Large archives may take several minutes; watch the row count below."
        ),
        rows=_tier1_rows(sess, rows_attr),
    )


def _run_tier1_mtr_worker(session_id: str, raw: bytes, orig_name: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
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
            _fd, saved_rows, _block = save_daily_file("amazon", orig_name or "mtr-upload.zip", df)
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

        resp = _session_lock_apply_sync(sess, work)
        _finish_tier1_bulk(sess, resp)
        if resp.ok:
            _auto_save_cache(sess)
    except Exception as e:
        _log.exception("tier1 mtr worker failed")
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = str(e)[:500]
        sess.tier1_bulk_started = 0.0
    finally:
        del raw
        gc.collect()


def _run_tier1_myntra_worker(session_id: str, zip_bytes: bytes, orig_fn: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
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
            _fd, saved_rows, _block = save_daily_file("myntra", orig_fn or "myntra-upload.zip", df)
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

        resp = _session_lock_apply_sync(sess, work)
        _finish_tier1_bulk(sess, resp)
        if resp.ok:
            _auto_save_cache(sess)
    except Exception as e:
        _log.exception("tier1 myntra worker failed")
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = str(e)[:500]
        sess.tier1_bulk_started = 0.0
    finally:
        del zip_bytes
        gc.collect()


def _run_tier1_meesho_worker(session_id: str, raw_bytes: bytes, fname: str, display_name: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    try:
        def work():
            if fname.endswith((".xlsx", ".xls")):
                df, msg = parse_meesho_order_export_xlsx(raw_bytes)
                if not df.empty:
                    df = apply_dsr_segment_from_upload_filename(df, display_name or None, "Meesho")
                    pre_total = len(sess.meesho_df)
                    _fd, saved_rows, _block = save_daily_file("meesho", display_name or "meesho-order.xlsx", df)
                    sess.meesho_df = _merge_platform_data(
                        sess.meesho_df, df, "meesho", source_filename=display_name or None,
                    )
                    sess.sales_df = build_sales_df(
                        mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                        flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                        sku_mapping=sess.sku_mapping, **_sales_overlay_build_kwargs(sess),
                    )
                    total = len(sess.meesho_df)
                    years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
                    skus = int((sess.meesho_df["SKU"].astype(str).str.strip() != "").sum())
                    kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                        parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
                    )
                    _session_data_changed(sess)
                    return UploadResponse(
                        ok=True,
                        message=(
                            f"Meesho order export (Excel): added {kept_rows:,} rows ({skus:,} with SKU). "
                            f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                        ),
                        rows=total, parsed_rows=len(df), kept_rows=kept_rows,
                        dropped_rows=dropped_rows, dropped_reasons=dropped_reasons or None,
                        years=years,
                    )
                return UploadResponse(
                    ok=False,
                    message=(
                        f"Excel is not a recognised Meesho sales export ({msg}). "
                        "Expected columns TxnDate, Transaction Type, Sku, Quantity."
                    ),
                )

            is_csv = fname.endswith(".csv") or (not fname.endswith(".zip") and raw_bytes[:3] != b"PK\x03")
            if is_csv:
                from ..services.meesho import parse_meesho_csv
                df, msg = parse_meesho_csv(raw_bytes)
                if df.empty:
                    return UploadResponse(ok=False, message=f"Meesho CSV parse error: {msg}")
                df = apply_dsr_segment_from_upload_filename(df, display_name or None, "Meesho")
                pre_total = len(sess.meesho_df)
                _fd, saved_rows, _block = save_daily_file("meesho", display_name or "meesho-orders.csv", df)
                sess.meesho_df = _merge_platform_data(
                    sess.meesho_df, df, "meesho", source_filename=display_name or None,
                )
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping, **_sales_overlay_build_kwargs(sess),
                )
                total = len(sess.meesho_df)
                years = sorted(sess.meesho_df["Date"].dt.year.dropna().unique().astype(int).tolist())
                kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                    parsed_rows=len(df), pre_total=pre_total, post_total=total, saved_rows=saved_rows,
                )
                _session_data_changed(sess)
                return UploadResponse(
                    ok=True,
                    message=(
                        f"Meesho Order CSV: added {kept_rows:,} rows. "
                        f"Parsed: {len(df):,}, Kept: {kept_rows:,}. Total: {total:,} rows."
                    ),
                    rows=total, parsed_rows=len(df), kept_rows=kept_rows,
                    dropped_rows=dropped_rows, dropped_reasons=dropped_reasons or None,
                    years=years,
                )

            df, zip_count, skipped = load_meesho_from_zip(raw_bytes, source_filename=display_name or None)
            if df.empty:
                return UploadResponse(
                    ok=False, message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )
            pre_total = len(sess.meesho_df)
            _fd, saved_rows, _block = save_daily_file("meesho", display_name or "meesho-upload.zip", df)
            sess.meesho_df = _merge_platform_data(
                sess.meesho_df, df, "meesho", source_filename=display_name or None,
            )
            sess.sales_df = build_sales_df(
                mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                sku_mapping=sess.sku_mapping, **_sales_overlay_build_kwargs(sess),
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
                ),
                rows=total, parsed_rows=len(df), kept_rows=kept_rows,
                dropped_rows=dropped_rows, dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None, years=years,
            )

        resp = _session_lock_apply_sync(sess, work)
        _finish_tier1_bulk(sess, resp)
        if resp.ok:
            _auto_save_cache(sess)
    except Exception as e:
        _log.exception("tier1 meesho worker failed")
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = str(e)[:500]
        sess.tier1_bulk_started = 0.0
    finally:
        del raw_bytes
        gc.collect()


def _run_tier1_flipkart_worker(session_id: str, tmp_path: str, orig_fn: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    try:
        def work():
            if not sess.sku_mapping:
                return UploadResponse(ok=False, message="Upload SKU Mapping first.")
            df, xlsx_count, skipped = load_flipkart_from_zip(
                tmp_path, sess.sku_mapping, source_filename=orig_fn or None,
            )
            gc.collect()
            if df.empty:
                return UploadResponse(
                    ok=False, message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )
            pre_total = len(sess.flipkart_df)
            _fd2, saved_rows, _block = save_daily_file("flipkart", orig_fn or "flipkart-upload.zip", df)
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
                ),
                rows=total, parsed_rows=len(df), kept_rows=kept_rows,
                dropped_rows=dropped_rows, dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None, years=years,
            )

        resp = _session_lock_apply_sync(sess, work)
        _finish_tier1_bulk(sess, resp)
        if resp.ok:
            _auto_save_cache(sess)
    except Exception as e:
        _log.exception("tier1 flipkart worker failed")
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = str(e)[:500]
        sess.tier1_bulk_started = 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        gc.collect()


def _run_tier1_snapdeal_worker(session_id: str, zip_bytes: bytes, display: str, snap_fn: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    try:
        def work():
            df, file_count, skipped, parse_info = load_snapdeal_from_zip(zip_bytes, sess.sku_mapping, display)
            sess.snapdeal_parse_info.update(parse_info)
            if df.empty:
                return UploadResponse(
                    ok=False, message=f"No data extracted. Issues: {'; '.join(skipped[:5])}",
                )
            pre_total = len(sess.snapdeal_df)
            _fd, saved_rows, _block = save_daily_file("snapdeal", snap_fn or "snapdeal-upload.zip", df)
            if sess.snapdeal_df.empty:
                sess.snapdeal_df = df
            else:
                sess.snapdeal_df = pd.concat([sess.snapdeal_df, df], ignore_index=True).drop_duplicates()
            post_total = len(sess.snapdeal_df)
            kept_rows, dropped_rows, dropped_reasons = _upload_quality_from_merge(
                parsed_rows=len(df), pre_total=pre_total, post_total=post_total, saved_rows=saved_rows,
            )
            val_warn = _collect_validation_warnings(skipped)
            if sess.sku_mapping:
                sess.sales_df = build_sales_df(
                    mtr_df=sess.mtr_df, myntra_df=sess.myntra_df, meesho_df=sess.meesho_df,
                    flipkart_df=sess.flipkart_df, snapdeal_df=sess.snapdeal_df,
                    sku_mapping=sess.sku_mapping, **_sales_overlay_build_kwargs(sess),
                )
                sess._quarterly_cache.clear()
            years = sorted(df["Date"].dt.year.dropna().unique().astype(int).tolist())
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=(
                    f"Snapdeal loaded: added {kept_rows:,} rows from {file_count} file(s). "
                    f"Parsed: {len(df):,}, Kept: {kept_rows:,}."
                ),
                rows=post_total, parsed_rows=len(df), kept_rows=kept_rows,
                dropped_rows=dropped_rows, dropped_reasons=dropped_reasons or None,
                validation_warnings=val_warn or None, years=years,
            )

        resp = _session_lock_apply_sync(sess, work)
        _finish_tier1_bulk(sess, resp)
        if resp.ok:
            _auto_save_cache(sess)
    except Exception as e:
        _log.exception("tier1 snapdeal worker failed")
        sess.tier1_bulk_status = "error"
        sess.tier1_bulk_message = str(e)[:500]
        sess.tier1_bulk_started = 0.0
    finally:
        del zip_bytes
        gc.collect()


def _clear_stuck_tier1_bulk(sess: AppSession, *, force: bool = False) -> bool:
    if getattr(sess, "tier1_bulk_status", "idle") != "running":
        return False
    started = float(getattr(sess, "tier1_bulk_started", 0) or 0)
    age = time.time() - started if started > 0 else 999999
    stuck_sec = int(os.environ.get("TIER1_BULK_STUCK_SEC", "1800"))
    if not force and age < stuck_sec:
        return False
    sess.tier1_bulk_status = "error"
    sess.tier1_bulk_message = (
        "Previous bulk upload did not finish (server was busy). "
        "Check row counts — re-upload if needed."
    )
    sess.tier1_bulk_started = 0.0
    return True


# ── SKU Mapping ───────────────────────────────────────────────

@router.post("/sku-mapping", response_model=UploadResponse)
async def upload_sku_mapping(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    sess = _get_session(request)
    try:
        file_bytes = await file.read()

        def work():
            parsed = parse_sku_mapping(file_bytes)
            base = resolve_sku_mapping_base(sess)
            mapping = merge_sku_mapping_upload(base, parsed)
            updated = sum(1 for k, v in parsed.items() if base.get(k) != v)
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
                    **_sales_overlay_build_kwargs(sess),
                )

            gaps = list_sku_mapping_gaps(sess.sales_df, mapping)
            msg = f"SKU mapping loaded: {len(parsed):,} rows from file"
            if base:
                msg += f" merged into master ({len(mapping):,} total"
                if updated:
                    msg += f", {updated:,} updated"
                msg += ")"
            else:
                msg += f" ({len(mapping):,} total entries)"
            if had_platform:
                msg += f"; sales rebuilt ({len(sess.sales_df):,} rows)"
            if gaps:
                msg += (
                    f". Warning: {len(gaps)} SKU(s) in sales are not in this map "
                    f"(as seller key or OMS value) — add them to the master or fix typos."
                )

            _session_data_changed(sess)
            try:
                from ..services.po_raise_remove import invalidate_po_calculate_result
                from ..services.po_shared_cache import invalidate_all_shared_caches

                invalidate_po_calculate_result(sess)
                invalidate_all_shared_caches()
            except Exception:
                pass
            return UploadResponse(
                ok=True,
                message=msg,
                sku_count=len(mapping),
                unmapped_skus=gaps or None,
            ), had_platform

        resp, had_platform = await _session_lock_apply(sess, work)
        background_tasks.add_task(_auto_save_sku_mapping_cache, sess)
        if had_platform:
            background_tasks.add_task(_auto_save_cache, sess)
        return resp
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse SKU mapping: {e}")


# ── Amazon MTR ────────────────────────────────────────────────

@router.post("/mtr", response_model=UploadResponse)
async def upload_mtr(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", "") or ""
    raw = await file.read()
    orig_name = file.filename or ""
    try:
        return _accept_tier1_bulk_async(
            sess,
            sid,
            platform="amazon",
            label="Amazon MTR",
            filename=orig_name,
            rows_attr="mtr_df",
            submit=lambda s: DAILY_UPLOAD_EXECUTOR.submit(_run_tier1_mtr_worker, s, raw, orig_name),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to accept MTR archive: {e}")


# ── Myntra ────────────────────────────────────────────────────

@router.post("/myntra", response_model=UploadResponse)
async def upload_myntra(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", "") or ""
    zip_bytes = await file.read()
    orig_fn = file.filename or ""
    try:
        return _accept_tier1_bulk_async(
            sess,
            sid,
            platform="myntra",
            label="Myntra",
            filename=orig_fn,
            rows_attr="myntra_df",
            submit=lambda s: DAILY_UPLOAD_EXECUTOR.submit(_run_tier1_myntra_worker, s, zip_bytes, orig_fn),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to accept Myntra ZIP: {e}")


# ── Meesho ────────────────────────────────────────────────────

@router.post("/meesho", response_model=UploadResponse)
async def upload_meesho(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", "") or ""
    raw_bytes = await file.read()
    fname = (file.filename or "").lower()
    display_name = file.filename or ""
    try:
        return _accept_tier1_bulk_async(
            sess,
            sid,
            platform="meesho",
            label="Meesho",
            filename=display_name,
            rows_attr="meesho_df",
            submit=lambda s: DAILY_UPLOAD_EXECUTOR.submit(
                _run_tier1_meesho_worker, s, raw_bytes, fname, display_name,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to accept Meesho file: {e}")


# ── Flipkart ──────────────────────────────────────────────────

@router.post("/flipkart", response_model=UploadResponse)
async def upload_flipkart(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", "") or ""
    tmp_path: Optional[str] = None
    orig_fn = file.filename or ""
    try:
        _fd, tmp_path = tempfile.mkstemp(suffix=".zip")
        os.close(_fd)
        chunk_size = 8 * 1024 * 1024
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
        path_copy = tmp_path
        return _accept_tier1_bulk_async(
            sess,
            sid,
            platform="flipkart",
            label="Flipkart",
            filename=orig_fn,
            rows_attr="flipkart_df",
            submit=lambda s: DAILY_UPLOAD_EXECUTOR.submit(_run_tier1_flipkart_worker, s, path_copy, orig_fn),
        )
    except Exception as e:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise HTTPException(status_code=422, detail=f"Failed to accept Flipkart ZIP: {e}")


# ── Snapdeal ──────────────────────────────────────────────────

@router.post("/snapdeal", response_model=UploadResponse)
async def upload_snapdeal(request: Request, file: UploadFile = File(...)):
    sess = _get_session(request)
    sid = getattr(request.state, "session_id", "") or ""
    zip_bytes = await file.read()
    display = file.filename or "upload"
    snap_fn = file.filename or ""
    try:
        return _accept_tier1_bulk_async(
            sess,
            sid,
            platform="snapdeal",
            label="Snapdeal",
            filename=snap_fn,
            rows_attr="snapdeal_df",
            submit=lambda s: DAILY_UPLOAD_EXECUTOR.submit(
                _run_tier1_snapdeal_worker, s, zip_bytes, display, snap_fn,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to accept Snapdeal ZIP: {e}")


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


def _set_inventory_upload_progress(sess: AppSession, pct: int, message: str) -> None:
    sess.inventory_upload_progress = max(0, min(100, int(pct)))
    sess.inventory_upload_message = message


def _clear_stuck_inventory_upload(sess: AppSession, *, force: bool = False) -> bool:
    """Reset a session stuck in inventory_upload_status=running."""
    st = getattr(sess, "inventory_upload_status", "idle")
    if st not in ("running", "error"):
        return False
    if st == "running":
        started = float(getattr(sess, "inventory_upload_started", 0) or 0)
        age = time.time() - started if started > 0 else 999999
        stuck_sec = int(os.environ.get("INVENTORY_UPLOAD_STUCK_SEC", "300"))
        if not force and age < stuck_sec:
            return False
    if force:
        msg = "Upload cleared — you can upload again."
    else:
        msg = (
            "Previous inventory upload did not finish (timed out or server was busy). "
            "Try uploading again."
        )
    sess.inventory_upload_status = "idle"
    sess.inventory_upload_progress = 0
    sess.inventory_upload_started = 0.0
    sess.inventory_upload_message = msg if not force else ""
    if not force:
        sess.inventory_upload_result = {"ok": False, "message": msg, "warnings": [msg]}
    return True


def _acquire_inventory_memory_lock(sess: AppSession, session_id: str) -> bool:
    """Wait briefly for the global upload lock, then parse anyway (avoid infinite queue)."""
    wait_sec = int(os.environ.get("INVENTORY_MEMORY_LOCK_WAIT_SEC", "120"))
    if _UPLOAD_MEMORY_LOCK.acquire(blocking=False):
        return True
    _set_inventory_upload_progress(
        sess, 8, f"Queued — waiting for server ({wait_sec}s max)…",
    )
    _log.info("inventory-auto queued behind another heavy job (session=%s)", session_id[:8])
    if _UPLOAD_MEMORY_LOCK.acquire(timeout=wait_sec):
        return True
    _log.warning(
        "inventory-auto proceeding without memory lock (session=%s)", session_id[:8],
    )
    _set_inventory_upload_progress(
        sess, 12, "Parsing inventory (server finishing cache load in background)…",
    )
    return False


def _persist_inventory_after_upload(sess: AppSession, session_id: str | None = None) -> bool:
    """Update warm cache + PostgreSQL immediately so reload/login sees the snapshot."""
    try:
        import backend.main as _main

        _main.merge_inventory_into_warm_cache(sess)
        hist = getattr(sess, "daily_inventory_history_df", None)
        if hist is not None and hasattr(hist, "empty") and not hist.empty:
            _main.merge_po_optional_sheets_into_warm_cache(sess)
    except Exception:
        _log.exception("merge_inventory_into_warm_cache failed")
    sid = (session_id or getattr(sess, "_persist_sid", None) or "").strip() or None
    if not sid:
        return False
    setattr(sess, "_persist_sid", sid)
    try:
        from ..db.forecast_session_pg import persist_session_bundle

        ok = persist_session_bundle(sid, sess)
        if not ok:
            _log.warning(
                "PostgreSQL persist after inventory upload returned false (session %s…)",
                sid[:8],
            )
        return ok
    except Exception:
        _log.exception("PostgreSQL persist after inventory upload")
        return False


def _schedule_inventory_github_cache_save(sess: AppSession) -> None:
    """GitHub cache save can take minutes — never block the upload response on it."""

    def _run() -> None:
        try:
            _auto_save_cache(sess)
        except Exception:
            _log.exception("background inventory GitHub cache save failed")

    threading.Thread(target=_run, name="inv-cache-save", daemon=True).start()


def _finish_inventory_server_save(sess: AppSession, session_id: str | None = None) -> None:
    """Sync warm + PG, then background GitHub (call after every successful inventory parse)."""
    pg_ok = _persist_inventory_after_upload(sess, session_id)
    if not pg_ok and session_id:
        warn = (
            "Inventory parsed but server save was delayed — open Inventory again in a few seconds "
            "or click Load Cache if totals look stale."
        )
        prev = getattr(sess, "inventory_upload_result", None) or {}
        if isinstance(prev, dict):
            merged = dict(prev)
            merged.setdefault("warnings", [])
            if isinstance(merged["warnings"], list):
                merged["warnings"] = list(dict.fromkeys([*merged["warnings"], warn]))
            sess.inventory_upload_result = merged
    _schedule_inventory_github_cache_save(sess)


def _build_inventory_upload_payload(
    *,
    df_variant: pd.DataFrame,
    debug: dict,
    detected: list[str],
    file_parts: list[tuple[str, bytes]],
    warnings: list[str] | None = None,
) -> dict:
    """Structured result for UI: what loaded, what skipped, SKU counts per source."""
    file_results: list[dict] = list(debug.get("rar_manifest") or [])
    if not file_results:
        for fname, _raw in file_parts:
            file_results.append({
                "filename": fname,
                "category": "upload",
                "status": "loaded" if fname in str(detected) else "loaded",
            })
    saved = sum(1 for r in file_results if r.get("status") == "loaded")
    skipped = sum(1 for r in file_results if r.get("status") == "skipped")
    sources: list[str] = []
    for key, label in (
        ("oms", "OMS"),
        ("combo_rar", "Combo SKUs"),
        ("flipkart", "Flipkart"),
        ("myntra", "Myntra"),
        ("amz", "Amazon"),
        ("fba", "FBA in-transit"),
    ):
        val = debug.get(key)
        if val is not None and str(val).strip() and not str(val).strip().startswith("0 SKUs"):
            sources.append(f"{label}: {val}")
    rows = len(df_variant)
    snap_label = debug.get("snapshot_date_label") or debug.get("snapshot_date") or ""
    parts = [f"{rows:,} SKUs in snapshot"]
    if snap_label:
        parts.insert(0, f"Snapshot as of {snap_label}")
    if sources:
        parts.append("Sources — " + "; ".join(sources))
    if skipped:
        parts.append(f"{skipped} file(s) inside archive skipped")
    if warnings:
        parts.append("; ".join(warnings[:3]))
    missing_mkt = inventory_missing_marketplace_warnings(debug)
    all_warnings = list(dict.fromkeys([*(warnings or []), *missing_mkt]))
    return {
        "ok": True,
        "message": " | ".join(parts),
        "rows": rows,
        "debug": debug,
        "detected": detected,
        "warnings": all_warnings,
        "file_results": file_results,
        "processed_files": len(file_parts),
        "saved_files": saved or len(file_parts),
        "skipped_files": skipped,
        "sources_summary": sources,
        "v": "inv-v10",
    }


def _inventory_parse_heavy(
    sess: AppSession,
    *,
    oms_bytes_list: list[bytes],
    fk_bytes: list[bytes] | None,
    myntra_bytes: list[bytes] | None,
    amz_bytes: bytes | None,
    sku_mapping: dict,
    warnings: list[str],
) -> tuple[Any, Any, dict]:
    """CPU/RAM-heavy inventory parse — no session lock (progress fields updated on sess)."""
    _set_inventory_upload_progress(sess, 45, "Merging OMS, Flipkart, Myntra, and Amazon stock…")
    df_variant, debug = load_inventory_consolidated(
        oms_bytes_list or None,
        fk_bytes,
        myntra_bytes,
        amz_bytes,
        sku_mapping,
        group_by_parent=False,
        return_debug=True,
    )
    if df_variant.empty:
        warnings.append("No SKUs parsed — check SKU mapping and file formats.")
    for m in debug.get("rar_manifest") or []:
        if m.get("status") == "skipped":
            warnings.append(f"{m.get('filename', '?')}: {m.get('reason', 'skipped')}")
    try:
        from ..services.helpers import get_parent_sku

        df_parent = df_variant.copy()
        inv_cols = [
            c
            for c in df_parent.columns
            if c.endswith("_Inventory")
            or c.endswith("_Live")
            or c.endswith("_InTransit")
            or c in ("Buffer_Stock", "Marketplace_Total", "Total_Inventory")
        ]
        df_parent["Parent_SKU"] = df_parent["OMS_SKU"].apply(get_parent_sku)
        df_parent = (
            df_parent.groupby("Parent_SKU")[inv_cols]
            .sum()
            .reset_index()
            .rename(columns={"Parent_SKU": "OMS_SKU"})
        )
    except Exception:
        df_parent = df_variant
    _set_inventory_upload_progress(sess, 80, "Building parent-SKU rollup…")
    return df_variant, df_parent, debug


def _inventory_apply_parse_result(
    sess: AppSession,
    *,
    df_variant: Any,
    df_parent: Any,
    debug: dict,
    file_parts: list[tuple[str, bytes]],
    detected: list[str],
    warnings: list[str],
) -> dict:
    """Commit parsed inventory to the session (brief lock)."""
    _set_inventory_upload_progress(sess, 95, "Finalizing snapshot…")
    missing_oms = upload_bundle_expects_oms(file_parts) and not oms_loaded_in_debug(debug)
    if missing_oms:
        warnings.append("OMS inventory CSV missing or empty inside the bundle.")
        # Do NOT blindly reject (keep old snapshot) when OMS parsing fails:
        # other marketplace stock layers may still be present and yield a usable
        # Total_Inventory. In that case, it is better to show the updated
        # marketplace-based snapshot than keep stale data.
        #
        # Only reject when the parsed snapshot is effectively empty/zero.
        parsed_non_zero = False
        try:
            if hasattr(df_variant, "empty") and not df_variant.empty and "Total_Inventory" in getattr(df_variant, "columns", []):
                import pandas as pd

                s = pd.to_numeric(df_variant["Total_Inventory"], errors="coerce").fillna(0).sum()
                parsed_non_zero = float(s) > 0
            elif hasattr(df_variant, "empty") and not df_variant.empty:
                # Fallback: if we parsed rows at all, treat as usable.
                parsed_non_zero = True
        except Exception:
            parsed_non_zero = False

        if not parsed_non_zero and restore_inventory_upload_backup(sess):
            msg = (
                "Upload rejected — OMS inventory file was not found in the bundle (and the parsed snapshot is empty). "
                "Your previous snapshot was kept. Include the OMS CSV inside the RAR (or upload it separately) and try again."
            )
            sess.inventory_upload_status = "error"
            sess.inventory_upload_progress = 0
            payload = {
                "ok": False,
                "message": msg,
                "rows": int(len(sess.inventory_df_variant)),
                "warnings": list(dict.fromkeys(warnings)),
                "debug": dict(getattr(sess, "inventory_debug", None) or debug),
            }
            sess.inventory_upload_result = payload
            sess.inventory_upload_message = msg
            sess.inventory_upload_started = 0.0
            return payload

    sess.inventory_df_variant = df_variant
    sess.inventory_df_parent = df_parent
    try:
        from ..services.manual_intransit_sheet import apply_manual_intransit_overlay_to_inventory

        _mit_df = getattr(sess, "manual_intransit_overlay_df", None)
        if _mit_df is not None and not _mit_df.empty:
            apply_manual_intransit_overlay_to_inventory(sess)
    except Exception:
        _log.exception("re-apply manual in-transit overlay after inventory snapshot failed")

    # Auto-apply any return columns found in the OMS snapshot as a return overlay.
    # Columns like "Amazon Returns Processing" / "Flipkart Returns" represent units
    # currently in return transit; they reduce net demand and adjust PO qty accordingly.
    try:
        from ..services.inventory import oms_returns_to_overlay

        snapshot_date = getattr(sess, "inventory_snapshot_date", None) or None
        oms_ov = oms_returns_to_overlay(df_variant, snapshot_date)
        if not oms_ov.empty:
            import pandas as _pd
            existing_ov = getattr(sess, "po_return_overlay_df", None)
            if existing_ov is not None and not existing_ov.empty:
                # Remove any previous OMS-sourced rows (keyed by source "oms") to avoid
                # double-counting when the inventory snapshot is re-uploaded.
                keep = existing_ov[
                    existing_ov.get("Return_Source", _pd.Series("", index=existing_ov.index)) != "oms"
                ] if "Return_Source" in existing_ov.columns else existing_ov
                oms_ov["Return_Source"] = "oms"
                sess.po_return_overlay_df = _pd.concat([keep, oms_ov], ignore_index=True)
            else:
                oms_ov["Return_Source"] = "oms"
                sess.po_return_overlay_df = oms_ov
            sess.return_overlay_as_of = snapshot_date or __import__("datetime").date.today().isoformat()
            total_return_units = int(oms_ov["Return_Units"].sum())
            _log.info(
                "OMS returns auto-applied as return overlay: %d SKUs / %d units across %s",
                oms_ov["OMS_SKU"].nunique(),
                total_return_units,
                oms_ov["Return_Platform"].unique().tolist(),
            )
    except Exception:
        _log.exception("OMS returns-to-overlay failed; inventory still saved")

    apply_inventory_snapshot_metadata(sess, file_parts, debug)
    try:
        from ..services.daily_inventory_history import append_snapshot_inventory_to_history

        hist = append_snapshot_inventory_to_history(sess)
        if hist.get("appended"):
            _log.info(
                "Daily snapshot appended to inventory history: %s (%s SKU-days)",
                hist.get("snapshot_date"),
                hist.get("rows"),
            )
    except Exception:
        _log.exception("append_snapshot_inventory_to_history failed")
    refresh_inventory_api_cache(sess)
    try:
        from ..db.forecast_ops_tables import persist_inventory_dataframe

        persist_inventory_dataframe(
            sess.inventory_df_variant,
            snapshot_date=getattr(sess, "inventory_snapshot_date", "") or None,
            snapshot_label=getattr(sess, "inventory_snapshot_date_label", "") or None,
            debug=dict(getattr(sess, "inventory_debug", None) or {}),
        )
    except Exception:
        _log.exception("PostgreSQL inventory table persist failed")
    sess._inventory_pre_upload_backup = None
    _session_data_changed(sess)
    payload = _build_inventory_upload_payload(
        df_variant=df_variant,
        debug=sess.inventory_debug,
        detected=detected,
        file_parts=file_parts,
        warnings=warnings,
    )
    if not df_variant.empty:
        sess.inventory_upload_status = "done"
        _set_inventory_upload_progress(sess, 100, payload["message"])
    else:
        sess.inventory_upload_status = "error"
        sess.inventory_upload_progress = 0
        payload["ok"] = False
    sess.inventory_upload_result = payload
    sess.inventory_upload_message = payload["message"]
    sess.inventory_upload_started = 0.0
    return payload


def _run_inventory_auto_from_parts(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Parse assembled inventory file bytes on the inventory worker thread."""
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    route_notes: list[str] = []
    try:
        from ..services.upload_file_sniff import partition_files_by_upload_target

        buckets, route_notes = partition_files_by_upload_target(file_parts, "snapshot_inventory")
        file_parts = list(buckets.get("snapshot_inventory") or [])
        sales_parts = list(buckets.get("daily_sales") or [])
        if sales_parts:
            DAILY_UPLOAD_EXECUTOR.submit(_run_daily_auto_ingest_pipeline, session_id, sales_parts)
        if route_notes:
            _log.info("inventory-auto auto-route session=%s %s", session_id[:8], route_notes)
    except Exception:
        _log.exception("inventory-auto worker upload partition failed")
    if not file_parts:
        if route_notes:
            msg = "Auto-routed to daily sales upload — " + "; ".join(route_notes)
            sess.inventory_upload_status = "done"
            sess.inventory_upload_message = msg
            sess.inventory_upload_result = {"ok": True, "message": msg, "auto_routed": route_notes}
        return
    fnames = ", ".join(n[:36] + "…" if len(n) > 36 else n for n, _ in file_parts[:3])
    if len(file_parts) > 3:
        fnames += f" (+{len(file_parts) - 3} more)"
    _mark_inventory_upload_running(
        sess, f"Classifying {len(file_parts)} file(s): {fnames}…", progress=5,
    )
    warnings: list[str] = []
    try:
        oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes, detected = _classify_inventory_file_parts(file_parts)
    except Exception as e:
        _log.exception("inventory-auto classify files")
        sess.inventory_upload_status = "error"
        sess.inventory_upload_progress = 0
        sess.inventory_upload_message = str(e)
        sess.inventory_upload_result = {"ok": False, "message": str(e)}
        return

    if not any([oms_bytes_list, fk_bytes, myntra_bytes, amz_bytes]):
        sess.inventory_upload_status = "error"
        sess.inventory_upload_progress = 0
        sess.inventory_upload_message = "No inventory files recognized."
        sess.inventory_upload_result = {"ok": False, "message": "No inventory files recognized."}
        return

    if amz_bytes and amz_bytes[:6] == _RAR_MAGIC:
        _set_inventory_upload_progress(
            sess, 20, "Extracting RAR archive and reading inner CSV files…",
        )
    else:
        _set_inventory_upload_progress(sess, 25, "Parsing marketplace inventory files…")

    sku_mapping = dict(sess.sku_mapping or {})

    def parse_work() -> tuple[Any, Any, dict]:
        lock_held = _acquire_inventory_memory_lock(sess, session_id)
        try:
            return _inventory_parse_heavy(
                sess,
                oms_bytes_list=oms_bytes_list,
                fk_bytes=fk_bytes,
                myntra_bytes=myntra_bytes,
                amz_bytes=amz_bytes,
                sku_mapping=sku_mapping,
                warnings=warnings,
            )
        finally:
            if lock_held:
                _UPLOAD_MEMORY_LOCK.release()

    try:
        df_variant, df_parent, debug = parse_work()

        def apply_work() -> dict:
            return _inventory_apply_parse_result(
                sess,
                df_variant=df_variant,
                df_parent=df_parent,
                debug=debug,
                file_parts=file_parts,
                detected=detected,
                warnings=warnings,
            )

        _session_lock_apply_sync(sess, apply_work)
        if sess.inventory_upload_status == "done":
            _finish_inventory_server_save(sess, session_id)
    except Exception as e:
        _log.exception("inventory-auto parse")
        sess.inventory_upload_status = "error"
        sess.inventory_upload_progress = 0
        sess.inventory_upload_started = 0.0
        sess.inventory_upload_message = f"Parse error: {e}"
        sess.inventory_upload_result = {"ok": False, "message": f"Parse error: {e}"}


def _run_inventory_auto_worker(session_id: str, file_parts: list[tuple[str, bytes]]) -> None:
    """Sync entry for INVENTORY_EXECUTOR (chunk finalize + direct upload)."""
    _run_inventory_auto_from_parts(session_id, file_parts)


_INVENTORY_AUTO_DIRECT_MAX_FILES = int(os.environ.get("INVENTORY_AUTO_DIRECT_MAX_FILES", "3"))


@router.post("/inventory-auto/reset-stuck")
async def reset_stuck_inventory_upload(request: Request):
    """Clear a session stuck on inventory_upload_status=running (e.g. after a long wait)."""
    sess = _get_session(request)
    cleared = _clear_stuck_inventory_upload(sess, force=True)
    return {
        "ok": True,
        "cleared": cleared,
        "message": (
            "Cleared stuck inventory upload — you can upload again."
            if cleared
            else "No stuck inventory upload to clear."
        ),
        "inventory_upload_status": getattr(sess, "inventory_upload_status", "idle"),
    }


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
        if not _clear_stuck_inventory_upload(sess, force=False):
            return JSONResponse(
                content={
                    "ok": False,
                    "stuck": True,
                    "message": (
                        "An inventory upload is still processing. Wait a few minutes, "
                        "or use “Clear stuck”, then try again."
                    ),
                }
            )
        _log.info(
            "Auto-cleared stale inventory upload before new upload (session=%s)",
            getattr(sess, "_persist_sid", "")[:8],
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
    route_notes: list[str] = []
    try:
        from ..services.upload_file_sniff import partition_files_by_upload_target

        buckets, route_notes = partition_files_by_upload_target(file_parts, "snapshot_inventory")
        file_parts = list(buckets.get("snapshot_inventory") or [])
        sales_parts = list(buckets.get("daily_sales") or [])
        if sales_parts and sid:
            DAILY_UPLOAD_EXECUTOR.submit(_run_daily_auto_ingest_pipeline, sid, sales_parts)
    except Exception:
        _log.exception("inventory-auto upload partition failed")

    if sid and file_parts:

        inv_msg = (
            "Upload accepted — parsing inventory on the server. "
            "Status updates below; you can stay on this page."
        )
        if route_notes:
            inv_msg = "Auto-routed: " + "; ".join(route_notes) + " " + inv_msg
        _mark_inventory_upload_running(sess, "Upload received — starting parse…")
        INVENTORY_EXECUTOR.submit(_run_inventory_auto_worker, sid, file_parts)
        return JSONResponse(
            content={
                "ok": True,
                "ingest_async": True,
                "message": inv_msg,
                "auto_routed": route_notes or None,
            }
        )
    if sid and route_notes:
        return JSONResponse(
            content={
                "ok": True,
                "ingest_async": True,
                "message": "Auto-routed to daily sales upload — " + "; ".join(route_notes),
                "auto_routed": route_notes,
            }
        )

    oms_bytes_list: list[bytes] = []
    fk_bytes_list: list[bytes] = []
    myntra_bytes_list: list[bytes] = []
    amz_bytes = None
    detected: list[str] = []
    direct_file_parts: list[tuple[str, bytes]] = []
    for file in files:
        raw = await file.read()
        fname = file.filename or ""
        direct_file_parts.append((fname or "upload", raw))
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
            apply_inventory_snapshot_metadata(sess, direct_file_parts, debug)
            refresh_inventory_api_cache(sess)
            _session_data_changed(sess)
            parts = [f"{len(df_variant):,} total SKUs"]
            snap = sess.inventory_snapshot_date_label or sess.inventory_snapshot_date
            if snap:
                parts.insert(0, f"Snapshot as of {snap}")
            for src, info in sess.inventory_debug.items():
                if src in ("snapshot_date", "snapshot_date_label", "snapshot_date_sources", "snapshot_uploaded_at"):
                    continue
                parts.append(f"{src}: {info}")
            return {
                "ok": True,
                "message": " | ".join(parts),
                "rows": len(df_variant),
                "debug": sess.inventory_debug,
                "detected": detected,
                "snapshot_date": sess.inventory_snapshot_date,
                "snapshot_date_label": sess.inventory_snapshot_date_label,
                "v": "inv-v8",
            }

        data = await _session_lock_apply(sess, work)
    except Exception as e:
        return JSONResponse(content={"ok": False, "message": f"Parse error: {e}"})

    sid = getattr(request.state, "session_id", None)
    _finish_inventory_server_save(sess, sid)
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
            apply_inventory_snapshot_metadata(sess, [], debug)
            refresh_inventory_api_cache(sess)
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

    sid = getattr(request.state, "session_id", None)
    _finish_inventory_server_save(sess, sid)
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

def _persist_existing_po_after_upload(sess, session_id: str | None) -> None:
    """Sync PostgreSQL + GitHub cache immediately after Existing PO upload."""
    if session_id:
        setattr(sess, "_persist_sid", session_id)
        try:
            from ..db.forecast_session_pg import persist_session_bundle_thread_safe

            persist_session_bundle_thread_safe(session_id, sess)
        except Exception:
            _log.exception("PostgreSQL persist after existing PO upload")
    try:
        import backend.main as _main

        _main.merge_existing_po_into_warm_cache(sess)
    except Exception:
        _log.exception("merge_existing_po_into_warm_cache after existing PO upload")
    try:
        _auto_save_cache(sess)
    except Exception:
        _log.exception("auto-save after existing PO upload")


def _parse_existing_po_into_session(sess, file_bytes: bytes, orig_fn: str) -> UploadResponse:
    from ..services.existing_po import audit_existing_po_upload

    df = parse_existing_po(file_bytes, orig_fn)
    audit = audit_existing_po_upload(file_bytes, orig_fn, df)
    from ..services.existing_po import apply_existing_po_upload_audit

    apply_existing_po_upload_audit(sess, audit)
    sess.existing_po_df = df
    sess.existing_po_filename = orig_fn
    from ..services.existing_po import read_existing_po_disk_meta

    disk_meta = read_existing_po_disk_meta() or {}
    disk_gen = int(disk_meta.get("existing_po_generation") or 0)
    sess_gen = int(getattr(sess, "existing_po_generation", 0) or 0)
    sess.existing_po_generation = max(sess_gen, disk_gen) + 1
    seed: dict = {}
    try:
        from ..services.po_raise_import import seed_ledger_from_manual_existing_po_upload

        seed = seed_ledger_from_manual_existing_po_upload(sess, replace_day=True)
        if seed.get("ok"):
            _log.info(
                "Manual Existing PO raise recorded: %s",
                seed.get("message") or seed,
            )
    except Exception:
        _log.exception("seed_ledger_from_manual_existing_po_upload failed")
    try:
        import datetime as _dt

        sess.existing_po_uploaded_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        sess.existing_po_uploaded_at = ""
    from ..services.po_raise_remove import invalidate_po_calculate_result
    from ..services.po_shared_cache import invalidate_all_shared_caches

    invalidate_po_calculate_result(sess)
    invalidate_all_shared_caches()
    from ..services.existing_po import persist_existing_po_to_disk

    persist_existing_po_to_disk(sess)
    _session_data_changed(sess)
    active = 0
    if "PO_Pipeline_Total" in df.columns:
        active = int((pd.to_numeric(df["PO_Pipeline_Total"], errors="coerce").fillna(0) > 0).sum())
    pipe = int(audit.get("pipeline_units") or 0)
    msg = (
        f"Existing PO loaded: {len(df):,} SKUs ({active:,} with pipeline, "
        f"{pipe:,} total balance units). "
        "Click Calculate PO on PO Engine to refresh pipeline columns."
    )
    warnings = list(audit.get("warnings") or [])
    if audit.get("sheet_total_row"):
        st = audit["sheet_total_row"]
        msg += (
            f" Sheet Total row: {st.get('total_balance_units', st.get('pipeline_units', 0)):,} balance"
        )
        if st.get("pending_cutting_units") is not None:
            msg += f", {st['pending_cutting_units']:,} pending cutting"
        if st.get("balance_to_dispatch_units") is not None:
            msg += f", {st['balance_to_dispatch_units']:,} balance to dispatch"
        msg += "."
    if not audit.get("totals_match", True):
        msg += " ⚠ Parsed totals differ from sheet Total row — review before Calculate PO."
    if seed.get("ok") and seed.get("manual_raise"):
        qty_note = (
            f", {int(seed.get('imported_skus', 0)):,} with qty"
            if seed.get("ledger_seeded")
            else ""
        )
        msg += (
            f" Recorded manual raise for {seed.get('raised_date', '?')} "
            f"({int(seed.get('raise_skus', 0)):,} SKUs on sheet{qty_note})."
        )
    return UploadResponse(
        ok=True,
        message=msg,
        rows=len(df),
        existing_po_uploaded_at=sess.existing_po_uploaded_at or None,
        existing_po_generation=sess.existing_po_generation,
        validation_warnings=warnings or None,
    )


def _run_existing_po_parse_worker(session_id: str, file_bytes: bytes, orig_fn: str) -> None:
    sess = _resolve_upload_session(session_id)
    if sess is None:
        return
    sess.existing_po_upload_status = "running"
    sess.existing_po_upload_message = f"Parsing {orig_fn}…"
    sess.existing_po_upload_progress = 15
    sess.existing_po_upload_started = time.time()
    try:
        with sess._daily_restore_lock:
            resp = _parse_existing_po_into_session(sess, file_bytes, orig_fn)
        sess.existing_po_upload_status = "done"
        sess.existing_po_upload_progress = 100
        sess.existing_po_upload_message = resp.message
        sess.existing_po_upload_result = resp.model_dump()
    except Exception as e:
        _log.warning("existing-po parse failed: %s", e, exc_info=True)
        msg = f"Failed to parse Existing PO: {e}"
        sess.existing_po_upload_status = "error"
        sess.existing_po_upload_progress = 0
        sess.existing_po_upload_message = msg
        sess.existing_po_upload_result = {"ok": False, "message": msg}
    finally:
        sess.existing_po_upload_started = 0.0
        _persist_existing_po_after_upload(sess, session_id)


@router.post("/existing-po", response_model=UploadResponse)
async def upload_existing_po(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    sess = _get_session(request)
    try:
        if getattr(sess, "existing_po_upload_status", "idle") == "running":
            return UploadResponse(
                ok=False,
                message="Existing PO upload already in progress — wait for parsing to finish.",
            )
        file_bytes = await file.read()
        orig_fn = file.filename or "existing_po.xlsx"
        try:
            from ..services.upload_file_sniff import check_upload_target

            wrong = check_upload_target("existing_po", file_bytes, orig_fn)
            if wrong:
                return UploadResponse(ok=False, message=wrong)
        except Exception:
            _log.exception("existing-po upload sniff failed")
        sid = getattr(request.state, "session_id", None)
        if sid:
            sess.existing_po_upload_status = "running"
            sess.existing_po_upload_message = f"Upload received — parsing {orig_fn}…"
            sess.existing_po_upload_progress = 5
            sess.existing_po_upload_started = time.time()
            HEAVY_EXECUTOR.submit(_run_existing_po_parse_worker, sid, file_bytes, orig_fn)
            return JSONResponse(
                content={
                    "ok": True,
                    "ingest_async": True,
                    "message": (
                        "Upload accepted — parsing existing PO on the server. "
                        "Status updates below; large sheets may take 1–3 minutes."
                    ),
                }
            )

        def work():
            return _parse_existing_po_into_session(sess, file_bytes, orig_fn)

        resp = await _session_lock_apply(sess, work)
        if resp.ok:
            sid = getattr(request.state, "session_id", None)
            background_tasks.add_task(_persist_existing_po_after_upload, sess, sid)
        return resp
    except Exception as e:
        _log.warning("existing-po parse failed: %s", e, exc_info=True)
        return UploadResponse(ok=False, message=f"Failed to parse Existing PO: {e}")


@router.post("/finishing-receipt", response_model=UploadResponse)
async def upload_finishing_receipt(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Admin: Finishing Dept receipt export — updates Existing PO Balance_to_Dispatch."""
    sess = _get_session(request)
    try:
        file_bytes = await file.read()
        orig_fn = file.filename or "finishing.xls"

        def work():
            from ..services.finishing_receipt import (
                apply_finishing_receipt_import,
                parse_finishing_receipt_workbook,
            )

            finishing_df, report = parse_finishing_receipt_workbook(
                file_bytes,
                orig_fn,
                sku_mapping=sess.sku_mapping or None,
            )
            out = apply_finishing_receipt_import(
                sess,
                finishing_df,
                report,
                filename=orig_fn,
            )
            _session_data_changed(sess)
            return UploadResponse(
                ok=True,
                message=out.get("message") or "Finishing receipt applied.",
                rows=out.get("rows"),
                existing_po_uploaded_at=getattr(sess, "existing_po_uploaded_at", None),
                existing_po_generation=out.get("existing_po_generation"),
                validation_warnings=(
                    [f"{out.get('left_units', 0):,} units still at finishing"]
                    if int(out.get("left_units") or 0) > 0
                    else None
                ),
            )

        resp = await _session_lock_apply(sess, work)
        if resp.ok:
            sid = getattr(request.state, "session_id", None)
            background_tasks.add_task(_persist_existing_po_after_upload, sess, sid)
        return resp
    except ValueError as e:
        return UploadResponse(ok=False, message=str(e))
    except Exception as e:
        _log.warning("finishing-receipt parse failed: %s", e, exc_info=True)
        return UploadResponse(ok=False, message=f"Failed to parse Finishing receipt: {e}")


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

_JUNK_UPLOAD_BASENAMES = frozenset({
    ".ds_store",
    "thumbs.db",
    "desktop.ini",
    ".localized",
})


def _upload_basename(filename: str) -> str:
    norm = (filename or "").replace("\\", "/").strip()
    return norm.rsplit("/", 1)[-1] if norm else ""


def _is_junk_upload_filename(filename: str) -> bool:
    """macOS/Windows metadata accidentally dropped with sales folders."""
    norm = (filename or "").replace("\\", "/").strip().lower()
    if not norm:
        return True
    if "__macosx/" in norm or norm.startswith("__macosx/"):
        return True
    base = _upload_basename(norm).lower()
    if base in _JUNK_UPLOAD_BASENAMES:
        return True
    if base.startswith("._"):
        return True
    return False


def _filter_junk_daily_upload_parts(
    file_parts: list[tuple[str, bytes]],
) -> tuple[list[tuple[str, bytes]], list[str]]:
    kept: list[tuple[str, bytes]] = []
    ignored: list[str] = []
    for fname, raw in file_parts:
        if _is_junk_upload_filename(fname):
            ignored.append(fname)
        else:
            kept.append((fname, raw))
    return kept, ignored


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


def _zip_is_myntra_monthly(file_bytes: bytes, outer_fn: str) -> bool:
    """Myntra PPMP / Seller Orders ZIP — must not be routed to Meesho."""
    fn = outer_fn.lower()
    if any(k in fn for k in ("myntra", "ppmp", "sjit", "seller_orders", "seller orders")):
        return True
    _names, joined = _zip_member_paths(file_bytes)
    if any(k in joined for k in ("myntra", "ppmp", "sjit", "seller_orders")):
        return True
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for n in zf.namelist():
                if not n.lower().endswith(".csv"):
                    continue
                head = zf.read(n)[:4000].decode("utf-8", errors="ignore").lower()
                if "order_created_date" in head or (
                    "sub order id" in head and "sku" in head
                ):
                    return True
                break
    except Exception:
        pass
    return False


def _zip_is_meesho_monthly(file_bytes: bytes, outer_fn: str) -> bool:
    if _zip_is_myntra_monthly(file_bytes, outer_fn):
        return False
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
    if _zip_is_myntra_monthly(file_bytes, fn):
        return "myntra"
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
    if (
        "myntra" in fn
        or "ppmp" in fn
        or "sjit" in fn
        or "seller_orders" in fn
        or "seller orders" in fn
        or "my ppmp" in fn
    ):
        return "myntra"
    if "b2b" in fn:
        return "amazon_b2b"
    if (
        "b2c" in fn
        or "mtr" in fn
        or "merchant" in fn
        or "tax report" in fn
        or " amazon" in fn
        or fn.endswith(" amazon.csv")
        or re.search(r"amazon\s+\d", fn)
    ):
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


def _record_file_skip(
    file_results: list[dict],
    warnings: list[str],
    fname: str,
    reason: str,
    *,
    platform: str = "",
) -> None:
    _log.info("daily-auto skip %s: %s", fname[:120], reason[:200])
    warnings.append(f"{fname}: {reason}")
    entry: dict = {"filename": fname, "status": "skipped", "reason": reason}
    if platform:
        entry["platform"] = platform
    file_results.append(entry)


def _save_daily_file_tracked(
    platform: str,
    fname: str,
    df: pd.DataFrame,
    *,
    detected: list[str],
    warnings: list[str],
    file_results: list[dict],
    detected_label: str,
    sess: AppSession | None = None,
) -> bool:
    """Persist to Tier-3 SQLite and record outcome. Returns True when saved."""
    if df is None or df.empty:
        _record_file_skip(file_results, warnings, fname, "No data extracted", platform=platform)
        return False
    _fd, rows, block = save_daily_file(platform, fname, df)
    if block or rows <= 0:
        reason = block or "Not saved to database"
        _record_file_skip(file_results, warnings, fname, reason, platform=platform)
        return False
    file_results.append(
        {
            "filename": fname,
            "status": "saved",
            "platform": platform,
            "rows": rows,
        }
    )
    detected.append(detected_label)
    if sess is not None:
        _after_daily_file_saved(sess, platform, df)
    return True


def _after_daily_file_saved(sess: AppSession, platform: str, df: pd.DataFrame) -> None:
    from ..session import mark_tier3_store_changed

    mark_tier3_store_changed(sess)
    if _daily_auto_fast_ingest_enabled():
        _track_daily_auto_platform(sess, platform)
        _buffer_daily_auto_parsed(sess, platform, df)


def _merge_slice_into_session(
    sess,
    platform: str,
    df_slice: pd.DataFrame,
    *,
    source_filename: Optional[str] = None,
    skip_session_merge: bool = False,
) -> None:
    """Merge one parsed upload slice into session — avoids reloading full Tier-3 SQLite."""
    if df_slice is None or df_slice.empty:
        return
    if skip_session_merge:
        _track_daily_auto_platform(sess, platform)
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
    elif platform == "snapdeal":
        sess.snapdeal_df = _merge_platform_data(
            sess.snapdeal_df, df_slice, "snapdeal", source_filename=source_filename,
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
            file_results: list[dict] = []
            expanded_files = 0
            fast_ingest = _daily_auto_fast_ingest_enabled()
            sess._daily_auto_platforms_touched = set()
            sess._daily_auto_parsed_buffers = {}

            file_parts, ignored_names = _filter_junk_daily_upload_parts(file_parts)
            if ignored_names:
                bit = ", ".join(_upload_basename(n) for n in ignored_names[:5])
                warnings.append(
                    f"Ignored {len(ignored_names)} system/metadata file(s): {bit}"
                )
            if not file_parts:
                msg = (
                    "No sales files to import — only system/metadata files were received "
                    f"({', '.join(_upload_basename(n) for n in ignored_names[:3])}). "
                    "Drop your Sales RAR/ZIP or marketplace CSV/XLSX exports instead."
                )
                return {
                    "ok": False,
                    "message": msg,
                    "detected_platforms": [],
                    "warnings": warnings,
                    "processed_files": len(ignored_names),
                    "detected_files": 0,
                    "unknown_files": len(ignored_names),
                    "expanded_files": len(ignored_names),
                    "saved_files": 0,
                    "file_results": [
                        {
                            "filename": n,
                            "status": "skipped",
                            "reason": "System/metadata file — not a sales report",
                        }
                        for n in ignored_names
                    ],
                }

            def _apply_parsed_slice(
                p: str,
                df: pd.DataFrame,
                src_fname: str,
                defer_queue: Optional[List[_DeferredSlice]] = None,
            ) -> None:
                if fast_ingest:
                    return
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
                nonlocal expanded_files
                expanded_files += 1
                platform = _detect_platform(fname, raw)
                try:
                    if platform == "amazon_mtr_zip":
                        df_mtr, _n, sk_mtr = load_mtr_from_zip(raw)
                        if _save_daily_file_tracked(
                            "amazon", fname, df_mtr,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Amazon MTR ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("amazon", df_mtr, fname, defer_queue)
                            if sk_mtr:
                                warnings.append(f"{fname}: {'; '.join(sk_mtr[:2])}")

                    elif platform == "amazon_b2c":
                        df, msg = parse_mtr_csv(raw, fname)
                        if _save_daily_file_tracked(
                            "amazon", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Amazon ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("amazon", df, fname, defer_queue)
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")

                    elif platform == "amazon_b2b":
                        df, msg = parse_mtr_csv(raw, fname)
                        if _save_daily_file_tracked(
                            "amazon", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Amazon B2B ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("amazon", df, fname, defer_queue)
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")

                    elif platform == "myntra":
                        from ..services.myntra import _parse_myntra_csv, load_myntra_from_zip
                        if fname.lower().endswith(".zip"):
                            df, _n_csv, sk_m = load_myntra_from_zip(
                                raw, sess.sku_mapping or {}, source_filename=fname,
                            )
                            msg = "OK"
                            if sk_m:
                                warnings.append(f"{fname}: {'; '.join(sk_m[:2])}")
                        else:
                            df, msg = _parse_myntra_csv(raw, fname, sess.sku_mapping)
                            if not df.empty:
                                df = apply_dsr_segment_from_upload_filename(df, fname, "Myntra")
                        if _save_daily_file_tracked(
                            "myntra", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Myntra ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("myntra", df, fname, defer_queue)
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")

                    elif platform == "meesho":
                        df, _count, _skipped = load_meesho_from_zip(raw, source_filename=fname)
                        if _save_daily_file_tracked(
                            "meesho", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Meesho ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            if _skipped:
                                warnings.append(f"{fname}: {'; '.join(_skipped[:2])}")

                    elif platform == "meesho_csv":
                        from ..services.meesho import parse_meesho_csv
                        df, msg = parse_meesho_csv(raw)
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                        if _save_daily_file_tracked(
                            "meesho", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Meesho ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")

                    elif platform == "meesho_order_xlsx":
                        df, msg = parse_meesho_order_export_xlsx(raw)
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Meesho")
                        if _save_daily_file_tracked(
                            "meesho", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Meesho order export ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("meesho", df, fname, defer_queue)
                            if msg != "OK":
                                warnings.append(f"{fname}: {msg}")

                    elif platform == "snapdeal":
                        df_sd, _fc, skipped_sd, parse_info = load_snapdeal_from_zip(
                            raw, sess.sku_mapping or {}, fname,
                        )
                        sess.snapdeal_parse_info.update(parse_info)
                        if _save_daily_file_tracked(
                            "snapdeal", fname, df_sd,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Snapdeal ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("snapdeal", df_sd, fname, defer_queue)
                            if skipped_sd:
                                warnings.append(f"{fname}: {'; '.join(skipped_sd[:2])}")

                    elif platform == "flipkart":
                        from ..services.flipkart import (
                            _parse_flipkart_xlsx, _parse_flipkart_orders_sheet,
                            _parse_flipkart_earn_more, _parse_flipkart_xlsb,
                        )
                        if fname.lower().endswith(".xlsb"):
                            df = _parse_flipkart_xlsb(raw, fname, sess.sku_mapping)
                            if df.empty:
                                _record_file_skip(
                                    file_results, warnings, fname,
                                    "No data extracted from Flipkart XLSB file",
                                    platform="flipkart",
                                )
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
                                    _record_file_skip(
                                        file_results, warnings, fname,
                                        f"No Sales Report, Orders, or earn_more_report sheet (sheets: {', '.join(xl_sheets[:4])})",
                                        platform="flipkart",
                                    )
                                    return
                        if not df.empty:
                            df = apply_dsr_segment_from_upload_filename(df, fname, "Flipkart")
                        if _save_daily_file_tracked(
                            "flipkart", fname, df,
                            detected=detected, warnings=warnings, file_results=file_results,
                            detected_label=f"Flipkart ({fname})",
                            sess=sess,
                        ):
                            _apply_parsed_slice("flipkart", df, fname, defer_queue)

                    else:
                        _record_file_skip(
                            file_results, warnings, fname,
                            "Could not detect platform (unknown format)",
                        )

                except Exception as e:
                    _record_file_skip(file_results, warnings, fname, str(e))

            n_files = len(file_parts)
            for _fi, (fname, raw) in enumerate(file_parts):
                # Live status visible on the frontend poll
                _short = fname[:40] + "…" if len(fname) > 40 else fname
                sess.daily_auto_ingest_message = (
                    f"Parsing file {_fi + 1}/{n_files}: {_short}…"
                    if n_files > 1
                    else f"Parsing {_short}…"
                )
                try:
                    fl = fname.lower()
                    # RAR archive — extract and process each file inside
                    if raw[:6] == _RAR_MAGIC or fl.endswith(".rar"):
                        try:
                            inner_files = _extract_rar_files(raw)
                            if not inner_files:
                                reason = "RAR archive extracted to zero files — archive may be corrupt or empty."
                                warnings.append(f"{fname}: {reason}")
                                file_results.append({
                                    "filename": fname,
                                    "status": "skipped",
                                    "reason": reason,
                                })
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
                            _log.info(
                                "RAR extract complete: %s — %d inner files, %d saved",
                                fname, expanded_files, sum(1 for r in file_results if r.get("status") == "saved"),
                            )
                        except Exception as e:
                            reason = f"Could not extract RAR: {e}"
                            warnings.append(f"{fname} (RAR extract): {e}")
                            file_results.append({
                                "filename": fname,
                                "status": "skipped",
                                "reason": reason,
                            })
                            _log.warning("RAR extraction failed for %s: %s", fname, e)
                    elif fl.endswith(".zip"):
                        if "snapdeal" in fl:
                            _handle_one(fname, raw)
                        elif _zip_is_myntra_monthly(raw, fname):
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
                                    expanded_files += 1
                                    if _save_daily_file_tracked(
                                        "flipkart", fname, df_fk,
                                        detected=detected, warnings=warnings, file_results=file_results,
                                        detected_label=f"Flipkart ZIP ({fname}, {n_fc} file(s))",
                                        sess=sess,
                                    ):
                                        _merge_slice_into_session(
                                            sess, "flipkart", df_fk, source_filename=fname,
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
                saved_files = sum(1 for r in file_results if r.get("status") == "saved")
                return {
                    "ok": False,
                    "message": warn_str,
                    "detected_platforms": [],
                    "warnings": warnings,
                    "processed_files": len(file_parts),
                    "detected_files": 0,
                    "unknown_files": max(0, expanded_files - saved_files) if expanded_files else len(file_parts),
                    "expanded_files": expanded_files or len(file_parts),
                    "saved_files": saved_files,
                    "file_results": file_results,
                }

            # Track loaded platforms; rebuild sales now or defer to background (daily-auto HTTP path).
            sess.daily_sales_sources = list(set(sess.daily_sales_sources + detected))
            sales_rebuild = "inline"
            if rebuild_sales:
                _n_cur = len(getattr(sess, "sales_df", None) or [])
                sess.daily_auto_ingest_message = (
                    f"Rebuilding combined sales ({_n_cur:,} rows)…" if _n_cur else "Rebuilding combined sales…"
                )
                touched = getattr(sess, "_daily_auto_platforms_touched", None) or None
                ok_rb, rb_msg = _rebuild_sales_sync(
                    sess,
                    refresh_sqlite=not (fast_ingest and touched),
                    platforms_touched=touched if fast_ingest and touched else None,
                )
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
            saved_files = sum(1 for r in file_results if r.get("status") == "saved")
            skipped_files = sum(1 for r in file_results if r.get("status") == "skipped")
            if skipped_files:
                msg_parts.append(f"{saved_files} saved, {skipped_files} skipped — see details below.")
            return {
                "ok": True,
                "message": " ".join(msg_parts),
                "detected_platforms": detected,
                "warnings": warnings,
                "processed_files": len(file_parts),
                "detected_files": len(detected),
                "unknown_files": max(0, (expanded_files or len(file_parts)) - saved_files),
                "expanded_files": expanded_files or len(file_parts),
                "saved_files": saved_files,
                "file_results": file_results,
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
    sid = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
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
        return JSONResponse(
            content={
                "ok": False,
                "message": "Session required — refresh the page and sign in again.",
                "detected_platforms": [],
                "warnings": [],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": n_files,
            },
            status_code=400,
        )

    route_notes: list[str] = []
    inv_parts: list[tuple[str, bytes]] = []
    try:
        from ..services.upload_file_sniff import partition_files_by_upload_target

        buckets, route_notes = partition_files_by_upload_target(file_parts, "daily_sales")
        file_parts = list(buckets.get("daily_sales") or [])
        inv_parts = list(buckets.get("snapshot_inventory") or [])
        if inv_parts:
            INVENTORY_EXECUTOR.submit(_run_inventory_auto_worker, sid, inv_parts)
    except Exception:
        _log.exception("daily-auto upload partition failed")

    msg = (
        "Upload accepted — parsing daily files on the server. "
        "Status updates below; you can stay on this page."
    )
    if route_notes:
        msg = "Auto-routed: " + "; ".join(route_notes) + " " + msg

    if file_parts:
        _mark_daily_auto_ingest_running(sess, "Parsing daily files and merging into session…")
        background_tasks.add_task(_queue_daily_auto_ingest, sid, file_parts)
    elif inv_parts:
        _mark_inventory_upload_running(sess, "Auto-routed inventory files — starting parse…")
    else:
        return JSONResponse(
            content={
                "ok": False,
                "message": "No recognizable daily files in upload.",
                "detected_platforms": [],
                "warnings": [],
                "processed_files": n_files,
                "detected_files": 0,
                "unknown_files": n_files,
            }
        )
    return JSONResponse(
        content={
            "ok": True,
            "ingest_async": True,
            "message": msg,
            "auto_routed": route_notes or None,
            "detected_platforms": [],
            "warnings": route_notes,
            "processed_files": n_files,
            "detected_files": 0,
            "unknown_files": n_files,
            "sales_rebuild": "pending" if file_parts else None,
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

    session_id = getattr(request.state, "session_id", None) or getattr(sess, "_persist_sid", None)
    if not session_id:
        return JSONResponse(content={"ok": False, "message": "Session not found — refresh and try again."})

    st = getattr(sess, "sales_rebuild_status", "idle") or "idle"
    if st == "running":
        return JSONResponse(
            content={
                "ok": True,
                "message": "Sales rebuild already in progress…",
                "sales_rebuild": "pending",
            }
        )

    ingest_st = getattr(sess, "daily_auto_ingest_status", "idle") or "idle"
    if ingest_st == "running":
        return JSONResponse(
            content={
                "ok": False,
                "message": "Daily upload still processing — wait for it to finish, then rebuild.",
                "sales_rebuild": "pending",
            }
        )

    sess.sales_rebuild_status = "running"
    sess.sales_rebuild_message = "Queued sales rebuild…"
    DAILY_UPLOAD_EXECUTOR.submit(_run_sales_rebuild_worker, session_id, refresh_sqlite=True)
    return JSONResponse(
        content={
            "ok": True,
            "message": "Sales rebuild started — refreshing from saved daily uploads…",
            "sales_rebuild": "pending",
        }
    )


# ── Clear platform data ────────────────────────────────────────

def _clear_inventory_platform(sess, _pd) -> None:
    from ..services.inventory import clear_inventory_snapshot

    clear_inventory_snapshot(sess)


_PLATFORM_CLEAR = {
    "mtr":      lambda sess, pd: setattr(sess, "mtr_df",      pd.DataFrame()),
    "myntra":   lambda sess, pd: setattr(sess, "myntra_df",   pd.DataFrame()),
    "meesho":   lambda sess, pd: setattr(sess, "meesho_df",   pd.DataFrame()),
    "flipkart": lambda sess, pd: setattr(sess, "flipkart_df", pd.DataFrame()),
    "snapdeal": lambda sess, pd: setattr(sess, "snapdeal_df", pd.DataFrame()),
    "sales":    lambda sess, pd: setattr(sess, "sales_df",    pd.DataFrame()),
    "inventory": _clear_inventory_platform,
}

@router.delete("/clear/{platform}")
async def clear_platform(platform: str, request: Request):
    """Clear a specific platform's data from the session."""
    from ..services.upload_policy import _DELETE_DENIED_MSG, may_delete_upload_data

    auth = getattr(request.state, "auth", None) or {}
    if not may_delete_upload_data(str(auth.get("role") or ""), str(auth.get("sub") or "")):
        return JSONResponse(
            content={"ok": False, "message": _DELETE_DENIED_MSG},
            status_code=403,
        )
    sess = _get_session(request)
    cleaner = _PLATFORM_CLEAR.get(platform)
    if not cleaner:
        return JSONResponse(content={"ok": False, "message": f"Unknown platform: {platform}"}, status_code=400)
    import pandas as pd

    def work():
        cleaner(sess, pd)
        if platform == "inventory":
            try:
                import backend.main as _main

                _main.merge_inventory_into_warm_cache(sess)
            except Exception:
                _log.exception("merge_inventory_into_warm_cache after inventory clear failed")
            _session_data_changed(sess)
            return {
                "ok": True,
                "message": (
                    "Snapshot inventory cleared from this session and server cache. "
                    "Upload a new file under Upload Data → Daily uploads → Snapshot inventory."
                ),
            }
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

    _mark_daily_auto_ingest_running(sess, "Parsing daily files and merging into session…")
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

    route_notes: list[str] = []
    try:
        from ..services.upload_file_sniff import partition_files_by_upload_target

        buckets, route_notes = partition_files_by_upload_target(file_parts, "snapshot_inventory")
        file_parts = list(buckets.get("snapshot_inventory") or [])
        sales_parts = list(buckets.get("daily_sales") or [])
        if sales_parts:
            DAILY_UPLOAD_EXECUTOR.submit(_run_daily_auto_ingest_pipeline, sid, sales_parts)
    except Exception:
        _log.exception("chunked inventory-auto upload partition failed")

    if not file_parts and route_notes:
        return {
            "ok": True,
            "ingest_async": True,
            "chunked": True,
            "message": "Auto-routed to daily sales — " + "; ".join(route_notes),
            "auto_routed": route_notes,
        }

    _mark_inventory_upload_running(sess, "Chunks assembled — starting parse…")
    INVENTORY_EXECUTOR.submit(_run_inventory_auto_worker, sid, file_parts)
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
    if body.target == "daily-auto":
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
                    },
                    status_code=409,
                )
        _mark_daily_auto_ingest_running(sess, "Receiving daily files…")
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
        INVENTORY_EXECUTOR.submit(_run_inventory_auto_worker, session_id, file_parts)


def _finalize_chunk_upload(session_id: str, upload_id: str) -> None:
    """Queue chunk finalize on the dedicated upload executor (never blocks behind warm-cache)."""
    DAILY_UPLOAD_EXECUTOR.submit(_finalize_chunk_upload_worker, session_id, upload_id)


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
        # Do NOT block here on daily_auto_ingest_status == "running":
        # chunk/init already checked and set status="running" for this very upload.
        # Re-checking causes every chunked upload to fail with "still processing"
        # if all chunks arrive within the 3-minute auto-clear window (self-deadlock).
        pass
    elif getattr(sess, "inventory_upload_status", "idle") == "running":
        if not _clear_stuck_inventory_upload(sess, force=False):
            return JSONResponse(
                content={
                    "ok": False,
                    "stuck": True,
                    "message": (
                        "An inventory upload is still processing. Wait for it to finish, "
                        "or use “Clear stuck upload”, then try again."
                    ),
                }
            )
        _log.info("Cleared stuck inventory upload before new chunk upload session=%s", sid[:8])

    if target == "daily-auto":
        _mark_daily_auto_ingest_running(sess, "Assembling uploaded files…")
    else:
        _mark_inventory_upload_running(sess, "Assembling uploaded chunks…", progress=1)

    # Use the dedicated daily/inventory upload executor so chunk finalization
    # does not wait behind unrelated heavy jobs (cache/restore/PO work).
    _finalize_chunk_upload(sid, body.upload_id)

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
