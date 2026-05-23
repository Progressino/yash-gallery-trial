"""
Yash Gallery ERP — FastAPI backend
Serves all business logic as a REST API.
Session state is stored server-side keyed by a UUID cookie.
"""
from dotenv import load_dotenv
load_dotenv()   # loads .env from cwd (run from repo root or backend/)

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from .concurrency import run_aux, run_heavy, _UPLOAD_MEMORY_LOCK

from .session import store
from .routers import upload, data, cache, po, shipment, auth as auth_router
from .routers.auth import verify_token, decode_token
from .services.permissions import karigar_may_access_api, may_access_erp_admin, KARIGAR_ROLE
from .routers.finance import router as finance_router
from .routers.item import router as item_router
from .routers.sales import router as sales_router
from .routers.purchase import router as purchase_router
from .routers.tna import router as tna_router
from .routers.production import router as production_router
from .routers.grey import router as grey_router
from .routers.stitching import router as stitching_router
from .routers.hrm import router as hrm_router
from .routers.erp_admin import router as erp_admin_router
from .routers.marketplace_connect import router as marketplace_router
from .db.finance_db import init_db
from .db.item_db import init_db as init_item_db
from .db.marketplace_db import init_db as init_marketplace_db
from .db.sales_db import init_db as init_sales_db
from .db.purchase_db import init_db as init_purchase_db
from .db.tna_db import init_db as init_tna_db
from .db.production_db import init_db as init_production_db
from .db.grey_db import init_db as init_grey_db
from .db.stitching_db import init_db as init_stitching_db
from .db.hrm_db import init_db as init_hrm_db
from .db.users_db import init_db as init_users_db
from .db.po_raised_db import init_db as init_po_raised_db
from .db.forecast_session_pg import init_db as init_forecast_session_pg

init_db()
init_item_db()
init_marketplace_db()
init_sales_db()
init_purchase_db()
init_tna_db()
init_production_db()
init_grey_db()
init_stitching_db()
init_hrm_db()
init_users_db()
init_po_raised_db()
init_forecast_session_pg()

log = logging.getLogger("erp.cache_warmer")

# ── Shared warm cache (app-level, not per-session) ────────────────
# Populated at startup and refreshed daily at 06:00 IST.
# New/empty sessions copy from here so users never wait for a GitHub download.
_warm_cache: dict = {}
_warm_cache_loaded_at: datetime | None = None
# Monotonically increasing counter: increments on every successful warm-cache
# update (Phase 1 = 1, Phase 2 = 2, next day refresh = 3, …).  Sessions track
# which generation they last received so the middleware can re-copy Phase-2 data
# to sessions that only got Phase-1 data without requiring a full page reload.
_warm_cache_generation: int = 0
# Set (signalled) once Phase 1 (SQLite) finishes. Cleared by clear_warm_cache().
_warm_cache_ready = threading.Event()

IST = timezone(timedelta(hours=5, minutes=30))

# Keys persisted to GitHub data-cache release (must match github_cache._CACHE_FILES)
_WARM_CACHE_KEYS = (
    "sales_df",
    "mtr_df",
    "meesho_df",
    "myntra_df",
    "flipkart_df",
    "snapdeal_df",
    "sku_mapping",
    "inventory_df_variant",
    "inventory_df_parent",
    "existing_po_df",
    # Optional PO inputs — must survive restarts / warm-cache copy so the UI does not
    # show "upload daily inventory again" and SKU-status merges stay in effect.
    "daily_inventory_history_df",
    "sku_status_lead_df",
    # Confirmed PO raises (Export & Confirm / CSV import) — must survive restarts like other PO sidecars.
    "po_raise_ledger_df",
)


def clear_warm_cache() -> None:
    """Drop app-level warm cache so the next load re-downloads from GitHub / rebuilds from SQLite."""
    global _warm_cache, _warm_cache_loaded_at, _warm_cache_generation
    _warm_cache = {}
    _warm_cache_loaded_at = None
    _warm_cache_generation = 0
    _warm_cache_ready.clear()  # Reset so next load is awaited correctly


def publish_warm_cache_from_session(sess) -> None:
    """Mirror session dataframes into _warm_cache after a full reload or rebuild."""
    global _warm_cache, _warm_cache_loaded_at
    import pandas as pd

    out: dict = {}
    for key in _WARM_CACHE_KEYS:
        val = getattr(sess, key, None)
        if val is None:
            continue
        if key == "sku_mapping" and not isinstance(val, dict):
            out[key] = {}
        elif hasattr(val, "empty"):
            out[key] = val if isinstance(val, pd.DataFrame) else pd.DataFrame()
        else:
            out[key] = val
    _warm_cache = out
    _warm_cache_loaded_at = datetime.now(IST)


_PO_SIDECAR_KEYS = ("daily_inventory_history_df", "sku_status_lead_df", "po_raise_ledger_df")


def session_needs_operational_data(sess) -> bool:
    """True when the browser session has no platform/sales data loaded yet."""
    if getattr(sess, "sales_df", None) is not None and not sess.sales_df.empty:
        return False
    if getattr(sess, "mtr_df", None) is not None and not sess.mtr_df.empty:
        return False
    for key in ("myntra_df", "meesho_df", "flipkart_df", "snapdeal_df"):
        df = getattr(sess, key, None)
        if df is not None and hasattr(df, "empty") and not df.empty:
            return False
    return True


def force_restore_session_from_server_cache(sess, warm_cache_generation: int) -> bool:
    """
    Fill an empty session from warm cache even when ``pause_auto_data_restore`` is set
    (e.g. after Delete All on another tab, or a stale PostgreSQL session blob).

    Clears ``daily_restored`` so Tier-3 SQLite can still top up if warm cache is partial.
  """
    if not session_needs_operational_data(sess):
        return False
    had_pause = bool(getattr(sess, "pause_auto_data_restore", False))
    sess.pause_auto_data_restore = False
    sess.daily_restored = False
    changed = False
    try:
        restore_po_sidecars_from_warm(sess)
        if _warm_cache and _copy_warm_cache_to_session(sess):
            sess._warm_cache_gen = warm_cache_generation
            sess._warm_cache_only = True
            changed = True
        elif _apply_warm_cache_if_needed(sess, warm_cache_generation):
            changed = True
    finally:
        if had_pause and session_needs_operational_data(sess):
            sess.pause_auto_data_restore = True
        elif had_pause:
            from .session import resume_auto_data_restore

            resume_auto_data_restore(sess)
    return changed


def restore_po_sidecars_from_warm(sess) -> bool:
    """Copy PO sidecar DataFrames from warm cache when the session is missing them.

    Does not acquire ``_daily_restore_lock`` — safe to call from ``/data/coverage`` while
    a Tier-3 upload holds the lock. Fixes PO UI showing \"No sheet loaded\" after restart
    when sales/inventory were already copied but optional PO sheets were skipped.
    """
    if not _warm_cache:
        return False
    changed = False
    for key in _PO_SIDECAR_KEYS:
        cur = getattr(sess, key, None)
        wc = _warm_cache.get(key)
        if wc is None or not hasattr(wc, "empty") or wc.empty:
            continue
        if key == "daily_inventory_history_df":
            from .services.daily_inventory_history import merge_inventory_history

            if cur is not None and hasattr(cur, "empty") and not cur.empty:
                merged = merge_inventory_history(cur, wc)
                if len(merged) != len(cur) or not merged.equals(cur):
                    setattr(sess, key, merged)
                    changed = True
            else:
                setattr(sess, key, wc.copy() if hasattr(wc, "copy") else wc)
                changed = True
            continue
        if cur is not None and hasattr(cur, "empty") and not cur.empty:
            continue
        setattr(sess, key, wc.copy() if hasattr(wc, "copy") else wc)
        changed = True
    if changed:
        sess._quarterly_cache.clear()
    return changed


def persist_po_sidecars_to_disk() -> None:
    """Write PO sidecar parquets into the disk warm-cache (survives container restart)."""
    import json
    import os

    if not _warm_cache:
        return
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        manifest_path = os.path.join(_DISK_CACHE_DIR, "_manifest.json")
        manifest: dict = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        keys = set(manifest.get("keys") or [])
        for key in _PO_SIDECAR_KEYS:
            df = _warm_cache.get(key)
            if df is None or not hasattr(df, "empty") or df.empty:
                continue
            from .services.helpers import _coerce_df_for_parquet

            path = os.path.join(_DISK_CACHE_DIR, f"{key}.parquet")
            _coerce_df_for_parquet(df).to_parquet(path, index=False)
            keys.add(key)
        if not keys:
            return
        manifest["keys"] = sorted(keys)
        manifest["saved_at"] = datetime.now(IST).isoformat()
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        log.info("PO sidecar disk cache updated (%s)", _DISK_CACHE_DIR)
    except Exception as _e:
        log.warning("PO sidecar disk persist failed: %s", _e)


def merge_po_optional_sheets_into_warm_cache(sess) -> None:
    """Copy PO-only sidecar frames into the shared warm cache (non-destructive merge).

    ``POST /api/po/daily-inventory-history``, ``POST /api/po/sku-status-lead``, and raise-ledger
    updates used to touch only the in-memory session. After a restart, ``_restore_daily_if_needed``
    could not find these keys in ``_warm_cache``, so coverage showed *No sheet loaded* / empty
    ledger even though the operator had already uploaded once. GitHub cache saves list these
    keys — we mirror them into ``_warm_cache`` after each change so new sessions inherit the
    same data without a full cache republish.
    """
    import pandas as pd

    global _warm_cache
    if not _warm_cache:
        _warm_cache = {}
    for key in _PO_SIDECAR_KEYS:
        df = getattr(sess, key, None)
        if df is None or not hasattr(df, "empty"):
            _warm_cache[key] = pd.DataFrame()
        elif df.empty:
            _warm_cache[key] = pd.DataFrame()
        else:
            _warm_cache[key] = df.copy()
    try:
        persist_po_sidecars_to_disk()
    except Exception:
        log.exception("persist_po_sidecars_to_disk after merge failed")


import os as _os_main
_DISK_CACHE_DIR     = _os_main.environ.get("WARM_CACHE_DIR", "/data/warm_cache")
_DISK_CACHE_MAX_AGE = int(_os_main.environ.get("WARM_CACHE_MAX_AGE_HOURS", "24"))


def _skip_phase2_when_disk_fresh() -> bool:
    """When Phase-0 disk cache is valid, skip GitHub download (saves ~1–3 GB RAM, 60–90s CPU)."""
    return _os_main.environ.get("WARM_CACHE_SKIP_PHASE2_WHEN_DISK_FRESH", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _merge_recent_sqlite_into_warm_cache(months: int = 4) -> None:
    """Light Tier-3 top-up into ``_warm_cache`` without GitHub Phase 2."""
    global _warm_cache
    from .services.daily_store import load_platform_data, merge_platform_data
    from .services.sales import build_sales_df
    import pandas as pd

    if not _warm_cache:
        return
    for plat, key in (
        ("amazon", "mtr_df"),
        ("myntra", "myntra_df"),
        ("meesho", "meesho_df"),
        ("flipkart", "flipkart_df"),
        ("snapdeal", "snapdeal_df"),
    ):
        df = load_platform_data(plat, months=months)
        if df is not None and not df.empty:
            cur = _warm_cache.get(key, pd.DataFrame())
            _warm_cache[key] = merge_platform_data(cur, df, plat)
    try:
        _warm_cache["sales_df"] = build_sales_df(
            mtr_df=_warm_cache.get("mtr_df", pd.DataFrame()),
            myntra_df=_warm_cache.get("myntra_df", pd.DataFrame()),
            meesho_df=_warm_cache.get("meesho_df", pd.DataFrame()),
            flipkart_df=_warm_cache.get("flipkart_df", pd.DataFrame()),
            snapdeal_df=_warm_cache.get("snapdeal_df", pd.DataFrame()),
            sku_mapping=_warm_cache.get("sku_mapping") or {},
        )
    except Exception:
        log.exception("merge_recent_sqlite_into_warm_cache: sales rebuild failed")


def _save_warm_cache_to_disk(cache_dict: dict) -> None:
    """Persist warm_cache DataFrames + sku_mapping to /data/warm_cache/ as parquet/JSON.
    Called after Phase 2 completes. ~2-3 s. Safe from any thread."""
    import os, json
    try:
        os.makedirs(_DISK_CACHE_DIR, exist_ok=True)
        saved: list[str] = []
        for key, val in cache_dict.items():
            if key == "sku_mapping":
                path = os.path.join(_DISK_CACHE_DIR, "sku_mapping.json")
                with open(path, "w") as f:
                    json.dump(val if isinstance(val, dict) else {}, f)
                saved.append(key)
            elif hasattr(val, "to_parquet") and hasattr(val, "empty") and not val.empty:
                path = os.path.join(_DISK_CACHE_DIR, f"{key}.parquet")
                val.to_parquet(path, index=False)
                saved.append(key)
        manifest = {"saved_at": datetime.now(IST).isoformat(), "keys": saved}
        with open(os.path.join(_DISK_CACHE_DIR, "_manifest.json"), "w") as f:
            json.dump(manifest, f)
        log.info("Warm-cache saved to disk (%d keys) → %s", len(saved), _DISK_CACHE_DIR)
    except Exception as _e:
        log.warning("Warm-cache disk save failed: %s", _e)


def _load_warm_cache_from_disk(ignore_age: bool = False) -> "tuple[bool, dict]":
    """Load warm_cache from /data/warm_cache/ if the manifest is < WARM_CACHE_MAX_AGE_HOURS old
    (unless ``ignore_age`` is True — used by ``/cache/reload-fresh`` to recover bulk history
    when the GitHub Release was overwritten by a small session snapshot).

    Returns (ok, loaded_dict). ok=False means disk cache is absent, stale, or corrupt."""
    import os, json
    try:
        manifest_path = os.path.join(_DISK_CACHE_DIR, "_manifest.json")
        if not os.path.exists(manifest_path):
            return False, {}
        with open(manifest_path) as f:
            manifest = json.load(f)
        saved_at_str = manifest.get("saved_at", "")
        if not saved_at_str:
            return False, {}
        saved_at = datetime.fromisoformat(saved_at_str)
        # Ensure tz-aware comparison
        if saved_at.tzinfo is None:
            saved_at = saved_at.replace(tzinfo=IST)
        age_hours = (datetime.now(IST) - saved_at).total_seconds() / 3600
        if not ignore_age and age_hours > _DISK_CACHE_MAX_AGE:
            log.info(
                "Disk cache is %.1fh old (limit %dh) — skipping, will reload from GitHub.",
                age_hours, _DISK_CACHE_MAX_AGE,
            )
            return False, {}
        if ignore_age and age_hours > _DISK_CACHE_MAX_AGE:
            log.warning(
                "reload-fresh recovery: using disk warm-cache despite age %.1fh (limit %dh) — "
                "merging with GitHub to restore rows missing from Release assets.",
                age_hours,
                _DISK_CACHE_MAX_AGE,
            )

        import pandas as pd
        keys = manifest.get("keys", [])
        loaded: dict = {}
        for key in keys:
            if key == "sku_mapping":
                path = os.path.join(_DISK_CACHE_DIR, "sku_mapping.json")
                if os.path.exists(path):
                    with open(path) as f:
                        loaded["sku_mapping"] = json.load(f)
            else:
                path = os.path.join(_DISK_CACHE_DIR, f"{key}.parquet")
                if os.path.exists(path):
                    loaded[key] = pd.read_parquet(path)

        if not loaded:
            return False, {}

        log.info(
            "Phase 0 disk cache: %.1fh old, %d keys loaded from %s",
            age_hours, len(loaded), _DISK_CACHE_DIR,
        )
        return True, loaded
    except Exception as _e:
        log.warning("Warm-cache disk load failed: %s", _e)
        return False, {}


def _warm_cache_loose_parquets_from_dir(disk_dir: "Path") -> dict:
    """Load ``mtr_df.parquet`` / etc. from ``disk_dir`` without ``_manifest.json``."""
    import json
    from pathlib import Path

    import pandas as pd

    out: dict = {}
    if not disk_dir.is_dir():
        return out
    for key in (
        "mtr_df",
        "myntra_df",
        "meesho_df",
        "flipkart_df",
        "snapdeal_df",
        "sales_df",
        "inventory_df_variant",
        "inventory_df_parent",
        "existing_po_df",
    ):
        p = disk_dir / f"{key}.parquet"
        if p.is_file():
            try:
                out[key] = pd.read_parquet(p)
            except Exception as ex:
                log.warning("warm-cache parquet read %s: %s", p, ex)
    sm = disk_dir / "sku_mapping.json"
    if sm.is_file():
        try:
            with open(sm) as f:
                out["sku_mapping"] = json.load(f)
        except Exception as ex:
            log.warning("warm-cache sku_mapping read %s: %s", sm, ex)
    return out


def warm_cache_disk_recovery_dict() -> dict:
    """
    Build the best on-disk snapshot for ``/cache/reload-fresh`` recovery.

    Combines, in order:
    1. Manifest-based load from ``WARM_CACHE_DIR`` (``_load_warm_cache_from_disk(ignore_age=True)``)
    2. Loose parquets in the same directory (when manifest is missing or incomplete)
    3. Optional ``WARM_CACHE_RECOVERY_DIR`` (e.g. operator-mounted backup volume)

    For each sales platform, frames from all sources are merged with
    ``merge_platform_data`` in **ascending row-count order** so the largest file
    wins on overlapping keys (avoids a small snapshot overwriting bulk).
    """
    import os
    from pathlib import Path

    import pandas as pd

    from .services.daily_store import merge_platform_data as _merge_pd

    base = Path(_DISK_CACHE_DIR)
    layers: list[dict] = []

    ok_m, data_m = _load_warm_cache_from_disk(ignore_age=True)
    if ok_m and data_m:
        layers.append(data_m)

    loose_main = _warm_cache_loose_parquets_from_dir(base)
    if loose_main:
        layers.append(loose_main)

    rec = (os.environ.get("WARM_CACHE_RECOVERY_DIR") or "").strip()
    if rec:
        loose_rec = _warm_cache_loose_parquets_from_dir(Path(rec))
        if loose_rec:
            log.info(
                "warm-cache recovery: merged WARM_CACHE_RECOVERY_DIR=%s (%d keys)",
                rec,
                len(loose_rec),
            )
            layers.append(loose_rec)

    if not layers:
        return {}

    pairs = [
        ("amazon", "mtr_df"),
        ("myntra", "myntra_df"),
        ("meesho", "meesho_df"),
        ("flipkart", "flipkart_df"),
        ("snapdeal", "snapdeal_df"),
    ]
    out: dict = {}
    for plat, key in pairs:
        frames = [
            L[key]
            for L in layers
            if isinstance(L.get(key), pd.DataFrame) and not L[key].empty
        ]
        if not frames:
            continue
        frames.sort(key=len)
        acc = frames[0]
        for nxt in frames[1:]:
            acc = _merge_pd(acc, nxt, plat)
        out[key] = acc

    sku: dict = {}
    for L in layers:
        d = L.get("sku_mapping")
        if isinstance(d, dict) and d:
            sku = {**sku, **d}
    if sku:
        out["sku_mapping"] = sku

    return out


def _do_load_warm_cache() -> bool:
    """
    Three-phase warm-cache load. Thread-safe (GIL).

    Phase 0 (~2-3 s): Load parquet files from /data/warm_cache/ (Docker volume,
    survives container restarts). Written after each successful Phase 2. On the
    NEXT deploy users get full historical data instantly instead of waiting
    60-90 s for GitHub. If cache is absent or >24 h old, falls through to Phase 1.

    Phase 1 (~2 s): Load platform DataFrames from the local SQLite daily store,
    publish to _warm_cache (only when Phase 0 is unavailable), and immediately
    signal _warm_cache_ready so the first page-load after a deploy returns data
    within a couple of seconds.

    Phase 2 (~60–90 s): Download GitHub historical cache in the same thread (no
    extra thread needed — lifespan already runs this in an executor). Strip any
    GitHub rows whose dates are already covered by the SQLite store (daily store
    always wins), merge, rebuild sales_df, update _warm_cache, and persist result
    to disk (Phase 0 for the next restart). Existing sessions that received
    Phase-0/Phase-1 data get the fuller historical view on their next request.
    """
    global _warm_cache, _warm_cache_loaded_at, _warm_cache_generation
    _phase2_lock_held = False   # tracked here so the outer finally can release safely
    try:
        from .services.github_cache import load_cache_from_drive
        from .services.daily_store import load_all_platforms, merge_platform_data as _merge
        from .services.sales import build_sales_df
        import pandas as pd

        def _sanitise_snapdeal(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "OMS_SKU" not in df.columns:
                return df
            bad = df["OMS_SKU"].astype(str).str.upper()
            return df[
                ~(bad.isin(["", "NAN", "NONE", "UNKNOWN", "N/A", "NA", "NULL"])
                  | df["OMS_SKU"].astype(str).str.match(r'^\d+$'))
            ].reset_index(drop=True)

        # ── Phase 0: local disk cache (fastest, ~2-3 s) ──────────────────────
        # Parquet files written to /data/warm_cache/ after each successful Phase 2.
        # /data is a Docker volume that persists across container restarts.
        # If the cache is fresh (< WARM_CACHE_MAX_AGE_HOURS old) we serve it
        # immediately and still run Phase 1+2 in the same thread to pick up any
        # new uploads that arrived since the last deploy.
        disk_ok, disk_data = _load_warm_cache_from_disk()
        if disk_ok and disk_data:
            _warm_cache = disk_data
            _warm_cache_loaded_at = datetime.now(IST)
            _warm_cache_generation += 1   # generation 1 = Phase-0 disk data
            _warm_cache_ready.set()       # ← unblocks first page-load immediately
            log.warning(
                "Warm-cache Phase 0 ready from disk (%d keys, gen=%d). "
                "Continuing Phase 1+2 to pick up new uploads…",
                len(disk_data), _warm_cache_generation,
            )

        # ── Acquire upload-memory lock before Phase 1 ─────────────────────────
        # Phase-0 disk data (~3-4 GB) is live in _warm_cache while Phase-1 SQLite
        # loads another ~1 GB. If a daily upload (1-2 GB) runs concurrently the
        # container exceeds its 7.5 GB limit. Hold the lock here — uploads will
        # queue in DAILY_UPLOAD_EXECUTOR — and release only after the Phase-0 data
        # has been swapped out (early-return path) or Phase-2 completes.
        import gc as _gc
        _UPLOAD_MEMORY_LOCK.acquire()
        _phase2_lock_held = True

        # ── Phase 1: SQLite data (local) ──────────────────────────────────────
        # If Phase 0 disk cache is present, only top-up recent months for speed.
        # If Phase 0 is absent, load full SQLite history so users still get full
        # data even when GitHub Phase 2 is slow/unavailable.
        from .services.daily_store import load_platform_data as _load_plat
        # Always cap Phase-1 at 4 months: Phase-2 (GitHub) covers older history.
        # Loading ALL history when disk_ok=False would hold 2-4 GB in p1_raw
        # while Phase-2 simultaneously downloads another 4 GB → OOM.
        _P1_MONTHS = 4

        phase1_ok = False
        try:
            p1_raw: dict = {}
            for _plat in ("amazon", "myntra", "meesho", "flipkart", "snapdeal"):
                _df = _load_plat(_plat, months=_P1_MONTHS)
                if not _df.empty:
                    p1_raw[_plat] = _df
            if p1_raw:
                p1: dict = {
                    "mtr_df":               p1_raw.get("amazon",   pd.DataFrame()),
                    "myntra_df":            p1_raw.get("myntra",   pd.DataFrame()),
                    "meesho_df":            p1_raw.get("meesho",   pd.DataFrame()),
                    "flipkart_df":          p1_raw.get("flipkart", pd.DataFrame()),
                    "snapdeal_df":          _sanitise_snapdeal(p1_raw.get("snapdeal", pd.DataFrame())),
                    "sales_df":             pd.DataFrame(),
                    "sku_mapping":          {},
                    "inventory_df_variant": pd.DataFrame(),
                    "inventory_df_parent":  pd.DataFrame(),
                }
                has_sales = any(
                    not v.empty
                    for v in [p1["mtr_df"], p1["myntra_df"], p1["meesho_df"],
                               p1["flipkart_df"], p1["snapdeal_df"]]
                )
                if has_sales:
                    p1["sales_df"] = build_sales_df(
                        mtr_df=p1["mtr_df"],       myntra_df=p1["myntra_df"],
                        meesho_df=p1["meesho_df"], flipkart_df=p1["flipkart_df"],
                        sku_mapping={},            snapdeal_df=p1["snapdeal_df"],
                    )
                # Only publish Phase-1 data to _warm_cache when Phase 0 disk cache
                # is absent — Phase 0 data is full historical and must not be
                # replaced by the 4-month-only Phase-1 snapshot.
                if not disk_ok:
                    _warm_cache = p1
                    _warm_cache_loaded_at = datetime.now(IST)
                    _warm_cache_generation += 1   # generation 1 = Phase-1 SQLite data
                    _warm_cache_ready.set()       # ← unblocks first page-load immediately
                phase1_ok = True
                log.info(
                    "Warm-cache Phase 1 complete: %d sales rows (%s from SQLite). "
                    "%s — fetching GitHub historical cache…",
                    len(p1.get("sales_df", pd.DataFrame())),
                    (f"last {_P1_MONTHS} months" if _P1_MONTHS is not None else "full history"),
                    "published to warm_cache" if not disk_ok else "NOT published (disk cache active)",
                )
        except Exception as e:
            log.warning("Warm-cache Phase 1 (SQLite) failed: %s — continuing to GitHub", e)

        if disk_ok and disk_data and _skip_phase2_when_disk_fresh():
            # Free Phase-1 scratch data (p1_raw / p1) — they are not needed
            # here since _warm_cache already holds the full Phase-0 disk cache.
            # Do NOT swap _warm_cache to p1: Phase-0 has full historical data
            # (2+ years); Phase-1 only has the last 4 months.
            try:
                del p1_raw, p1
            except Exception:
                pass
            disk_data = None
            _gc.collect()
            # Release lock — Phase-1 RAM is freed; uploads may proceed safely
            # against the ~3.5 GB Phase-0 warm cache.
            _UPLOAD_MEMORY_LOCK.release()
            _phase2_lock_held = False
            log.info(
                "Warm-cache: fresh disk cache — skipping GitHub Phase 2 "
                "(set WARM_CACHE_SKIP_PHASE2_WHEN_DISK_FRESH=0 to force full reload)."
            )
            try:
                _merge_recent_sqlite_into_warm_cache(months=_P1_MONTHS or 4)
            except Exception:
                log.exception("Warm-cache SQLite top-up after disk load failed")
            try:
                _save_warm_cache_to_disk(_warm_cache)
            except Exception:
                pass
            return True

        # ── Phase 2: GitHub historical cache (network, slow) ──────────────────
        # Provides data for dates not yet in the SQLite daily store.
        #
        # ── Free Phase-0 data before Phase-2 to prevent OOM ──────────────────
        # When disk_ok=True, _warm_cache holds the full Phase-0 historical data
        # (2-4 GB with large MTR). Phase-2 downloads another 2-4 GB on top.
        # Running both simultaneously exceeds the 7.5 GB container limit.
        #
        # Fix: before Phase-2 downloads, swap _warm_cache from Phase-0 (heavy)
        # to Phase-1 SQLite data (lightweight, ~200 MB). If Phase-1 had no data,
        # clear _warm_cache entirely. Phase-0 data is dereferenced and freed
        # before the GitHub download allocates.
        #
        # _UPLOAD_MEMORY_LOCK is already held (acquired before Phase 1 above).
        if disk_ok:
            if phase1_ok and p1_raw:
                # Temporarily serve Phase-1 SQLite data while Phase-2 loads
                _warm_cache = p1
                _warm_cache_loaded_at = datetime.now(IST)
                _warm_cache_ready.set()
                log.info(
                    "Phase-2 pre-load: swapped Phase-0 → Phase-1 SQLite data "
                    "to free memory before GitHub download."
                )
            else:
                _warm_cache = {}
                log.info(
                    "Phase-2 pre-load: cleared Phase-0 data (no Phase-1 available) "
                    "to free memory before GitHub download."
                )
        else:
            # disk_ok=False: _warm_cache may hold Phase-1 SQLite data (set above).
            # Clear it now so Phase-2 download doesn't stack on top of it.
            # Users see a loading state for the ~60-90 s Phase-2 takes;
            # Phase-1 data was already served before this point.
            _warm_cache = {}
            log.info(
                "Phase-2 pre-load: cleared Phase-1 data (no disk cache) "
                "to free memory before GitHub download."
            )
        try:
            del p1_raw, p1
        except Exception:
            pass
        if disk_ok and disk_data is not None:
            disk_data = None
        _gc.collect()
        # Lock already held since before Phase 1 (acquired above).
        ok, msg, loaded = load_cache_from_drive()
        if not ok or not loaded:
            log.warning("Warm-cache Phase 2 (GitHub) failed: %s", msg)
            if not phase1_ok:
                # Nothing at all — no disk cache, no SQLite, no GitHub
                if disk_ok and disk_data:
                    log.info("Using Phase 0 disk cache only (GitHub unavailable, SQLite empty).")
                    return True
                return False
            return True   # Phase 1 data is still good

        # Sanitise Snapdeal from GitHub
        loaded["snapdeal_df"] = _sanitise_snapdeal(loaded.get("snapdeal_df", pd.DataFrame()))

        # Date-superseding merge: for every date covered by the SQLite daily store,
        # strip that date from the GitHub cache before merging so the freshly-parsed
        # SQLite rows always win (prevents double-counting after parser fixes).
        # Also strip dates that were *deleted* from SQLite by cleanup migrations —
        # the GitHub cache may still carry those stale rows.
        try:
            from .services.daily_store import get_blocked_dates as _get_blocked_dates
            # Re-read recent SQLite (4-month window) to pick up uploads that
            # arrived during Phase-2 download. Use the same month cap as Phase-1:
            # GitHub covers older history; loading all history here would add
            # ~2 GB on top of the already-loaded GitHub dict and cause OOM.
            daily = {
                _p: _df
                for _p in ("amazon", "myntra", "meesho", "flipkart", "snapdeal")
                if not (_df := _load_plat(_p, months=4)).empty
            }
            merged_any = False
            for plat, key in [
                ("amazon",   "mtr_df"),
                ("myntra",   "myntra_df"),
                ("meesho",   "meesho_df"),
                ("flipkart", "flipkart_df"),
                ("snapdeal", "snapdeal_df"),
            ]:
                github_df = loaded.get(key, pd.DataFrame())
                if not github_df.empty and "Date" in github_df.columns:
                    try:
                        daily_df = daily.get(plat, pd.DataFrame())
                        # Dates present in SQLite — those rows always win
                        daily_dates: set = set()
                        if not daily_df.empty and "Date" in daily_df.columns:
                            daily_dates = set(
                                pd.to_datetime(daily_df["Date"], errors="coerce").dt.normalize()
                            )
                        # Dates removed by cleanup migrations — must also be blocked
                        # from the GitHub cache so stale rows don't sneak back in.
                        for bd_str in _get_blocked_dates(plat):
                            try:
                                daily_dates.add(pd.Timestamp(bd_str).normalize())
                            except Exception:
                                pass
                        if daily_dates:
                            github_dates = pd.to_datetime(
                                github_df["Date"], errors="coerce"
                            ).dt.normalize()
                            github_df = github_df[~github_dates.isin(daily_dates)].copy()
                            loaded[key] = github_df
                    except Exception:
                        pass
                if daily.get(plat) is not None and not daily[plat].empty:
                    loaded[key] = _merge(loaded.get(key, pd.DataFrame()), daily[plat], plat)
                    merged_any = True
            if merged_any:
                log.info("Phase 2: SQLite merged on top of GitHub for %d platform(s).", merged_any)
        except Exception as e:
            log.warning("Phase 2 daily-store merge warning: %s", e)

        # ── Trim platform DFs in loaded to 18-month window before build_sales_df ──
        # Large uploads (e.g. 107 MB MTR) can push build_sales_df memory to 6+ GB.
        # Trim each platform DF to the most recent 18 months; PO calc uses ≤15 months.
        try:
            from .services.platform_session_window import SESSION_PLATFORM_MAX_DAYS

            _trim_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=SESSION_PLATFORM_MAX_DAYS)
            for _trim_key, _date_col in [
                ("mtr_df", "Date"), ("myntra_df", "Date"),
                ("meesho_df", "Date"), ("flipkart_df", "Date"),
                ("snapdeal_df", "Date"),
            ]:
                _df = loaded.get(_trim_key)
                if _df is not None and not _df.empty and _date_col in _df.columns:
                    _dates = pd.to_datetime(_df[_date_col], errors="coerce")
                    _mask = _dates >= _trim_cutoff
                    if _mask.sum() < len(_df):
                        loaded[_trim_key] = _df[_mask].reset_index(drop=True)
                        log.info(
                            "Phase 2 trim: %s %d→%d rows (%d-day window)",
                            _trim_key, len(_df), len(loaded[_trim_key]), SESSION_PLATFORM_MAX_DAYS,
                        )
            _gc.collect()
        except Exception as _trim_err:
            log.warning("Phase 2 platform trim warning (non-fatal): %s", _trim_err)

        # ── Free daily + old warm cache before build_sales_df (memory peak) ────
        # build_sales_df is the heaviest allocation in Phase 2. Free everything we
        # no longer need so it doesn't run alongside _warm_cache + loaded + daily.
        try:
            del daily
        except Exception:
            pass
        # Save fallback sales_df in case build_sales_df fails below.
        _old_sales_df: "pd.DataFrame" = (
            _warm_cache.get("sales_df", pd.DataFrame())
            if _warm_cache else pd.DataFrame()
        )
        # Swap out the old warm cache before the peak allocation.
        # Active sessions already hold their own data; the brief empty window is safe.
        _warm_cache = {}
        _gc.collect()

        # Rebuild unified sales_df from the fully-merged platform data
        try:
            loaded["sales_df"] = build_sales_df(
                mtr_df=loaded.get("mtr_df", pd.DataFrame()),
                myntra_df=loaded.get("myntra_df", pd.DataFrame()),
                meesho_df=loaded.get("meesho_df", pd.DataFrame()),
                flipkart_df=loaded.get("flipkart_df", pd.DataFrame()),
                sku_mapping=loaded.get("sku_mapping") or {},
                snapdeal_df=loaded.get("snapdeal_df"),
            )
            log.info("Phase 2 sales_df: %d rows", len(loaded["sales_df"]))
        except Exception as e:
            log.warning("Phase 2 sales_df rebuild failed: %s", e)
            if phase1_ok:
                loaded.setdefault("sales_df", _old_sales_df)
        del _old_sales_df
        _gc.collect()
        _warm_cache = loaded
        _warm_cache_loaded_at = datetime.now(IST)
        _warm_cache_generation += 1   # generation 2+ = Phase-2 GitHub+SQLite data
        log.info("Warm cache fully loaded (Phase 2, gen=%d) at %s — %s",
                 _warm_cache_generation, _warm_cache_loaded_at.strftime("%H:%M IST"), msg)

        # ── Phase 0 save: persist to disk for sub-3-second load on next restart ──
        # /data/warm_cache/ lives on the Docker volume that survives container restarts.
        try:
            _save_warm_cache_to_disk(_warm_cache)
        except Exception as _disk_err:
            log.warning("Warm-cache disk save failed (non-fatal): %s", _disk_err)

        return True

    except Exception as e:
        log.exception("Warm cache load error: %s", e)
        return False

    finally:
        # Always signal so waiting code is never blocked forever
        # (no-op if Phase 1 already called set())
        _warm_cache_ready.set()
        # Release Phase-2 memory lock if it was acquired
        try:
            if _phase2_lock_held:
                _UPLOAD_MEMORY_LOCK.release()
        except Exception:
            pass


def _copy_warm_cache_to_session(sess) -> bool:
    """Copy _warm_cache into an AppSession. Returns True if data was available."""
    if not _warm_cache:
        return False
    for key, val in _warm_cache.items():
        # Trim daily_inventory_history_df to _MAX_HISTORY_DAYS when copying from
        # warm cache — prevents OOM if the cached version predates upload-time trimming.
        if key == "daily_inventory_history_df" and val is not None and hasattr(val, "empty") and not val.empty:
            try:
                from .services.daily_inventory_upload_run import _MAX_HISTORY_DAYS, _trim_history_to_recent
                trimmed, _note = _trim_history_to_recent(val, _MAX_HISTORY_DAYS)
                if _note:
                    log.info("warm-cache copy: trimmed daily_inventory_history_df: %s", _note)
                setattr(sess, key, trimmed)
                continue
            except Exception:
                pass  # fall through to plain copy on any import / trim error
        if key in ("mtr_df", "myntra_df", "meesho_df", "flipkart_df", "snapdeal_df") and val is not None and hasattr(val, "empty") and not val.empty:
            try:
                from .services.platform_session_window import trim_platform_df
                before = len(val)
                val = trim_platform_df(val)
                if len(val) < before:
                    log.info("warm-cache copy: trimmed %s %d→%d rows", key, before, len(val))
            except Exception:
                pass
        setattr(sess, key, val)
    sess._quarterly_cache.clear()
    # Warm cache already contains rebuilt sales + merged platform history.
    # Mark restored to avoid triggering a heavy synchronous SQLite restore on first
    # /data/* request right after login (the main cause of "syncing..." slowness).
    sess.daily_restored = True
    return True


def _apply_warm_cache_if_needed(sess, warm_cache_generation: int) -> bool:
    """Decide whether to copy warm cache into session (sync; may run in thread pool)."""
    restore_po_sidecars_from_warm(sess)
    if getattr(sess, "pause_auto_data_restore", False) and not session_needs_operational_data(sess):
        return False
    _session_gen = getattr(sess, "_warm_cache_gen", 0)
    _wc_only = getattr(sess, "_warm_cache_only", False)
    _phase2_ready = warm_cache_generation >= 2 and _session_gen < warm_cache_generation

    def _warm_cache_has_more() -> bool:
        if not _warm_cache:
            return False
        wc_sm = _warm_cache.get("sku_mapping")
        if isinstance(wc_sm, dict) and wc_sm and not getattr(sess, "sku_mapping", None):
            return True
        for key in [
            "mtr_df", "myntra_df", "meesho_df", "flipkart_df",
            "snapdeal_df", "sales_df", "inventory_df_variant",
            *_PO_SIDECAR_KEYS,
        ]:
            wc = _warm_cache.get(key)
            cur = getattr(sess, key, None)
            wc_has_data = hasattr(wc, "empty") and not wc.empty
            cur_has_data = hasattr(cur, "empty") and not cur.empty
            if wc_has_data and not cur_has_data:
                return True
        return False

    if not getattr(sess, "sku_mapping", None):
        try:
            from .services.sku_mapping import restore_sku_mapping_to_session

            restore_sku_mapping_to_session(sess)
        except Exception:
            pass

    if sess.mtr_df.empty and sess.sales_df.empty:
        if _copy_warm_cache_to_session(sess):
            sess._warm_cache_gen = warm_cache_generation
            sess._warm_cache_only = True
            return True
    elif _warm_cache_has_more():
        if _copy_warm_cache_to_session(sess):
            sess._warm_cache_gen = warm_cache_generation
            sess._warm_cache_only = True
            return True
    elif _wc_only and _phase2_ready:
        if _copy_warm_cache_to_session(sess):
            sess._warm_cache_gen = warm_cache_generation
            sess._warm_cache_only = True
            return True
    return False


def _cookie_secure_from_request(request: Request) -> bool:
    xf = request.headers.get("x-forwarded-proto")
    if xf:
        return xf.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def _do_marketplace_sync_all() -> None:
    """
    Called daily at 6AM IST. Pulls last 2 days of data from all connected
    marketplace APIs and merges results into warm cache + SQLite daily store.
    """
    from .db.marketplace_db import get_credentials, save_sync_log
    from .services.daily_store import merge_platform_data, save_daily_file
    from datetime import date as _date
    import pandas as pd

    # Map: platform → (sync_fn, cache_key, db_platform)
    _PLATFORM_MAP = {
        "amazon":   ("amazon_sp_api",  "sync_amazon_data",   "mtr_df",      "amazon"),
        "flipkart": ("flipkart_api",   "sync_flipkart_data", "flipkart_df", "flipkart"),
        "myntra":   ("myntra_api",     "sync_myntra_data",   "myntra_df",   "myntra"),
        "meesho":   ("meesho_api",     "sync_meesho_data",   "meesho_df",   "meesho"),
    }

    for platform, (module, fn_name, cache_key, db_platform) in _PLATFORM_MAP.items():
        creds = get_credentials(platform)
        if not creds:
            continue
        log.info("Scheduled %s sync starting…", platform)
        try:
            import importlib
            mod = importlib.import_module(f".services.{module}", package="backend")
            sync_fn = getattr(mod, fn_name)
            df, msg = sync_fn(creds, days_back=2)
        except Exception as e:
            log.exception("Scheduled %s sync failed: %s", platform, e)
            save_sync_log(platform, "error", 0, message=str(e))
            continue

        if df.empty:
            save_sync_log(platform, "partial", 0, message=msg)
            log.info("Scheduled %s: no new data — %s", platform, msg)
            continue

        api_sync_fname = f"api-{platform}-{_date.today()}.csv"
        try:
            save_daily_file(db_platform, api_sync_fname, df)
        except Exception as e:
            log.warning("Scheduled %s: daily store save failed: %s", platform, e)

        try:
            existing = _warm_cache.get(cache_key, pd.DataFrame())
            _warm_cache[cache_key] = merge_platform_data(
                existing, df, db_platform, source_filename=api_sync_fname,
            )
            log.info("Scheduled %s: warm cache updated with %d rows", platform, len(df))
        except Exception as e:
            log.warning("Scheduled %s: warm cache merge failed: %s", platform, e)

        date_col  = "Date"
        date_from = str(df[date_col].min().date()) if date_col in df.columns and not df.empty else ""
        date_to   = str(df[date_col].max().date()) if date_col in df.columns and not df.empty else ""
        save_sync_log(platform, "success", len(df), date_from, date_to, msg)
        log.info("Scheduled %s sync complete: %s", platform, msg)


async def _warm_cache_scheduler():
    """Background task: refresh cache + run Amazon sync at 06:00 IST every day."""
    while True:
        now = datetime.now(IST)
        target = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        log.info("Next warm-cache refresh at %s IST (in %.0fs)", target.strftime("%Y-%m-%d %H:%M"), wait_seconds)
        await asyncio.sleep(wait_seconds)
        log.info("Running scheduled warm-cache refresh…")
        loop = asyncio.get_event_loop()
        await run_heavy(_do_load_warm_cache)
        # Run all marketplace API syncs after cache refresh
        await run_heavy(_do_marketplace_sync_all)


def _bootstrap_stitching_on_startup() -> None:
    try:
        from .services.stitching_costing import bootstrap_stitching_data

        out = bootstrap_stitching_data()
        if out.get("merged"):
            log.info("Stitching bootstrap merged sheets: %s", out.get("added_rows"))
        elif out.get("skipped"):
            log.debug("Stitching bootstrap skipped: %s", out.get("skipped"))
    except Exception:
        log.exception("Stitching bootstrap failed")


async def _session_eviction_loop() -> None:
    """Drop idle browser sessions from RAM (each can duplicate warm-cache-sized frames)."""
    interval = int(_os_main.environ.get("SESSION_EVICT_INTERVAL_SEC", "1800"))
    while True:
        await asyncio.sleep(max(300, interval))
        try:
            n = store.evict_stale()
            if n:
                log.info("Evicted %d idle session(s) from memory", n)
        except Exception:
            log.exception("session eviction failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load cache in background so server is ready immediately
    # Never use the default thread pool (Starlette run_in_threadpool / sync routes).
    # Warm-cache load can run many minutes and would starve /api/auth/login → 504.
    asyncio.create_task(run_heavy(_bootstrap_stitching_on_startup))
    asyncio.create_task(run_heavy(_do_load_warm_cache))
    # Schedule daily 6AM IST refresh
    task = asyncio.create_task(_warm_cache_scheduler())
    evict_task = asyncio.create_task(_session_eviction_loop())
    yield
    task.cancel()
    evict_task.cancel()


app = FastAPI(
    title="Yash Gallery ERP API",
    version="1.0.0",
    description="FastAPI backend for the Yash Gallery ERP system",
    lifespan=lifespan,
)

# ── CORS (allow Vite dev server + production domain) ─────────
import os as _os
_extra = _os.environ.get("EXTRA_CORS_ORIGIN", "").strip()
_origins = [
    "http://localhost:5173",        # Vite dev
    "http://localhost:3000",        # alternate dev
    "https://progressino.com",      # production root
    "https://www.progressino.com",  # www
    "https://app.progressino.com",  # app subdomain
    "https://yashgallery.com",      # yashgallery production
    "https://www.yashgallery.com",  # www
    "https://app.yashgallery.com",  # app subdomain
]
if _extra:
    _origins.append(_extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,       # needed for the session cookie
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


def _session_skip_heavy_warm(path: str) -> bool:
    """Poll/start endpoints must stay fast while a heavy PO job runs in the background."""
    if path.startswith("/api/po/calculate"):
        return True
    return path.startswith(
        (
            "/api/po/daily-inventory-history/upload-status",
        )
    )


def _session_po_calculate_light(path: str, method: str) -> bool:
    """PO calc poll/start/result — avoid PG restore + warm-cache copy (stay fast during long jobs)."""
    if method == "POST" and path == "/api/po/calculate":
        return True
    if method == "GET" and path in (
        "/api/po/calculate/status",
        "/api/po/calculate/result",
    ):
        return True
    return False


def _session_uses_stitching_db_only(path: str) -> bool:
    """SQLite-only modules — do not block on forecast session blobs."""
    return path.startswith("/api/stitching/") or path.startswith("/api/hrm/")

# ── Auth middleware (outermost — runs first) ──────────────────
_AUTH_EXEMPT = {
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/otp/resend",
    "/api/auth/otp/verify",
    "/api/health",
}

# Skip heavy session restore / warm-cache copy (login was blocked for minutes behind PG blobs).
_SESSION_LIGHTWEIGHT = frozenset({
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/otp/resend",
    "/api/auth/otp/verify",
    "/api/auth/me",
})

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path in _AUTH_EXEMPT or not request.url.path.startswith("/api/"):
        return await call_next(request)
    token = request.cookies.get("auth_token")
    payload = decode_token(token) if token else None
    if not payload or not payload.get("sub"):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

    request.state.auth = payload
    role = payload.get("role", "Admin")
    path = request.url.path

    if role == KARIGAR_ROLE and not karigar_may_access_api(path, request.method):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"detail": "Access denied for karigar role"})

    if path.startswith("/api/erp-admin") and not may_access_erp_admin(role):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=403, content={"detail": "Admin access required"})

    _po_upload_policy = (
        path.startswith("/api/po/returns/")
        or path.startswith("/api/po/daily-inventory-history")
        or path.startswith("/api/po/sku-status-lead")
    )
    if path.startswith("/api/upload/") or path.startswith("/api/cache/") or (
        path.startswith("/api/data/daily-uploads/") and request.method.upper() == "DELETE"
    ) or (_po_upload_policy and request.method.upper() in ("POST", "DELETE", "PUT")):
        from .services.upload_policy import check_upload_api_access

        blocked = check_upload_api_access(role, request.method, path)
        if blocked:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=403, content={"detail": blocked})

    return await call_next(request)


# ── Session cookie middleware ─────────────────────────────────
SESSION_COOKIE = "session_id"

@app.middleware("http")
async def session_middleware(request: Request, call_next):
    path = request.url.path

    # Auth + health must never wait on multi-GB PostgreSQL session blobs or warm-cache copies.
    if path in _SESSION_LIGHTWEIGHT:
        request.state.session_id = None
        request.state.session = None
        return await call_next(request)

    if _session_uses_stitching_db_only(path):
        sid = request.cookies.get(SESSION_COOKIE)
        request.state.session_id = sid
        request.state.session = None
        response: Response = await call_next(request)
        if sid:
            response.set_cookie(
                key=SESSION_COOKIE,
                value=sid,
                httponly=True,
                samesite="lax",
                secure=_cookie_secure_from_request(request),
                max_age=14 * 24 * 3600,
            )
        return response

    if _session_po_calculate_light(path, request.method):
        sid = request.cookies.get(SESSION_COOKIE)
        sess = store.get(sid) if sid else None
        if sess is None and request.method == "POST" and path == "/api/po/calculate":
            sid, sess = await run_aux(store.get_or_create, sid)
        request.state.session_id = sid
        request.state.session = sess
        response: Response = await call_next(request)
        if sid:
            response.set_cookie(
                key=SESSION_COOKIE,
                value=sid,
                httponly=True,
                samesite="lax",
                secure=_cookie_secure_from_request(request),
                max_age=14 * 24 * 3600,
            )
        return response

    sid = request.cookies.get(SESSION_COOKIE)
    # Run in executor so a slow PostgreSQL session restore doesn't block the event loop.
    sid, session = await run_aux(store.get_or_create, sid)
    request.state.session_id = sid
    request.state.session = session
    setattr(session, "_persist_sid", sid)

    copied_warm = False
    if not _session_skip_heavy_warm(path):
        try:
            copied_warm = await run_aux(
                _apply_warm_cache_if_needed, session, _warm_cache_generation,
            )
        except Exception:
            log.exception("warm-cache apply failed")

    # SKU bundle merge is expensive; skip on read-only data APIs when sales are already loaded.
    _skip_sku_bundle = (
        request.method == "GET"
        and path.startswith("/api/data/")
        and not getattr(session, "sales_df", None) is None
        and not getattr(session, "sales_df").empty
    )
    if not _skip_sku_bundle:
        try:
            from .services.sku_mapping import ensure_default_sku_mapping_from_bundle

            await run_aux(ensure_default_sku_mapping_from_bundle, session)
        except Exception:
            pass

    response: Response = await call_next(request)

    try:
        from .db.forecast_session_pg import debounced_persist_session, pg_session_persist_enabled

        if pg_session_persist_enabled() and sid:
            if copied_warm:
                debounced_persist_session(sid, session, delay=15.0)
            elif request.method in frozenset({"POST", "PUT", "DELETE", "PATCH"}) and path.startswith(
                "/api/"
            ):
                debounced_persist_session(sid, session, delay=30.0)
    except Exception:
        pass

    # Set / refresh cookie on every response
    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure_from_request(request),
        max_age=14 * 24 * 3600,  # 14 days — must stay stable while PostgreSQL stores session blobs by this id
    )
    return response


# ── Routers ───────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/api/auth",       tags=["auth"])
app.include_router(upload.router,      prefix="/api/upload",     tags=["upload"])
app.include_router(data.router,        prefix="/api/data",       tags=["data"])
app.include_router(cache.router,       prefix="/api/cache",      tags=["cache"])
app.include_router(po.router,          prefix="/api/po",         tags=["po"])
app.include_router(shipment.router,    prefix="/api/shipment",   tags=["shipment"])
app.include_router(finance_router,     prefix="/api/finance",    tags=["finance"])
app.include_router(item_router,        prefix="/api/items",      tags=["items"])
app.include_router(sales_router,       prefix="/api/sales",      tags=["sales"])
app.include_router(purchase_router,    prefix="/api/purchase",   tags=["purchase"])
app.include_router(tna_router,         prefix="/api/tna",        tags=["tna"])
app.include_router(production_router,  prefix="/api/production", tags=["production"])
app.include_router(grey_router,        prefix="/api/grey",       tags=["grey"])
app.include_router(stitching_router,   prefix="/api/stitching",  tags=["stitching"])
app.include_router(hrm_router,         prefix="/api/hrm",        tags=["hrm"])
app.include_router(erp_admin_router,   prefix="/api/erp-admin",  tags=["erp-admin"])
app.include_router(marketplace_router, prefix="/api/marketplace", tags=["marketplace"])


@app.get("/api/health")
def health():
    from .app_version import get_build_info

    info = get_build_info()
    return {
        "status": "ok",
        "version": info["version"],
        "git_sha": info["git_sha"],
        "built_at": info["built_at"],
        "label": info["label"],
        "sessions": store.count,
        "warm_cache": bool(_warm_cache),
        "warm_cache_loaded_at": _warm_cache_loaded_at.isoformat() if _warm_cache_loaded_at else None,
        # generation: 0=loading, 1=Phase1 SQLite ready, 2=Phase2 GitHub+SQLite ready
        "warm_cache_generation": _warm_cache_generation,
    }
