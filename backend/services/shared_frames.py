"""Process-wide warm-cache frame access — avoid per-session copies of huge DataFrames."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

_log = __import__("logging").getLogger(__name__)

# Frames that must not be duplicated per browser session (~300MB+ each at scale).
LARGE_FRAME_KEYS = frozenset(
    {
        "sales_df",
        "mtr_df",
        "myntra_df",
        "meesho_df",
        "flipkart_df",
        "snapdeal_df",
        "inventory_df_variant",
        "inventory_df_parent",
        "daily_inventory_history_df",
    }
)


def shared_frames_enabled() -> bool:
    raw = (os.environ.get("SESSION_SHARED_FRAMES") or "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    try:
        from ..db.forecast_ops_tables import normalized_tables_enabled

        if normalized_tables_enabled():
            return True
    except Exception:
        pass
    try:
        import backend.main as _main

        return _main.warm_cache_po_session_only()
    except Exception:
        return False


def _warm_cache() -> dict:
    try:
        import backend.main as _main

        return _main._warm_cache or {}
    except Exception:
        return {}


def session_uses_shared_frames(sess) -> bool:
    return bool(getattr(sess, "_shared_frames", False)) and shared_frames_enabled()


def attach_shared_frames(sess, *, warm_cache_generation: int) -> None:
    """Point session at process warm cache — do not copy large frames."""
    sess._shared_frames = True
    sess._warm_cache_gen = int(warm_cache_generation or 0)
    sess._warm_cache_only = True
    wc = _warm_cache()
    if isinstance(wc.get("sku_mapping"), dict) and wc["sku_mapping"]:
        sess.sku_mapping = wc["sku_mapping"]

    # Never clobber a newer in-session inventory snapshot with a stale warm frame.
    # Race: upload sets sess frame+meta, then a concurrent attach overwrites only the
    # frame from warm while leaving the new uploaded_at — sync then republishes the
    # old numbers under the new meta (Amazon ledger mismatch / "concurrent session").
    skip_inventory_frames = False
    try:
        from .inventory import inventory_snapshot_upload_epoch

        meta_key = "inventory_session_meta"
        try:
            import backend.main as _main

            meta_key = getattr(_main, "_INVENTORY_META_WARM_KEY", meta_key)
        except Exception:
            pass
        warm_meta = wc.get(meta_key) if isinstance(wc.get(meta_key), dict) else {}
        warm_at = inventory_snapshot_upload_epoch(
            str((warm_meta or {}).get("inventory_snapshot_uploaded_at") or "")
        )
        sess_at = inventory_snapshot_upload_epoch(
            getattr(sess, "inventory_snapshot_uploaded_at", "") or ""
        )
        sess_inv = getattr(sess, "inventory_df_variant", None)
        sess_has = (
            sess_inv is not None
            and hasattr(sess_inv, "empty")
            and not sess_inv.empty
        )
        if sess_has and sess_at and (not warm_at or sess_at > warm_at + 1e-6):
            skip_inventory_frames = True
        elif getattr(sess, "inventory_upload_status", "idle") == "running":
            skip_inventory_frames = True
    except Exception:
        skip_inventory_frames = False

    _INV_KEYS = frozenset({"inventory_df_variant", "inventory_df_parent"})
    for key in LARGE_FRAME_KEYS:
        if skip_inventory_frames and key in _INV_KEYS:
            continue
        val = wc.get(key)
        if val is not None and hasattr(val, "empty") and not val.empty:
            setattr(sess, key, val)
    sess._quarterly_cache.clear()


def warm_frame(key: str, sess=None) -> pd.DataFrame:
    """Unified sales / platform / inventory accessor."""
    wc = _warm_cache()
    wc_df = wc.get(key)
    if sess is not None:
        df = getattr(sess, key, None)
        if df is not None and hasattr(df, "empty") and not df.empty:
            if not session_uses_shared_frames(sess):
                return df
            # Session holds a distinct copy (upload / test seed) — prefer it over warm cache.
            if wc_df is None or df is not wc_df:
                return df
    if wc_df is not None and hasattr(wc_df, "empty"):
        return wc_df
    if sess is not None:
        df = getattr(sess, key, None)
        if df is not None and hasattr(df, "empty"):
            return df
    return pd.DataFrame()


def session_sales_df(sess) -> pd.DataFrame:
    return warm_frame("sales_df", sess)


def session_inventory_variant(sess) -> pd.DataFrame:
    return warm_frame("inventory_df_variant", sess)


def session_inventory_parent(sess) -> pd.DataFrame:
    return warm_frame("inventory_df_parent", sess)


def session_platform_df(sess, platform_key: str) -> pd.DataFrame:
    return warm_frame(platform_key, sess)


def platform_frame_for_window(
    attr: str,
    sess=None,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Platform frame for a calendar window — reads full-history disk parquet when RAM is trimmed."""
    from pathlib import Path

    s = str(start_date or "")[:10]
    e = str(end_date or "")[:10]
    has_window = len(s) == 10 and len(e) == 10

    mem = warm_frame(attr, sess)
    if mem is not None and hasattr(mem, "empty") and not mem.empty:
        if not has_window:
            return mem
        date_col = (
            "Date"
            if "Date" in mem.columns
            else ("TxnDate" if "TxnDate" in mem.columns else None)
        )
        if date_col:
            d = pd.to_datetime(mem[date_col], errors="coerce")
            in_w = mem[(d >= pd.Timestamp(s)) & (d <= pd.Timestamp(e))]
            # If RAM only holds a recent slice (or other incomplete window), fall through
            # to full-history disk parquet instead of returning an empty frame.
            if not in_w.empty:
                # Stale session/RAM Meesho can still hold pre-OMS-fill blanks; prefer disk
                # when the window is mostly unusable for SKU deepdive.
                if attr == "meesho_df" and _meesho_window_mostly_blank(in_w):
                    pass  # fall through to disk
                else:
                    return in_w
        else:
            return mem

    disk_dir = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    path = disk_dir / f"{attr}.parquet"
    if not path.is_file():
        return mem if mem is not None else pd.DataFrame()
    try:
        if has_window:
            try:
                df = pd.read_parquet(
                    path,
                    filters=[
                        ("Date", ">=", pd.Timestamp(s)),
                        ("Date", "<=", pd.Timestamp(e) + pd.Timedelta(days=1)),
                    ],
                )
                if df is not None and not getattr(df, "empty", True):
                    return df
            except Exception:
                pass
            # Fallback: read full parquet then slice (filters unsupported / schema drift).
            full = pd.read_parquet(path)
            if full is None or getattr(full, "empty", True):
                return full if full is not None else pd.DataFrame()
            date_col = (
                "Date"
                if "Date" in full.columns
                else ("TxnDate" if "TxnDate" in full.columns else None)
            )
            if not date_col:
                return full
            d = pd.to_datetime(full[date_col], errors="coerce")
            return full[(d >= pd.Timestamp(s)) & (d <= pd.Timestamp(e))]
        return pd.read_parquet(path)
    except Exception:
        return mem if mem is not None else pd.DataFrame()


def _meesho_window_mostly_blank(df: pd.DataFrame) -> bool:
    """True when ≥25% of rows lack usable SKU/OMS (stale TCS cache / pre-fill RAM)."""
    if df is None or getattr(df, "empty", True):
        return True
    bad = {"", "NAN", "NONE", "NAT", "MEESHO_TOTAL"}
    sku = (
        df["SKU"].astype(str).str.strip().str.upper()
        if "SKU" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    oms = (
        df["OMS_SKU"].astype(str).str.strip().str.upper()
        if "OMS_SKU" in df.columns
        else pd.Series("", index=df.index, dtype=str)
    )
    blank = sku.isin(bad) & oms.isin(bad)
    n = int(len(df))
    return n > 0 and int(blank.sum()) / n >= 0.25


def resolve_meesho_frame(
    preferred: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """
    Best Meesho frame for PO / quarterly / Deepdive parity.

    Prefer ``preferred`` (session/warm) unless it is empty or mostly blank OMS —
    then fall back to filled ``meesho_df.parquet`` on disk (post OMS-fill cache).
    """
    mem = preferred
    if mem is None:
        mem = warm_frame("meesho_df")
    disk_dir = Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))
    path = disk_dir / "meesho_df.parquet"

    def _read_disk() -> pd.DataFrame:
        if not path.is_file():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    if mem is None or getattr(mem, "empty", True):
        disk = _read_disk()
        return disk if disk is not None else pd.DataFrame()

    if not _meesho_window_mostly_blank(mem):
        return mem

    disk = _read_disk()
    if disk is None or getattr(disk, "empty", True):
        return mem
    # Disk wins when it is materially less blank (or mem was ≥25% blank).
    if not _meesho_window_mostly_blank(disk):
        return disk
    try:
        bad = {"", "NAN", "NONE", "NAT", "MEESHO_TOTAL"}

        def _blank_n(df: pd.DataFrame) -> int:
            sku = (
                df["SKU"].astype(str).str.strip().str.upper()
                if "SKU" in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            oms = (
                df["OMS_SKU"].astype(str).str.strip().str.upper()
                if "OMS_SKU" in df.columns
                else pd.Series("", index=df.index, dtype=str)
            )
            return int((sku.isin(bad) & oms.isin(bad)).sum())

        if _blank_n(disk) + 500 < _blank_n(mem):
            return disk
    except Exception:
        pass
    return mem


def frame_row_count(key: str, sess) -> int:
    df = warm_frame(key, sess)
    try:
        return int(len(df)) if df is not None and hasattr(df, "__len__") else 0
    except Exception:
        return 0


def should_skip_session_copy(key: str) -> bool:
    return shared_frames_enabled() and key in LARGE_FRAME_KEYS


def assign_frame_no_copy(sess, key: str, val: Any) -> None:
    """Set session attribute without copying when shared-frame mode is on."""
    if val is None:
        return
    if hasattr(val, "empty") and val.empty:
        return
    if should_skip_session_copy(key):
        setattr(sess, key, val)
        return
    if hasattr(val, "copy"):
        setattr(sess, key, val.copy())
    else:
        setattr(sess, key, val)
