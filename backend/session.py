"""
Server-side session store — mirrors st.session_state exactly.
Each browser gets a UUID cookie; the UUID maps to an AppSession.
"""
import logging
import os
import threading
import uuid
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

_log = logging.getLogger(__name__)

# Cap blocking PG restore so /api/auth/login is not queued behind a multi‑GB session blob.
_PG_RESTORE_TIMEOUT_SEC = float(os.environ.get("SESSION_PG_RESTORE_TIMEOUT_SEC", "3"))


@dataclass
class AppSession:
    created: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # ── Data mirrors ─────────────────────────────────────────
    sku_mapping: dict = field(default_factory=dict)
    sales_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    mtr_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    myntra_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    meesho_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    flipkart_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapdeal_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_df_variant: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_df_parent: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_orders_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    existing_po_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    po_raise_ledger_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    po_return_overlay_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sku_status_lead_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_inventory_history_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    transfer_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    cogs_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Settings ─────────────────────────────────────────────
    amazon_date_basis: str = "Shipment Date"
    include_replacements: bool = False

    # ── Metadata ─────────────────────────────────────────────
    daily_sales_sources: list = field(default_factory=list)
    daily_sales_rows: int = 0
    load_warnings: list = field(default_factory=list)

    # ── Parse diagnostics ────────────────────────────────────
    snapdeal_parse_info: dict = field(default_factory=dict)  # raw cols + detected fields per file
    inventory_debug: dict = field(default_factory=dict)      # debug info from last inventory upload

    # ── Daily-store restore flag ──────────────────────────────
    daily_restored: bool = False   # True once daily SQLite data has been loaded into session

    # Async sales rebuild after Tier-3 daily-auto (avoids 502 on long build_sales_df).
    sales_rebuild_status: str = "idle"   # idle | running | done | error
    sales_rebuild_message: str = ""

    # Async Tier-3 ingest (RAR extract + daily SQLite) — avoids 502 while proxy waits.
    daily_auto_ingest_status: str = "idle"   # idle | running | done | error
    daily_auto_ingest_message: str = ""

    # Async PO calculate (large catalogs exceed reverse-proxy timeouts).
    po_calculate_status: str = "idle"   # idle | running | done | error
    po_calculate_message: str = ""
    po_calculate_result: dict = field(default_factory=dict)
    po_calculate_result_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Async daily inventory history upload (wide Excel can take minutes).
    daily_inventory_upload_status: str = "idle"  # idle | running | done | error
    daily_inventory_upload_message: str = ""
    daily_inventory_upload_result: dict = field(default_factory=dict)

    # After "Clear all app data", block warm-cache copy, Tier-3 SQLite restore, and
    # frontend auto Load-Cache until the user uploads again or clicks Load Cache.
    pause_auto_data_restore: bool = False

    # ── PO Engine cache ───────────────────────────────────────
    # Keyed by (group_by_parent, n_quarters) — cleared when sales data changes
    _quarterly_cache: dict = field(default_factory=dict)

    # Serialize Tier-3 SQLite restore + build_sales_df + Tier-1 upload mutations so parallel
    # requests on the same browser session do not corrupt DataFrames (see ``upload._session_lock_apply``).
    _daily_restore_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False,
    )


def wipe_app_session(sess: AppSession) -> None:
    """Clear every loaded dataset in this browser session (fresh upload from scratch)."""
    sess.sku_mapping = {}
    sess.sales_df = pd.DataFrame()
    sess.mtr_df = pd.DataFrame()
    sess.myntra_df = pd.DataFrame()
    sess.meesho_df = pd.DataFrame()
    sess.flipkart_df = pd.DataFrame()
    sess.snapdeal_df = pd.DataFrame()
    sess.inventory_df_variant = pd.DataFrame()
    sess.inventory_df_parent = pd.DataFrame()
    sess.daily_orders_df = pd.DataFrame()
    sess.existing_po_df = pd.DataFrame()
    sess.po_raise_ledger_df = pd.DataFrame()
    sess.po_return_overlay_df = pd.DataFrame()
    sess.sku_status_lead_df = pd.DataFrame()
    sess.daily_inventory_history_df = pd.DataFrame()
    sess.transfer_df = pd.DataFrame()
    sess.cogs_df = pd.DataFrame()
    sess.amazon_date_basis = "Shipment Date"
    sess.include_replacements = False
    sess.daily_sales_sources = []
    sess.daily_sales_rows = 0
    sess.load_warnings = []
    sess.snapdeal_parse_info = {}
    sess.inventory_debug = {}
    sess.daily_restored = False
    sess.sales_rebuild_status = "idle"
    sess.sales_rebuild_message = ""
    sess.daily_auto_ingest_status = "idle"
    sess.daily_auto_ingest_message = ""
    sess.po_calculate_status = "idle"
    sess.po_calculate_message = ""
    sess.po_calculate_result = {}
    sess.po_calculate_result_df = pd.DataFrame()
    sess.daily_inventory_upload_status = "idle"
    sess.daily_inventory_upload_message = ""
    sess.daily_inventory_upload_result = {}
    sess.pause_auto_data_restore = True
    sess._quarterly_cache.clear()


def resume_auto_data_restore(sess: AppSession) -> None:
    """Re-enable automatic restores / merges after an explicit load or new upload."""
    sess.pause_auto_data_restore = False
    sess.daily_restored = False
    # Mark that the session now has user/explicit data — prevents Phase-2 warm-cache
    # auto-re-copy from overwriting data the user just uploaded.
    sess._warm_cache_only = False


def _load_session_from_pg_timed(session_id: str, timeout_sec: float):
    """Load session from PostgreSQL with a hard timeout (avoids blocking login for minutes)."""
    from .db.forecast_session_pg import load_session_from_pg

    if timeout_sec <= 0:
        return load_session_from_pg(session_id)
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(load_session_from_pg, session_id)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            _log.warning(
                "PostgreSQL session restore timed out after %.1fs (session %s…) — using empty session; background restore scheduled",
                timeout_sec,
                session_id[:8],
            )
            return None


def _schedule_pg_session_restore(session_id: str) -> None:
    """Finish a large PG restore without blocking the request thread."""

    def _run() -> None:
        try:
            from .db.forecast_session_pg import load_session_from_pg, pg_session_persist_enabled

            if not pg_session_persist_enabled():
                return
            loaded = load_session_from_pg(session_id)
            if loaded is None:
                return
            with store._pg_restore_lock:
                cur = store._sessions.get(session_id)
                if cur is None:
                    store._sessions[session_id] = loaded
                    loaded.last_accessed = time.time()
                    return
                # Do not clobber active user work — only replace empty / warm-cache-only shells.
                if getattr(cur, "pause_auto_data_restore", False):
                    return
                if not getattr(cur, "_warm_cache_only", True) and not (
                    cur.mtr_df.empty and cur.sales_df.empty
                ):
                    return
                loaded.last_accessed = time.time()
                store._sessions[session_id] = loaded
                _log.info("Background PostgreSQL session restore completed (%s…)", session_id[:8])
        except Exception:
            _log.exception("Background PostgreSQL session restore failed")

    threading.Thread(target=_run, name=f"pg-restore-{session_id[:8]}", daemon=True).start()


class SessionStore:
    """Thread-safe in-memory session store."""

    _TTL_SECONDS = 12 * 3600  # 12 hours idle timeout

    def __init__(self):
        self._sessions: dict[str, AppSession] = {}
        self._pg_restore_lock = threading.Lock()

    def get_or_create(self, sid: Optional[str]) -> tuple[str, AppSession]:
        if sid and sid in self._sessions:
            sess = self._sessions[sid]
            sess.last_accessed = time.time()
            return sid, sess

        # Cookie present but not in RAM (process restart, second worker, or cold start):
        # restore from PostgreSQL so ~1M-row uploads are not "lost".
        if sid:
            try:
                from .db.forecast_session_pg import pg_session_persist_enabled

                if pg_session_persist_enabled():
                    loaded = _load_session_from_pg_timed(sid, _PG_RESTORE_TIMEOUT_SEC)
                    if loaded is not None:
                        self._sessions[sid] = loaded
                        loaded.last_accessed = time.time()
                        return sid, loaded
                    _schedule_pg_session_restore(sid)
            except Exception:
                _log.exception("PostgreSQL session restore failed")
            # Reuse the same session_id for an empty session so the next persist
            # uses the PK the browser already has (legacy behaviour rotated UUIDs).
            empty = AppSession()
            empty.last_accessed = time.time()
            self._sessions[sid] = empty
            return sid, empty

        new_id = str(uuid.uuid4())
        self._sessions[new_id] = AppSession()
        self._sessions[new_id].last_accessed = time.time()
        return new_id, self._sessions[new_id]

    def get(self, sid: str) -> Optional[AppSession]:
        sess = self._sessions.get(sid)
        if sess:
            sess.last_accessed = time.time()
        return sess

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)

    def evict_stale(self) -> int:
        cutoff = time.time() - self._TTL_SECONDS
        stale = [k for k, v in self._sessions.items() if v.last_accessed < cutoff]
        for k in stale:
            del self._sessions[k]
        return len(stale)

    @property
    def count(self) -> int:
        return len(self._sessions)


# Singleton — imported by all routers
store = SessionStore()
