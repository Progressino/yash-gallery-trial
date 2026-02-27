"""
Server-side session store — mirrors st.session_state exactly.
Each browser gets a UUID cookie; the UUID maps to an AppSession.
"""
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


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
    inventory_df_variant: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory_df_parent: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_orders_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    existing_po_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    transfer_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ── Settings ─────────────────────────────────────────────
    amazon_date_basis: str = "Shipment Date"
    include_replacements: bool = False

    # ── Metadata ─────────────────────────────────────────────
    daily_sales_sources: list = field(default_factory=list)
    daily_sales_rows: int = 0
    load_warnings: list = field(default_factory=list)


class SessionStore:
    """Thread-safe in-memory session store."""

    _TTL_SECONDS = 4 * 3600  # 4 hours idle timeout

    def __init__(self):
        self._sessions: dict[str, AppSession] = {}

    def get_or_create(self, sid: Optional[str]) -> tuple[str, AppSession]:
        if sid and sid in self._sessions:
            sess = self._sessions[sid]
            sess.last_accessed = time.time()
            return sid, sess
        new_id = str(uuid.uuid4())
        self._sessions[new_id] = AppSession()
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
