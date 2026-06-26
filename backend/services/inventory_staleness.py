"""Warnings when snapshot inventory or daily inventory history is behind calendar."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")
_DEFAULT_MAX_LAG_DAYS = 1


def today_ist_iso() -> str:
    return datetime.now(_IST).date().isoformat()


def data_lag_days(reference_iso: str, data_through_iso: str) -> int | None:
    ref = (reference_iso or "").strip()[:10]
    thru = (data_through_iso or "").strip()[:10]
    if not ref or not thru:
        return None
    try:
        d0 = datetime.strptime(ref, "%Y-%m-%d").date()
        d1 = datetime.strptime(thru, "%Y-%m-%d").date()
    except ValueError:
        return None
    return int((d0 - d1).days)


def data_is_stale(
    reference_iso: str,
    data_through_iso: str | None,
    *,
    max_expected_lag_days: int = _DEFAULT_MAX_LAG_DAYS,
) -> bool:
    if not data_through_iso:
        return True
    lag = data_lag_days(reference_iso, data_through_iso)
    if lag is None:
        return False
    return lag > max_expected_lag_days


def daily_inventory_history_bounds(df) -> tuple[str, str]:
    if df is None or getattr(df, "empty", True):
        return "", ""
    import pandas as pd

    dates = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    if dates.isna().all():
        return "", ""
    min_d = dates.min()
    max_d = dates.max()
    return (
        str(pd.Timestamp(min_d).date()) if pd.notna(min_d) else "",
        str(pd.Timestamp(max_d).date()) if pd.notna(max_d) else "",
    )


def build_inventory_staleness(
    *,
    reference_date: str | None = None,
    inventory_loaded: bool = False,
    inventory_snapshot_date: str | None = None,
    daily_inventory_history_loaded: bool = False,
    daily_inventory_history_max_date: str | None = None,
    max_expected_lag_days: int = _DEFAULT_MAX_LAG_DAYS,
) -> dict:
    """Return lag days + user-facing warnings for coverage / PO banners."""
    ref = (reference_date or today_ist_iso()).strip()[:10]
    warnings: list[str] = []

    hist_lag = None
    hist_stale = False
    if daily_inventory_history_loaded:
        hist_lag = data_lag_days(ref, daily_inventory_history_max_date or "")
        hist_stale = data_is_stale(
            ref,
            daily_inventory_history_max_date,
            max_expected_lag_days=max_expected_lag_days,
        )
        if hist_stale:
            label = daily_inventory_history_max_date or "unknown"
            warnings.append(
                f"Daily inventory history matrix ends {label} "
                f"({hist_lag or '?'} day(s) behind today). "
                "Re-upload the wide inventory history Excel (Upload → History & setup) "
                "or update the matrix — Eff_Days in PO may be wrong without fresh history."
            )

    snap_lag = None
    snap_stale = False
    if inventory_loaded:
        snap_lag = data_lag_days(ref, inventory_snapshot_date or "")
        snap_stale = data_is_stale(
            ref,
            inventory_snapshot_date,
            max_expected_lag_days=max_expected_lag_days,
        )
        # When the wide history matrix is current, PO stock/Eff_Days do not depend on
        # a separate daily snapshot date — avoid a misleading "unknown" banner.
        if (
            snap_stale
            and not inventory_snapshot_date
            and daily_inventory_history_loaded
            and daily_inventory_history_max_date
            and not hist_stale
        ):
            snap_stale = False
            snap_lag = hist_lag
        if snap_stale:
            label = inventory_snapshot_date or "unknown"
            warnings.append(
                f"Daily snapshot inventory is from {label} "
                f"({snap_lag or '?'} day(s) behind today). "
                "Upload today's OMS + marketplace inventory on Upload → Daily uploads "
                "so PO stock and cover days stay accurate."
            )
    if inventory_loaded and not daily_inventory_history_loaded:
        warnings.append(
            "No daily inventory history matrix loaded. Upload the wide "
            "'Daily Inventory History' Excel (same format as your OMS/Amazon sheets) "
            "under Upload → History & setup so PO Eff_Days uses real in-stock days."
        )

    return {
        "reference_date": ref,
        "inventory_snapshot_lag_days": snap_lag,
        "inventory_snapshot_stale": snap_stale,
        "daily_inventory_history_lag_days": hist_lag,
        "daily_inventory_history_stale": hist_stale,
        "inventory_staleness_warnings": warnings,
    }
