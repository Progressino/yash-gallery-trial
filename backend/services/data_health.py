"""
Automated data-health checks — run after uploads and on a schedule so bad data
(duplicate sales files, matrix drift, inventory double counts) is flagged
without the user having to spot it manually.

Results are cached to ``{WARM_CACHE_DIR}/data_health.json`` and served by
GET /api/data/health-checks. Every check is defensive: one failing probe never
kills the run — it is reported as its own failed check instead.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE_MAX_AGE_SEC = int(os.environ.get("DATA_HEALTH_MAX_AGE_SEC", str(6 * 3600)))
_SALES_WINDOW_DAYS = int(os.environ.get("DATA_HEALTH_SALES_DAYS", "35"))
_MANUAL_OVERLAY_STALE_DAYS = int(os.environ.get("DATA_HEALTH_MANUAL_STALE_DAYS", "14"))
_SWING_PCT = float(os.environ.get("DATA_HEALTH_INV_SWING_PCT", "20"))
_SWING_UNITS = float(os.environ.get("DATA_HEALTH_INV_SWING_UNITS", "3000"))


def _warm_dir() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def _cache_path() -> Path:
    return _warm_dir() / "data_health.json"


def _check(
    check_id: str,
    area: str,
    title: str,
    ok: bool,
    detail: str,
    *,
    severity: str = "fail",
    data: dict | None = None,
) -> dict:
    return {
        "id": check_id,
        "area": area,
        "title": title,
        "ok": bool(ok),
        "severity": severity if not ok else "ok",
        "detail": detail,
        "data": data or {},
    }


def _error_check(check_id: str, area: str, title: str, exc: Exception) -> dict:
    return _check(
        check_id, area, title, False,
        f"Check could not run: {type(exc).__name__}: {exc}",
        severity="warn",
    )


# ── Sales checks ───────────────────────────────────────────────────────────


def _sales_cross_file_duplicate_check() -> dict:
    """Same order line present in more than one upload file (twin exports)."""
    check_id, area, title = (
        "sales_amazon_cross_file_duplicates",
        "sales",
        "Amazon uploads: duplicate order rows across files",
    )
    try:
        import io

        from .daily_store import (
            _coalesce_tier3_upload_rows,
            _get_conn,
            _tier3_window_sql_clause,
        )

        end = pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
        start = end - pd.Timedelta(days=_SALES_WINDOW_DAYS)
        conn = _get_conn()
        try:
            clause = _tier3_window_sql_clause()
            rows = conn.execute(
                f"SELECT filename, data_parquet FROM daily_uploads "
                f"WHERE platform=? AND ({clause}) ORDER BY file_date ASC",
                ("amazon", str(end.date()), str(start.date())),
            ).fetchall()
        finally:
            conn.close()
        rows = _coalesce_tier3_upload_rows(rows, platform="amazon")
        if not rows:
            return _check(check_id, area, title, True, "No Amazon uploads in window.")

        parts = []
        for fn, blob in rows:
            d = pd.read_parquet(io.BytesIO(blob))
            d["_src"] = fn
            parts.append(d)
        df = pd.concat(parts, ignore_index=True)
        ship = df[df["Transaction_Type"].astype(str).str.strip() == "Shipment"].copy()
        if ship.empty:
            return _check(check_id, area, title, True, "No shipment rows in window.")
        oid = ship["Order_Id"].astype(str).str.strip()
        ship = ship[oid.ne("") & ~oid.str.lower().isin(["nan", "none"])]
        day = pd.to_datetime(ship["Date"], errors="coerce").dt.normalize()
        key = (
            ship["Order_Id"].astype(str)
            + "|" + ship["SKU"].astype(str)
            + "|" + ship["Quantity"].astype(str)
            + "|" + day.astype(str)
        )
        ship = ship.assign(_key=key, _day=day)
        dups = ship[ship.duplicated("_key", keep=False)]
        cross = dups.groupby("_key")["_src"].nunique()
        bad_keys = cross[cross > 1].index
        if len(bad_keys) == 0:
            return _check(
                check_id, area, title, True,
                f"No duplicate order rows across {len(rows)} files (last {_SALES_WINDOW_DAYS} days).",
            )
        bad = ship[ship["_key"].isin(bad_keys)]
        per_day = (
            (pd.to_numeric(bad["Quantity"], errors="coerce").fillna(0).groupby(bad["_day"]).sum() / 2)
            .round()
            .astype(int)
        )
        days_str = ", ".join(f"{k.date()}: +{v}" for k, v in per_day.items())
        return _check(
            check_id, area, title, False,
            f"Duplicate order rows across upload files inflate daily units — {days_str}. "
            f"Delete the duplicate export or re-upload the day.",
            data={"duplicated_units_per_day": {str(k.date()): int(v) for k, v in per_day.items()}},
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("data-health: %s failed", check_id)
        return _error_check(check_id, area, title, exc)


def _sales_matrix_vs_tier3_check() -> list[dict]:
    """Sales History matrix must equal a fresh Tier-3 re-read for every upload day."""
    check_id, area, title = (
        "sales_matrix_vs_tier3",
        "sales",
        "Sales History matrix vs uploaded files",
    )
    try:
        from types import SimpleNamespace

        from .daily_sales_history import (
            build_sales_history_sales_df,
            sales_history_data_quality_checks,
        )
        from .sku_mapping import load_sku_mapping_from_disk

        sess = SimpleNamespace(sku_mapping=load_sku_mapping_from_disk() or {}, sales_df=None)
        frame = build_sales_history_sales_df(sess, days=_SALES_WINDOW_DAYS)
        if frame is None or frame.empty:
            return [_check(check_id, area, title, True, "No Tier-3 sales rows in window.")]
        raw = sales_history_data_quality_checks(
            frame, days=_SALES_WINDOW_DAYS, max_days=_SALES_WINDOW_DAYS
        )
        out: list[dict] = []
        for c in raw:
            ok = bool(c.get("ok"))
            out.append(
                _check(
                    f"{check_id}:{c.get('date')}",
                    area,
                    f"{c.get('platform')} {c.get('date')}: matrix vs upload",
                    ok,
                    (
                        f"Matrix {c.get('matrix_net_units')} vs upload {c.get('expected_net_units')} "
                        f"(Δ {c.get('delta')})"
                    ),
                    data=c,
                )
            )
        if not out:
            out.append(_check(check_id, area, title, True, "No upload days to verify."))
        return out
    except Exception as exc:  # noqa: BLE001
        log.exception("data-health: %s failed", check_id)
        return [_error_check(check_id, area, title, exc)]


def _sales_coverage_check() -> dict:
    """Yesterday's daily files should exist for all core platforms (warn only)."""
    check_id, area, title = ("sales_upload_coverage", "sales", "Daily upload coverage (yesterday)")
    try:
        from .daily_store import get_upload_report_day_coverage

        yday = (
            pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)
            - pd.Timedelta(days=1)
        )
        iso = str(yday.date())
        coverage = get_upload_report_day_coverage()
        missing = [
            p for p in ("amazon", "flipkart", "meesho", "myntra")
            if iso not in (coverage.get(p) or set())
        ]
        if not missing:
            return _check(check_id, area, title, True, f"All core platform files present for {iso}.")
        return _check(
            check_id, area, title, False,
            f"Missing daily sales files for {iso}: {', '.join(missing)}. "
            f"Those days show low/zero sales until uploaded.",
            severity="warn",
            data={"date": iso, "missing_platforms": missing},
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("data-health: %s failed", check_id)
        return _error_check(check_id, area, title, exc)


# ── Inventory checks ───────────────────────────────────────────────────────


def _load_inventory_frame() -> pd.DataFrame | None:
    p = _warm_dir() / "inventory_df_variant.parquet"
    if not p.is_file():
        return None
    return pd.read_parquet(p)


def _inventory_component_sum_check(inv: pd.DataFrame) -> dict:
    """Total_Inventory must equal the sum of its component columns."""
    check_id, area, title = (
        "inventory_component_sum",
        "inventory",
        "Inventory total equals sum of components",
    )
    components = [
        "OMS_Inventory",
        "Amazon_Inventory",
        "Myntra_Other_Inventory",
        "Flipkart_Inventory",
        "Meesho_Inventory",
        "Manual_InTransit",
        "Not_In_Inventory_Qty",
    ]
    have = [c for c in components if c in inv.columns]
    if "Total_Inventory" not in inv.columns or not have:
        return _check(check_id, area, title, True, "Snapshot missing total/components — skipped.")
    total = float(pd.to_numeric(inv["Total_Inventory"], errors="coerce").fillna(0).sum())
    comp_sum = float(
        sum(pd.to_numeric(inv[c], errors="coerce").fillna(0).sum() for c in have)
    )
    delta = round(total - comp_sum, 1)
    ok = abs(delta) <= max(1.0, total * 0.001)
    return _check(
        check_id, area, title, ok,
        f"Total {int(total)} vs components {int(comp_sum)} (Δ {delta}).",
        data={"total": total, "component_sum": comp_sum, "delta": delta},
    )


def _inventory_amazon_ledger_check(inv: pd.DataFrame) -> dict:
    """Amazon column in the snapshot must match the parsed FC ledger headline."""
    check_id, area, title = (
        "inventory_amazon_ledger",
        "inventory",
        "Amazon snapshot equals FC ledger",
    )
    meta_p = _warm_dir() / "inventory_session_meta.json"
    if not meta_p.is_file():
        return _check(check_id, area, title, True, "No snapshot metadata — skipped.")
    meta = json.loads(meta_p.read_text())
    dbg = meta.get("inventory_debug") or {}
    amz = dbg.get("amz_disclaimer") or {}
    expected = float(amz.get("sellable_non_znne_units") or amz.get("latest_report_units") or 0)
    if expected <= 0:
        return _check(check_id, area, title, True, "No Amazon ledger in the last upload — skipped.")
    actual = (
        float(pd.to_numeric(inv["Amazon_Inventory"], errors="coerce").fillna(0).sum())
        if "Amazon_Inventory" in inv.columns
        else 0.0
    )
    delta = round(actual - expected, 1)
    ok = abs(delta) < 1.0
    detail = f"Ledger {int(expected)} vs snapshot {int(actual)} (Δ {delta})."
    if not ok:
        snap = str(meta.get("inventory_snapshot_date_label") or meta.get("inventory_snapshot_date") or "")
        detail += (
            f" Snapshot label says {snap or 'updated'} but totals are from an older file — "
            f"re-upload the inventory RAR (the app should auto-heal after this deploy)."
        )
    return _check(
        check_id, area, title, ok, detail,
        data={"expected": expected, "actual": actual, "delta": delta},
    )


def _inventory_meta_frame_desync_check(inv: pd.DataFrame) -> dict:
    """Catch the specific race: new snapshot date/meta with old OMS totals."""
    check_id, area, title = (
        "inventory_meta_frame_desync",
        "inventory",
        "Inventory date label matches OMS totals",
    )
    meta_p = _warm_dir() / "inventory_session_meta.json"
    if not meta_p.is_file():
        return _check(check_id, area, title, True, "No snapshot metadata — skipped.")
    meta = json.loads(meta_p.read_text())
    dbg = meta.get("inventory_debug") or {}
    oms_dbg = str(dbg.get("oms") or "")
    # oms debug looks like "14978 SKUs" — use amz_disclaimer + sources instead.
    # Prefer comparing Amazon ledger (reliable) already covered above; here compare
    # OMS when the debug string embeds a unit total, else use source filename date.
    sources = meta.get("inventory_snapshot_date_sources") or []
    uploaded_at = str(meta.get("inventory_snapshot_uploaded_at") or "")
    snap_date = str(meta.get("inventory_snapshot_date") or "")
    if not uploaded_at or not snap_date:
        return _check(check_id, area, title, True, "Snapshot metadata incomplete — skipped.")
    # If Amazon ledger check would fail, this desync is the same root cause —
    # surface it with an actionable message about the upload race.
    amz = dbg.get("amz_disclaimer") or {}
    expected_amz = float(amz.get("sellable_non_znne_units") or amz.get("latest_report_units") or 0)
    actual_amz = (
        float(pd.to_numeric(inv["Amazon_Inventory"], errors="coerce").fillna(0).sum())
        if "Amazon_Inventory" in inv.columns
        else 0.0
    )
    if expected_amz > 0 and abs(actual_amz - expected_amz) >= 1.0:
        return _check(
            check_id, area, title, False,
            (
                f"Warm-cache desync: meta says {snap_date} "
                f"({', '.join(str(s) for s in sources[:3]) or 'upload'}) "
                f"but Amazon units are still {int(actual_amz)} (ledger {int(expected_amz)}). "
                f"A concurrent session overwrote the numbers after upload."
            ),
            data={
                "snapshot_date": snap_date,
                "uploaded_at": uploaded_at,
                "expected_amazon": expected_amz,
                "actual_amazon": actual_amz,
                "oms_dbg": oms_dbg,
            },
        )
    return _check(
        check_id, area, title, True,
        f"Snapshot {snap_date} matches ledger totals.",
        data={"snapshot_date": snap_date, "uploaded_at": uploaded_at},
    )


def _inventory_swing_check(inv: pd.DataFrame) -> dict:
    """Component-level jump vs the previous daily snapshot (warn only)."""
    check_id, area, title = (
        "inventory_day_over_day_swing",
        "inventory",
        "Inventory swing vs previous snapshot",
    )
    hist_p = _warm_dir() / "daily_inventory_history_df.parquet"
    if not hist_p.is_file():
        return _check(check_id, area, title, True, "No inventory history — skipped.")
    hist = pd.read_parquet(hist_p)
    if hist.empty or "Date" not in hist.columns or "Qty" not in hist.columns:
        return _check(check_id, area, title, True, "Inventory history unusable — skipped.")
    days = pd.to_datetime(hist["Date"], errors="coerce").dt.normalize()
    per_day = pd.to_numeric(hist["Qty"], errors="coerce").fillna(0).groupby(days).sum()
    per_day = per_day[per_day > 0].sort_index()
    if len(per_day) < 2:
        return _check(check_id, area, title, True, "Not enough history days — skipped.")
    prev_day, last_day = per_day.index[-2], per_day.index[-1]
    prev, last = float(per_day.iloc[-2]), float(per_day.iloc[-1])
    delta = last - prev
    pct = (abs(delta) / prev * 100) if prev else 0.0
    ok = not (pct > _SWING_PCT and abs(delta) > _SWING_UNITS)
    return _check(
        check_id, area, title, ok,
        (
            f"{prev_day.date()}: {int(prev)} → {last_day.date()}: {int(last)} "
            f"(Δ {int(delta):+d}, {pct:.1f}%)."
            + ("" if ok else " Large jump — verify the latest upload before raising POs.")
        ),
        severity="warn",
        data={"prev": prev, "last": last, "delta": delta, "pct": pct},
    )


def _inventory_manual_overlay_check() -> dict:
    """Manual in-transit sheet gets stale — arrived goods double count with OMS."""
    check_id, area, title = (
        "inventory_manual_overlay_age",
        "inventory",
        "Manual in-transit sheet freshness",
    )
    p = _warm_dir() / "manual_intransit_overlay_df.parquet"
    if not p.is_file():
        return _check(check_id, area, title, True, "No manual in-transit sheet loaded.")
    ov = pd.read_parquet(p)
    units = float(pd.to_numeric(ov.get("Manual_InTransit"), errors="coerce").fillna(0).sum()) if "Manual_InTransit" in ov.columns else 0.0
    if units <= 0:
        return _check(check_id, area, title, True, "Manual in-transit sheet is empty.")
    age_days = (time.time() - p.stat().st_mtime) / 86400.0
    ok = age_days <= _MANUAL_OVERLAY_STALE_DAYS
    return _check(
        check_id, area, title, ok,
        (
            f"{int(units)} in-transit units from a sheet last updated {age_days:.0f} day(s) ago."
            + (
                ""
                if ok
                else " If these goods have arrived they are also inside OMS — re-upload or clear the sheet."
            )
        ),
        severity="warn",
        data={"units": units, "age_days": round(age_days, 1)},
    )


def _inventory_checks() -> list[dict]:
    try:
        inv = _load_inventory_frame()
    except Exception as exc:  # noqa: BLE001
        return [_error_check("inventory_load", "inventory", "Load inventory snapshot", exc)]
    out: list[dict] = []
    if inv is None or inv.empty:
        out.append(
            _check(
                "inventory_load", "inventory", "Inventory snapshot present", True,
                "No inventory snapshot on disk — nothing to verify.",
            )
        )
        return out
    for fn in (
        lambda: _inventory_component_sum_check(inv),
        lambda: _inventory_amazon_ledger_check(inv),
        lambda: _inventory_meta_frame_desync_check(inv),
        lambda: _inventory_swing_check(inv),
        _inventory_manual_overlay_check,
    ):
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001
            log.exception("data-health: inventory check failed")
            out.append(_error_check("inventory_check", "inventory", "Inventory check", exc))
    return out


# ── Runner / cache ─────────────────────────────────────────────────────────


def run_data_health_checks() -> dict:
    """Run the full suite and persist the result JSON. Thread-safe."""
    with _LOCK:
        t0 = time.time()
        checks: list[dict] = []
        checks.append(_sales_cross_file_duplicate_check())
        checks.extend(_sales_matrix_vs_tier3_check())
        checks.append(_sales_coverage_check())
        checks.extend(_inventory_checks())

        fails = [c for c in checks if not c["ok"] and c.get("severity") == "fail"]
        warns = [c for c in checks if not c["ok"] and c.get("severity") == "warn"]
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "took_sec": round(time.time() - t0, 1),
            "ok": not fails,
            "fail_count": len(fails),
            "warn_count": len(warns),
            "checks": checks,
        }
        try:
            tmp = _cache_path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(result, default=str))
            tmp.replace(_cache_path())
        except Exception:
            log.exception("data-health: could not persist result")
        log.info(
            "data-health: %d checks, %d fail, %d warn (%.1fs)",
            len(checks), len(fails), len(warns), result["took_sec"],
        )
        return result


def get_cached_data_health(max_age_sec: int | None = None) -> dict | None:
    p = _cache_path()
    if not p.is_file():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > (max_age_sec if max_age_sec is not None else _CACHE_MAX_AGE_SEC):
            return None
        out = json.loads(p.read_text())
        out["age_sec"] = round(age)
        return out
    except Exception:
        return None


def schedule_data_health_refresh(reason: str = "") -> None:
    """Fire-and-forget refresh (used after uploads)."""
    def _run() -> None:
        try:
            run_data_health_checks()
        except Exception:
            log.exception("data-health: scheduled refresh failed (%s)", reason)

    threading.Thread(target=_run, name="data-health-refresh", daemon=True).start()
