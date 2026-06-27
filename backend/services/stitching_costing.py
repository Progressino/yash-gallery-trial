"""Stitching Costing business logic (ported from Streamlit v4.3)."""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from ..db.stitching_db import DATA_KEYS, DEFAULT_SHEETS, HOUR_COLS, get_all_sheets, get_sheet_df, save_sheet_df

IST = timezone(timedelta(hours=5, minutes=30))
HOUR_LBLS = [
    "9-10", "10-11", "11-12", "12-13", "13-14",
    "14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21",
]

_PRODUCTION_LOG_LOCK = threading.RLock()

# Factory SOP — benchmark karigar ₹480/day; LTL tolerance % by daily-rate band.
BENCHMARK_DAILY_RATE_RS = 480.0
# Default bands if sheet empty: (min inclusive, max exclusive, tolerance %)
_DEFAULT_LTL_TOLERANCE_BANDS: list[tuple[float, float, float]] = [
    (200.0, 300.0, 35.0),
    (300.0, 400.0, 12.0),
]


def _default_ltl_bands_rows() -> list[dict[str, float]]:
    return [
        {"Min_Rs": lo, "Max_Rs": hi, "Tolerance_Pct": pct}
        for lo, hi, pct in _DEFAULT_LTL_TOLERANCE_BANDS
    ]


def load_ltl_tolerance_bands() -> list[tuple[float, float, float]]:
    """Salary-range LTL tolerance bands from sheet (min inclusive, max exclusive, %)."""
    df = get_sheet_df("ltl_tolerance_bands")
    bands: list[tuple[float, float, float]] = []
    if not df.empty:
        for _, row in df.iterrows():
            try:
                lo = float(row.get("Min_Rs", 0) or 0)
                hi = float(row.get("Max_Rs", 0) or 0)
                pct = float(row.get("Tolerance_Pct", 0) or 0)
            except (TypeError, ValueError):
                continue
            if hi > lo and pct >= 0:
                bands.append((lo, hi, pct))
    if not bands:
        return list(_DEFAULT_LTL_TOLERANCE_BANDS)
    return sorted(bands, key=lambda b: b[0])


def save_ltl_tolerance_bands(bands: list[dict[str, Any]]) -> dict:
    rows: list[dict[str, float]] = []
    for b in bands:
        try:
            lo = float(b.get("Min_Rs", b.get("from_rs", 0)) or 0)
            hi = float(b.get("Max_Rs", b.get("to_rs", 0)) or 0)
            pct = float(b.get("Tolerance_Pct", b.get("tolerance_pct", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if hi <= lo or pct < 0:
            continue
        rows.append({"Min_Rs": lo, "Max_Rs": hi, "Tolerance_Pct": pct})
    if not rows:
        return {"ok": False, "message": "At least one valid band (Min < Max, tolerance %) is required"}
    save_sheet_df("ltl_tolerance_bands", pd.DataFrame(rows))
    return {"ok": True, "message": f"Saved {len(rows)} tolerance band(s)", "bands": rows}


def ltl_tolerance_bands_for_api() -> list[dict[str, float]]:
    return [
        {"from_rs": lo, "to_rs": hi, "tolerance_pct": pct}
        for lo, hi, pct in load_ltl_tolerance_bands()
    ]


def ltl_tolerance_factor(daily_rate: float) -> float:
    """Return multiplier applied to base target (1 − tolerance %)."""
    dr = float(daily_rate or 0)
    bands = load_ltl_tolerance_bands()
    for lo, hi, pct in bands:
        if lo <= dr < hi:
            return 1.0 - (pct / 100.0)
    if dr >= bands[-1][0]:
        return 1.0 - (bands[-1][2] / 100.0)
    if bands and dr < bands[0][0]:
        return 1.0 - (bands[0][2] / 100.0)
    return 1.0 - (bands[0][2] / 100.0) if bands else 0.65


def ltl_tolerance_pct_for_rate(daily_rate: float) -> float:
    dr = float(daily_rate or 0)
    bands = load_ltl_tolerance_bands()
    for lo, hi, pct in bands:
        if lo <= dr < hi:
            return pct
    if dr >= bands[-1][0]:
        return bands[-1][2]
    return bands[0][2] if bands else 35.0


def normalize_operation_name(name: Any) -> str:
    return " ".join(str(name or "").split())


def _style_session_key(style: Any) -> str:
    """Case-insensitive style key for upsert / load / reports."""
    return str(style or "").strip().lower()


def _resolve_canonical_style(style: str, sm: pd.DataFrame | None = None) -> str:
    """Persist the spelling from style_master when available."""
    raw = str(style or "").strip()
    if not raw:
        return ""
    key = _style_session_key(raw)
    sheet = sm if sm is not None else get_sheet_df("style_master")
    if not sheet.empty and "Style" in sheet.columns:
        for candidate in sheet["Style"].astype(str):
            cand = str(candidate).strip()
            if cand and _style_session_key(cand) == key:
                return cand
    return raw


def _production_session_keys(date_str: str, karigar_id: str, challan_no: str, style: str) -> tuple[str, str, str, str]:
    return (
        clean_key(date_str),
        clean_key(karigar_id),
        clean_key(challan_no),
        _style_session_key(style),
    )


def _drop_production_session_rows(
    log_df: pd.DataFrame,
    date_str: str,
    karigar_id: str,
    challan_no: str,
    style: str,
) -> pd.DataFrame:
    """Remove every operation row for this karigar + challan + style + date (upsert before re-save)."""
    if log_df.empty:
        return log_df
    d, k, c, s = _production_session_keys(date_str, karigar_id, challan_no, style)
    work = log_df.copy()
    work["_ck_date"] = work["Date"].apply(clean_key)
    work["_ck_kar"] = work["Karigar_ID"].apply(clean_key)
    work["_ck_challan"] = work["Challan_No"].apply(clean_key)
    work["_ck_style"] = work["Style"].apply(_style_session_key)
    keep = ~(
        (work["_ck_date"] == d)
        & (work["_ck_kar"] == k)
        & (work["_ck_challan"] == c)
        & (work["_ck_style"] == s)
    )
    return work[keep].drop(
        columns=["_ck_date", "_ck_kar", "_ck_challan", "_ck_style"],
        errors="ignore",
    )


KARIGAR_EXPENSE_WORK_TYPES = (
    "Part Change",
    "Alter",
    "Trainee",
    "Other Task",
    "Helper",
    "Other",
)


def _mask_production_log(
    log_df: pd.DataFrame,
    *,
    date_str: str,
    karigar_id: str,
    challan_no: str | None = None,
    style: str | None = None,
    operation: str | None = None,
    require_karigar: bool = True,
) -> pd.Series:
    """Boolean mask for production_log rows to update or delete."""
    if log_df.empty:
        return pd.Series(dtype=bool)
    work = log_df.copy()
    d, k, c, s = _production_session_keys(
        date_str,
        karigar_id,
        challan_no or "",
        style or "",
    )
    work["_ck_date"] = work["Date"].apply(clean_key)
    work["_ck_kar"] = work["Karigar_ID"].apply(clean_key)
    mask = work["_ck_date"] == d
    if require_karigar and k:
        mask = mask & (work["_ck_kar"] == k)
    if challan_no:
        work["_ck_challan"] = work["Challan_No"].apply(clean_key)
        mask = mask & (work["_ck_challan"] == c)
    if style:
        work["_ck_style"] = work["Style"].apply(_style_session_key)
        mask = mask & (work["_ck_style"] == s)
    if operation:
        op = normalize_operation_name(operation)
        work["_op_norm"] = work.get("Operation", pd.Series(dtype=str)).apply(normalize_operation_name)
        mask = mask & (work["_op_norm"] == op)
    return mask


def _latest_production_row_index(
    log_df: pd.DataFrame,
    *,
    date_str: str,
    karigar_id: str,
    challan_no: str = "",
    style: str = "",
    operation: str = "",
) -> int | None:
    """Pick the newest save for this session/operation (matches report dedupe)."""
    if log_df.empty:
        return None
    attempts = [
        dict(
            date_str=date_str,
            karigar_id=karigar_id,
            challan_no=challan_no or None,
            style=style or None,
            operation=operation or None,
            require_karigar=True,
        ),
    ]
    if karigar_id and challan_no and style and operation:
        attempts.append(
            dict(
                date_str=date_str,
                karigar_id="",
                challan_no=challan_no,
                style=style,
                operation=operation,
                require_karigar=False,
            )
        )
    for kwargs in attempts:
        mask = _mask_production_log(log_df, **kwargs)
        if not mask.any():
            continue
        subset = log_df.loc[mask].copy()
        if "Save_Time" in subset.columns:
            subset = subset.sort_values("Save_Time", ascending=False, na_position="last")
        return int(subset.index[0])
    return None


def list_karigar_directory() -> list[dict[str, str]]:
    """Karigars for expense/production dropdowns — master + employees + recent production."""
    out: dict[str, dict[str, str]] = {}
    inactive: set[str] = set()

    def _add(kid: Any, name: Any = "") -> None:
        key = clean_key(kid)
        if not key:
            return
        nm = str(name or "").strip() or key
        if key not in out:
            out[key] = {"Karigar_ID": key, "Name": nm}
        elif len(nm) > len(out[key]["Name"]):
            out[key]["Name"] = nm

    km = get_sheet_df("karigar_master")
    if not km.empty and "Karigar_ID" in km.columns:
        for _, row in km.iterrows():
            key = clean_key(row.get("Karigar_ID"))
            if key and _is_karigar_inactive(row):
                inactive.add(key)
            _add(row.get("Karigar_ID"), row.get("Name", ""))

    em = get_sheet_df("employee_master")
    if not em.empty and "E_Code" in em.columns:
        for _, row in em.iterrows():
            typ = str(row.get("Type", "") or "").strip().lower()
            if typ not in ("karigar", "stitching"):
                continue
            key = clean_key(row.get("E_Code"))
            if key and str(row.get("Active", "")).strip().lower() in ("0", "false", "no", "inactive"):
                inactive.add(key)
            _add(row.get("E_Code"), row.get("Name", ""))

    pl = get_sheet_df("production_log")
    if not pl.empty and "Karigar_ID" in pl.columns:
        for _, row in pl.iterrows():
            _add(row.get("Karigar_ID"), row.get("Karigar_Name", ""))

    rows = []
    for r in out.values():
        kid = clean_key(r.get("Karigar_ID"))
        if kid in inactive:
            continue
        rows.append(r)
    return sorted(rows, key=lambda r: (r["Name"].lower(), r["Karigar_ID"]))


def set_karigar_active(karigar_id: str, active: bool) -> dict:
    """Soft deactivate a karigar/contractor so they don't show in payroll/dropdowns."""
    kid = clean_key(karigar_id)
    if not kid:
        return {"ok": False, "message": "Karigar ID is required"}

    today_str = datetime.now(IST).strftime("%Y-%m-%d")
    changed = 0
    km = get_sheet_df("karigar_master")
    if not km.empty and "Karigar_ID" in km.columns:
        mask = km["Karigar_ID"].apply(clean_key) == kid
        if mask.any():
            idx = km[mask].index[0]
            km.at[idx, "Active"] = bool(active)
            if not active:
                km.at[idx, "Resign_Date"] = today_str
            else:
                km.at[idx, "Resign_Date"] = ""
            save_sheet_df("karigar_master", km)
            changed += 1

    em = get_sheet_df("employee_master")
    if not em.empty and "E_Code" in em.columns:
        mask = em["E_Code"].apply(clean_key) == kid
        if mask.any():
            idx = em[mask].index[0]
            em.at[idx, "Active"] = bool(active)
            save_sheet_df("employee_master", em)
            changed += 1

    if changed <= 0:
        return {"ok": False, "message": f"Karigar {kid} not found in masters"}
    state = "active" if active else "resigned"
    return {"ok": True, "message": f"Marked {kid} as {state}.", "karigar_id": kid, "active": bool(active)}


def list_archived_karigars() -> list[dict]:
    """Return karigars marked inactive/resigned."""
    km = get_sheet_df("karigar_master")
    if km.empty or "Karigar_ID" not in km.columns:
        return []
    rows = []
    for _, row in km.iterrows():
        if _is_karigar_inactive(row):
            rows.append({
                "Karigar_ID": str(row.get("Karigar_ID", "")).strip(),
                "Name": str(row.get("Name", "")).strip(),
                "Skill": str(row.get("Skill", "")).strip(),
                "Daily_Rate_Rs": float(row.get("Daily_Rate_Rs", 0) or 0),
                "Resign_Date": str(row.get("Resign_Date", "") or ""),
            })
    return sorted(rows, key=lambda r: r["Name"].lower())


def delete_production_entries(
    *,
    date_str: str,
    karigar_id: str,
    challan_no: str = "",
    style: str = "",
    operation: str = "",
) -> dict:
    """
    Remove rows from production_log. All four production reports read from this sheet,
    so one delete updates history, Report 1, Report 2 summary, and hour-wise detail.
    """
    with _PRODUCTION_LOG_LOCK:
        log_df = get_sheet_df("production_log")
        if log_df.empty:
            return {"ok": False, "message": "No production entries to delete.", "removed": 0}
        mask = _mask_production_log(
            log_df,
            date_str=date_str,
            karigar_id=karigar_id,
            challan_no=challan_no or None,
            style=style or None,
            operation=operation or None,
        )
        removed = int(mask.sum())
        if removed <= 0:
            scope = operation or (f"{challan_no}/{style}" if challan_no and style else "all sessions")
            return {
                "ok": False,
                "message": f"No matching production rows for {karigar_id} on {date_str} ({scope}).",
                "removed": 0,
            }
        log_df = log_df[~mask].reset_index(drop=True)
        save_sheet_df("production_log", log_df)
        scope = (
            f"operation {operation}"
            if operation
            else (f"challan {challan_no} / {style}" if challan_no and style else f"all entries for {karigar_id}")
        )
        return {
            "ok": True,
            "message": f"Deleted {removed} production row(s) — {scope}. All reports refreshed.",
            "removed": removed,
        }


def list_production_sessions_admin(
    date_str: str,
    karigar_id: str | None = None,
) -> list[dict]:
    """Flat list of production sessions for admin edit/delete (one row per operation, latest save)."""
    pl = get_sheet_df("production_log")
    if pl.empty:
        return []
    day = pl[pl["Date"].apply(clean_key) == clean_key(date_str)].copy()
    if karigar_id:
        day = day[day["Karigar_ID"].apply(clean_key) == clean_key(karigar_id)]
    if day.empty:
        return []
    day = _production_log_latest_rows(day)
    rows: list[dict] = []
    for _, r in day.iterrows():
        rows.append(
            {
                "Date": str(r.get("Date", date_str)),
                "Karigar_ID": str(r.get("Karigar_ID", "")),
                "Karigar_Name": str(r.get("Karigar_Name", "")),
                "Challan_No": str(r.get("Challan_No", "")),
                "Style": str(r.get("Style", "")),
                "Operation": str(r.get("Operation", "")),
                "Total_Pieces": int(safe_num(pd.Series([r.get("Total_Pieces", 0)])).iloc[0]),
                "Save_Time": str(r.get("Save_Time", "") or ""),
                "Efficiency_%": float(safe_num(pd.Series([r.get("Efficiency_%", 0)])).iloc[0]),
            }
        )
    return rows


def get_ltl_setup_table() -> dict:
    """Manual LTL overrides + formula preview for admin LTL Setup tab."""
    df = get_sheet_df("target_ltl_override")
    overrides: list[dict] = []
    if not df.empty:
        for _, row in df.iterrows():
            st = str(row.get("Style", "")).strip()
            op = normalize_operation_name(row.get("Operation", ""))
            kid = clean_key(row.get("Karigar_ID", ""))
            manual = int(float(row.get("Manual_LTL") or 0))
            applied = resolve_applied_ltl(st, op, kid, base_target=None)
            overrides.append(
                {
                    "Style": st,
                    "Operation": op,
                    "Karigar_ID": kid,
                    "Manual_LTL": manual,
                    "Formula_LTL": applied["formula_ltl"],
                    "Final_Applied_LTL": applied["applied_ltl"],
                    "LTL_Source": applied["ltl_source"],
                    "Notes": str(row.get("Notes", "") or ""),
                    "Updated_At": str(row.get("Updated_At", "") or ""),
                }
            )
    preview = target_control_preview(str(date.today()))
    band_df = get_sheet_df("ltl_tolerance_bands")
    band_rows = band_df.fillna("").to_dict(orient="records") if not band_df.empty else _default_ltl_bands_rows()
    return {
        "ok": True,
        "tolerance_bands": preview.get("tolerance_bands", ltl_tolerance_bands_for_api()),
        "tolerance_band_rows": band_rows,
        "period_defaults": {"daily": 1, "weekly": 6, "monthly": 26},
        "overrides": overrides,
        "preview_rows": preview.get("rows", [])[:200],
    }


def _new_expense_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]


def upsert_karigar_expense(
    *,
    date_str: str,
    karigar_id: str,
    work_type: str,
    challan_no: str = "",
    challan_nos: list[str] | None = None,
    style: str = "",
    amount_rs: float,
    hours: float = 0,
    notes: str = "",
    operation: str = "",
    output: str = "",
    expense_id: str = "",
) -> dict:
    """Record non-production work (part change, alter, trainee, etc.) against a challan for payroll."""
    wt = str(work_type or "").strip()
    if wt not in KARIGAR_EXPENSE_WORK_TYPES:
        return {
            "ok": False,
            "message": f"Work type must be one of: {', '.join(KARIGAR_EXPENSE_WORK_TYPES)}",
        }
    kid = clean_key(karigar_id)
    if not kid:
        return {"ok": False, "message": "Karigar_ID is required."}
    amt = round(float(amount_rs or 0), 2)
    if amt <= 0 and float(hours or 0) <= 0:
        return {"ok": False, "message": "Enter Amount_Rs or Hours."}
    daily = get_daily_rate_for_date(kid, date_str)
    hourly = round(daily / 8, 2) if daily > 0 else 0.0
    auto_amount = False
    if amt <= 0 and hours > 0 and hourly > 0:
        amt = round(float(hours) * hourly, 2)
        auto_amount = True
    km = get_sheet_df("karigar_master")
    kname = kid
    if not km.empty and "Karigar_ID" in km.columns:
        hit = km[km["Karigar_ID"].astype(str).map(clean_key) == kid]
        if not hit.empty:
            kname = str(hit.iloc[0].get("Name", kid))
    challans_list = [str(c).strip() for c in (challan_nos or []) if str(c).strip()]
    if not challans_list and challan_no:
        challans_list = [str(challan_no).strip()]
    combined_challan = ", ".join(challans_list)

    def _style_for_challan(cn: str) -> str:
        if style:
            return _resolve_canonical_style(style)
        if not cn:
            return ""
        ch = get_sheet_df("challan_master")
        if not ch.empty and "Challan_No" in ch.columns:
            hit = ch[ch["Challan_No"].astype(str).map(clean_key) == clean_key(cn)]
            if not hit.empty:
                return str(hit.iloc[0].get("Style", "")).strip()
        return ""

    resolved_style = _style_for_challan(challans_list[0] if challans_list else "") or (
        _resolve_canonical_style(style) if style else ""
    )
    op_name = normalize_operation_name(operation)
    output_text = str(output or "").strip()
    note_parts = [str(notes or "").strip()]
    if output_text:
        note_parts.append(f"Output: {output_text}")
    combined_notes = " | ".join(p for p in note_parts if p)

    base_row = {
        "Date": date_str[:10],
        "Karigar_ID": kid,
        "Karigar_Name": kname,
        "Work_Type": wt,
        "Challan_No": combined_challan,
        "Style": resolved_style,
        "Operation": op_name,
        "Output": output_text,
        "Hours": round(float(hours or 0), 2),
        "Amount_Rs": amt,
        "Daily_Rate_Rs": daily,
        "Hourly_Rate_Rs": hourly,
        "Auto_Amount": auto_amount,
        "Notes": combined_notes,
        "Updated_At": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    df = get_sheet_df("karigar_expenses")
    if df.empty:
        df = pd.DataFrame()
    for col in ("Operation", "Output"):
        if col not in df.columns:
            df[col] = ""

    if expense_id:
        row = {**base_row, "Expense_ID": expense_id.strip()}
        mask = df["Expense_ID"].astype(str) == expense_id.strip()
        if not mask.any():
            return {"ok": False, "message": f"Expense {expense_id} not found."}
        idx = df[mask].index[0]
        for k, v in row.items():
            df.at[idx, k] = v
        save_sheet_df("karigar_expenses", df)
        return {"ok": True, "message": "Expense updated.", "row": row}

    targets = challans_list if challans_list else [""]
    per_amt = round(amt / len(targets), 2) if len(targets) > 1 and amt > 0 else amt
    added: list[dict] = []
    for cn in targets:
        row = {
            **base_row,
            "Expense_ID": _new_expense_id(),
            "Challan_No": cn,
            "Style": _style_for_challan(cn) or resolved_style,
            "Amount_Rs": per_amt,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        added.append(row)
    save_sheet_df("karigar_expenses", df)
    msg = f"Expense added ({len(added)} row(s))." if len(added) > 1 else "Expense added."
    return {"ok": True, "message": msg, "rows": added, "row": added[0] if added else base_row}


def delete_karigar_expense(expense_id: str) -> dict:
    df = get_sheet_df("karigar_expenses")
    if df.empty or "Expense_ID" not in df.columns:
        return {"ok": False, "message": "Expense not found."}
    mask = df["Expense_ID"].astype(str) == str(expense_id).strip()
    if not mask.any():
        return {"ok": False, "message": "Expense not found."}
    df = df[~mask].reset_index(drop=True)
    save_sheet_df("karigar_expenses", df)
    return {"ok": True, "message": "Expense deleted."}


def list_karigar_expenses(
    date_from: str,
    date_to: str,
    karigar_id: str | None = None,
) -> list[dict]:
    df = get_sheet_df("karigar_expenses")
    if df.empty:
        return []
    work = df.copy()
    work["Date_dt"] = pd.to_datetime(work["Date"], errors="coerce")
    work = work[
        (work["Date_dt"] >= pd.Timestamp(date_from))
        & (work["Date_dt"] <= pd.Timestamp(date_to))
    ]
    if karigar_id:
        work = work[work["Karigar_ID"].apply(clean_key) == clean_key(karigar_id)]
    # Recompute hourly/daily and (when auto-derived) amount for consistency across SKUs.
    for c in ("Hours", "Amount_Rs", "Daily_Rate_Rs", "Hourly_Rate_Rs"):
        if c in work.columns:
            work[c] = safe_num(work[c])
    if not work.empty and "Karigar_ID" in work.columns and "Date" in work.columns:
        for idx in work.index:
            kid = clean_key(work.at[idx, "Karigar_ID"])
            dstr = str(work.at[idx, "Date"] or "")[:10]
            if not kid or not dstr:
                continue
            daily = get_daily_rate_for_date(kid, dstr)
            hourly = round(daily / 8, 2) if daily > 0 else float(work.at[idx, "Hourly_Rate_Rs"] or 0)
            hrs = float(work.at[idx, "Hours"] or 0)
            amt = float(work.at[idx, "Amount_Rs"] or 0)
            expected = round(hrs * hourly, 2) if hrs > 0 and hourly > 0 else amt
            auto_flag = bool(work.at[idx, "Auto_Amount"]) if "Auto_Amount" in work.columns else False
            # If amount is auto-calculated (or looks auto), keep it in sync with current rate-at-date.
            if hrs > 0 and hourly > 0 and (auto_flag or amt <= 0 or abs(amt - expected) <= 1.0):
                work.at[idx, "Amount_Rs"] = expected
            if daily > 0:
                work.at[idx, "Daily_Rate_Rs"] = daily
            work.at[idx, "Hourly_Rate_Rs"] = hourly
    return work.fillna("").to_dict(orient="records")


def _karigar_pieces_on_date(
    log_df: pd.DataFrame,
    date_str: str,
    karigar_id: str,
    *,
    additional_by_op: dict[str, int] | None = None,
) -> int:
    """Sum pieces already logged today for this karigar plus pieces in the current save."""
    total = 0
    if log_df is not None and not log_df.empty and "Total_Pieces" in log_df.columns:
        mask = (
            log_df["Date"].apply(clean_key) == clean_key(date_str)
        ) & (log_df["Karigar_ID"].apply(clean_key) == clean_key(karigar_id))
        if mask.any():
            total += int(safe_num(log_df.loc[mask, "Total_Pieces"]).sum())
    if additional_by_op:
        total += sum(max(0, int(v)) for v in additional_by_op.values())
    return max(total, 0)


def compute_financial_audit(
    base_target: int | float,
    pieces: int | float,
    daily_rate: float,
    *,
    allocated_actual_amount: float | None = None,
    allocated_budgeted_amount: float | None = None,
) -> dict[str, Any]:
    """
    Factory financial audit (benchmark ₹480 / operation target).

    Budget rate/pc = 480 / Base Target
    Budgeted amount = pieces × budget rate/pc (capped at ₹480/day across ops when allocated)
    Actual amount = karigar daily rate (allocated across ops when multiple rows/day)
    P&L = Budgeted − Actual (profit if positive)
    """
    bt = max(int(base_target or 0), 1)
    pcs = max(int(pieces or 0), 0)
    dr = float(daily_rate or 0)
    budget_rate = round(BENCHMARK_DAILY_RATE_RS / bt, 4)
    budgeted = (
        round(float(allocated_budgeted_amount), 2)
        if allocated_budgeted_amount is not None
        else round(pcs * budget_rate, 2)
    )
    if allocated_actual_amount is not None:
        actual_amt = round(float(allocated_actual_amount), 2)
    else:
        actual_amt = round(dr, 2)
    actual_rate_pc = round(actual_amt / pcs, 4) if pcs > 0 else 0.0
    pl = round(budgeted - actual_amt, 2)
    return {
        "budget_rate_per_piece": budget_rate,
        "budgeted_amount": budgeted,
        "actual_rate_per_piece": actual_rate_pc,
        "actual_amount": actual_amt,
        "pl_rs": pl,
        "daily_rate_rs": round(dr, 2),
    }


def _financial_from_log_row(row: pd.Series, *, date_str: str, kid: str) -> dict[str, Any]:
    """Use persisted audit columns when present; else recompute for legacy rows."""
    pcs = int(safe_num(pd.Series([row.get("Total_Pieces", 0)])).iloc[0])
    base_target = int(float(row.get("Base_Target") or row.get("Target") or 0))
    daily_rate = float(row.get("Daily_Rate_Rs") or 0)
    if daily_rate <= 0:
        daily_rate = _get_daily_salary(kid, date_str)
    if "Budget_Rate_Per_Piece" in row.index and pd.notna(row.get("Budgeted_Expense_Rs")):
        return {
            "budget_rate_per_piece": float(row.get("Budget_Rate_Per_Piece") or 0),
            "budgeted_amount": float(row.get("Budgeted_Expense_Rs") or 0),
            "actual_rate_per_piece": float(row.get("Actual_Rate_Per_Piece") or 0),
            "actual_amount": float(row.get("Actual_Expense_Rs") or 0),
            "pl_rs": float(row.get("PL_Rs") or 0),
            "daily_rate_rs": daily_rate,
        }
    return compute_financial_audit(base_target, pcs, daily_rate)


def compute_formula_ltl(base_target: int | float, daily_rate: float) -> int:
    """ROUND((Operation Target × tolerance factor) × (Daily Rate / 480), 0)."""
    bt = int(base_target or 0)
    dr = float(daily_rate or 0)
    if bt <= 0 or dr <= 0:
        return 0
    tol = ltl_tolerance_factor(dr)
    return int(round((bt * tol) * (dr / BENCHMARK_DAILY_RATE_RS), 0))


def _ltl_override_key(style: str, operation: str, karigar_id: str) -> tuple[str, str, str]:
    return (
        clean_key(style),
        normalize_operation_name(operation).lower(),
        clean_key(karigar_id),
    )


def _load_ltl_override_map() -> dict[tuple[str, str, str], int]:
    df = get_sheet_df("target_ltl_override")
    out: dict[tuple[str, str, str], int] = {}
    if df.empty:
        return out
    for _, row in df.iterrows():
        try:
            val = int(float(row.get("Manual_LTL") or 0))
        except (ValueError, TypeError):
            continue
        if val <= 0:
            continue
        key = _ltl_override_key(
            str(row.get("Style", "")),
            str(row.get("Operation", "")),
            str(row.get("Karigar_ID", "")),
        )
        out[key] = val
    return out


def resolve_applied_ltl(
    style: str,
    operation: str,
    karigar_id: str,
    *,
    as_of_date: str | None = None,
    base_target: int | None = None,
    daily_rate: float | None = None,
    override_map: dict[tuple[str, str, str], int] | None = None,
) -> dict[str, Any]:
    """Resolve Formula LTL, optional manual override, and Final Applied LTL."""
    op = normalize_operation_name(operation)
    bt = int(base_target) if base_target is not None else 0
    if bt <= 0:
        sm = get_sheet_df("style_master")
        if not sm.empty:
            hit = sm[
                (sm["Style"].astype(str).str.strip() == str(style).strip())
                & (sm["Operation"].astype(str).str.strip().str.lower() == op.lower())
            ]
            if not hit.empty:
                bt = int(float(hit.iloc[0]["Target"] or 0))
    dr = float(daily_rate) if daily_rate is not None else 0.0
    if dr <= 0:
        dr = float(get_daily_rate_for_date(karigar_id, as_of_date))

    formula_ltl = compute_formula_ltl(bt, dr)
    omap = override_map if override_map is not None else _load_ltl_override_map()
    manual = omap.get(_ltl_override_key(style, op, karigar_id))
    if manual is not None and manual > 0:
        applied = int(manual)
        source = "override"
        target_type = "Manual Override"
    else:
        applied = formula_ltl
        source = "formula"
        target_type = "Automated Formula"

    return {
        "style": str(style).strip(),
        "operation": op,
        "karigar_id": clean_key(karigar_id),
        "daily_rate_rs": round(dr, 2),
        "tolerance_pct": ltl_tolerance_pct_for_rate(dr),
        "tolerance_factor": ltl_tolerance_factor(dr),
        "base_target": bt,
        "formula_ltl": formula_ltl,
        "manual_override": manual,
        "applied_ltl": applied,
        "ltl_source": source,
        "target_type": target_type,
    }


def target_control_preview(
    date_str: str,
    *,
    style: str = "",
    karigar_id: str = "",
    operation: str = "",
    period: str = "daily",
) -> dict:
    """Central Target Control ledger — style matrix × karigar with formula + overrides."""
    period_key = str(period or "daily").strip().lower()
    period_days = {"daily": 1, "weekly": 6, "monthly": 26}.get(period_key, 1)
    sm = get_sheet_df("style_master")
    km = get_sheet_df("karigar_master")
    if sm.empty or km.empty:
        return {
            "date": date_str,
            "benchmark_daily_rate_rs": BENCHMARK_DAILY_RATE_RS,
            "period": period_key,
            "period_days": period_days,
            "tolerance_bands": ltl_tolerance_bands_for_api(),
            "rows": [],
        }

    sm = sm.copy()
    sm["Style"] = sm["Style"].astype(str).str.strip()
    sm["Operation"] = sm["Operation"].astype(str).str.strip()
    if "Operation_Type" not in sm.columns:
        sm["Operation_Type"] = "Medium"
    sm["Operation_Type"] = sm["Operation_Type"].astype(str).str.strip().replace("", "Medium")
    if style:
        sm = sm[sm["Style"] == style.strip()]
    if operation:
        op_f = normalize_operation_name(operation).lower()
        sm = sm[sm["Operation"].str.lower() == op_f]

    km = km.copy()
    km["Karigar_ID"] = km["Karigar_ID"].apply(clean_key)
    if karigar_id:
        km = km[km["Karigar_ID"] == clean_key(karigar_id)]

    omap = _load_ltl_override_map()
    km_names = {clean_key(str(r["Karigar_ID"])): str(r.get("Name", "")) for _, r in km.iterrows()}
    rows: list[dict] = []
    for _, srow in sm.iterrows():
        st = str(srow["Style"])
        op = str(srow["Operation"])
        bt = int(float(srow["Target"] or 0))
        for kid in km["Karigar_ID"].astype(str):
            info = resolve_applied_ltl(
                st, op, kid,
                as_of_date=date_str,
                base_target=bt,
                override_map=omap,
            )
            rows.append(
                {
                    "Style": st,
                    "Operation": op,
                    "Operation_Type": str(srow.get("Operation_Type", "Medium") or "Medium"),
                    "Karigar_ID": kid,
                    "Karigar_Name": km_names.get(clean_key(kid), ""),
                    "Daily_Rate_Rs": info["daily_rate_rs"],
                    "Base_Target": info["base_target"],
                    "Tolerance_%": info["tolerance_pct"],
                    "Formula_LTL": info["formula_ltl"],
                    "Manual_Override": info["manual_override"] if info["manual_override"] else "",
                    "Final_Applied_LTL": info["applied_ltl"],
                    "Final_Applied_LTL_Period": int(info["applied_ltl"] * period_days),
                    "Target_For_Period": int(info["base_target"] * period_days),
                    "Period": period_key,
                    "Target_Type": info["target_type"],
                    "LTL_Source": info["ltl_source"],
                }
            )

    rows.sort(key=lambda r: (r["Style"], r["Operation"], r["Karigar_ID"]))
    return {
        "date": date_str,
        "period": period_key,
        "period_days": period_days,
        "benchmark_daily_rate_rs": BENCHMARK_DAILY_RATE_RS,
        "tolerance_bands": ltl_tolerance_bands_for_api(),
        "rows": rows,
    }


def upsert_ltl_override(
    style: str,
    operation: str,
    karigar_id: str,
    manual_ltl: int | None,
    *,
    notes: str = "",
) -> dict:
    """Set or clear managerial manual LTL override for (Style, Operation, Karigar)."""
    st = str(style).strip()
    op = normalize_operation_name(operation)
    kid = clean_key(karigar_id)
    if not st or not op or not kid:
        return {"ok": False, "message": "Style, Operation, and Karigar_ID are required"}

    df = get_sheet_df("target_ltl_override")
    if df.empty:
        df = pd.DataFrame(columns=["Style", "Operation", "Karigar_ID", "Manual_LTL", "Notes", "Updated_At"])

    for c in ["Style", "Operation", "Karigar_ID", "Manual_LTL", "Notes", "Updated_At"]:
        if c not in df.columns:
            df[c] = ""

    mask = (
        (df["Style"].astype(str).str.strip() == st)
        & (df["Operation"].astype(str).str.strip().str.lower() == op.lower())
        & (df["Karigar_ID"].apply(clean_key) == kid)
    )

    if manual_ltl is None or int(manual_ltl) <= 0:
        if mask.any():
            df = df[~mask].reset_index(drop=True)
            save_sheet_df("target_ltl_override", df)
        return {"ok": True, "message": "Manual override cleared — formula LTL will apply"}

    row = {
        "Style": st,
        "Operation": op,
        "Karigar_ID": kid,
        "Manual_LTL": int(manual_ltl),
        "Notes": str(notes or "").strip(),
        "Updated_At": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    if mask.any():
        idx = df[mask].index[0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = dedupe_sheet("target_ltl_override", df)
    save_sheet_df("target_ltl_override", df)
    applied = resolve_applied_ltl(st, op, kid, base_target=None)
    return {
        "ok": True,
        "message": f"Manual LTL {int(manual_ltl)} saved for {st} / {op} / {kid}",
        "applied_ltl": applied["applied_ltl"],
    }


def bulk_upsert_ltl_override_all_styles(
    operation: str,
    karigar_id: str,
    manual_ltl: int | None,
    *,
    notes: str = "",
) -> dict:
    """Apply manual LTL override to every style that has this operation."""
    op = normalize_operation_name(operation)
    kid = clean_key(karigar_id)
    if not op or not kid:
        return {"ok": False, "message": "Operation and Karigar_ID are required"}
    sm = get_sheet_df("style_master")
    if sm.empty:
        return {"ok": False, "message": "Style master empty"}
    styles = sorted(
        {
            str(s).strip()
            for s in sm.loc[
                sm["Operation"].astype(str).str.strip().str.lower() == op.lower(), "Style"
            ].astype(str)
            if str(s).strip()
        }
    )
    if not styles:
        return {"ok": False, "message": f"No styles found for operation {op}"}
    count = 0
    for st in styles:
        out = upsert_ltl_override(st, op, kid, manual_ltl, notes=notes)
        if out.get("ok"):
            count += 1
    action = "cleared" if manual_ltl is None or int(manual_ltl or 0) <= 0 else f"set to {int(manual_ltl)}"
    return {
        "ok": True,
        "message": f"Manual LTL {action} for {count} style(s), operation {op}, karigar {kid}",
        "styles_updated": count,
    }


def list_karigar_challans_for_expense(karigar_id: str, *, days_back: int = 90) -> list[dict[str, str]]:
    """Challans this karigar has worked on recently (for expense multi-select)."""
    kid = clean_key(karigar_id)
    if not kid:
        return []
    pl = get_sheet_df("production_log")
    if pl.empty:
        return []
    cutoff = (date.today() - timedelta(days=max(1, days_back))).isoformat()
    work = pl.copy()
    work["_kar"] = work["Karigar_ID"].apply(clean_key)
    work["_date"] = work["Date"].astype(str).str[:10]
    work = work[(work["_kar"] == kid) & (work["_date"] >= cutoff)]
    if work.empty:
        return []
    work = _production_log_latest_rows(work)
    out: dict[str, dict[str, str]] = {}
    for _, r in work.iterrows():
        cn = str(r.get("Challan_No", "")).strip()
        if not cn:
            continue
        st = str(r.get("Style", "")).strip()
        key = clean_key(cn)
        if key not in out:
            out[key] = {"Challan_No": cn, "Style": st, "Last_Date": str(r.get("Date", ""))[:10]}
    return sorted(out.values(), key=lambda x: (x["Last_Date"], x["Challan_No"]), reverse=True)


def _resolve_style_operation(name: str, op_info: dict[str, dict]) -> str | None:
    """Map UI operation text to style_master operation key (trim + case-insensitive)."""
    n = normalize_operation_name(name)
    if not n:
        return None
    if n in op_info:
        return n
    lower = n.lower()
    for key in op_info:
        if key.lower() == lower:
            return key
    return None


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def sticker_in_col(hcol: str) -> str:
    return f"SI_{hcol}"


def sticker_out_col(hcol: str) -> str:
    return f"SO_{hcol}"


def resolve_hour_pieces(entry: dict) -> int:
    """Pieces from stickers or manual count (single hour; use resolve_session_hour_pieces for a full day)."""
    if bool(entry.get("manual_pieces")):
        return int(entry.get("pieces") or 0)
    sin = int(entry.get("sticker_in") or 0)
    sout = int(entry.get("sticker_out") or 0)
    if sin == 0 and sout == 0:
        return int(entry.get("pieces") or 0)
    if sin == 0 and sout > 0:
        return sout
    return abs(sin - sout)


def resolve_session_hour_pieces(hour_entries: list[dict]) -> dict[str, int]:
    """
    Resolve pieces per hour in time order.
    Sticker-out-only with non-decreasing counts = cumulative machine counter (use delta).
    """
    by_col = {
        (e.get("hour_col") or e.get("hour") or ""): e
        for e in hour_entries
        if (e.get("hour_col") or e.get("hour") or "") in HOUR_COLS
    }
    prev_out = 0
    out: dict[str, int] = {}
    for hc in HOUR_COLS:
        if hc == "H_13_14":
            continue
        e = by_col.get(hc)
        if not e:
            out[hc] = 0
            continue
        if bool(e.get("manual_pieces")):
            out[hc] = int(e.get("pieces") or 0)
            continue
        sin = int(e.get("sticker_in") or 0)
        sout = int(e.get("sticker_out") or 0)
        if sin == 0 and sout == 0:
            out[hc] = int(e.get("pieces") or 0)
            continue
        if sin == 0 and sout > 0:
            pcs = sout - prev_out if prev_out > 0 and sout >= prev_out else sout
            out[hc] = max(0, pcs)
            prev_out = max(prev_out, sout)
            continue
        out[hc] = abs(sin - sout)
    return out


def clean_key(val: Any) -> str:
    if val is None:
        return ""
    try:
        f = float(val)
        if pd.isna(f):
            return ""
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(val).strip()


def _active_bool_from_cell(val: Any) -> bool | None:
    if val is True or val == 1:
        return True
    if val is False or val == 0:
        return False
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "active", "y"):
        return True
    if s in ("0", "false", "no", "inactive", "resigned", "n"):
        return False
    return None


def _is_karigar_inactive(row: dict | pd.Series) -> bool:
    active = _active_bool_from_cell(row.get("Active", ""))
    resign = str(row.get("Resign_Date", "") or "").strip()
    has_resign = bool(resign) and resign.lower() not in ("nan", "nat", "none", "")
    if active is False:
        return True
    if active is True:
        return False
    return has_resign


def filter_non_karigar_employees(rows: list[dict]) -> list[dict]:
    """Exclude karigars that belong in karigar master from employee master list."""
    km = get_sheet_df("karigar_master")
    karigar_ids: set[str] = set()
    if not km.empty and "Karigar_ID" in km.columns:
        karigar_ids = {clean_key(x) for x in km["Karigar_ID"] if clean_key(x)}
    out: list[dict] = []
    for row in rows:
        ecode = clean_key(row.get("E_Code", ""))
        typ = str(row.get("Type", "") or "").strip().lower()
        if ecode and ecode in karigar_ids:
            continue
        if typ in ("karigar", "stitching"):
            continue
        out.append(row)
    return out


def calc_salary(in_str: str, out_str: str, daily_rate: float, ot_mult: float = 1.0) -> dict:
    """Karigar attendance policy — see ``karigar_attendance`` module."""
    from .karigar_attendance import calc_salary as _calc

    return _calc(in_str, out_str, daily_rate, ot_mult=ot_mult)


def dashboard_summary(planning_date: str | None = None) -> dict:
    today = planning_date or str(date.today())
    pl = get_sheet_df("production_log")
    km = get_sheet_df("karigar_master")
    cm = get_sheet_df("challan_master")
    rate_map = build_daily_rate_map(today)

    tdpl = pl[pl["Date"].astype(str) == today] if not pl.empty and "Date" in pl.columns else pd.DataFrame()
    active_k = int(tdpl["Karigar_ID"].nunique()) if not tdpl.empty else 0
    pieces = int(safe_num(tdpl["Total_Pieces"]).sum()) if not tdpl.empty else 0
    avg_eff = float(safe_num(tdpl["Efficiency_%"]).mean()) if not tdpl.empty and "Efficiency_%" in tdpl.columns else 0.0
    pv = float(safe_num(tdpl["Piece_Value_Rs"]).sum()) if not tdpl.empty else 0.0
    pend_c = 0
    if not cm.empty:
        cm2 = cm.copy()
        cm2["Pend"] = safe_num(cm2["Total_Qty"]) - safe_num(cm2.get("Received_Qty", 0))
        pend_c = int(len(cm2[cm2["Pend"] > 0]))

    aids = set(tdpl["Karigar_ID"].astype(str).map(clean_key)) if not tdpl.empty else set()
    karigar_status = []
    if not km.empty:
        for _, r in km.iterrows():
            kid = str(r["Karigar_ID"])
            kid_key = clean_key(kid)
            karigar_status.append({
                "Karigar_ID": kid,
                "Name": str(r.get("Name", "")),
                "Skill": str(r.get("Skill", "")),
                "Daily_Rate_Rs": rate_map.get(kid_key, 0.0),
                "Status": "Working" if kid_key in aids else "Idle",
            })

    challan_register = []
    if not cm.empty:
        for _, r in cm.iterrows():
            pend = int(safe_num(pd.Series([r["Total_Qty"]])).iloc[0] - safe_num(pd.Series([r.get("Received_Qty", 0)])).iloc[0])
            challan_register.append({
                "Challan_No": str(r["Challan_No"]),
                "Style": str(r.get("Style", "")),
                "Party": str(r.get("Party", "")),
                "Total_Qty": int(safe_num(pd.Series([r["Total_Qty"]])).iloc[0]),
                "Pending": pend,
                "Status": "Done" if pend <= 0 else f"{pend} pending",
            })

    today_production = []
    if not tdpl.empty:
        for _, r in tdpl.iterrows():
            today_production.append({
                k: (float(r[k]) if k in ("Efficiency_%", "Piece_Value_Rs", "Rate_Rs") and pd.notna(r.get(k)) else r.get(k))
                for k in ["Karigar_Name", "Challan_No", "Style", "Operation", "Total_Pieces", "Target", "Efficiency_%", "Piece_Value_Rs"]
                if k in tdpl.columns
            })

    return {
        "date": today,
        "metrics": {
            "active_karigar": active_k,
            "total_karigar": len(km),
            "pieces_today": pieces,
            "avg_efficiency": round(avg_eff, 1),
            "piece_value_today": round(pv, 2),
            "total_challans": len(cm),
            "pending_challans": pend_c,
        },
        "karigar_status": karigar_status,
        "challan_register": challan_register,
        "today_production": today_production,
    }


def load_production_entry(date_str: str, karigar_id: str, challan_no: str, style: str) -> dict:
    """Load existing hour-wise entry for composite key."""
    style = _resolve_canonical_style(style)
    pl = get_sheet_df("production_log")
    hours: dict[str, dict] = {
        h: {"operation": "", "pieces": 0, "sticker_in": 0, "sticker_out": 0, "manual_pieces": False}
        for h in HOUR_COLS
    }
    if pl.empty:
        return {"hours": hours, "found": False}

    pl = pl.copy()
    pl["_date"] = pl["Date"].apply(clean_key)
    pl["_kar"] = pl["Karigar_ID"].apply(clean_key)
    pl["_challan"] = pl["Challan_No"].apply(clean_key)
    pl["_style"] = pl["Style"].apply(_style_session_key)
    existing = pl[
        (pl["_date"] == clean_key(date_str))
        & (pl["_kar"] == clean_key(karigar_id))
        & (pl["_challan"] == clean_key(challan_no))
        & (pl["_style"] == _style_session_key(style))
    ]
    if existing.empty:
        return {"hours": hours, "found": False}

    if "Save_Time" in existing.columns:
        existing = existing.sort_values("Save_Time", ascending=False, na_position="last")
    if "Operation" in existing.columns:
        existing["_op_norm"] = existing["Operation"].apply(normalize_operation_name)
        existing = existing.drop_duplicates(subset=["_op_norm"], keep="first")

    for _, row in existing.iterrows():
        op_name = normalize_operation_name(row.get("Operation", ""))
        for hcol in HOUR_COLS:
            raw = row.get(hcol, 0)
            try:
                val = 0 if pd.isna(raw) else int(float(raw))
            except (ValueError, TypeError):
                val = 0
            si = 0
            so = 0
            try:
                si = int(float(row.get(sticker_in_col(hcol), 0) or 0))
            except (ValueError, TypeError):
                si = 0
            try:
                so = int(float(row.get(sticker_out_col(hcol), 0) or 0))
            except (ValueError, TypeError):
                so = 0
            if val > 0 or si > 0 or so > 0:
                manual = si == 0 and so == 0
                hours[hcol] = {
                    "operation": op_name,
                    "pieces": val,
                    "sticker_in": si,
                    "sticker_out": so,
                    "manual_pieces": manual,
                }
    return {"hours": hours, "found": True, "rows_loaded": len(existing)}


def _session_hour_entries_from_log(
    date_str: str,
    karigar_id: str,
    challan_no: str,
    style: str,
) -> tuple[list[dict[str, Any]], str]:
    """Rebuild hour_entries for save_production_entry from persisted log rows."""
    pl = get_sheet_df("production_log")
    if pl.empty:
        return [], ""
    pl = pl.copy()
    pl["_date"] = pl["Date"].apply(clean_key)
    pl["_kar"] = pl["Karigar_ID"].apply(clean_key)
    pl["_challan"] = pl["Challan_No"].apply(clean_key)
    pl["_style"] = pl["Style"].apply(_style_session_key)
    existing = pl[
        (pl["_date"] == clean_key(date_str))
        & (pl["_kar"] == clean_key(karigar_id))
        & (pl["_challan"] == clean_key(challan_no))
        & (pl["_style"] == _style_session_key(style))
    ]
    if existing.empty:
        return [], ""
    if "Save_Time" in existing.columns:
        existing = existing.sort_values("Save_Time", ascending=False, na_position="last")
    if "Operation" in existing.columns:
        existing["_op_norm"] = existing["Operation"].apply(normalize_operation_name)
        existing = existing.drop_duplicates(subset=["_op_norm"], keep="first")

    kname = str(existing.iloc[0].get("Karigar_Name", karigar_id))
    hour_entries: list[dict[str, Any]] = []
    for _, row in existing.iterrows():
        op_name = normalize_operation_name(row.get("Operation", ""))
        for hcol in HOUR_COLS:
            if hcol == "H_13_14":
                continue
            try:
                pcs = int(float(row.get(hcol, 0) or 0))
            except (ValueError, TypeError):
                pcs = 0
            try:
                si = int(float(row.get(sticker_in_col(hcol), 0) or 0))
            except (ValueError, TypeError):
                si = 0
            try:
                so = int(float(row.get(sticker_out_col(hcol), 0) or 0))
            except (ValueError, TypeError):
                so = 0
            if pcs > 0 or si > 0 or so > 0:
                hour_entries.append(
                    {
                        "hour_col": hcol,
                        "operation": op_name,
                        "pieces": pcs,
                        "sticker_in": si,
                        "sticker_out": so,
                        "manual_pieces": si == 0 and so == 0,
                    }
                )
    return hour_entries, kname


def recalculate_production_for_karigar_from_date(karigar_id: str, effective_from: str) -> dict:
    """Re-save all production sessions for a karigar on/after effective_from (rate/LTL refresh)."""
    kid = clean_key(karigar_id)
    eff = str(effective_from or "")[:10]
    if not kid or not eff:
        return {"ok": False, "message": "Karigar and effective date required"}

    pl = get_sheet_df("production_log")
    if pl.empty:
        return {"ok": True, "sessions": 0, "message": "No production entries to recalculate"}

    work = pl.copy()
    work["_date"] = work["Date"].astype(str).str[:10]
    work["_kar"] = work["Karigar_ID"].apply(clean_key)
    subset = work[(work["_kar"] == kid) & (work["_date"] >= eff)]
    if subset.empty:
        return {"ok": True, "sessions": 0, "message": "No sessions on or after effective date"}

    latest = _production_log_latest_rows(subset)
    sessions: set[tuple[str, str, str]] = set()
    for _, r in latest.iterrows():
        sessions.add(
            (
                str(r.get("Date", ""))[:10],
                str(r.get("Challan_No", "")),
                str(r.get("Style", "")),
            )
        )

    recalced = 0
    for date_str, challan_no, style in sorted(sessions):
        hour_entries, kname = _session_hour_entries_from_log(date_str, kid, challan_no, style)
        if not hour_entries:
            continue
        out = save_production_entry(
            date_str=date_str,
            karigar_id=kid,
            karigar_name=kname or kid,
            challan_no=challan_no,
            style=style,
            hour_entries=hour_entries,
            saved_by="system",
            saved_by_name="rate-recalc",
        )
        if out.get("ok"):
            recalced += 1

    return {
        "ok": True,
        "sessions": recalced,
        "message": f"Recalculated {recalced} production session(s) from {eff} using current rates and LTL.",
    }


def save_production_entry(
    *,
    date_str: str,
    karigar_id: str,
    karigar_name: str,
    challan_no: str,
    style: str,
    hour_entries: list[dict],
    saved_by: str = "erp",
    saved_by_name: str = "",
) -> dict:
    """hour_entries: [{hour_col, operation, pieces, sticker_in, sticker_out, manual_pieces}, ...]"""
    with _PRODUCTION_LOG_LOCK:
        style = _resolve_canonical_style(style)
        sm = get_sheet_df("style_master")
        style_ops = sm[sm["Style"].astype(str).str.strip().str.lower() == _style_session_key(style)] if not sm.empty else pd.DataFrame()
        op_info: dict[str, dict] = {}
        for _, row in style_ops.iterrows():
            op_key = normalize_operation_name(row["Operation"])
            if not op_key:
                continue
            op_info[op_key] = {
                "Target": int(row["Target"]),
                "Rate_Rs": float(row["Rate_Rs"]),
                "Hourly_Target": max(1, int(row["Target"])),
            }

        from .karigar_attendance import production_hour_cols_for_date

        allowed_hours = set(production_hour_cols_for_date(date_str))
        h_vals = resolve_session_hour_pieces(hour_entries)
        si_vals: dict[str, int] = {}
        so_vals: dict[str, int] = {}
        op_vals: dict[str, str | None] = {}
        for e in hour_entries:
            hc = e.get("hour_col") or e.get("hour")
            if hc not in HOUR_COLS or hc not in allowed_hours:
                continue
            si_vals[hc] = int(e.get("sticker_in") or 0)
            so_vals[hc] = int(e.get("sticker_out") or 0)
            raw_op = e.get("operation") or None
            op_vals[hc] = _resolve_style_operation(raw_op, op_info) if raw_op else None

        if len(op_info) == 1:
            only_op = next(iter(op_info))
            for hc in HOUR_COLS:
                if hc == "H_13_14":
                    continue
                if h_vals.get(hc, 0) > 0 and not op_vals.get(hc):
                    op_vals[hc] = only_op

        op_totals: dict[str, dict] = defaultdict(lambda: {"pieces": 0, "hours": 0, "value": 0.0})
        for hc in production_hour_cols_for_date(date_str):
            if hc == "H_13_14":
                continue
            op = op_vals.get(hc)
            pcs = h_vals.get(hc, 0)
            if op and op in op_info and pcs > 0:
                od = op_info[op]
                op_totals[op]["pieces"] += pcs
                op_totals[op]["hours"] += 1
                op_totals[op]["value"] += pcs * od["Rate_Rs"]

        if not op_totals:
            return {"ok": False, "message": "No pieces entered"}

        log_df = get_sheet_df("production_log")
        log_df = _drop_production_session_rows(log_df, date_str, karigar_id, challan_no, style)

        save_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        chal_meta = challan_snapshot(challan_no)
        ltl_overrides = _load_ltl_override_map()
        daily_rate = get_daily_rate_for_date(karigar_id, date_str)
        new_op_pieces = {op: int(data["pieces"]) for op, data in op_totals.items()}
        day_piece_total = _karigar_pieces_on_date(
            log_df, date_str, karigar_id, additional_by_op=new_op_pieces,
        )
        raw_budget_by_op: dict[str, float] = {}
        for op_name, data in op_totals.items():
            bt = int(op_info[op_name]["Target"])
            raw_budget_by_op[op_name] = data["pieces"] * (BENCHMARK_DAILY_RATE_RS / max(bt, 1))
        raw_budget_total = sum(raw_budget_by_op.values())
        budget_scale = (
            min(1.0, BENCHMARK_DAILY_RATE_RS / raw_budget_total)
            if raw_budget_total > BENCHMARK_DAILY_RATE_RS
            else 1.0
        )
        new_rows: list[dict] = []

        for op_name, data in op_totals.items():
            od = op_info[op_name]
            base_target = int(od["Target"])
            ltl_info = resolve_applied_ltl(
                style,
                op_name,
                karigar_id,
                as_of_date=date_str,
                base_target=base_target,
                override_map=ltl_overrides,
            )
            applied_ltl = int(ltl_info["applied_ltl"])
            wh = int(data["hours"])
            adj_target = max(applied_ltl * wh, 1)
            op_eff = round(data["pieces"] / adj_target * 100, 1) if applied_ltl > 0 else 0.0
            hour_row = {
                hcol: (h_vals.get(hcol, 0) if op_vals.get(hcol) == op_name else 0)
                for hcol in HOUR_COLS
            }
            sticker_row: dict[str, int] = {}
            for hcol in HOUR_COLS:
                if op_vals.get(hcol) == op_name:
                    sticker_row[sticker_in_col(hcol)] = si_vals.get(hcol, 0)
                    sticker_row[sticker_out_col(hcol)] = so_vals.get(hcol, 0)
                else:
                    sticker_row[sticker_in_col(hcol)] = 0
                    sticker_row[sticker_out_col(hcol)] = 0
            pcs = int(data["pieces"])
            piece_value = round(pcs * od["Rate_Rs"], 2)
            share = (pcs / day_piece_total) if day_piece_total > 0 else 1.0
            allocated_actual = round(daily_rate * share, 2)
            fin = compute_financial_audit(
                base_target,
                pcs,
                daily_rate,
                allocated_actual_amount=allocated_actual,
                allocated_budgeted_amount=raw_budget_by_op.get(op_name, 0) * budget_scale,
            )
            new_rows.append(
                {
                    "Date": date_str,
                    "Karigar_ID": karigar_id,
                    "Karigar_Name": karigar_name,
                    "Challan_No": challan_no,
                    "Challan_Party": chal_meta.get("Party", ""),
                    "Challan_Description": chal_meta.get("Challan_Description", ""),
                    "Style": style,
                    "Operation": op_name,
                    **hour_row,
                    **sticker_row,
                    "Total_Pieces": pcs,
                    "Base_Target": base_target,
                    "Formula_LTL": int(ltl_info["formula_ltl"]),
                    "Applied_LTL": applied_ltl,
                    "LTL_Source": ltl_info["ltl_source"],
                    "Target": applied_ltl,
                    "Rate_Rs": od["Rate_Rs"],
                    "Daily_Rate_Rs": fin["daily_rate_rs"],
                    "Efficiency_%": op_eff,
                    "Piece_Value_Rs": piece_value,
                    "Budget_Rate_Per_Piece": fin["budget_rate_per_piece"],
                    "Budgeted_Expense_Rs": fin["budgeted_amount"],
                    "Actual_Rate_Per_Piece": fin["actual_rate_per_piece"],
                    "Actual_Expense_Rs": fin["actual_amount"],
                    "PL_Rs": fin["pl_rs"],
                    "Saved_By": saved_by,
                    "Saved_By_Name": saved_by_name,
                    "Save_Time": save_time,
                }
            )

        log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
        save_sheet_df("production_log", log_df)
        rows_added = len(new_rows)
        return {
            "ok": True,
            "message": f"Saved {rows_added} operation row(s)",
            "rows_added": rows_added,
            "save_time": save_time,
        }


def style_costing_report(
    *,
    month: str = "All",
    style: str = "All",
    party: str = "All",
) -> dict:
    cm = get_sheet_df("challan_master")
    pl = get_sheet_df("production_log")
    sm = get_sheet_df("style_master")

    if cm.empty:
        return {"rows": [], "summary": {}, "style_rollup": []}

    cm_sc = cm.copy()
    if "Date" not in cm_sc.columns:
        cm_sc["Date"] = pd.NaT
    cm_sc["Date_dt"] = pd.to_datetime(cm_sc["Date"], errors="coerce")
    if month != "All":
        cm_sc = cm_sc[cm_sc["Date_dt"].dt.strftime("%Y-%m") == month]
    if style != "All":
        cm_sc = cm_sc[cm_sc["Style"] == style]
    if party != "All" and "Party" in cm_sc.columns:
        cm_sc = cm_sc[cm_sc["Party"] == party]

    for col in ["Total_Qty", "Rate_Per_Pc", "Deposit_Rs", "Received_Qty"]:
        if col in cm_sc.columns:
            cm_sc[col] = safe_num(cm_sc[col])
    if "Received_Qty" not in cm_sc.columns:
        cm_sc["Received_Qty"] = 0.0
    if "Deposit_Rs" not in cm_sc.columns:
        cm_sc["Deposit_Rs"] = 0.0
    cm_sc["Pending"] = cm_sc["Total_Qty"] - cm_sc["Received_Qty"]
    cm_sc["Is_Pending"] = cm_sc["Pending"] > 0

    if not sm.empty:
        target_rate = sm.groupby("Style")["Rate_Rs"].sum().reset_index()
        target_rate.columns = ["Style", "Target_Labour_Rate_Pc"]
        cm_sc = cm_sc.merge(target_rate, on="Style", how="left").fillna({"Target_Labour_Rate_Pc": 0})
    else:
        cm_sc["Target_Labour_Rate_Pc"] = 0

    if not pl.empty and "Piece_Value_Rs" in pl.columns and "Challan_No" in pl.columns:
        pl_sc = pl.copy()
        pl_sc["Piece_Value_Rs"] = safe_num(pl_sc["Piece_Value_Rs"])
        actual_exp = pl_sc.groupby("Challan_No")["Piece_Value_Rs"].sum().reset_index()
        actual_exp.columns = ["Challan_No", "Actual_Labour_Rs"]
        actual_exp["Challan_No"] = actual_exp["Challan_No"].astype(str)
        cm_sc["Challan_No"] = cm_sc["Challan_No"].astype(str)
        cm_sc = cm_sc.merge(actual_exp, on="Challan_No", how="left").fillna({"Actual_Labour_Rs": 0})
    else:
        cm_sc["Actual_Labour_Rs"] = 0

    # Show costing on received qty when material is in; keep ordered totals for reference.
    cm_sc["Party_Value_Ordered_Rs"] = (cm_sc["Rate_Per_Pc"] * cm_sc["Total_Qty"]).round(2)
    cm_sc["Party_Value_Received_Rs"] = (cm_sc["Rate_Per_Pc"] * cm_sc["Received_Qty"]).round(2)
    cm_sc["Target_Labour_Ordered_Rs"] = (cm_sc["Target_Labour_Rate_Pc"] * cm_sc["Total_Qty"]).round(2)
    cm_sc["Target_Labour_Received_Rs"] = (cm_sc["Target_Labour_Rate_Pc"] * cm_sc["Received_Qty"]).round(2)
    has_recv = cm_sc["Received_Qty"] > 0
    cm_sc["Party_Value_Rs"] = cm_sc["Party_Value_Received_Rs"].where(
        has_recv, cm_sc["Party_Value_Ordered_Rs"]
    ).round(2)
    cm_sc["Target_Labour_Rs"] = cm_sc["Target_Labour_Received_Rs"].where(
        has_recv, cm_sc["Target_Labour_Ordered_Rs"]
    ).round(2)
    cm_sc["Total_Expense_Rs"] = (cm_sc["Actual_Labour_Rs"] + cm_sc["Deposit_Rs"]).round(2)
    cm_sc["PL_Rs"] = (cm_sc["Party_Value_Rs"] - cm_sc["Total_Expense_Rs"]).round(2)
    costing_qty = cm_sc["Received_Qty"].where(has_recv, cm_sc["Total_Qty"]).replace(0, 1)
    cm_sc["PL_Per_Pc"] = (cm_sc["PL_Rs"] / costing_qty).round(2)
    cm_sc["Margin_%"] = (cm_sc["PL_Rs"] / cm_sc["Party_Value_Rs"].replace(0, 1) * 100).round(1)
    cm_sc["Cost_vs_Target_%"] = (
        cm_sc["Actual_Labour_Rs"] / cm_sc["Target_Labour_Rs"].replace(0, 1) * 100
    ).round(1)
    recv_qty = safe_num(cm_sc["Received_Qty"]).astype(float)
    total_qty = safe_num(cm_sc["Total_Qty"]).astype(float)
    actual_lab = safe_num(cm_sc["Actual_Labour_Rs"]).astype(float)
    target_lab = safe_num(cm_sc["Target_Labour_Rs"]).astype(float)
    cm_sc["Actual_Cost"] = np.where(recv_qty > 0, np.round(actual_lab / recv_qty, 2), 0.0)
    cm_sc["Target_Cost"] = np.where(total_qty > 0, np.round(target_lab / total_qty, 2), 0.0)

    summary = {
        "challans": len(cm_sc),
        "pending": int(cm_sc["Is_Pending"].sum()),
        "party_value": float(cm_sc["Party_Value_Rs"].sum()),
        "actual_expense": float(cm_sc["Total_Expense_Rs"].sum()),
        "net_pl": float(cm_sc["PL_Rs"].sum()),
    }

    style_rollup = (
        cm_sc.groupby("Style")
        .agg(
            Challans=("Challan_No", "nunique"),
            Qty_Ordered=("Total_Qty", "sum"),
            Qty_Received=("Received_Qty", "sum"),
            Actual_Labour=("Actual_Labour_Rs", "sum"),
            Party_Value=("Party_Value_Rs", "sum"),
            Party_Value_Ordered=("Party_Value_Ordered_Rs", "sum"),
            Total_Expense=("Total_Expense_Rs", "sum"),
            PL=("PL_Rs", "sum"),
            Pending_Challans=("Is_Pending", "sum"),
        )
        .reset_index()
    )
    style_rollup["Qty"] = style_rollup["Qty_Received"].where(
        style_rollup["Qty_Received"] > 0, style_rollup["Qty_Ordered"]
    )
    style_rollup["Margin_%"] = (style_rollup["PL"] / style_rollup["Party_Value"].replace(0, 1) * 100).round(1)
    style_rollup["Result"] = style_rollup["PL"].apply(
        lambda x: "Profit" if x > 0 else ("Loss" if x < 0 else "Break-even")
    )

    rows = cm_sc.fillna("").to_dict(orient="records")
    return {
        "rows": rows,
        "summary": summary,
        "style_rollup": style_rollup.fillna("").to_dict(orient="records"),
        "pending": [r for r in rows if r.get("Is_Pending")],
        "completed": [r for r in rows if not r.get("Is_Pending")],
    }


def challan_detail_report(challan_no: str) -> dict[str, Any]:
    """Full challan snapshot: master, costing P&L, production log, and karigar expenses."""
    cn_raw = str(challan_no or "").strip()
    cn_key = clean_key(cn_raw)
    if not cn_key:
        return {"ok": False, "message": "Challan number required"}

    cm = get_sheet_df("challan_master")
    if cm.empty or "Challan_No" not in cm.columns:
        return {"ok": False, "message": f"Challan {cn_raw} not found"}
    hit = cm[cm["Challan_No"].astype(str).map(clean_key) == cn_key]
    if hit.empty:
        return {"ok": False, "message": f"Challan {cn_raw} not found"}
    master_row = hit.iloc[-1].fillna("").to_dict()

    costing_rows = style_costing_report().get("rows", [])
    costing = next(
        (r for r in costing_rows if clean_key(str(r.get("Challan_No", ""))) == cn_key),
        {},
    )

    pl = get_sheet_df("production_log")
    production_detail: list[dict[str, Any]] = []
    by_operation: list[dict[str, Any]] = []
    by_karigar: list[dict[str, Any]] = []
    prod_summary = {"pieces": 0, "piece_value_rs": 0.0, "karigars": 0, "operations": 0}

    if not pl.empty and "Challan_No" in pl.columns:
        work = pl[pl["Challan_No"].astype(str).map(clean_key) == cn_key].copy()
        if not work.empty:
            work = _production_log_latest_rows(work)
            for c in ("Total_Pieces", "Piece_Value_Rs", "Budgeted_Expense_Rs", "Actual_Expense_Rs", "PL_Rs"):
                if c in work.columns:
                    work[c] = safe_num(work[c])
            snap = challan_snapshot(cn_raw)
            if "Challan_Party" not in work.columns:
                work["Challan_Party"] = snap.get("Party", "")
            if "Challan_Description" not in work.columns:
                work["Challan_Description"] = snap.get("Challan_Description", "")

            prod_summary = {
                "pieces": int(safe_num(work.get("Total_Pieces", 0)).sum()),
                "piece_value_rs": round(float(safe_num(work.get("Piece_Value_Rs", 0)).sum()), 2),
                "karigars": int(work["Karigar_ID"].nunique()) if "Karigar_ID" in work.columns else 0,
                "operations": int(work["Operation"].nunique()) if "Operation" in work.columns else 0,
            }

            detail_cols = [
                c
                for c in [
                    "Date",
                    "Karigar_ID",
                    "Karigar_Name",
                    "Operation",
                    "Style",
                    "Total_Pieces",
                    "Avg_Efficiency_%",
                    "Piece_Value_Rs",
                    "Budgeted_Expense_Rs",
                    "Actual_Expense_Rs",
                    "PL_Rs",
                    "Challan_Party",
                    "Challan_Description",
                ]
                if c in work.columns
            ]
            production_detail = work[detail_cols].fillna("").to_dict(orient="records")

            if "Operation" in work.columns:
                op = work.copy()
                op["_op_norm"] = op["Operation"].apply(normalize_operation_name)
                by_operation = (
                    op.groupby("_op_norm", as_index=False)
                    .agg(
                        Operation=("_op_norm", "first"),
                        Pieces=("Total_Pieces", "sum"),
                        Piece_Value_Rs=("Piece_Value_Rs", "sum"),
                        PL_Rs=("PL_Rs", "sum"),
                    )
                    .fillna("")
                    .to_dict(orient="records")
                )

            if "Karigar_ID" in work.columns:
                kg = work.copy()
                kg["_kid"] = kg["Karigar_ID"].apply(clean_key)
                by_karigar = (
                    kg.groupby("_kid", as_index=False)
                    .agg(
                        Karigar_ID=("_kid", "first"),
                        Karigar_Name=("Karigar_Name", "first"),
                        Pieces=("Total_Pieces", "sum"),
                        Piece_Value_Rs=("Piece_Value_Rs", "sum"),
                        PL_Rs=("PL_Rs", "sum"),
                    )
                    .fillna("")
                    .to_dict(orient="records")
                )

    exp = get_sheet_df("karigar_expenses")
    expenses: list[dict[str, Any]] = []
    expense_total = 0.0
    if not exp.empty and "Challan_No" in exp.columns:
        ex = exp[exp["Challan_No"].astype(str).map(clean_key) == cn_key].copy()
        if not ex.empty and "Amount_Rs" in ex.columns:
            ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
            expense_total = round(float(ex["Amount_Rs"].sum()), 2)
            exp_cols = [
                c
                for c in [
                    "Date",
                    "Karigar_ID",
                    "Karigar_Name",
                    "Task_Type",
                    "Description",
                    "Amount_Rs",
                    "Style",
                ]
                if c in ex.columns
            ]
            sort_col = "Date" if "Date" in ex.columns else exp_cols[0]
            expenses = ex.sort_values(sort_col, ascending=False)[exp_cols].fillna("").to_dict(orient="records")

    return {
        "ok": True,
        "challan_no": cn_raw,
        "master": master_row,
        "costing": costing,
        "production": {
            "summary": prod_summary,
            "by_operation": by_operation,
            "by_karigar": by_karigar,
            "detail": production_detail,
        },
        "expenses": expenses,
        "expense_total_rs": expense_total,
    }


def record_challan_deposit(
    challan_no: str,
    deposit_date: str,
    qty: int,
    deposit_rs: float = 0.0,
) -> dict[str, Any]:
    """Log a material deposit with date and update challan received qty."""
    cn_raw = str(challan_no or "").strip()
    cn_key = clean_key(cn_raw)
    if not cn_key:
        return {"ok": False, "message": "Challan number required"}
    dep_date = str(deposit_date or "").strip()[:10]
    if not dep_date:
        return {"ok": False, "message": "Deposit date is required"}
    add_qty = max(0, int(qty))
    if add_qty <= 0:
        return {"ok": False, "message": "Enter quantity deposited"}

    cm = get_sheet_df("challan_master")
    if cm.empty or "Challan_No" not in cm.columns:
        return {"ok": False, "message": f"Challan {cn_raw} not found"}
    mask = cm["Challan_No"].astype(str).map(clean_key) == cn_key
    if not mask.any():
        return {"ok": False, "message": f"Challan {cn_raw} not found"}
    idx = cm[mask].index[-1]
    total = int(safe_num(pd.Series([cm.at[idx, "Total_Qty"]])).iloc[0])
    received = int(safe_num(pd.Series([cm.at[idx, "Received_Qty"]])).iloc[0]) if "Received_Qty" in cm.columns else 0
    pending = max(0, total - received)
    if pending <= 0:
        return {"ok": False, "message": "Challan is already fully received"}
    new_received = min(total, received + add_qty)
    add_rs = max(0.0, float(deposit_rs or 0))
    current_deposit = float(cm.at[idx, "Deposit_Rs"] or 0) if "Deposit_Rs" in cm.columns else 0.0
    cm.at[idx, "Received_Qty"] = new_received
    if "Deposit_Rs" in cm.columns:
        cm.at[idx, "Deposit_Rs"] = round(current_deposit + add_rs, 2)
    save_sheet_df("challan_master", cm)

    log = get_sheet_df("challan_deposit_log")
    entry = {
        "Challan_No": str(cm.at[idx, "Challan_No"]),
        "Style": str(cm.at[idx, "Style"] if "Style" in cm.columns else ""),
        "Party": str(cm.at[idx, "Party"] if "Party" in cm.columns else ""),
        "Deposit_Date": dep_date,
        "Qty": add_qty,
        "Deposit_Rs": round(add_rs, 2),
        "Received_After": new_received,
        "Recorded_At": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }
    log = pd.concat([log, pd.DataFrame([entry])], ignore_index=True) if not log.empty else pd.DataFrame([entry])
    save_sheet_df("challan_deposit_log", log)
    return {
        "ok": True,
        "message": f"Challan {cn_raw}: received {new_received} of {total} (+{add_qty} on {dep_date}).",
        "received_qty": new_received,
        "deposit_entry": entry,
    }


def challan_deposit_summary(date_from: str, date_to: str) -> dict[str, Any]:
    """Pieces deposited per challan in a date range (for weekly/monthly reports)."""
    log = get_sheet_df("challan_deposit_log")
    if log.empty:
        return {"date_from": date_from, "date_to": date_to, "rows": [], "summary": {"total_qty": 0, "deposit_count": 0}}
    work = log.copy()
    work["Deposit_Date_dt"] = pd.to_datetime(work["Deposit_Date"], errors="coerce")
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    work = work[(work["Deposit_Date_dt"] >= d0) & (work["Deposit_Date_dt"] <= d1)]
    if work.empty:
        return {"date_from": date_from, "date_to": date_to, "rows": [], "summary": {"total_qty": 0, "deposit_count": 0}}
    work["Qty"] = safe_num(work["Qty"])
    work["Deposit_Rs"] = safe_num(work.get("Deposit_Rs", 0))
    by_challan = (
        work.groupby(work["Challan_No"].astype(str).str.strip(), as_index=False)
        .agg(
            Challan_No=("Challan_No", "first"),
            Style=("Style", "first"),
            Party=("Party", "first"),
            Deposits=("Qty", "count"),
            Total_Qty_Deposited=("Qty", "sum"),
            Total_Deposit_Rs=("Deposit_Rs", "sum"),
            First_Deposit=("Deposit_Date", "min"),
            Last_Deposit=("Deposit_Date", "max"),
        )
    )
    detail = work.sort_values("Deposit_Date", ascending=False).fillna("").to_dict(orient="records")
    rows = by_challan.fillna("").to_dict(orient="records")
    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "detail": detail,
        "summary": {
            "deposit_count": int(len(work)),
            "total_qty": int(work["Qty"].sum()),
            "total_deposit_rs": round(float(work["Deposit_Rs"].sum()), 2),
            "challan_count": len(rows),
        },
    }


def karigar_detail_report(karigar_id: str, date_from: str, date_to: str) -> dict[str, Any]:
    """Monthly karigar snapshot: master, attendance, production, payroll, expenses."""
    kid = clean_key(karigar_id)
    if not kid:
        return {"ok": False, "message": "Karigar ID required"}

    master: dict[str, Any] = {}
    km = get_sheet_df("karigar_master")
    if not km.empty and "Karigar_ID" in km.columns:
        hit = km[km["Karigar_ID"].astype(str).map(clean_key) == kid]
        if not hit.empty:
            master = hit.iloc[-1].fillna("").to_dict()

    att_rows: list[dict[str, Any]] = []
    att = get_sheet_df("karigar_attendance")
    if not att.empty and "E_Code" in att.columns:
        aw = att[att["E_Code"].astype(str).map(clean_key) == kid].copy()
        if not aw.empty and "Date" in aw.columns:
            aw["Date_dt"] = pd.to_datetime(aw["Date"], errors="coerce")
            d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
            aw = aw[(aw["Date_dt"] >= d0) & (aw["Date_dt"] <= d1)]
            att_rows = aw.sort_values("Date", ascending=False).fillna("").to_dict(orient="records")

    prod_rows: list[dict[str, Any]] = []
    prod_summary = {"pieces": 0, "piece_value_rs": 0.0, "operations": 0, "challans": 0}
    work = _production_log_in_range(date_from, date_to)
    if not work.empty and "Karigar_ID" in work.columns:
        pw = work[work["Karigar_ID"].astype(str).map(clean_key) == kid].copy()
        if not pw.empty:
            for c in ("Total_Pieces", "Piece_Value_Rs", "PL_Rs"):
                if c in pw.columns:
                    pw[c] = safe_num(pw[c])
            prod_summary = {
                "pieces": int(safe_num(pw.get("Total_Pieces", 0)).sum()),
                "piece_value_rs": round(float(safe_num(pw.get("Piece_Value_Rs", 0)).sum()), 2),
                "operations": int(pw["Operation"].nunique()) if "Operation" in pw.columns else 0,
                "challans": int(pw["Challan_No"].nunique()) if "Challan_No" in pw.columns else 0,
            }
            prod_cols = [
                c for c in [
                    "Date", "Challan_No", "Style", "Operation", "Total_Pieces",
                    "Piece_Value_Rs", "PL_Rs", "Avg_Efficiency_%",
                ] if c in pw.columns
            ]
            prod_rows = pw.sort_values("Date", ascending=False)[prod_cols].fillna("").to_dict(orient="records")

    payroll_rows = payroll_report(date_from, date_to).get("rows", [])
    payroll = next((r for r in payroll_rows if clean_key(str(r.get("Karigar_ID", ""))) == kid), {})

    exp_rows: list[dict[str, Any]] = []
    exp_total = 0.0
    exp = get_sheet_df("karigar_expenses")
    if not exp.empty and "Karigar_ID" in exp.columns:
        ex = exp[exp["Karigar_ID"].astype(str).map(clean_key) == kid].copy()
        if not ex.empty and "Date" in ex.columns:
            ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
            d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
            ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
            if not ex.empty and "Amount_Rs" in ex.columns:
                ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
                exp_total = round(float(ex["Amount_Rs"].sum()), 2)
                exp_cols = [c for c in ["Date", "Challan_No", "Style", "Task_Type", "Description", "Amount_Rs"] if c in ex.columns]
                exp_rows = ex.sort_values("Date", ascending=False)[exp_cols].fillna("").to_dict(orient="records")

    piece_val = float(prod_summary.get("piece_value_rs", 0))
    paid = float(payroll.get("Total", 0) or 0)
    pl = round(piece_val - paid - exp_total, 2)
    status = "Profit" if pl > 0.01 else ("Loss" if pl < -0.01 else "Break-even")

    return {
        "ok": True,
        "karigar_id": kid,
        "date_from": date_from,
        "date_to": date_to,
        "master": master,
        "summary": {
            "days_worked": int(payroll.get("Days", 0) or 0),
            "total_payroll": round(paid, 2),
            "pieces": prod_summary["pieces"],
            "piece_value_rs": prod_summary["piece_value_rs"],
            "operations": prod_summary["operations"],
            "challans": prod_summary["challans"],
            "expense_rs": exp_total,
            "profit_loss_rs": pl,
            "status": status,
        },
        "attendance": att_rows,
        "production": prod_rows,
        "payroll": payroll,
        "expenses": exp_rows,
    }


STYLE_MASTER_COLUMN_SETS: dict[str, list[str]] = {
    "master_operations": ["Operation", "Target", "Rate_Rs"],
    "production_by_operation": [
        "Operation",
        "Pieces",
        "Avg_Efficiency_%",
        "Piece_Value_Rs",
        "Budgeted_Rs",
        "Actual_Rs",
        "PL_Rs",
        "Sessions",
    ],
    "production_by_karigar": [
        "Karigar_ID",
        "Karigar_Name",
        "Pieces",
        "Avg_Efficiency_%",
        "Piece_Value_Rs",
        "Budgeted_Rs",
        "Actual_Rs",
        "PL_Rs",
        "Sessions",
    ],
    "production_by_date": [
        "Date",
        "Pieces",
        "Piece_Value_Rs",
        "Budgeted_Rs",
        "Actual_Rs",
        "PL_Rs",
        "Karigars",
        "Sessions",
    ],
    "production_detail": [
        "Date",
        "Karigar_ID",
        "Karigar_Name",
        "Challan_No",
        "Operation",
        "Total_Pieces",
        "Efficiency_%",
        "Piece_Value_Rs",
        "Budgeted_Expense_Rs",
        "Actual_Expense_Rs",
        "PL_Rs",
    ],
    "costing_challans": [
        "Challan_No",
        "Party",
        "Total_Qty",
        "Received_Qty",
        "Pending",
        "Party_Value_Rs",
        "Actual_Labour_Rs",
        "Target_Labour_Rs",
        "Total_Expense_Rs",
        "PL_Rs",
        "Margin_%",
        "Date",
    ],
}


def _filter_df_by_style(df: pd.DataFrame, style: str) -> pd.DataFrame:
    if df.empty or "Style" not in df.columns:
        return df.iloc[0:0]
    sk = clean_key(style)
    return df[df["Style"].map(clean_key) == sk].copy()


def _filter_df_by_date(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    if df.empty or "Date" not in df.columns or (not date_from and not date_to):
        return df
    out = df.copy()
    out["Date_dt"] = pd.to_datetime(out["Date"], errors="coerce")
    if date_from:
        out = out[out["Date_dt"] >= pd.Timestamp(date_from)]
    if date_to:
        out = out[out["Date_dt"] <= pd.Timestamp(date_to)]
    return out.drop(columns=["Date_dt"], errors="ignore")


def _aggregate_production(pl_f: pd.DataFrame) -> dict[str, Any]:
    """Build production summary and breakdown tables for one style."""
    empty: dict[str, Any] = {
        "summary": {},
        "by_operation": [],
        "by_karigar": [],
        "by_date": [],
        "detail": [],
    }
    if pl_f.empty:
        return empty

    num_cols = [
        "Total_Pieces",
        "Efficiency_%",
        "Piece_Value_Rs",
        "Budgeted_Expense_Rs",
        "Actual_Expense_Rs",
        "PL_Rs",
    ]
    for c in num_cols:
        if c in pl_f.columns:
            pl_f[c] = safe_num(pl_f[c])

    summary = {
        "pieces": int(pl_f["Total_Pieces"].sum()) if "Total_Pieces" in pl_f.columns else 0,
        "piece_value_rs": round(float(pl_f["Piece_Value_Rs"].sum()), 2)
        if "Piece_Value_Rs" in pl_f.columns
        else 0.0,
        "budgeted_rs": round(float(pl_f["Budgeted_Expense_Rs"].sum()), 2)
        if "Budgeted_Expense_Rs" in pl_f.columns
        else 0.0,
        "actual_rs": round(float(pl_f["Actual_Expense_Rs"].sum()), 2)
        if "Actual_Expense_Rs" in pl_f.columns
        else 0.0,
        "pl_rs": round(float(pl_f["PL_Rs"].sum()), 2) if "PL_Rs" in pl_f.columns else 0.0,
        "avg_efficiency_pct": round(float(pl_f["Efficiency_%"].mean()), 1)
        if "Efficiency_%" in pl_f.columns
        else 0.0,
        "sessions": len(pl_f),
        "karigars": int(pl_f["Karigar_ID"].nunique()) if "Karigar_ID" in pl_f.columns else 0,
        "challans": int(pl_f["Challan_No"].nunique()) if "Challan_No" in pl_f.columns else 0,
        "date_first": str(pl_f["Date"].min()) if "Date" in pl_f.columns else "",
        "date_last": str(pl_f["Date"].max()) if "Date" in pl_f.columns else "",
    }

    agg_map: dict[str, tuple[str, str]] = {}
    if "Total_Pieces" in pl_f.columns:
        agg_map["Pieces"] = ("Total_Pieces", "sum")
    if "Efficiency_%" in pl_f.columns:
        agg_map["Avg_Efficiency_%"] = ("Efficiency_%", "mean")
    if "Piece_Value_Rs" in pl_f.columns:
        agg_map["Piece_Value_Rs"] = ("Piece_Value_Rs", "sum")
    if "Budgeted_Expense_Rs" in pl_f.columns:
        agg_map["Budgeted_Rs"] = ("Budgeted_Expense_Rs", "sum")
    if "Actual_Expense_Rs" in pl_f.columns:
        agg_map["Actual_Rs"] = ("Actual_Expense_Rs", "sum")
    if "PL_Rs" in pl_f.columns:
        agg_map["PL_Rs"] = ("PL_Rs", "sum")
    if "Operation" in pl_f.columns:
        agg_map["Sessions"] = ("Operation", "count")

    def _group_table(group_col: str, extra_cols: list[str] | None = None) -> list[dict]:
        if group_col not in pl_f.columns or not agg_map:
            return []
        g = pl_f.groupby(group_col).agg(**agg_map).round(2).reset_index()
        if extra_cols:
            for col in extra_cols:
                if col in pl_f.columns and col != group_col:
                    firsts = pl_f.groupby(group_col)[col].first().reset_index()
                    g = g.merge(firsts, on=group_col, how="left")
        return g.fillna("").to_dict(orient="records")

    by_operation = _group_table("Operation")
    by_karigar = _group_table("Karigar_ID", ["Karigar_Name"])
    by_date = _group_table("Date")
    if by_date and "Karigar_ID" in pl_f.columns:
        k_per_day = pl_f.groupby("Date")["Karigar_ID"].nunique().reset_index()
        k_per_day.columns = ["Date", "Karigars"]
        by_date_df = pd.DataFrame(by_date).merge(k_per_day, on="Date", how="left")
        by_date = by_date_df.fillna("").to_dict(orient="records")

    detail_cols = [c for c in STYLE_MASTER_COLUMN_SETS["production_detail"] if c in pl_f.columns]
    detail = (
        pl_f[detail_cols].sort_values(["Date", "Karigar_ID"], ascending=[False, True]).fillna("").head(200).to_dict(
            orient="records"
        )
        if detail_cols
        else []
    )

    return {
        "summary": summary,
        "by_operation": by_operation,
        "by_karigar": by_karigar,
        "by_date": by_date,
        "detail": detail,
    }


def style_master_report(
    style: str,
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    view: str = "full",
    include_production_detail: bool = False,
) -> dict:
    """Per-style report for Master Data: operations, production totals, challan costing."""
    style = str(style).strip()
    if not style:
        return {"ok": False, "message": "Style required"}

    view = (view or "full").strip().lower()
    if view not in ("full", "summary", "master", "production", "costing"):
        view = "full"

    sm = get_sheet_df("style_master")
    pl = get_sheet_df("production_log")

    sm_f = _filter_df_by_style(sm, style)
    master_ops: list[dict] = []
    master_totals = {
        "operation_count": 0,
        "sum_target": 0,
        "sum_rate_rs": 0.0,
        "benchmark_labour_per_piece_rs": 0.0,
        "benchmark_daily_budget_rs": BENCHMARK_DAILY_RATE_RS,
    }
    if not sm_f.empty:
        sm_f = sm_f.copy()
        sm_f["Target"] = safe_num(sm_f["Target"])
        sm_f["Rate_Rs"] = safe_num(sm_f["Rate_Rs"])
        if "Operation" in sm_f.columns:
            sm_f["Operation"] = sm_f["Operation"].map(normalize_operation_name)
        master_ops = sm_f[["Operation", "Target", "Rate_Rs"]].fillna("").to_dict(orient="records")
        rate_sum = float(sm_f["Rate_Rs"].sum())
        master_totals = {
            "operation_count": len(sm_f),
            "sum_target": int(sm_f["Target"].sum()),
            "sum_rate_rs": round(rate_sum, 2),
            "benchmark_labour_per_piece_rs": round(rate_sum, 2),
            "benchmark_daily_budget_rs": BENCHMARK_DAILY_RATE_RS,
        }

    pl_f = _filter_df_by_date(_filter_df_by_style(pl, style), date_from, date_to)
    production = _aggregate_production(pl_f)
    if not include_production_detail:
        production["detail"] = []

    month = "All"
    if date_from and date_to and date_from[:7] == date_to[:7]:
        month = date_from[:7]
    costing_rep = style_costing_report(month=month, style=style, party="All")
    costing_rows = costing_rep.get("rows") or []
    if date_from or date_to:
        filtered = []
        for row in costing_rows:
            d = str(row.get("Date") or "")
            if date_from and d and d < date_from:
                continue
            if date_to and d and d > date_to:
                continue
            filtered.append(row)
        costing_rows = filtered
        costing_summary = {
            "challans": len(costing_rows),
            "pending": sum(1 for r in costing_rows if r.get("Is_Pending")),
            "party_value": round(sum(float(r.get("Party_Value_Rs") or 0) for r in costing_rows), 2),
            "actual_expense": round(sum(float(r.get("Total_Expense_Rs") or 0) for r in costing_rows), 2),
            "net_pl": round(sum(float(r.get("PL_Rs") or 0) for r in costing_rows), 2),
            "target_labour": round(sum(float(r.get("Target_Labour_Rs") or 0) for r in costing_rows), 2),
            "actual_labour": round(sum(float(r.get("Actual_Labour_Rs") or 0) for r in costing_rows), 2),
        }
    else:
        costing_summary = dict(costing_rep.get("summary") or {})
        costing_summary.setdefault(
            "target_labour",
            round(sum(float(r.get("Target_Labour_Rs") or 0) for r in costing_rows), 2),
        )
        costing_summary.setdefault(
            "actual_labour",
            round(sum(float(r.get("Actual_Labour_Rs") or 0) for r in costing_rows), 2),
        )

    prod_s = production.get("summary") or {}
    totals = {
        "master_operations": master_totals["operation_count"],
        "labour_rate_per_piece_rs": master_totals["benchmark_labour_per_piece_rs"],
        "production_pieces": prod_s.get("pieces", 0),
        "production_piece_value_rs": prod_s.get("piece_value_rs", 0),
        "production_pl_rs": prod_s.get("pl_rs", 0),
        "costing_party_value_rs": costing_summary.get("party_value", 0),
        "costing_net_pl_rs": costing_summary.get("net_pl", 0),
        "costing_challans": costing_summary.get("challans", 0),
    }

    out: dict[str, Any] = {
        "ok": True,
        "style": style,
        "date_from": date_from or "",
        "date_to": date_to or "",
        "view": view,
        "totals": totals,
        "column_sets": STYLE_MASTER_COLUMN_SETS,
    }
    if view in ("full", "summary", "master"):
        out["master"] = {"operations": master_ops, "totals": master_totals}
    if view in ("full", "summary", "production"):
        out["production"] = production
    if view in ("full", "summary", "costing"):
        out["costing"] = {"summary": costing_summary, "challans": costing_rows}
    return out


def efficiency_report(date_from: str, date_to: str, styles: list[str] | None = None) -> dict:
    pl = get_sheet_df("production_log")
    if pl.empty:
        return {"karigar_wise": [], "operation_wise": [], "metrics": {}}
    df = pl.copy()
    for c in ["Total_Pieces", "Target", "Efficiency_%", "Piece_Value_Rs"]:
        if c in df.columns:
            df[c] = safe_num(df[c])
    df["Date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    mask = (df["Date_dt"] >= pd.Timestamp(date_from)) & (df["Date_dt"] <= pd.Timestamp(date_to))
    if styles:
        mask &= df["Style"].isin(styles)
    df_f = df[mask]
    if df_f.empty:
        return {"karigar_wise": [], "operation_wise": [], "metrics": {}}

    ke = (
        df_f.groupby("Karigar_Name")
        .agg(Avg_Eff=("Efficiency_%", "mean"), Pieces=("Total_Pieces", "sum"), Value=("Piece_Value_Rs", "sum"), Ops=("Operation", "count"))
        .round(2)
        .reset_index()
    )
    ke["Grade"] = ke["Avg_Eff"].apply(
        lambda x: "A" if x >= 100 else ("B" if x >= 85 else ("C" if x >= 70 else "D"))
    )
    oe = (
        df_f.groupby("Operation")
        .agg(Avg_Eff=("Efficiency_%", "mean"), Pieces=("Total_Pieces", "sum"), Value=("Piece_Value_Rs", "sum"))
        .round(2)
        .reset_index()
        .sort_values("Avg_Eff")
    )
    return {
        "metrics": {
            "avg_efficiency": round(float(df_f["Efficiency_%"].mean()), 1),
            "total_piece_value": round(float(df_f["Piece_Value_Rs"].sum()), 2),
            "total_pieces": int(df_f["Total_Pieces"].sum()),
        },
        "karigar_wise": ke.to_dict(orient="records"),
        "operation_wise": oe.to_dict(orient="records"),
        "bottlenecks": oe[oe["Avg_Eff"] < 80]["Operation"].tolist() if not oe.empty else [],
    }


def payroll_report(date_from: str, date_to: str) -> dict:
    """Attendance pay + other-work expenses (part change, alter, trainee, etc.)."""
    att = get_sheet_df("karigar_attendance")
    exp = get_sheet_df("karigar_expenses")
    if att.empty and exp.empty:
        return {"rows": [], "total_payroll": 0, "total_attendance": 0, "total_other_work": 0}

    frames: list[pd.DataFrame] = []
    if not att.empty:
        ap = att.copy()
        ap["Date_dt"] = pd.to_datetime(ap["Date"], errors="coerce")
        ap = ap[(ap["Date_dt"] >= pd.Timestamp(date_from)) & (ap["Date_dt"] <= pd.Timestamp(date_to))]
        if not ap.empty:
            for c in ["Payable_Hrs", "Normal_Pay", "OT_Hours", "OT_Pay", "Total_Pay"]:
                if c not in ap.columns:
                    ap[c] = 0
                ap[c] = safe_num(ap[c])
            pr = (
                ap.groupby("E_Code")
                .agg(
                    Name=("Name", "first"),
                    Days=("Date", "nunique"),
                    Hrs=("Payable_Hrs", "sum"),
                    Normal=("Normal_Pay", "sum"),
                    OT_Hrs=("OT_Hours", "sum"),
                    OT_Pay=("OT_Pay", "sum"),
                    Attendance_Total=("Total_Pay", "sum"),
                )
                .round(2)
                .reset_index()
            )
            pr = pr.rename(columns={"E_Code": "Karigar_ID"})
            frames.append(pr)

    if not exp.empty:
        ex = exp.copy()
        ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
        ex = ex[(ex["Date_dt"] >= pd.Timestamp(date_from)) & (ex["Date_dt"] <= pd.Timestamp(date_to))]
        if not ex.empty:
            for c in ["Hours", "Amount_Rs"]:
                if c in ex.columns:
                    ex[c] = safe_num(ex[c])
            er = (
                ex.groupby("Karigar_ID")
                .agg(
                    Name=("Karigar_Name", "first"),
                    Expense_Days=("Date", "nunique"),
                    Other_Hours=("Hours", "sum"),
                    Other_Work_Pay=("Amount_Rs", "sum"),
                )
                .round(2)
                .reset_index()
            )
            frames.append(er)

    if not frames:
        return {"rows": [], "total_payroll": 0, "total_attendance": 0, "total_other_work": 0}

    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="Karigar_ID", how="outer", suffixes=("", "_y"))
    if "Name_y" in merged.columns:
        merged["Name"] = merged["Name"].combine_first(merged["Name_y"])
        merged = merged.drop(columns=["Name_y"], errors="ignore")
    merged = merged.fillna(0)
    for col in ("Days", "Hrs", "Normal", "OT_Hrs", "OT_Pay", "Attendance_Total", "Expense_Days", "Other_Hours", "Other_Work_Pay"):
        if col not in merged.columns:
            merged[col] = 0
    merged["Days"] = merged[["Days", "Expense_Days"]].max(axis=1)
    merged["Total_Payroll"] = (
        safe_num(merged["Attendance_Total"]) + safe_num(merged["Other_Work_Pay"])
    ).round(2)
    merged = merged.rename(
        columns={
            "Attendance_Total": "Attendance_Pay",
            "OT_Pay": "OT_Pay",
            "Normal": "Normal",
            "Total_Payroll": "Total",
        }
    )
    out_cols = [
        "Karigar_ID",
        "Name",
        "Days",
        "Hrs",
        "Normal",
        "OT_Hrs",
        "OT_Pay",
        "Attendance_Pay",
        "Other_Hours",
        "Other_Work_Pay",
        "Total",
    ]
    merged["Total"] = safe_num(merged.get("Total", 0)).fillna(0)
    # Filter out zero rows (requested): don't show people with 0 payroll in the range.
    merged = merged[merged["Total"] > 0].copy()
    rows = merged[[c for c in out_cols if c in merged.columns]].fillna("").to_dict(orient="records")
    return {
        "rows": rows,
        "total_payroll": float(safe_num(merged["Total"]).sum()),
        "total_attendance": float(safe_num(merged.get("Attendance_Pay", 0)).sum()),
        "total_other_work": float(safe_num(merged.get("Other_Work_Pay", 0)).sum()),
    }


def export_all_zip_bytes() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k in DATA_KEYS:
            df = get_sheet_df(k)
            zf.writestr(f"{k}.csv", df.to_csv(index=False))
    buf.seek(0)
    return buf.getvalue()


def import_zip_bytes(zb: bytes) -> dict:
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        for nm in zf.namelist():
            base = os.path.basename(nm)
            if not base.lower().endswith(".csv"):
                continue
            k = base[:-4]
            if k in DATA_KEYS:
                df = pd.read_csv(io.StringIO(zf.read(nm).decode()))
                save_sheet_df(k, df)
    return {"ok": True, "message": "All data restored"}


def hour_labels(for_date: str | None = None) -> list[dict]:
    from .karigar_attendance import production_hour_cols_for_date

    cols = production_hour_cols_for_date(for_date) if for_date else HOUR_COLS
    allowed = set(cols)
    return [{"col": c, "label": l} for c, l in zip(HOUR_COLS, HOUR_LBLS) if c in allowed]


def _get_daily_salary(karigar_id: str, as_of_date: str | None = None) -> float:
    """Resolve karigar daily rate for a calendar date (rate history + master fallback)."""
    return get_daily_rate_for_date(karigar_id, as_of_date)


def build_daily_rate_map(as_of_date: str | None = None) -> dict[str, float]:
    """Resolve daily rates for all karigars in one pass (dashboard / payroll)."""
    as_of = (as_of_date or str(date.today()))[:10]
    cutoff = pd.to_datetime(as_of, errors="coerce")
    rates: dict[str, float] = {}

    km = get_sheet_df("karigar_master")
    if not km.empty and "Karigar_ID" in km.columns:
        kids = km["Karigar_ID"].map(clean_key)
        vals = safe_num(km["Daily_Rate_Rs"]) if "Daily_Rate_Rs" in km.columns else pd.Series(0, index=km.index)
        for kid, val in zip(kids, vals, strict=False):
            if kid:
                rates[str(kid)] = float(val)

    em = get_sheet_df("employee_master")
    if not em.empty and "E_Code" in em.columns:
        kids = em["E_Code"].map(clean_key)
        vals = safe_num(em["Daily_Rate_Rs"]) if "Daily_Rate_Rs" in em.columns else pd.Series(0, index=em.index)
        for kid, val in zip(kids, vals, strict=False):
            if kid and kid not in rates:
                rates[str(kid)] = float(val)

    hist = get_sheet_df("karigar_rate_history")
    if (
        not hist.empty
        and "Karigar_ID" in hist.columns
        and "Effective_From" in hist.columns
        and pd.notna(cutoff)
    ):
        h = hist.copy()
        h["_kid"] = h["Karigar_ID"].apply(clean_key)
        h["_eff"] = pd.to_datetime(h["Effective_From"], errors="coerce")
        h = h[h["_eff"].notna() & (h["_eff"] <= cutoff)]
        if not h.empty:
            h = h.sort_values("_eff", ascending=False)
            for kid, sub in h.groupby("_kid", sort=False):
                rates[str(kid)] = float(safe_num(sub["Daily_Rate_Rs"]).iloc[0])

    return rates


def get_daily_rate_for_date(karigar_id: str, as_of_date: str | None = None) -> float:
    kid = clean_key(karigar_id)
    if not kid:
        return 0.0
    return float(build_daily_rate_map(as_of_date).get(kid, 0.0))


def _daily_salary_for_row(kid: str, date_str: str, row: pd.Series | None = None) -> float:
    daily = _get_daily_salary(kid, date_str)
    if daily <= 0 and row is not None and "Daily_Rate_Rs" in row.index:
        try:
            daily = float(row.get("Daily_Rate_Rs") or 0)
        except (TypeError, ValueError):
            daily = 0.0
    return max(float(daily), 0.0)


def _production_log_latest_rows(day_pl: pd.DataFrame) -> pd.DataFrame:
    """Keep only the newest save per karigar/challan/style/operation (hides legacy duplicates)."""
    if day_pl.empty:
        return day_pl
    work = day_pl.copy()
    work["_ck_date"] = work["Date"].apply(clean_key)
    work["_ck_kar"] = work["Karigar_ID"].apply(clean_key)
    work["_ck_challan"] = work["Challan_No"].apply(clean_key)
    work["_ck_style"] = work["Style"].apply(_style_session_key)
    work["_op_norm"] = work.get("Operation", pd.Series(dtype=str)).apply(normalize_operation_name)
    if "Save_Time" in work.columns:
        work = work.sort_values("Save_Time", ascending=False, na_position="last")
    work = work.drop_duplicates(
        subset=["_ck_date", "_ck_kar", "_ck_challan", "_ck_style", "_op_norm"],
        keep="first",
    )
    return work.drop(
        columns=["_ck_date", "_ck_kar", "_ck_challan", "_ck_style", "_op_norm"],
        errors="ignore",
    )


def production_entry_reports(date_str: str, karigar_id: str | None = None) -> dict:
    """Day reports + save history (Streamlit Production Entry bottom section)."""
    pl = get_sheet_df("production_log")
    empty = {
        "date": date_str,
        "karigar_id": karigar_id or "",
        "history": [],
        "recent_saves": [],
        "report1": [],
        "report2_hourly": [],
        "report2_summary": [],
        "grand_total": None,
    }
    if pl.empty:
        return empty

    day_pl = pl[pl["Date"].apply(clean_key) == clean_key(date_str)].copy()
    if karigar_id:
        day_pl = day_pl[day_pl["Karigar_ID"].apply(clean_key) == clean_key(karigar_id)]

    if day_pl.empty:
        return empty

    day_pl = _production_log_latest_rows(day_pl)

    for c in [
        "Total_Pieces",
        "Target",
        "Rate_Rs",
        "Efficiency_%",
        "Piece_Value_Rs",
        "Budgeted_Expense_Rs",
        "Actual_Expense_Rs",
        "PL_Rs",
    ]:
        if c in day_pl.columns:
            day_pl[c] = safe_num(day_pl[c])

    def _chal_party(row) -> str:
        p = str(row.get("Challan_Party") or "").strip()
        if p:
            return p
        return challan_snapshot(str(row.get("Challan_No") or "")).get("Party", "")

    def _chal_desc(row) -> str:
        d = str(row.get("Challan_Description") or "").strip()
        if d:
            return d
        return challan_snapshot(str(row.get("Challan_No") or "")).get("Challan_Description", "")

    if "Challan_Party" not in day_pl.columns:
        day_pl["Challan_Party"] = day_pl.apply(_chal_party, axis=1)
    if "Challan_Description" not in day_pl.columns:
        day_pl["Challan_Description"] = day_pl.apply(_chal_desc, axis=1)

    hist_cols = [
        c
        for c in [
            "Date",
            "Save_Time",
            "Saved_By_Name",
            "Karigar_Name",
            "Karigar_ID",
            "Challan_No",
            "Challan_Party",
            "Challan_Description",
            "Style",
            "Operation",
            "Total_Pieces",
            "Budget_Rate_Per_Piece",
            "Budgeted_Expense_Rs",
            "Actual_Rate_Per_Piece",
            "Actual_Expense_Rs",
            "PL_Rs",
            "Profit_Loss",
            "Efficiency_%",
        ]
        if c in day_pl.columns
    ]
    history = day_pl.copy()
    if "Save_Time" in history.columns:
        history = history.sort_values("Save_Time", ascending=False)
    if "Profit_Loss" not in history.columns and "PL_Rs" in history.columns:
        history["Profit_Loss"] = history["PL_Rs"]
    history_rows = history[hist_cols].fillna("").to_dict(orient="records") if hist_cols else []

    recent = history_rows[:5] if history_rows else []

    from .karigar_attendance import production_hour_cols_for_date

    valid_hour_cols = [
        h
        for h in production_hour_cols_for_date(date_str)
        if h != "H_13_14" and h in day_pl.columns
    ]

    def working_hours(row) -> int:
        return sum(1 for h in valid_hour_cols if safe_num(pd.Series([row.get(h, 0)])).iloc[0] > 0)

    report1_rows = []
    for _, row in day_pl.iterrows():
        wh = working_hours(row)
        base_target = float(row.get("Base_Target") or row.get("Target") or 0)
        applied_ltl = float(row.get("Applied_LTL") or row.get("Target") or 0)
        rate = float(row.get("Rate_Rs") or 0)
        kid = str(row.get("Karigar_ID", ""))
        daily_salary = _daily_salary_for_row(kid, date_str, row)
        from .karigar_attendance import hourly_rate_from_daily

        hourly_salary = round(hourly_rate_from_daily(daily_salary, date_str), 2)
        adj_target = round(applied_ltl * wh, 0)
        total_pcs = float(row.get("Total_Pieces") or 0)
        normal_target_pcs = int(round(base_target * wh))
        ltl_target_pcs = int(adj_target)
        efficiency = round(total_pcs / adj_target * 100, 1) if adj_target > 0 else 0.0
        normal_eff = round(total_pcs / normal_target_pcs * 100, 1) if normal_target_pcs > 0 else 0.0
        fin = _financial_from_log_row(row, date_str=date_str, kid=kid)

        hour_vals = {}
        for hcol, hlbl in zip(HOUR_COLS, HOUR_LBLS):
            if hcol in row.index and hcol != "H_13_14":
                v = int(safe_num(pd.Series([row.get(hcol, 0)])).iloc[0])
                if v > 0:
                    hour_vals[hlbl] = v

        report1_rows.append(
            {
                "Date": str(row.get("Date", date_str) or date_str),
                "Save_Time": str(row.get("Save_Time", "") or ""),
                "Karigar_Name": row.get("Karigar_Name", ""),
                "Karigar_ID": kid,
                "Challan_No": row.get("Challan_No", ""),
                "Challan_Party": row.get("Challan_Party", ""),
                "Challan_Description": row.get("Challan_Description", ""),
                "Style": row.get("Style", ""),
                "Operation": row.get("Operation", ""),
                "Base_Target": int(base_target),
                "Formula_LTL": int(float(row.get("Formula_LTL") or 0)),
                "Applied_LTL": int(applied_ltl),
                "LTL_Source": str(row.get("LTL_Source", "") or ""),
                "hours": hour_vals,
                "Working_Hours": wh,
                "Total_Pieces": int(total_pcs),
                "Adj_Target": int(adj_target),
                "Normal_Target_Pcs": normal_target_pcs,
                "LTL_Target_Pcs": ltl_target_pcs,
                "Normal_Variance_Pcs": int(total_pcs) - normal_target_pcs,
                "LTL_Variance_Pcs": int(total_pcs) - ltl_target_pcs,
                "Normal_Efficiency_%": normal_eff,
                "Efficiency_%": efficiency,
                "LTL_Efficiency_%": efficiency,
                "Daily_Salary_Rs": daily_salary,
                "Hourly_Salary_Rs": hourly_salary,
                "Budget_Rate_Per_Piece": fin["budget_rate_per_piece"],
                "Budgeted_Expense_Rs": fin["budgeted_amount"],
                "Actual_Rate_Per_Piece": fin["actual_rate_per_piece"],
                "Actual_Expense_Rs": fin["actual_amount"],
                "PL_Rs": fin["pl_rs"],
                "Profit_Loss": fin["pl_rs"],
                "Saved_By_Name": row.get("Saved_By_Name", ""),
            }
        )

    report2_rows = []
    for _, row in day_pl.iterrows():
        kid = str(row.get("Karigar_ID", ""))
        kar_name = str(row.get("Karigar_Name", ""))
        op_name = str(row.get("Operation", ""))
        rate_rs = float(row.get("Rate_Rs") or 0)
        applied_ltl = float(row.get("Applied_LTL") or row.get("Target") or 0)

        daily_rate = _daily_salary_for_row(kid, date_str, row)
        from .karigar_attendance import hourly_rate_from_daily

        hourly_sal = round(hourly_rate_from_daily(daily_rate, date_str), 2)

        for hcol, hlbl in zip(HOUR_COLS, HOUR_LBLS):
            if hcol == "H_13_14" or hcol not in row.index or hcol not in valid_hour_cols:
                continue
            pcs = int(safe_num(pd.Series([row.get(hcol, 0)])).iloc[0])
            if pcs <= 0:
                continue
            actual_piece_val = round(pcs * rate_rs, 2)
            target_piece_val = round(applied_ltl * rate_rs, 2)
            net_pl = round(actual_piece_val - hourly_sal, 2)
            report2_rows.append(
                {
                    "Date": str(row.get("Date", date_str) or date_str),
                    "Save_Time": str(row.get("Save_Time", "") or ""),
                    "Karigar": kar_name,
                    "Karigar_ID": kid,
                    "Challan_No": str(row.get("Challan_No", "")),
                    "Challan_Description": str(row.get("Challan_Description", "") or ""),
                    "Style": str(row.get("Style", "")),
                    "Hour": hlbl,
                    "Operation": op_name,
                    "Daily_Salary_Rs": daily_rate,
                    "Hourly_Salary_Rs": hourly_sal,
                    "Pieces_Done": pcs,
                    "Applied_LTL": int(applied_ltl),
                    "Hourly_Target_Pcs": int(applied_ltl),
                    "Actual_Piece_Val_Rs": actual_piece_val,
                    "Target_Piece_Val_Rs": target_piece_val,
                    "Net_PL_Rs": net_pl,
                    "Status": "Profit" if net_pl >= 0 else "Loss",
                }
            )

    report2_summary = []
    grand_total = None
    if report2_rows:
        r2_df = pd.DataFrame(report2_rows)
        r2_sum = (
            r2_df.groupby(["Karigar", "Operation"])
            .agg(
                Date=("Date", "first"),
                Save_Time=("Save_Time", "max"),
                Karigar_ID=("Karigar_ID", "first"),
                Challan_No=("Challan_No", "first"),
                Style=("Style", "first"),
                Hours_Worked=("Hour", "count"),
                Total_Pieces=("Pieces_Done", "sum"),
                Applied_LTL=("Applied_LTL", "first"),
                Hourly_Target_Pcs=("Hourly_Target_Pcs", "first"),
                Daily_Salary_Rs=("Daily_Salary_Rs", "first"),
                Total_Salary_Cost=("Hourly_Salary_Rs", "sum"),
                Total_Actual_Val=("Actual_Piece_Val_Rs", "sum"),
                Total_Target_Val=("Target_Piece_Val_Rs", "sum"),
                Total_Net_PL=("Net_PL_Rs", "sum"),
            )
            .round(2)
            .reset_index()
        )
        r2_sum["Avg_Pieces_Per_Hr"] = (
            r2_sum["Total_Pieces"] / r2_sum["Hours_Worked"].replace(0, 1)
        ).round(1)
        r2_sum["Efficiency_%"] = (
            r2_sum["Total_Pieces"]
            / (r2_sum["Hourly_Target_Pcs"] * r2_sum["Hours_Worked"]).replace(0, 1)
            * 100
        ).round(1)
        r2_sum["Result"] = r2_sum["Total_Net_PL"].apply(lambda x: "Profit" if x >= 0 else "Loss")
        report2_summary = r2_sum.fillna("").to_dict(orient="records")
        grand_total = {
            "total_salary_cost": round(float(r2_sum["Total_Salary_Cost"].sum()), 2),
            "total_actual_val": round(float(r2_sum["Total_Actual_Val"].sum()), 2),
            "total_target_val": round(float(r2_sum["Total_Target_Val"].sum()), 2),
            "total_net_pl": round(float(r2_sum["Total_Net_PL"].sum()), 2),
            "overall_profit": float(r2_sum["Total_Net_PL"].sum()) >= 0,
        }

    karigar_summary: list[dict[str, Any]] = []
    if report1_rows:
        by_k: dict[str, dict[str, Any]] = {}
        for r in report1_rows:
            kid = str(r.get("Karigar_ID", "")).strip()
            if not kid:
                continue
            if kid not in by_k:
                by_k[kid] = {
                    "Karigar_ID": kid,
                    "Karigar_Name": r.get("Karigar_Name", kid),
                    "Sessions": 0,
                    "Working_Hours": 0,
                    "Total_Pieces": 0,
                    "Normal_Target_Pcs": 0,
                    "LTL_Target_Pcs": 0,
                    "Normal_Variance_Pcs": 0,
                    "LTL_Variance_Pcs": 0,
                }
            agg = by_k[kid]
            agg["Sessions"] += 1
            agg["Working_Hours"] += int(r.get("Working_Hours") or 0)
            agg["Total_Pieces"] += int(r.get("Total_Pieces") or 0)
            agg["Normal_Target_Pcs"] += int(r.get("Normal_Target_Pcs") or 0)
            agg["LTL_Target_Pcs"] += int(r.get("LTL_Target_Pcs") or 0)
            agg["Normal_Variance_Pcs"] += int(r.get("Normal_Variance_Pcs") or 0)
            agg["LTL_Variance_Pcs"] += int(r.get("LTL_Variance_Pcs") or 0)
        for agg in by_k.values():
            tp = int(agg["Total_Pieces"])
            nt = int(agg["Normal_Target_Pcs"])
            lt = int(agg["LTL_Target_Pcs"])
            agg["Normal_Efficiency_%"] = round(tp / nt * 100, 1) if nt > 0 else 0.0
            agg["LTL_Efficiency_%"] = round(tp / lt * 100, 1) if lt > 0 else 0.0
            karigar_summary.append(agg)
        karigar_summary.sort(key=lambda x: str(x.get("Karigar_Name", "")))

    return {
        "date": date_str,
        "karigar_id": karigar_id or "",
        "history": history_rows,
        "recent_saves": recent,
        "report1": report1_rows,
        "report2_hourly": report2_rows,
        "report2_summary": report2_summary,
        "grand_total": grand_total,
        "karigar_summary": karigar_summary,
    }


def performance_report(date_from: str, date_to: str) -> dict:
    """Piece value vs salary — karigar performance (Streamlit Performance tab)."""
    pl = get_sheet_df("production_log")
    att = get_sheet_df("karigar_attendance")
    if pl.empty or att.empty:
        return {
            "ok": False,
            "message": "Need both production entries and attendance records.",
            "rows": [],
            "summary": {},
        }

    pl3 = pl.copy()
    for c in ["Total_Pieces", "Piece_Value_Rs", "Efficiency_%"]:
        if c in pl3.columns:
            pl3[c] = safe_num(pl3[c])
    pl3["Date_dt"] = pd.to_datetime(pl3["Date"], errors="coerce")
    pl3 = pl3[(pl3["Date_dt"] >= pd.Timestamp(date_from)) & (pl3["Date_dt"] <= pd.Timestamp(date_to))]
    if pl3.empty:
        return {"ok": True, "message": "No production in date range", "rows": [], "summary": {}}

    psm = (
        pl3.groupby("Karigar_ID")
        .agg(
            Piece_Value=("Piece_Value_Rs", "sum"),
            Total_Pieces=("Total_Pieces", "sum"),
            Avg_Eff=("Efficiency_%", "mean"),
        )
        .reset_index()
    )

    att3 = att.copy()
    if "Total_Pay" not in att3.columns:
        return {
            "ok": False,
            "message": "Attendance lacks salary data. Add punches in Attendance tab.",
            "rows": [],
            "summary": {},
        }
    att3["Date_dt"] = pd.to_datetime(att3["Date"], errors="coerce")
    att3 = att3[(att3["Date_dt"] >= pd.Timestamp(date_from)) & (att3["Date_dt"] <= pd.Timestamp(date_to))]
    for c in ["Total_Pay", "Payable_Hrs"]:
        if c in att3.columns:
            att3[c] = safe_num(att3[c])

    ss = (
        att3.groupby("E_Code")
        .agg(
            Name=("Name", "first"),
            Days=("Date", "nunique"),
            Hrs=("Payable_Hrs", "sum"),
            Salary=("Total_Pay", "sum"),
        )
        .round(2)
        .reset_index()
    )
    ss["E_Code"] = ss["E_Code"].astype(str)
    psm2 = psm.rename(columns={"Karigar_ID": "E_Code"}).copy()
    psm2["E_Code"] = psm2["E_Code"].astype(str)
    perf = ss.merge(psm2, on="E_Code", how="outer").fillna(0)
    perf["Piece_Value"] = perf["Piece_Value"].round(2)
    perf["Attendance_Pay"] = perf["Salary"].round(2)
    perf["Other_Work_Pay"] = 0.0
    try:
        pr = payroll_report(date_from, date_to)
        pr_df = pd.DataFrame(pr.get("rows") or [])
        if not pr_df.empty and "Karigar_ID" in pr_df.columns:
            pr_df["E_Code"] = pr_df["Karigar_ID"].astype(str)
            for c in ("Attendance_Pay", "Other_Work_Pay", "Total"):
                if c in pr_df.columns:
                    pr_df[c] = safe_num(pr_df[c])
            perf = perf.merge(
                pr_df[["E_Code", "Attendance_Pay", "Other_Work_Pay", "Total"]],
                on="E_Code",
                how="left",
                suffixes=("_att", ""),
            )
            if "Attendance_Pay_att" in perf.columns:
                perf["Attendance_Pay"] = perf["Attendance_Pay"].combine_first(perf["Attendance_Pay_att"]).fillna(0)
                perf = perf.drop(columns=["Attendance_Pay_att"], errors="ignore")
            perf["Other_Work_Pay"] = safe_num(perf.get("Other_Work_Pay", 0)).fillna(0)
            perf["Total_Payroll_Paid"] = safe_num(perf.get("Total", perf["Attendance_Pay"])).fillna(0).round(2)
        else:
            perf["Total_Payroll_Paid"] = perf["Attendance_Pay"].round(2)
    except Exception:
        perf["Total_Payroll_Paid"] = perf["Attendance_Pay"].round(2)
        perf["Other_Work_Pay"] = 0.0
    perf["Salary"] = perf["Total_Payroll_Paid"]
    perf["Surplus"] = (perf["Piece_Value"] - perf["Total_Payroll_Paid"]).round(2)
    perf["In_Production"] = perf["Total_Pieces"] > 0
    perf["In_Payroll"] = perf["Total_Payroll_Paid"] > 0
    perf["Payroll_Only_Expense"] = perf["In_Payroll"] & ~perf["In_Production"]
    perf["ROI_%"] = (perf["Piece_Value"] / perf["Total_Payroll_Paid"].replace(0, 1) * 100).round(1)
    perf["Avg_Eff"] = perf["Avg_Eff"].round(1)
    perf["Grade"] = perf["Avg_Eff"].apply(
        lambda x: "A–Excellent"
        if x >= 100
        else ("B–Good" if x >= 85 else ("C–Average" if x >= 70 else "D–Needs Improvement"))
    )

    summary = {
        "total_piece_value": float(perf["Piece_Value"].sum()),
        "total_salary": float(perf["Total_Payroll_Paid"].sum()),
        "total_attendance_pay": float(perf["Attendance_Pay"].sum()),
        "total_other_work_pay": float(perf["Other_Work_Pay"].sum()),
        "net_surplus": float(perf["Surplus"].sum()),
    }
    op_type_breakup: list[dict[str, Any]] = []
    try:
        if not pl3.empty and "Style" in pl3.columns and "Operation" in pl3.columns:
            sm = get_sheet_df("style_master")
            if not sm.empty:
                sm2 = sm.copy()
                if "Operation_Type" not in sm2.columns:
                    sm2["Operation_Type"] = "Medium"
                sm2["Operation"] = sm2["Operation"].map(normalize_operation_name)
                sm2["Style"] = sm2["Style"].astype(str).str.strip()
                sm2["Operation_Type"] = sm2["Operation_Type"].astype(str).str.strip().replace("", "Medium")
                p3 = pl3.copy()
                p3["Operation"] = p3["Operation"].map(normalize_operation_name)
                p3["Style"] = p3["Style"].astype(str).str.strip()
                pm = p3.merge(sm2[["Style", "Operation", "Operation_Type"]], on=["Style", "Operation"], how="left")
                pm["Operation_Type"] = pm["Operation_Type"].fillna("Medium")
                ot = (
                    pm.groupby("Operation_Type")
                    .agg(
                        Count=("Operation", "count"),
                        Pieces=("Total_Pieces", "sum"),
                        Piece_Value=("Piece_Value_Rs", "sum"),
                        Avg_Eff=("Efficiency_%", "mean"),
                    )
                    .reset_index()
                    .round(2)
                )
                op_type_breakup = ot.to_dict(orient="records")
    except Exception:
        op_type_breakup = []
    return {
        "ok": True,
        "rows": perf.fillna("").to_dict(orient="records"),
        "summary": summary,
        "operation_type_breakup": op_type_breakup,
    }


def comparison_dashboard_report(date_from: str, date_to: str) -> dict:
    """P&L comparison: karigar vs benchmark/LTL, SKU profit/loss, challan budget status."""
    pl = get_sheet_df("production_log")
    empty = {
        "date_from": date_from,
        "date_to": date_to,
        "summary": {
            "karigars_at_loss": 0,
            "karigars_at_profit": 0,
            "skus_at_loss": 0,
            "challans_over_budget": 0,
            "total_budgeted_rs": 0.0,
            "total_actual_rs": 0.0,
            "total_net_pl_rs": 0.0,
        },
        "karigar_comparison": [],
        "sku_comparison": [],
        "challan_comparison": [],
        "karigar_sku_detail": [],
    }
    if pl.empty:
        return empty

    work = pl.copy()
    work["Date_dt"] = pd.to_datetime(work["Date"], errors="coerce")
    d0 = pd.Timestamp(date_from)
    d1 = pd.Timestamp(date_to)
    work = work[(work["Date_dt"] >= d0) & (work["Date_dt"] <= d1)]
    if work.empty:
        return empty

    work = _production_log_latest_rows(work)
    fin_cols = ["Budgeted_Expense_Rs", "Actual_Expense_Rs", "PL_Rs", "Piece_Value_Rs", "Applied_LTL", "Formula_LTL"]

    def _agg_group(group: pd.DataFrame, key_name: str, key_val: str) -> dict:
        budgeted = float(safe_num(group.get("Budgeted_Expense_Rs", 0)).sum()) if "Budgeted_Expense_Rs" in group.columns else 0.0
        actual = float(safe_num(group.get("Actual_Expense_Rs", 0)).sum()) if "Actual_Expense_Rs" in group.columns else 0.0
        pl_rs = float(safe_num(group.get("PL_Rs", 0)).sum()) if "PL_Rs" in group.columns else budgeted - actual
        piece_val = float(safe_num(group.get("Piece_Value_Rs", 0)).sum()) if "Piece_Value_Rs" in group.columns else 0.0
        pieces = int(safe_num(group.get("Total_Pieces", 0)).sum()) if "Total_Pieces" in group.columns else 0
        if budgeted <= 0 and actual > 0:
            for _, r in group.iterrows():
                fin = _financial_from_log_row(r, date_str=str(r.get("Date", ""))[:10], kid=str(r.get("Karigar_ID", "")))
                budgeted += fin["budgeted_amount"]
                actual += fin["actual_amount"]
                pl_rs += fin["pl_rs"]
        running_ltl = 0.0
        if "Applied_LTL" in group.columns:
            running_ltl = float(safe_num(group["Applied_LTL"]).mean())
        return {
            key_name: key_val,
            "Pieces": pieces,
            "Piece_Value_Rs": round(piece_val, 2),
            "Budgeted_Rs": round(budgeted, 2),
            "Actual_Rs": round(actual, 2),
            "Net_PL_Rs": round(pl_rs, 2),
            "Running_LTL": round(running_ltl, 1),
            "Status": "Profit" if pl_rs >= 0 else "Loss",
            "Variance_%": round((actual / budgeted * 100) if budgeted > 0 else 0, 1),
        }

    karigar_comparison: list[dict] = []
    if "Karigar_ID" in work.columns:
        for kid, grp in work.groupby(work["Karigar_ID"].apply(clean_key)):
            if not kid:
                continue
            nm = str(grp.iloc[0].get("Karigar_Name", kid))
            row = _agg_group(grp, "Karigar_ID", str(kid))
            row["Karigar_Name"] = nm
            karigar_comparison.append(row)
    karigar_comparison.sort(key=lambda r: r["Net_PL_Rs"])

    sku_comparison: list[dict] = []
    if "Style" in work.columns:
        for st, grp in work.groupby(work["Style"].astype(str).str.strip()):
            if not st:
                continue
            sku_comparison.append(_agg_group(grp, "Style", st))
    sku_comparison.sort(key=lambda r: r["Net_PL_Rs"])

    challan_comparison: list[dict] = []
    if "Challan_No" in work.columns:
        for cn, grp in work.groupby(work["Challan_No"].astype(str).str.strip()):
            if not cn:
                continue
            row = _agg_group(grp, "Challan_No", cn)
            row["Style"] = str(grp.iloc[0].get("Style", ""))
            challan_comparison.append(row)
    challan_comparison.sort(key=lambda r: r["Net_PL_Rs"])

    karigar_sku_detail: list[dict] = []
    if "Karigar_ID" in work.columns and "Style" in work.columns:
        work["_kid"] = work["Karigar_ID"].apply(clean_key)
        work["_st"] = work["Style"].astype(str).str.strip()
        for (kid, st), grp in work.groupby(["_kid", "_st"]):
            if not kid or not st:
                continue
            row = _agg_group(grp, "Style", st)
            row["Karigar_ID"] = kid
            row["Karigar_Name"] = str(grp.iloc[0].get("Karigar_Name", kid))
            karigar_sku_detail.append(row)

    loss_k = sum(1 for r in karigar_comparison if r["Net_PL_Rs"] < 0)
    loss_s = sum(1 for r in sku_comparison if r["Net_PL_Rs"] < 0)
    over_ch = sum(1 for r in challan_comparison if r["Actual_Rs"] > r["Budgeted_Rs"] and r["Budgeted_Rs"] > 0)

    return {
        "date_from": date_from,
        "date_to": date_to,
        "summary": {
            "karigars_at_loss": loss_k,
            "karigars_at_profit": len(karigar_comparison) - loss_k,
            "skus_at_loss": loss_s,
            "challans_over_budget": over_ch,
            "total_budgeted_rs": round(sum(r["Budgeted_Rs"] for r in karigar_comparison), 2),
            "total_actual_rs": round(sum(r["Actual_Rs"] for r in karigar_comparison), 2),
            "total_net_pl_rs": round(sum(r["Net_PL_Rs"] for r in karigar_comparison), 2),
        },
        "karigar_comparison": karigar_comparison,
        "sku_comparison": sku_comparison,
        "challan_comparison": challan_comparison,
        "karigar_sku_detail": karigar_sku_detail,
    }


def karigar_profitability_report(date_from: str, date_to: str) -> dict:
    """
    Karigar-wise: full payroll paid vs piece value vs ₹480 benchmark P&L (LTL in Running_LTL).
    Answers whether each karigar is profitable on payroll and on factory benchmark.
    """
    payroll = payroll_report(date_from, date_to)
    comp = comparison_dashboard_report(date_from, date_to)
    pr_by_id = {str(r.get("Karigar_ID", "")): r for r in payroll.get("rows", [])}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _row(
        kid: str,
        name: str,
        prow: dict,
        krow: dict | None,
    ) -> dict[str, Any]:
        att = float(prow.get("Attendance_Pay", 0) or 0)
        other = float(prow.get("Other_Work_Pay", 0) or 0)
        paid = float(prow.get("Total", 0) or att + other)
        piece = float((krow or {}).get("Piece_Value_Rs", 0) or 0)
        budgeted = float((krow or {}).get("Budgeted_Rs", 0) or 0)
        actual = float((krow or {}).get("Actual_Rs", 0) or 0)
        net_pl = float((krow or {}).get("Net_PL_Rs", 0) or 0)
        ltl = float((krow or {}).get("Running_LTL", 0) or 0)
        pay_vs_piece = round(piece - paid, 2)
        return {
            "Karigar_ID": kid,
            "Karigar_Name": name or kid,
            "Days": int(prow.get("Days", 0) or 0),
            "Attendance_Pay": round(att, 2),
            "Other_Work_Pay": round(other, 2),
            "Total_Payroll_Paid": round(paid, 2),
            "Piece_Value_Rs": round(piece, 2),
            "Pay_vs_Piece_Rs": pay_vs_piece,
            "Profitable_On_Payroll": "Yes" if piece >= paid - 0.01 else "No",
            "Budgeted_Rs": round(budgeted, 2),
            "Actual_Cost_Rs": round(actual, 2),
            "Net_PL_Benchmark": round(net_pl, 2),
            "Profitable_On_Benchmark": "Yes" if net_pl >= -0.01 else "No",
            "Running_LTL": round(ltl, 1),
            "LTL_Note": "LTL targets active" if ltl > 0 else "Base target (no LTL row)",
            "Pieces": int((krow or {}).get("Pieces", 0) or 0),
            "Payroll_Only": "Yes" if paid > 0 and piece <= 0 else "",
        }

    for krow in comp.get("karigar_comparison", []):
        kid = str(krow.get("Karigar_ID", "")).strip()
        if not kid:
            continue
        seen.add(kid)
        prow = pr_by_id.get(kid, {})
        rows.append(
            _row(kid, str(krow.get("Karigar_Name", prow.get("Name", ""))), prow, krow)
        )

    for kid, prow in pr_by_id.items():
        if kid in seen or not kid:
            continue
        rows.append(_row(kid, str(prow.get("Name", "")), prow, None))

    rows.sort(key=lambda r: float(r.get("Net_PL_Benchmark", 0)))
    prof_pay = sum(1 for r in rows if r.get("Profitable_On_Payroll") == "Yes")
    prof_bench = sum(1 for r in rows if r.get("Profitable_On_Benchmark") == "Yes")
    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "summary": {
            "karigar_count": len(rows),
            "profitable_on_payroll": prof_pay,
            "profitable_on_benchmark": prof_bench,
            "total_payroll_paid": float(payroll.get("total_payroll", 0) or 0),
            "total_piece_value": round(sum(float(r.get("Piece_Value_Rs", 0)) for r in rows), 2),
            "total_net_pl_benchmark": round(sum(float(r.get("Net_PL_Benchmark", 0)) for r in rows), 2),
        },
    }


def challan_labour_payroll_report(date_from: str, date_to: str) -> dict:
    """Challan + style + karigar: production budget/actual and payroll paid (expense + allocated attendance)."""
    pl = get_sheet_df("production_log")
    exp = get_sheet_df("karigar_expenses")
    empty = {"date_from": date_from, "date_to": date_to, "rows": [], "summary": {}}
    if pl.empty:
        return empty

    work = pl.copy()
    work["Date_dt"] = pd.to_datetime(work["Date"], errors="coerce")
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    work = work[(work["Date_dt"] >= d0) & (work["Date_dt"] <= d1)]
    if work.empty:
        return empty
    work = _production_log_latest_rows(work)
    for c in ("Total_Pieces", "Piece_Value_Rs", "Budgeted_Expense_Rs", "Actual_Expense_Rs", "PL_Rs"):
        if c in work.columns:
            work[c] = safe_num(work[c])

    payroll = payroll_report(date_from, date_to)
    att_by_k: dict[str, float] = {}
    for r in payroll.get("rows", []):
        kid = str(r.get("Karigar_ID", ""))
        if kid:
            att_by_k[kid] = float(r.get("Attendance_Pay", 0) or 0)

    karigar_piece: dict[str, float] = {}
    if "Karigar_ID" in work.columns and "Piece_Value_Rs" in work.columns:
        for kid, g in work.groupby(work["Karigar_ID"].apply(clean_key)):
            if kid:
                karigar_piece[kid] = float(safe_num(g["Piece_Value_Rs"]).sum())

    exp_direct: dict[tuple, float] = {}
    if not exp.empty:
        ex = exp.copy()
        ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
        ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
        if not ex.empty and "Amount_Rs" in ex.columns:
            ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
            for _, er in ex.iterrows():
                key = (
                    str(er.get("Challan_No", "")).strip(),
                    str(er.get("Style", "")).strip(),
                    str(er.get("Karigar_ID", "")).strip(),
                )
                exp_direct[key] = exp_direct.get(key, 0.0) + float(er.get("Amount_Rs", 0) or 0)

    rows: list[dict[str, Any]] = []
    group_cols = ["Challan_No", "Style", "Karigar_ID"]
    for col in group_cols:
        if col not in work.columns:
            return empty

    for (cn, st, kid), grp in work.groupby(
        [work["Challan_No"].astype(str).str.strip(), work["Style"].astype(str).str.strip(), work["Karigar_ID"].apply(clean_key)]
    ):
        if not cn and not st:
            continue
        kid = str(kid or "")
        piece_val = float(safe_num(grp.get("Piece_Value_Rs", 0)).sum())
        budgeted = float(safe_num(grp.get("Budgeted_Expense_Rs", 0)).sum())
        actual = float(safe_num(grp.get("Actual_Expense_Rs", 0)).sum())
        net_pl = float(safe_num(grp.get("PL_Rs", 0)).sum())
        if budgeted <= 0 and actual > 0:
            for _, r in grp.iterrows():
                fin = _financial_from_log_row(
                    r, date_str=str(r.get("Date", ""))[:10], kid=kid
                )
                budgeted += fin["budgeted_amount"]
                actual += fin["actual_amount"]
                net_pl += fin["pl_rs"]
        direct = exp_direct.get((cn, st, kid), 0.0)
        k_total = karigar_piece.get(kid, 0.0)
        att_pool = att_by_k.get(kid, 0.0)
        att_alloc = round(att_pool * (piece_val / k_total), 2) if k_total > 0 and piece_val > 0 else 0.0
        total_paid = round(direct + att_alloc, 2)
        rows.append(
            {
                "Challan_No": cn,
                "Style": st,
                "Karigar_ID": kid,
                "Karigar_Name": str(grp.iloc[0].get("Karigar_Name", kid)),
                "Pieces": int(safe_num(grp.get("Total_Pieces", 0)).sum()),
                "Piece_Value_Rs": round(piece_val, 2),
                "Budgeted_Labour_Rs": round(budgeted, 2),
                "Actual_Cost_Rs": round(actual, 2),
                "Net_PL_Benchmark": round(net_pl, 2),
                "Expense_On_Challan_Rs": round(direct, 2),
                "Attendance_Allocated_Rs": att_alloc,
                "Total_Payroll_Paid": total_paid,
                "Pay_vs_Budget": round(budgeted - total_paid, 2),
                "Profitable_On_Payroll": "Yes" if piece_val >= total_paid - 0.01 else "No",
                "Profitable_On_Benchmark": "Yes" if net_pl >= -0.01 else "No",
            }
        )

    rows.sort(key=lambda r: (r.get("Challan_No", ""), r.get("Style", "")))
    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "summary": {
            "challan_lines": len(rows),
            "total_payroll_paid": round(sum(float(r["Total_Payroll_Paid"]) for r in rows), 2),
            "total_budgeted": round(sum(float(r["Budgeted_Labour_Rs"]) for r in rows), 2),
            "total_net_pl": round(sum(float(r["Net_PL_Benchmark"]) for r in rows), 2),
        },
    }


def _production_log_in_range(date_from: str, date_to: str) -> pd.DataFrame:
    pl = get_sheet_df("production_log")
    if pl.empty:
        return pl
    work = pl.copy()
    work["Date_dt"] = pd.to_datetime(work["Date"], errors="coerce")
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    work = work[(work["Date_dt"] >= d0) & (work["Date_dt"] <= d1)]
    if work.empty:
        return work
    return _production_log_latest_rows(work)


def _collect_hourly_financial_rows(work: pd.DataFrame) -> list[dict[str, Any]]:
    """Hour-level piece value vs hourly salary (Report 2 logic) for a production_log slice."""
    if work.empty:
        return []
    from .karigar_attendance import hourly_rate_from_daily, production_hour_cols_for_date

    rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        date_str = str(row.get("Date", ""))[:10]
        kid = str(row.get("Karigar_ID", ""))
        kar_name = str(row.get("Karigar_Name", ""))
        op_name = str(row.get("Operation", ""))
        rate_rs = float(row.get("Rate_Rs") or 0)
        applied_ltl = float(row.get("Applied_LTL") or row.get("Target") or 0)
        base_target = float(row.get("Base_Target") or applied_ltl)
        daily_rate = _daily_salary_for_row(kid, date_str, row)
        hourly_sal = round(hourly_rate_from_daily(daily_rate, date_str), 2)
        valid_hour_cols = [
            h
            for h in production_hour_cols_for_date(date_str)
            if h != "H_13_14" and h in row.index
        ]
        for hcol, hlbl in zip(HOUR_COLS, HOUR_LBLS):
            if hcol == "H_13_14" or hcol not in valid_hour_cols:
                continue
            pcs = int(safe_num(pd.Series([row.get(hcol, 0)])).iloc[0])
            if pcs <= 0:
                continue
            actual_piece_val = round(pcs * rate_rs, 2)
            target_piece_val = round(applied_ltl * rate_rs, 2)
            normal_target_val = round(base_target * rate_rs, 2)
            net_pl = round(actual_piece_val - hourly_sal, 2)
            rows.append(
                {
                    "Date": date_str,
                    "Karigar_ID": kid,
                    "Karigar_Name": kar_name,
                    "Challan_No": str(row.get("Challan_No", "")),
                    "Style": str(row.get("Style", "")),
                    "Operation": op_name,
                    "Hour": hlbl,
                    "Pieces_Done": pcs,
                    "Base_Target": int(base_target),
                    "Applied_LTL": int(applied_ltl),
                    "Rate_Rs": rate_rs,
                    "Actual_Piece_Val_Rs": actual_piece_val,
                    "Target_Piece_Val_Rs": target_piece_val,
                    "Normal_Target_Val_Rs": normal_target_val,
                    "Hourly_Salary_Rs": hourly_sal,
                    "Net_PL_Rs": net_pl,
                }
            )
    return rows


def other_tasks_report(
    date_from: str,
    date_to: str,
    *,
    karigar_id: str | None = None,
) -> dict:
    """Other-task / helper / trainee / alter lines with summary by Work_Type."""
    exp = get_sheet_df("karigar_expenses")
    empty = {
        "date_from": date_from,
        "date_to": date_to,
        "karigar_id": karigar_id or "",
        "lines": [],
        "by_work_type": [],
        "summary": {"lines": 0, "total_amount_rs": 0.0, "karigars": 0},
    }
    if exp.empty:
        return empty

    ex = exp.copy()
    ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
    d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
    ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
    if karigar_id:
        ex = ex[ex["Karigar_ID"].apply(clean_key) == clean_key(karigar_id)]
    if ex.empty:
        return empty

    if "Amount_Rs" in ex.columns:
        ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
    else:
        ex["Amount_Rs"] = 0

    line_cols = [
        c
        for c in [
            "Date",
            "Karigar_ID",
            "Karigar_Name",
            "Work_Type",
            "Challan_No",
            "Style",
            "Operation",
            "Hours",
            "Amount_Rs",
            "Notes",
        ]
        if c in ex.columns
    ]
    lines = ex[line_cols].fillna("").to_dict(orient="records")

    by_type = (
        ex.groupby(ex["Work_Type"].astype(str).str.strip().replace("", "Other"))
        .agg(
            Lines=("Amount_Rs", "count"),
            Karigars=("Karigar_ID", "nunique"),
            Amount_Rs=("Amount_Rs", "sum"),
            **({"Hours": ("Hours", "sum")} if "Hours" in ex.columns else {}),
        )
        .reset_index()
        .rename(columns={"Work_Type": "Work_Type"})
    )
    if "Hours" not in by_type.columns:
        by_type["Hours"] = 0
    by_type["Hours"] = safe_num(by_type["Hours"]).round(2)
    by_type["Amount_Rs"] = safe_num(by_type["Amount_Rs"]).round(2)
    by_type_rows = by_type.fillna("").to_dict(orient="records")

    return {
        "date_from": date_from,
        "date_to": date_to,
        "karigar_id": karigar_id or "",
        "lines": lines,
        "by_work_type": by_type_rows,
        "summary": {
            "lines": len(lines),
            "total_amount_rs": round(float(safe_num(ex["Amount_Rs"]).sum()), 2),
            "karigars": int(ex["Karigar_ID"].nunique()) if "Karigar_ID" in ex.columns else 0,
        },
    }


def style_challan_expense_report(date_from: str, date_to: str) -> dict:
    """
    Style + challan labour expense: actual hourly salary vs target piece value.
    Includes other-task amounts tagged to style/challan.
    """
    work = _production_log_in_range(date_from, date_to)
    hourly = _collect_hourly_financial_rows(work)
    empty = {
        "date_from": date_from,
        "date_to": date_to,
        "rows": [],
        "style_rollup": [],
        "trainee_expense": 0.0,
        "summary": {},
    }
    if not hourly:
        grp = pd.DataFrame(
            columns=[
                "Style",
                "Challan_No",
                "Actual_Expense_Rs",
                "Target_Rs",
                "Normal_Target_Rs",
                "Piece_Value_Rs",
                "Hours_Worked",
                "Pieces",
            ]
        )
    else:
        hourly_df = pd.DataFrame(hourly)
        hourly_df["Style"] = hourly_df["Style"].astype(str).str.strip()
        hourly_df["Challan_No"] = hourly_df["Challan_No"].astype(str).str.strip()
        grp = (
            hourly_df.groupby(["Style", "Challan_No"], as_index=False)
            .agg(
                Actual_Expense_Rs=("Hourly_Salary_Rs", "sum"),
                Target_Rs=("Target_Piece_Val_Rs", "sum"),
                Normal_Target_Rs=("Normal_Target_Val_Rs", "sum"),
                Piece_Value_Rs=("Actual_Piece_Val_Rs", "sum"),
                Hours_Worked=("Hour", "count"),
                Pieces=("Pieces_Done", "sum"),
            )
        )

    exp = get_sheet_df("karigar_expenses")
    other_by_key: dict[tuple[str, str], float] = {}
    trainee_total = 0.0
    if not exp.empty:
        ex = exp.copy()
        ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
        d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
        ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
        if not ex.empty and "Amount_Rs" in ex.columns:
            ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
            for _, er in ex.iterrows():
                wt = str(er.get("Work_Type", "") or "").strip()
                amt = float(er.get("Amount_Rs", 0) or 0)
                if wt.lower() == "trainee":
                    trainee_total += amt
                st = str(er.get("Style", "") or "").strip()
                cn = str(er.get("Challan_No", "") or "").strip()
                if st or cn:
                    key = (st, cn)
                    other_by_key[key] = other_by_key.get(key, 0.0) + amt

    if not grp.empty:
        grp["Other_Task_Rs"] = grp.apply(
            lambda r: other_by_key.get(
                (str(r["Style"]).strip(), str(r["Challan_No"]).strip()), 0.0
            ),
            axis=1,
        )
        grp["Actual_Expense_Rs"] = (
            safe_num(grp["Actual_Expense_Rs"]) + safe_num(grp["Other_Task_Rs"])
        ).round(2)
        grp["Loss_Rs"] = (grp["Actual_Expense_Rs"] - grp["Target_Rs"]).round(2)
        grp["Normal_Loss_Rs"] = (grp["Actual_Expense_Rs"] - grp["Normal_Target_Rs"]).round(2)
        grp["Result"] = grp["Loss_Rs"].apply(lambda x: "Loss" if x > 0 else ("Profit" if x < 0 else "Break-even"))

    style_rollup = []
    if not grp.empty:
        sr = (
            grp.groupby("Style")
            .agg(
                Challans=("Challan_No", "nunique"),
                Actual_Expense_Rs=("Actual_Expense_Rs", "sum"),
                Target_Rs=("Target_Rs", "sum"),
                Normal_Target_Rs=("Normal_Target_Rs", "sum"),
                Piece_Value_Rs=("Piece_Value_Rs", "sum"),
                Loss_Rs=("Loss_Rs", "sum"),
            )
            .reset_index()
            .round(2)
        )
        style_rollup = sr.fillna("").to_dict(orient="records")

    rows = grp.fillna("").to_dict(orient="records") if not grp.empty else []

    # Expense-only lines (no production) still appear as rows
    seen = {(str(r.get("Style", "")), str(r.get("Challan_No", ""))) for r in rows}
    for (st, cn), amt in other_by_key.items():
        if (st, cn) in seen or amt <= 0:
            continue
        rows.append(
            {
                "Style": st,
                "Challan_No": cn,
                "Actual_Expense_Rs": round(amt, 2),
                "Target_Rs": 0.0,
                "Normal_Target_Rs": 0.0,
                "Piece_Value_Rs": 0.0,
                "Other_Task_Rs": round(amt, 2),
                "Loss_Rs": round(amt, 2),
                "Normal_Loss_Rs": round(amt, 2),
                "Result": "Loss",
                "Hours_Worked": 0,
                "Pieces": 0,
            }
        )

    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "style_rollup": style_rollup,
        "trainee_expense": round(trainee_total, 2),
        "summary": {
            "style_lines": len(rows),
            "total_actual_expense": round(
                sum(float(r.get("Actual_Expense_Rs", 0) or 0) for r in rows), 2
            ),
            "total_target": round(sum(float(r.get("Target_Rs", 0) or 0) for r in rows), 2),
            "total_loss": round(sum(float(r.get("Loss_Rs", 0) or 0) for r in rows), 2),
            "trainee_expense": round(trainee_total, 2),
        },
    }


def karigar_hourly_pl_report(date_from: str, date_to: str) -> dict:
    """
    Karigar P&L from hourly production (Actual piece value − hourly salary)
    plus attendance, other-task, trainee, and operating-staff expenses.
    """
    work = _production_log_in_range(date_from, date_to)
    hourly = _collect_hourly_financial_rows(work)

    payroll = payroll_report(date_from, date_to)
    other = other_tasks_report(date_from, date_to)

    by_k: dict[str, dict[str, Any]] = {}

    if hourly:
        hdf = pd.DataFrame(hourly)
        for kid, grp in hdf.groupby(hdf["Karigar_ID"].apply(clean_key)):
            if not kid:
                continue
            nm = str(grp.iloc[0].get("Karigar_Name", kid))
            by_k[kid] = {
                "Karigar_ID": kid,
                "Karigar_Name": nm,
                "Hours_Worked": int(len(grp)),
                "Pieces": int(safe_num(grp["Pieces_Done"]).sum()),
                "Actual_Piece_Val_Rs": round(float(safe_num(grp["Actual_Piece_Val_Rs"]).sum()), 2),
                "Hourly_Salary_Rs": round(float(safe_num(grp["Hourly_Salary_Rs"]).sum()), 2),
                "Net_PL_Rs": round(float(safe_num(grp["Net_PL_Rs"]).sum()), 2),
                "Target_Piece_Val_Rs": round(float(safe_num(grp["Target_Piece_Val_Rs"]).sum()), 2),
                "Normal_Target_Val_Rs": round(float(safe_num(grp["Normal_Target_Val_Rs"]).sum()), 2),
            }

    for prow in payroll.get("rows", []):
        kid = str(prow.get("Karigar_ID", "")).strip()
        if not kid:
            continue
        row = by_k.setdefault(
            kid,
            {
                "Karigar_ID": kid,
                "Karigar_Name": str(prow.get("Name", kid)),
                "Hours_Worked": 0,
                "Pieces": 0,
                "Actual_Piece_Val_Rs": 0.0,
                "Hourly_Salary_Rs": 0.0,
                "Net_PL_Rs": 0.0,
                "Target_Piece_Val_Rs": 0.0,
                "Normal_Target_Val_Rs": 0.0,
            },
        )
        row["Karigar_Name"] = str(prow.get("Name", row.get("Karigar_Name", kid)))
        row["Attendance_Pay"] = round(float(prow.get("Attendance_Pay", 0) or 0), 2)
        row["Other_Work_Pay"] = round(float(prow.get("Other_Work_Pay", 0) or 0), 2)
        row["Total_Payroll_Paid"] = round(float(prow.get("Total", 0) or 0), 2)
        row["Days"] = int(prow.get("Days", 0) or 0)

    expense_by_k: dict[str, dict[str, float]] = {}
    exp = get_sheet_df("karigar_expenses")
    if not exp.empty:
        ex = exp.copy()
        ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
        d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
        ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
        if not ex.empty and "Amount_Rs" in ex.columns:
            ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
            for _, er in ex.iterrows():
                kid = str(er.get("Karigar_ID", "")).strip()
                if not kid:
                    continue
                wt = str(er.get("Work_Type", "") or "Other").strip() or "Other"
                expense_by_k.setdefault(kid, {})
                expense_by_k[kid][wt] = expense_by_k[kid].get(wt, 0.0) + float(er.get("Amount_Rs", 0) or 0)

    rows: list[dict[str, Any]] = []
    all_kids = set(by_k.keys()) | set(expense_by_k.keys())
    for kid in sorted(all_kids):
        base = by_k.get(
            kid,
            {
                "Karigar_ID": kid,
                "Karigar_Name": kid,
                "Hours_Worked": 0,
                "Pieces": 0,
                "Actual_Piece_Val_Rs": 0.0,
                "Hourly_Salary_Rs": 0.0,
                "Net_PL_Rs": 0.0,
                "Target_Piece_Val_Rs": 0.0,
                "Normal_Target_Val_Rs": 0.0,
            },
        )
        exp_map = expense_by_k.get(kid, {})
        trainee = round(exp_map.get("Trainee", 0.0), 2)
        helper = round(exp_map.get("Helper", 0.0), 2)
        other_task = round(
            exp_map.get("Other Task", 0.0) + exp_map.get("Other", 0.0) + exp_map.get("Alter", 0.0)
            + exp_map.get("Part Change", 0.0),
            2,
        )
        total_other = round(sum(exp_map.values()), 2)
        att = float(base.get("Attendance_Pay", 0) or 0)
        if att <= 0 and kid in {str(r.get("Karigar_ID", "")) for r in payroll.get("rows", [])}:
            att = float(next((r.get("Attendance_Pay", 0) for r in payroll["rows"] if str(r.get("Karigar_ID")) == kid), 0) or 0)
        paid = float(base.get("Total_Payroll_Paid", 0) or att + total_other)
        piece = float(base.get("Actual_Piece_Val_Rs", 0) or 0)
        net_hr = float(base.get("Net_PL_Rs", 0) or 0)
        rows.append(
            {
                **base,
                "Attendance_Pay": round(att, 2),
                "Other_Work_Pay": round(float(base.get("Other_Work_Pay", total_other) or total_other), 2),
                "Trainee_Pay": trainee,
                "Helper_Pay": helper,
                "Other_Task_Pay": other_task,
                "Total_Payroll_Paid": round(paid, 2),
                "Pay_vs_Piece_Rs": round(piece - paid, 2),
                "Net_PL_Rs": net_hr,
                "Profitable_On_Hourly_PL": "Yes" if net_hr >= -0.01 else "No",
                "Profitable_On_Payroll": "Yes" if piece >= paid - 0.01 else "No",
            }
        )

    rows.sort(key=lambda r: float(r.get("Net_PL_Rs", 0)))

    op_staff = 0.0
    oa = get_sheet_df("operating_attendance")
    if not oa.empty and "Total_Pay" in oa.columns:
        oa2 = oa.copy()
        oa2["Date_dt"] = pd.to_datetime(oa2["Date"], errors="coerce")
        d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
        oa2 = oa2[(oa2["Date_dt"] >= d0) & (oa2["Date_dt"] <= d1)]
        if not oa2.empty:
            op_staff = round(float(safe_num(oa2["Total_Pay"]).sum()), 2)

    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "other_tasks_by_type": other.get("by_work_type", []),
        "summary": {
            "karigar_count": len(rows),
            "total_actual_piece_val": round(sum(float(r.get("Actual_Piece_Val_Rs", 0)) for r in rows), 2),
            "total_hourly_salary": round(sum(float(r.get("Hourly_Salary_Rs", 0)) for r in rows), 2),
            "total_net_pl_hourly": round(sum(float(r.get("Net_PL_Rs", 0)) for r in rows), 2),
            "total_payroll_paid": round(sum(float(r.get("Total_Payroll_Paid", 0)) for r in rows), 2),
            "trainee_expense": round(
                sum(float(r.get("Trainee_Pay", 0)) for r in rows), 2
            ),
            "operating_staff_pay": op_staff,
            "other_task_total": float(other.get("summary", {}).get("total_amount_rs", 0) or 0),
        },
    }


def karigar_salary_report(date_from: str, date_to: str) -> dict:
    """Karigar salary judgment: P&L per karigar with operations worked."""
    payroll = payroll_report(date_from, date_to)
    pr_by_id = {str(r.get("Karigar_ID", "")): r for r in payroll.get("rows", [])}

    work = _production_log_in_range(date_from, date_to)
    prod_by_k: dict[str, dict[str, Any]] = {}
    if not work.empty:
        for c in ("Total_Pieces", "Piece_Value_Rs"):
            if c in work.columns:
                work[c] = safe_num(work[c])
        if "Karigar_ID" in work.columns:
            for kid, grp in work.groupby(work["Karigar_ID"].apply(clean_key)):
                if not kid:
                    continue
                ops = set()
                if "Operation" in grp.columns:
                    ops = {str(o).strip() for o in grp["Operation"].dropna().unique() if str(o).strip()}
                piece_val = float(safe_num(grp.get("Piece_Value_Rs", 0)).sum()) if "Piece_Value_Rs" in grp.columns else 0.0
                pieces = int(safe_num(grp.get("Total_Pieces", 0)).sum()) if "Total_Pieces" in grp.columns else 0
                prod_by_k[kid] = {
                    "Piece_Value_Rs": round(piece_val, 2),
                    "Pieces": pieces,
                    "Operations": sorted(ops),
                }

    km = get_sheet_df("karigar_master")
    rate_by_id: dict[str, float] = {}
    if not km.empty and "Karigar_ID" in km.columns:
        for _, row in km.iterrows():
            kid = clean_key(row.get("Karigar_ID"))
            if kid:
                rate_by_id[kid] = float(row.get("Daily_Rate_Rs", 0) or 0)

    all_kids = set(pr_by_id.keys()) | set(prod_by_k.keys())
    rows: list[dict[str, Any]] = []
    for kid in sorted(all_kids):
        prow = pr_by_id.get(kid, {})
        prod = prod_by_k.get(kid, {})
        days = int(prow.get("Days", 0) or 0)
        paid = float(prow.get("Total", 0) or 0)
        piece_val = float(prod.get("Piece_Value_Rs", 0))
        pl = round(piece_val - paid, 2)
        ops = prod.get("Operations", [])
        rows.append({
            "Karigar_ID": kid,
            "Karigar_Name": str(prow.get("Name", kid)),
            "Daily_Rate_Rs": rate_by_id.get(kid, 0),
            "Days_Worked": days,
            "Total_Payroll_Paid": round(paid, 2),
            "Total_Piece_Value": round(piece_val, 2),
            "Pieces": int(prod.get("Pieces", 0)),
            "Operations_Count": len(ops),
            "Operations_List": ", ".join(ops),
            "Profit_Loss_Rs": pl,
            "Status": "Profit" if pl > 0.01 else ("Loss" if pl < -0.01 else "Break-even"),
            "Avg_Daily_Output_Rs": round(piece_val / days, 2) if days > 0 else 0.0,
        })
    rows.sort(key=lambda r: float(r["Profit_Loss_Rs"]))
    total_paid = sum(float(r["Total_Payroll_Paid"]) for r in rows)
    total_piece = sum(float(r["Total_Piece_Value"]) for r in rows)
    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "summary": {
            "karigar_count": len(rows),
            "total_payroll_paid": round(total_paid, 2),
            "total_piece_value": round(total_piece, 2),
            "total_profit_loss": round(total_piece - total_paid, 2),
            "profitable_count": sum(1 for r in rows if r["Status"] == "Profit"),
            "loss_count": sum(1 for r in rows if r["Status"] == "Loss"),
        },
    }


def challan_wise_summary_report(date_from: str, date_to: str) -> dict:
    """Challan-wise P&L: party value vs actual labour + expenses."""
    cm = get_sheet_df("challan_master")
    if cm.empty:
        return {"date_from": date_from, "date_to": date_to, "rows": [], "summary": {}}

    work = _production_log_in_range(date_from, date_to)
    hourly = _collect_hourly_financial_rows(work)

    labour_by_challan: dict[str, dict[str, Any]] = {}
    if hourly:
        hdf = pd.DataFrame(hourly)
        for cn, grp in hdf.groupby(hdf["Challan_No"].astype(str).str.strip()):
            cn = str(cn).strip()
            if not cn:
                continue
            karigars = set()
            ops = set()
            if "Karigar_ID" in grp.columns:
                karigars = {str(k).strip() for k in grp["Karigar_ID"].dropna().unique() if str(k).strip()}
            if "Operation" in grp.columns:
                ops = {str(o).strip() for o in grp["Operation"].dropna().unique() if str(o).strip()}
            labour_by_challan[cn] = {
                "Actual_Labour_Rs": round(float(safe_num(grp["Hourly_Salary_Rs"]).sum()), 2),
                "Pieces_Produced": int(safe_num(grp["Pieces_Done"]).sum()),
                "Karigars_Involved": len(karigars),
                "Operations_Count": len(ops),
            }

    exp = get_sheet_df("karigar_expenses")
    exp_by_challan: dict[str, float] = {}
    if not exp.empty:
        ex = exp.copy()
        ex["Date_dt"] = pd.to_datetime(ex["Date"], errors="coerce")
        d0, d1 = pd.Timestamp(date_from), pd.Timestamp(date_to)
        ex = ex[(ex["Date_dt"] >= d0) & (ex["Date_dt"] <= d1)]
        if not ex.empty and "Amount_Rs" in ex.columns:
            ex["Amount_Rs"] = safe_num(ex["Amount_Rs"])
            for _, er in ex.iterrows():
                cn = str(er.get("Challan_No", "") or "").strip()
                if cn:
                    exp_by_challan[cn] = exp_by_challan.get(cn, 0.0) + float(er.get("Amount_Rs", 0) or 0)

    rows: list[dict[str, Any]] = []
    for _, crow in cm.iterrows():
        cn = str(crow.get("Challan_No", "")).strip()
        if not cn:
            continue
        total_qty = int(crow.get("Total_Qty", 0) or 0)
        rate = float(crow.get("Rate_Per_Pc", 0) or 0)
        party_value = round(total_qty * rate, 2)
        lab = labour_by_challan.get(cn, {})
        actual_labour = float(lab.get("Actual_Labour_Rs", 0))
        other_exp = exp_by_challan.get(cn, 0.0)
        total_cost = round(actual_labour + other_exp, 2)
        pl = round(party_value - total_cost, 2)
        margin = round((pl / party_value) * 100, 1) if party_value > 0 else 0.0
        rows.append({
            "Challan_No": cn,
            "Style": str(crow.get("Style", "")).strip(),
            "Party": str(crow.get("Party", "")).strip(),
            "Total_Qty": total_qty,
            "Received_Qty": int(crow.get("Received_Qty", 0) or 0),
            "Rate_Per_Pc": rate,
            "Party_Value_Rs": party_value,
            "Actual_Labour_Rs": round(actual_labour, 2),
            "Other_Expenses_Rs": round(other_exp, 2),
            "Total_Cost_Rs": total_cost,
            "PL_Rs": pl,
            "Margin_Pct": margin,
            "Pieces_Produced": int(lab.get("Pieces_Produced", 0)),
            "Karigars_Involved": int(lab.get("Karigars_Involved", 0)),
            "Operations_Count": int(lab.get("Operations_Count", 0)),
        })
    rows.sort(key=lambda r: float(r["PL_Rs"]))
    total_party = sum(float(r["Party_Value_Rs"]) for r in rows)
    total_cost = sum(float(r["Total_Cost_Rs"]) for r in rows)
    return {
        "date_from": date_from,
        "date_to": date_to,
        "rows": rows,
        "summary": {
            "challan_count": len(rows),
            "total_party_value": round(total_party, 2),
            "total_cost": round(total_cost, 2),
            "total_pl": round(total_party - total_cost, 2),
            "overall_margin_pct": round(((total_party - total_cost) / total_party) * 100, 1) if total_party > 0 else 0.0,
        },
    }


def stitching_reports_hub(date_from: str, date_to: str) -> dict:
    """Dynamics-style report pack: payroll, performance, P&L compare, karigar profit, challan labour."""
    return {
        "date_from": date_from,
        "date_to": date_to,
        "generated_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        "payroll": payroll_report(date_from, date_to),
        "performance": performance_report(date_from, date_to),
        "comparison": comparison_dashboard_report(date_from, date_to),
        "karigar_profitability": karigar_profitability_report(date_from, date_to),
        "challan_labour": challan_labour_payroll_report(date_from, date_to),
        "other_tasks": other_tasks_report(date_from, date_to),
        "style_challan_expense": style_challan_expense_report(date_from, date_to),
        "karigar_hourly_pl": karigar_hourly_pl_report(date_from, date_to),
        "karigar_salary": karigar_salary_report(date_from, date_to),
        "challan_wise": challan_wise_summary_report(date_from, date_to),
        "challan_deposits": challan_deposit_summary(date_from, date_to),
    }


def _replace_id_in_sheet(sheet_key: str, col: str, old_id: str, new_id: str) -> int:
    """Replace clean_key-matched IDs in one sheet column. Returns rows updated."""
    df = get_sheet_df(sheet_key)
    if df.empty or col not in df.columns:
        return 0
    mask = df[col].apply(clean_key) == clean_key(old_id)
    n = int(mask.sum())
    if n:
        df.loc[mask, col] = new_id
        save_sheet_df(sheet_key, df)
    return n


def rename_karigar_id(old_id: str, new_id: str) -> dict:
    """Rename Karigar_ID and update E_Code / attendance / production references."""
    old = clean_key(old_id)
    new = clean_key(new_id)
    if not old or not new:
        return {"ok": False, "message": "Invalid karigar code"}
    if old == new:
        return {"ok": True, "message": "No change"}
    km = get_sheet_df("karigar_master")
    if km.empty or "Karigar_ID" not in km.columns:
        return {"ok": False, "message": "Karigar master empty"}
    if not (km["Karigar_ID"].apply(clean_key) == old).any():
        return {"ok": False, "message": f"Karigar {old} not found"}
    if (km["Karigar_ID"].apply(clean_key) == new).any():
        return {"ok": False, "message": f"Karigar ID {new} already exists"}

    updated = 0
    for sheet, col in (
        ("karigar_master", "Karigar_ID"),
        ("employee_master", "E_Code"),
        ("karigar_attendance", "E_Code"),
        ("production_log", "Karigar_ID"),
        ("karigar_rate_history", "Karigar_ID"),
        ("target_ltl_override", "Karigar_ID"),
        ("karigar_expenses", "Karigar_ID"),
        ("operating_attendance", "E_Code"),
    ):
        updated += _replace_id_in_sheet(sheet, col, old, new)

    return {
        "ok": True,
        "message": f"Karigar ID {old} → {new} ({updated} reference(s) updated)",
        "old_id": old,
        "new_id": new,
        "references_updated": updated,
    }


def rename_employee_e_code(old_code: str, new_code: str) -> dict:
    """Rename employee E_Code; also renames linked karigar when codes match."""
    old = clean_key(old_code)
    new = clean_key(new_code)
    if not old or not new:
        return {"ok": False, "message": "Invalid employee code"}
    if old == new:
        return {"ok": True, "message": "No change"}
    em = get_sheet_df("employee_master")
    if em.empty or "E_Code" not in em.columns:
        return {"ok": False, "message": "Employee master empty"}
    if not (em["E_Code"].apply(clean_key) == old).any():
        return {"ok": False, "message": f"Employee {old} not found"}
    if (em["E_Code"].apply(clean_key) == new).any():
        return {"ok": False, "message": f"E_Code {new} already exists"}

    km = get_sheet_df("karigar_master")
    if not km.empty and (km["Karigar_ID"].apply(clean_key) == old).any():
        return rename_karigar_id(old, new)

    updated = 0
    for sheet, col in (
        ("employee_master", "E_Code"),
        ("karigar_attendance", "E_Code"),
        ("operating_attendance", "E_Code"),
    ):
        updated += _replace_id_in_sheet(sheet, col, old, new)

    return {
        "ok": True,
        "message": f"E_Code {old} → {new} ({updated} reference(s) updated)",
        "old_id": old,
        "new_id": new,
        "references_updated": updated,
    }


def challan_snapshot(challan_no: str) -> dict[str, str]:
    """Party + human-readable description from challan_master for production log."""
    cn = str(challan_no or "").strip()
    if not cn:
        return {"Party": "", "Challan_Description": ""}
    cm = get_sheet_df("challan_master")
    if cm.empty or "Challan_No" not in cm.columns:
        return {"Party": "", "Challan_Description": cn}
    hit = cm[cm["Challan_No"].astype(str).map(clean_key) == clean_key(cn)]
    if hit.empty:
        return {"Party": "", "Challan_Description": cn}
    row = hit.iloc[-1]
    party = str(row.get("Party") or "").strip()
    style = str(row.get("Style") or "").strip()
    total = int(safe_num(pd.Series([row.get("Total_Qty", 0)])).iloc[0])
    received = int(safe_num(pd.Series([row.get("Received_Qty", 0)])).iloc[0])
    rate = float(safe_num(pd.Series([row.get("Rate_Per_Pc", 0)])).iloc[0])
    parts = [p for p in [party, style, f"Qty {total}"] if p]
    if received and received != total:
        parts.append(f"Recv {received}")
    if rate > 0:
        parts.append(f"₹{rate}/pc")
    desc = " · ".join(parts) if parts else cn
    return {"Party": party, "Challan_Description": desc}


def update_employee_master(
    e_code: str,
    *,
    name: str | None = None,
    emp_type: str | None = None,
    daily_rate_rs: float | None = None,
) -> dict:
    code = clean_key(e_code)
    df = get_sheet_df("employee_master")
    if df.empty or "E_Code" not in df.columns:
        return {"ok": False, "message": "Employee master empty"}
    mask = df["E_Code"].apply(clean_key) == code
    if not mask.any():
        return {"ok": False, "message": f"Employee {code} not found"}
    idx = df[mask].index[0]
    if name is not None:
        df.at[idx, "Name"] = str(name).strip()
    if emp_type is not None:
        df.at[idx, "Type"] = str(emp_type).strip()
    if daily_rate_rs is not None:
        rate = float(daily_rate_rs)
        df.at[idx, "Daily_Rate_Rs"] = rate
        df.at[idx, "Hourly_Rate_Rs"] = round(rate / 8, 2)
    save_sheet_df("employee_master", df)
    return {"ok": True, "message": f"Employee {code} updated"}


def update_style_operation(
    style: str,
    operation: str,
    *,
    target: int | None,
    rate_rs: float | None,
    operation_type: str | None = None,
) -> dict:
    df = get_sheet_df("style_master")
    if df.empty:
        return {"ok": False, "message": "Style master empty"}
    mask = (df["Style"].astype(str) == str(style)) & (df["Operation"].astype(str) == str(operation))
    if not mask.any():
        return {"ok": False, "message": f"Operation {operation} not found for {style}"}
    idx = df[mask].index[0]
    if target is not None:
        df.at[idx, "Target"] = int(target)
    if rate_rs is not None:
        df.at[idx, "Rate_Rs"] = float(rate_rs)
    if operation_type is not None:
        ot = str(operation_type).strip().title()
        if ot not in ("Easy", "Medium", "Hard"):
            return {"ok": False, "message": "Operation type must be Easy, Medium, or Hard"}
        if "Operation_Type" not in df.columns:
            df["Operation_Type"] = "Medium"
        df.at[idx, "Operation_Type"] = ot
    save_sheet_df("style_master", df)
    return {"ok": True, "message": "Updated"}


CHALLAN_KEY_COLS = ["Challan_No"]


def normalize_challan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Trim keys and dedupe by challan number after import or manual add."""
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    rename_map = {
        "challan no": "Challan_No",
        "challan_no": "Challan_No",
        "Challan No": "Challan_No",
        "style": "Style",
        "party": "Party",
        "total qty": "Total_Qty",
        "total_qty": "Total_Qty",
        "received qty": "Received_Qty",
        "received_qty": "Received_Qty",
        "rate per pc": "Rate_Per_Pc",
        "rate_per_pc": "Rate_Per_Pc",
        "deposit rs": "Deposit_Rs",
        "deposit_rs": "Deposit_Rs",
        "delivery by": "Delivery_By",
        "delivery_by": "Delivery_By",
    }
    out.columns = [rename_map.get(str(c).strip().lower(), str(c).strip()) for c in out.columns]
    for c in ["Challan_No", "Style", "Party", "Delivery_By"]:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    if "Challan_No" not in out.columns:
        return out
    out = out[out["Challan_No"].astype(str).str.len() > 0]
    out["Challan_No"] = out["Challan_No"].astype(str).str.strip()
    return out.drop_duplicates(subset=["Challan_No"], keep="last").reset_index(drop=True)


def delete_challan(challan_no: str) -> dict:
    df = get_sheet_df("challan_master")
    if df.empty or "Challan_No" not in df.columns:
        return {"ok": False, "message": "No challans"}
    ck = clean_key(challan_no)
    if not ck:
        return {"ok": False, "message": "Challan number required"}
    mask = df["Challan_No"].apply(clean_key) == ck
    if not mask.any():
        return {"ok": False, "message": f"Challan {challan_no} not found"}
    removed = int(mask.sum())
    df = df[~mask].reset_index(drop=True)
    save_sheet_df("challan_master", df)
    return {"ok": True, "message": f"Deleted challan {challan_no}", "removed": removed}


MASTER_KEY_COLS: dict[str, list[str]] = {
    "style_master": ["Style", "Operation"],
    "karigar_master": ["Karigar_ID"],
    "target_ltl_override": ["Style", "Operation", "Karigar_ID"],
    "employee_master": ["E_Code"],
    "challan_master": CHALLAN_KEY_COLS,
    "karigar_expenses": ["Expense_ID"],
}

MASTER_EDITABLE_KEYS = frozenset(MASTER_KEY_COLS.keys())


def dedupe_sheet(key: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep last row per natural key so imports/forms cannot duplicate calculations."""
    if df is None:
        df = get_sheet_df(key)
    cols = MASTER_KEY_COLS.get(key)
    if df is None or df.empty or not cols:
        return df if df is not None else pd.DataFrame()
    present = [c for c in cols if c in df.columns]
    if len(present) != len(cols):
        return df
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.strip()
    return out.drop_duplicates(subset=cols, keep="last").reset_index(drop=True)


def merge_sheet_dataframes(key: str, existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Union rows by master key — keep server rows; add keys only present in incoming."""
    if incoming is None or incoming.empty:
        return dedupe_sheet(key, existing)
    if existing is None or existing.empty:
        return dedupe_sheet(key, incoming)
    cols = MASTER_KEY_COLS.get(key)
    if not cols:
        return dedupe_sheet(key, pd.concat([existing, incoming], ignore_index=True))
    present_in = [c for c in cols if c in incoming.columns]
    present_ex = [c for c in cols if c in existing.columns]
    if len(present_in) != len(cols) or len(present_ex) != len(cols):
        return dedupe_sheet(key, pd.concat([existing, incoming], ignore_index=True))
    ex = existing.copy()
    inc = incoming.copy()
    for c in cols:
        ex[c] = ex[c].astype(str).str.strip()
        inc[c] = inc[c].astype(str).str.strip()

    def _row_key(row: pd.Series) -> tuple[str, ...]:
        return tuple(str(row[c]) for c in cols)

    ex_keys = {_row_key(ex.iloc[i]) for i in range(len(ex))}
    only_new = inc[~inc.apply(_row_key, axis=1).isin(ex_keys)]
    if only_new.empty:
        return dedupe_sheet(key, ex)
    return dedupe_sheet(key, pd.concat([ex, only_new], ignore_index=True))


def looks_like_seed_only_master() -> bool:
    """True when style master is still the built-in demo (data lost after deploy)."""
    sm = get_sheet_df("style_master")
    if sm.empty:
        return True
    seed = pd.DataFrame(DEFAULT_SHEETS["style_master"])
    if len(sm) > len(seed) + 5:
        return False
    if "Style" not in sm.columns:
        return False
    seed_styles = set(seed["Style"].astype(str).str.strip())
    cur_styles = set(sm["Style"].astype(str).str.strip())
    if cur_styles - seed_styles:
        return False
    return len(sm) <= len(seed) + 2


def bootstrap_stitching_data() -> dict[str, Any]:
    """Merge Google Sheet master data on startup when DB is seed-only or bootstrap is enabled."""
    if os.environ.get("STITCHING_GSHEET_BOOTSTRAP", "1").strip().lower() in ("0", "false", "no"):
        return {"ok": True, "skipped": "bootstrap disabled"}
    from .stitching_gsheet import gsheet_status, sync_from_gsheet_merge

    if not gsheet_status().get("available"):
        return {"ok": True, "skipped": "gsheet unavailable"}
    if looks_like_seed_only_master():
        return sync_from_gsheet_merge()
    return {"ok": True, "skipped": "custom data present"}


def _append_rate_history(karigar_id: str, daily_rate: float, effective_from: str) -> None:
    kid = clean_key(karigar_id)
    eff = (effective_from or str(date.today()))[:10]
    hist = get_sheet_df("karigar_rate_history")
    row = {
        "Karigar_ID": kid,
        "Effective_From": eff,
        "Daily_Rate_Rs": float(daily_rate),
    }
    if hist.empty:
        save_sheet_df("karigar_rate_history", pd.DataFrame([row]))
        return
    dup = (
        (hist["Karigar_ID"].apply(clean_key) == kid)
        & (hist["Effective_From"].astype(str).str[:10] == eff)
    )
    if dup.any():
        hist = hist.copy()
        hist.loc[dup, "Daily_Rate_Rs"] = float(daily_rate)
    else:
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("karigar_rate_history", dedupe_sheet("karigar_rate_history", hist))


def _sync_employee_from_karigar(
    karigar_id: str,
    name: str,
    daily_rate: float,
) -> None:
    kid = clean_key(karigar_id)
    hourly = round(float(daily_rate) / 8, 2)
    em = get_sheet_df("employee_master")
    row = {
        "E_Code": kid,
        "Name": name,
        "Type": "Karigar",
        "Daily_Rate_Rs": float(daily_rate),
        "Hourly_Rate_Rs": hourly,
    }
    if em.empty:
        save_sheet_df("employee_master", pd.DataFrame([row]))
        return
    em = em.copy()
    em["_ek"] = em["E_Code"].apply(clean_key)
    mask = em["_ek"] == kid
    if mask.any():
        idx = em[mask].index[0]
        em.at[idx, "Name"] = name
        em.at[idx, "Type"] = "Karigar"
        em.at[idx, "Daily_Rate_Rs"] = float(daily_rate)
        em.at[idx, "Hourly_Rate_Rs"] = hourly
        em = em.drop(columns=["_ek"], errors="ignore")
    else:
        em = em.drop(columns=["_ek"], errors="ignore")
        em = pd.concat([em, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("employee_master", dedupe_sheet("employee_master", em))


def add_style_operation_row(
    style: str,
    operation: str,
    target: int,
    rate_rs: float,
    *,
    operation_type: str = "Medium",
) -> dict:
    style_s = str(style).strip()
    op_s = str(operation).strip()
    if not style_s or not op_s:
        return {"ok": False, "message": "Style and operation are required"}
    df = get_sheet_df("style_master")
    if not df.empty and "Style" in df.columns and "Operation" in df.columns:
        dup = (df["Style"].astype(str).str.strip() == style_s) & (
            df["Operation"].astype(str).str.strip() == op_s
        )
        if dup.any():
            return {"ok": False, "message": f"{style_s} / {op_s} already exists — edit or delete the duplicate first"}
    operation_type = str(operation_type or "Medium").strip().title()
    if operation_type not in ("Easy", "Medium", "Hard"):
        return {"ok": False, "message": "Operation type must be Easy, Medium, or Hard"}
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "Style": style_s,
                        "Operation": op_s,
                        "Target": int(target),
                        "Rate_Rs": float(rate_rs),
                        "Operation_Type": operation_type,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_sheet_df("style_master", dedupe_sheet("style_master", df))
    return {"ok": True, "message": "Style operation added"}


def delete_production_hour_entry(
    *,
    date_str: str,
    karigar_id: str,
    challan_no: str,
    style: str,
    operation: str,
    hour_label: str,
) -> dict:
    """Delete one hour from production and purge related hourly aggregates if it becomes empty."""
    if hour_label not in HOUR_LBLS:
        return {"ok": False, "message": f"Unknown hour {hour_label}"}
    hour_col = HOUR_COLS[HOUR_LBLS.index(hour_label)]
    with _PRODUCTION_LOG_LOCK:
        pl = get_sheet_df("production_log")
        if pl.empty:
            return {"ok": False, "message": "No production entries to delete."}
        work = pl.copy()
        idx = _latest_production_row_index(
            work,
            date_str=date_str,
            karigar_id=karigar_id,
            challan_no=challan_no,
            style=style,
            operation=operation,
        )
        if idx is None:
            return {"ok": False, "message": "No matching operation row found"}
        latest_row = work.loc[idx].copy()
        resolved_kid = str(latest_row.get("Karigar_ID", karigar_id))
        dup_mask = _mask_production_log(
            work,
            date_str=date_str,
            karigar_id=resolved_kid,
            challan_no=challan_no,
            style=style,
            operation=operation,
            require_karigar=bool(clean_key(resolved_kid)),
        )
        work = work[~dup_mask].reset_index(drop=True)

        latest_row[hour_col] = 0
        latest_row[sticker_in_col(hour_col)] = 0
        latest_row[sticker_out_col(hour_col)] = 0
        remaining = 0
        for hc in HOUR_COLS:
            if hc == "H_13_14":
                continue
            remaining += int(safe_num(pd.Series([latest_row.get(hc, 0)])).iloc[0])

        session_mask = _mask_production_log(
            work,
            date_str=str(latest_row.get("Date", date_str)),
            karigar_id=resolved_kid,
            challan_no=str(latest_row.get("Challan_No", challan_no)),
            style=str(latest_row.get("Style", style)),
            require_karigar=bool(clean_key(resolved_kid)),
        )
        session_rows = work[session_mask].copy()
        if remaining > 0:
            session_rows = pd.concat([session_rows, pd.DataFrame([latest_row])], ignore_index=True)

        if remaining <= 0 and session_rows.empty:
            save_sheet_df("production_log", work)
            return {
                "ok": True,
                "message": f"Deleted hour {hour_label}; operation row removed (no remaining pieces). Related reports updated.",
                "removed": 1,
                "hour_deleted": hour_label,
            }

        hour_entries: list[dict[str, Any]] = []
        for _, srow in session_rows.iterrows():
            op_name = normalize_operation_name(srow.get("Operation", ""))
            for hc in HOUR_COLS:
                if hc == "H_13_14":
                    continue
                pcs = int(safe_num(pd.Series([srow.get(hc, 0)])).iloc[0])
                sin = int(safe_num(pd.Series([srow.get(sticker_in_col(hc), 0)])).iloc[0])
                sout = int(safe_num(pd.Series([srow.get(sticker_out_col(hc), 0)])).iloc[0])
                if pcs > 0 or sin > 0 or sout > 0:
                    hour_entries.append(
                        {
                            "hour_col": hc,
                            "operation": op_name,
                            "pieces": pcs,
                            "sticker_in": sin,
                            "sticker_out": sout,
                            "manual_pieces": sin == 0 and sout == 0,
                        }
                    )

        work = _drop_production_session_rows(
            work,
            str(latest_row.get("Date", date_str)),
            resolved_kid,
            str(latest_row.get("Challan_No", challan_no)),
            str(latest_row.get("Style", style)),
        )
        save_sheet_df("production_log", work)
        if hour_entries:
            save_production_entry(
                date_str=str(latest_row.get("Date", date_str)),
                karigar_id=resolved_kid,
                karigar_name=str(latest_row.get("Karigar_Name", "")),
                challan_no=str(latest_row.get("Challan_No", challan_no)),
                style=str(latest_row.get("Style", style)),
                hour_entries=hour_entries,
                saved_by="admin",
                saved_by_name="hour-delete",
            )
        return {
            "ok": True,
            "message": f"Deleted hour {hour_label} and recalculated related session/report rows.",
            "removed": 1,
            "hour_deleted": hour_label,
        }


def add_karigar_master(
    karigar_id: str,
    name: str,
    skill: str,
    daily_rate_rs: float,
    effective_from: str | None = None,
) -> dict:
    kid = clean_key(karigar_id)
    if not kid:
        return {"ok": False, "message": "Karigar ID is required"}
    df = get_sheet_df("karigar_master")
    if not df.empty and "Karigar_ID" in df.columns:
        if (df["Karigar_ID"].apply(clean_key) == kid).any():
            return {"ok": False, "message": f"Karigar {kid} already exists — update rate or delete first"}
    eff = (effective_from or str(date.today()))[:10]
    row = {
        "Karigar_ID": kid,
        "Name": str(name).strip(),
        "Skill": str(skill).strip() or "Stitching",
        "Daily_Rate_Rs": float(daily_rate_rs),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("karigar_master", dedupe_sheet("karigar_master", df))
    _sync_employee_from_karigar(kid, row["Name"], float(daily_rate_rs))
    _append_rate_history(kid, float(daily_rate_rs), eff)
    return {"ok": True, "message": f"Karigar {kid} added (rate ₹{daily_rate_rs}/day from {eff})"}


def update_karigar_master(
    karigar_id: str,
    *,
    name: str | None = None,
    skill: str | None = None,
    daily_rate_rs: float | None = None,
    effective_from: str | None = None,
) -> dict:
    kid = clean_key(karigar_id)
    df = get_sheet_df("karigar_master")
    if df.empty or "Karigar_ID" not in df.columns:
        return {"ok": False, "message": "Karigar master empty"}
    mask = df["Karigar_ID"].apply(clean_key) == kid
    if not mask.any():
        return {"ok": False, "message": f"Karigar {kid} not found"}
    idx = df[mask].index[0]
    if name is not None:
        df.at[idx, "Name"] = str(name).strip()
    if skill is not None:
        df.at[idx, "Skill"] = str(skill).strip()
    rate_changed = False
    if daily_rate_rs is not None:
        df.at[idx, "Daily_Rate_Rs"] = float(daily_rate_rs)
        rate_changed = True
    save_sheet_df("karigar_master", df)
    nm = str(df.at[idx, "Name"])
    rate = float(df.at[idx, "Daily_Rate_Rs"] or 0)
    _sync_employee_from_karigar(kid, nm, rate)
    if rate_changed:
        eff = (effective_from or str(date.today()))[:10]
        _append_rate_history(kid, rate, eff)
        recalc = recalculate_production_for_karigar_from_date(kid, eff)
        extra = ""
        if recalc.get("sessions", 0):
            extra = f" Recalculated {recalc['sessions']} production session(s)."
        return {
            "ok": True,
            "message": f"Rate ₹{rate}/day applies from {eff}.{extra}",
            "recalculated_sessions": recalc.get("sessions", 0),
        }
    return {"ok": True, "message": "Karigar updated"}


def delete_master_rows(sheet_key: str, rows: list[dict]) -> dict:
    if sheet_key not in MASTER_EDITABLE_KEYS:
        return {"ok": False, "message": f"Cannot delete rows from {sheet_key}"}
    if not rows:
        return {"ok": False, "message": "No rows selected"}
    key_cols = MASTER_KEY_COLS[sheet_key]
    df = get_sheet_df(sheet_key)
    if df.empty:
        return {"ok": False, "message": "Sheet is empty"}
    removed = 0
    karigar_ids: list[str] = []
    for ident in rows:
        mask = pd.Series(True, index=df.index)
        for col in key_cols:
            if col not in df.columns:
                continue
            val = clean_key(ident.get(col, ""))
            mask &= df[col].apply(clean_key) == val
        n = int(mask.sum())
        if n:
            removed += n
            if sheet_key == "karigar_master" and "Karigar_ID" in ident:
                karigar_ids.append(clean_key(ident["Karigar_ID"]))
            df = df[~mask]
    if removed == 0:
        return {"ok": False, "message": "No matching rows found"}
    save_sheet_df(sheet_key, df.reset_index(drop=True))
    if sheet_key == "karigar_master" and karigar_ids:
        em = get_sheet_df("employee_master")
        if not em.empty and "E_Code" in em.columns:
            em = em[~em["E_Code"].apply(clean_key).isin(karigar_ids)].reset_index(drop=True)
            save_sheet_df("employee_master", em)
        hist = get_sheet_df("karigar_rate_history")
        if not hist.empty and "Karigar_ID" in hist.columns:
            hist = hist[~hist["Karigar_ID"].apply(clean_key).isin(karigar_ids)].reset_index(drop=True)
            save_sheet_df("karigar_rate_history", hist)
    return {"ok": True, "message": f"Deleted {removed} row(s)", "removed": removed}
