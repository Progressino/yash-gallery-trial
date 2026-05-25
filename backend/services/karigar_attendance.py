"""Karigar attendance policy, punch parsing, and biometric sheet import."""
from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, time, timedelta
from io import BytesIO
from typing import Any

import pandas as pd

from ..db.stitching_db import get_sheet_df, save_sheet_df
from .stitching_costing import clean_key, get_daily_rate_for_date

# Official shift (IST wall clock)
WORK_START = time(9, 0)
WORK_END = time(18, 0)
LUNCH_START = time(13, 0)
LUNCH_END = time(13, 30)
TEA1_START = time(11, 0)
TEA1_END = time(11, 15)
TEA2_START = time(16, 0)
TEA2_END = time(16, 15)
TEA_BREAK_MINUTES_EACH = 15
DEFAULT_LUNCH_BREAK_MINUTES = 30
DEFAULT_TEA_BREAK_MINUTES = TEA_BREAK_MINUTES_EACH * 2  # two tea breaks × 15 min
OT_START = time(18, 0)
OT_END = time(21, 0)
OT_MIN_MINUTES = 25
OT_BREAK_MINUTES = 15
STANDARD_PAY_MINUTES = 8 * 60  # salary based on 8 working hours

WORK_BLOCKS = (
    (WORK_START, TEA1_START),
    (TEA1_END, LUNCH_START),
    (LUNCH_END, TEA2_START),
    (TEA2_END, WORK_END),
)


def _parse_clock(val: Any) -> time | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, datetime):
        return val.time()
    if isinstance(val, time):
        return val
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return None
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            continue
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        return time(int(m.group(1)), int(m.group(2)))
    return None


def _dt_on(base: date, t: time) -> datetime:
    return datetime.combine(base, t)


def _merge_intervals(intervals: list[tuple[datetime, datetime]]) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[datetime, datetime]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _overlap_minutes(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 60.0


def _column_lookup(columns: list[str]) -> dict[str, str]:
    """Map normalized header (IN1, OUT2, …) to actual column name."""
    colmap: dict[str, str] = {}
    for c in columns:
        norm = re.sub(r"[^A-Z0-9]", "", str(c).strip().upper())
        if norm:
            colmap[norm] = c
    return colmap


def extract_punch_pairs(row: pd.Series, columns: list[str]) -> list[tuple[time, time | None]]:
    """Read IN-1/OUT-1 … IN-5/OUT-5 from a biometric export row."""
    pairs: list[tuple[time, time | None]] = []
    colmap = _column_lookup(columns)
    for n in range(1, 6):
        tin = tout = None
        for in_key in (f"IN{n}", f"IN-{n}"):
            if in_key.replace("-", "") in colmap:
                tin = _parse_clock(row.get(colmap[in_key.replace("-", "")]))
                break
        for out_key in (f"OUT{n}", f"OUT-{n}"):
            norm = out_key.replace("-", "")
            if norm in colmap:
                tout = _parse_clock(row.get(colmap[norm]))
                break
        if tin is not None:
            pairs.append((tin, tout))
    return pairs


def serialize_punch_pairs(pairs: list[tuple[time, time | None]]) -> str:
    """JSON list of [in, out] strings for storage on attendance rows."""
    payload = []
    for tin, tout in pairs:
        payload.append(
            [
                tin.strftime("%H:%M") if tin else "",
                tout.strftime("%H:%M") if tout else "",
            ]
        )
    return json.dumps(payload)


def deserialize_punch_pairs(raw: Any) -> list[tuple[time, time | None]]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s in ("[]", "nan"):
            return []
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
    elif isinstance(raw, list):
        data = raw
    else:
        return []
    pairs: list[tuple[time, time | None]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        tin = _parse_clock(item[0])
        tout = _parse_clock(item[1]) if len(item) > 1 and item[1] else None
        if tin is not None:
            pairs.append((tin, tout))
    return pairs


def _intervals_from_pairs(pairs: list[tuple[time, time | None]], base: date) -> list[tuple[datetime, datetime]]:
    intervals: list[tuple[datetime, datetime]] = []
    default_out = WORK_END
    for tin, tout in pairs:
        start = _dt_on(base, tin)
        if tout is None:
            end = _dt_on(base, default_out)
        else:
            end = _dt_on(base, tout)
            if end < start:
                end += timedelta(days=1)
        if end > start:
            intervals.append((start, end))
    return _merge_intervals(intervals)


def _left_before_lunch_return_at_tea(intervals: list[tuple[datetime, datetime]], base: date) -> bool:
    lunch = _dt_on(base, LUNCH_START)
    tea = _dt_on(base, TEA2_START)
    left_before_lunch = any(end <= lunch for _, end in intervals)
    returned_at_tea = any(start >= tea for start, _ in intervals)
    return left_before_lunch and returned_at_tea


def _present_during(intervals: list[tuple[datetime, datetime]], start: datetime, end: datetime) -> bool:
    return any(_overlap_minutes(s, e, start, end) > 0 for s, e in intervals)


def needs_miss_punch(punch_pairs: list[tuple[time, time | None]]) -> bool:
    """True when any IN punch is missing its OUT (or row has no punches)."""
    if not punch_pairs:
        return True
    return any(tout is None for _, tout in punch_pairs)


def calc_salary_from_punches(
    punch_pairs: list[tuple[time, time | None]],
    daily_rate: float,
    *,
    on_date: str | None = None,
    status: str = "P",
    waive_lunch_break: bool = False,
    waive_tea_break: bool = False,
    lunch_break_minutes: float | None = None,
    tea_break_minutes: float | None = None,
) -> dict[str, Any]:
    """
    Apply karigar attendance / OT policy to clock-in/out punches.

    Regular pay uses 8 hours/day; hourly rate = daily_rate / 8.
    OT uses the same hourly rate (no multiplier).
    """
    hourly = round(float(daily_rate or 0) / 8, 4)
    zero = {
        "Status": status,
        "Total_Presence_Hrs": 0.0,
        "Payable_Hrs": 0.0,
        "Hourly_Rate_Rs": round(hourly, 2),
        "Normal_Pay": 0.0,
        "OT_Hours": 0.0,
        "OT_Pay": 0.0,
        "Total_Pay": 0.0,
        "Lunch_Deduction_Hrs": 0.0,
        "Tea_Deduction_Hrs": 0.0,
        "Break_Penalty_Hrs": 0.0,
        "Late_Early_Deduction_Hrs": 0.0,
    }
    if status in ("A", "WO") or daily_rate <= 0 or not punch_pairs:
        return zero

    base = pd.to_datetime(on_date or date.today()).date()
    intervals = _intervals_from_pairs(punch_pairs, base)
    if not intervals:
        return zero

    # First punch before 09:00 counts as 09:00 for regular hours.
    adjusted: list[tuple[datetime, datetime]] = []
    for idx, (start, end) in enumerate(intervals):
        if idx == 0 and start.time() < WORK_START:
            start = _dt_on(base, WORK_START)
        adjusted.append((start, end))
    intervals = _merge_intervals(adjusted)

    total_presence_min = sum((e - s).total_seconds() / 60.0 for s, e in intervals)

    work_minutes = 0.0
    for block_start, block_end in WORK_BLOCKS:
        bs = _dt_on(base, block_start)
        be = _dt_on(base, block_end)
        for s, e in intervals:
            work_minutes += _overlap_minutes(s, e, bs, be)

    lunch_penalty = 0.0
    tea_penalty = 0.0
    special_break_penalty = _left_before_lunch_return_at_tea(intervals, base)
    lunch_window = (_dt_on(base, LUNCH_START), _dt_on(base, LUNCH_END))
    tea1_window = (_dt_on(base, TEA1_START), _dt_on(base, TEA1_END))
    tea2_window = (_dt_on(base, TEA2_START), _dt_on(base, TEA2_END))
    # Lunch: deduct if they stayed clocked in through lunch (no break). Tea: deduct if absent from floor.
    worked_through_lunch = _present_during(intervals, *lunch_window)
    tea1_absent = not _present_during(intervals, *tea1_window)
    tea2_absent = not _present_during(intervals, *tea2_window)

    if waive_lunch_break:
        lunch_penalty = 0.0
    elif lunch_break_minutes is not None:
        lunch_penalty = max(0.0, float(lunch_break_minutes))
    elif special_break_penalty:
        lunch_penalty = float(DEFAULT_LUNCH_BREAK_MINUTES)
    elif worked_through_lunch:
        lunch_penalty = float(DEFAULT_LUNCH_BREAK_MINUTES)

    if waive_tea_break:
        tea_penalty = 0.0
    elif tea_break_minutes is not None:
        tea_penalty = max(0.0, float(tea_break_minutes))
    elif special_break_penalty:
        tea_penalty = float(DEFAULT_TEA_BREAK_MINUTES)
    else:
        if tea1_absent:
            tea_penalty += float(TEA_BREAK_MINUTES_EACH)
        if tea2_absent:
            tea_penalty += float(TEA_BREAK_MINUTES_EACH)

    first_in = min(s for s, _ in intervals).time()
    last_out = max(e for _, e in intervals).time()
    late_min = 0.0
    if first_in > WORK_START:
        late_min = (
            datetime.combine(base, first_in) - datetime.combine(base, WORK_START)
        ).total_seconds() / 60.0
    # Overtime: after 18:00 until actual out (cap 21:00), same hourly rate as regular (daily/8).
    last_out_dt = max(e for _, e in intervals)
    ot_start = _dt_on(base, OT_START)
    ot_end_cap = min(last_out_dt, _dt_on(base, OT_END))
    ot_minutes = 0.0
    if ot_end_cap > ot_start:
        ot_minutes = (ot_end_cap - ot_start).total_seconds() / 60.0
    if ot_minutes <= OT_MIN_MINUTES:
        ot_minutes = 0.0
        ot_hrs_bill = 0.0
    else:
        ot_minutes = min(ot_minutes, 180.0)
        ot_hrs_bill = float(math.ceil(ot_minutes / 60.0 - 1e-9))
    ot_hrs = ot_hrs_bill
    ot_pay = round(ot_hrs_bill * hourly, 2)

    early_min = 0.0
    if last_out < WORK_END and ot_minutes <= 0:
        early_min = (
            datetime.combine(base, WORK_END) - datetime.combine(base, last_out)
        ).total_seconds() / 60.0

    work_net_minutes = max(work_minutes - lunch_penalty - tea_penalty, 0.0)
    block_coverages = []
    for block_start, block_end in WORK_BLOCKS:
        bs = _dt_on(base, block_start)
        be = _dt_on(base, block_end)
        block_len = (be - bs).total_seconds() / 60.0
        got = sum(_overlap_minutes(s, e, bs, be) for s, e in intervals)
        block_coverages.append(got / block_len if block_len > 0 else 0.0)
    on_time_full_day = late_min <= 0 and early_min <= 0 and all(c >= 0.85 for c in block_coverages)

    if on_time_full_day:
        payable_minutes = float(STANDARD_PAY_MINUTES)
    else:
        shift_net_minutes = max(
            float(STANDARD_PAY_MINUTES) - lunch_penalty - tea_penalty - late_min - early_min,
            0.0,
        )
        payable_minutes = min(shift_net_minutes, work_net_minutes)
    payable_minutes = min(payable_minutes, float(STANDARD_PAY_MINUTES))
    payable_hrs = round(payable_minutes / 60.0, 2)
    normal_pay = round((payable_minutes / STANDARD_PAY_MINUTES) * daily_rate, 2)

    punch_pairs_json = serialize_punch_pairs(punch_pairs)

    late_deduction_rs = round((late_min / 60.0) * hourly, 2)
    early_deduction_rs = round((early_min / 60.0) * hourly, 2)

    return {
        "Status": status,
        "Total_Presence_Hrs": round(total_presence_min / 60.0, 2),
        "Payable_Hrs": payable_hrs,
        "Hourly_Rate_Rs": round(hourly, 2),
        "Normal_Pay": normal_pay,
        "OT_Hours": ot_hrs,
        "OT_Pay": ot_pay,
        "Total_Pay": round(normal_pay + ot_pay, 2),
        "Lunch_Deduction_Hrs": round(lunch_penalty / 60.0, 2),
        "Tea_Deduction_Hrs": round(tea_penalty / 60.0, 2),
        "Break_Penalty_Hrs": round((lunch_penalty + tea_penalty) / 60.0, 2),
        "Late_Deduction_Hrs": round(late_min / 60.0, 2),
        "Early_Deduction_Hrs": round(early_min / 60.0, 2),
        "Late_Deduction_Rs": late_deduction_rs,
        "Early_Deduction_Rs": early_deduction_rs,
        "Late_Early_Deduction_Hrs": round((late_min + early_min) / 60.0, 2),
        "In_Punch": first_in.strftime("%H:%M"),
        "Out_Punch": last_out.strftime("%H:%M"),
        "Punch_Pairs": punch_pairs_json,
        "Punch_Count": len(punch_pairs),
        "Needs_Miss_Punch": needs_miss_punch(punch_pairs),
    }


def punch_pairs_from_request(items: list[dict[str, str]]) -> list[tuple[time, time | None]]:
    """Build punch pairs from API payload [{in_time, out_time}, …]."""
    pairs: list[tuple[time, time | None]] = []
    for item in items:
        tin = _parse_clock(item.get("in_time") or item.get("In_Punch"))
        tout = _parse_clock(item.get("out_time") or item.get("Out_Punch"))
        if tin is not None:
            pairs.append((tin, tout))
    return pairs


def _pairs_from_patch(
    punch_pairs: list[tuple[time, time | None]] | None,
    in_punch: str,
    out_punch: str,
) -> list[tuple[time, time | None]] | None:
    if punch_pairs:
        if not any(tin is not None for tin, _ in punch_pairs):
            return None
        return punch_pairs
    tin = _parse_clock(in_punch)
    tout = _parse_clock(out_punch)
    if tin is None:
        return None
    return [(tin, tout)]


def update_karigar_attendance_row(
    *,
    on_date: str,
    e_code: str,
    in_punch: str = "",
    out_punch: str = "",
    punch_pairs: list[tuple[time, time | None]] | None = None,
    waive_lunch_break: bool = False,
    waive_tea_break: bool = False,
    lunch_break_minutes: float | None = None,
    tea_break_minutes: float | None = None,
) -> dict[str, Any]:
    """Recalculate one attendance row after miss-punch / break corrections."""
    e_code, name, master_rate = match_employee_code(e_code, "")
    daily = master_rate or get_daily_rate_for_date(e_code, on_date)
    if daily <= 0:
        return {"ok": False, "message": f"No daily rate for E_Code {e_code}."}
    pairs = _pairs_from_patch(punch_pairs, in_punch, out_punch)
    if pairs is None:
        return {"ok": False, "message": "Valid In punch required."}
    if needs_miss_punch(pairs):
        return {"ok": False, "message": "Each In punch needs an Out time before saving."}
    calc = calc_salary_from_punches(
        pairs,
        daily,
        on_date=on_date,
        status="P",
        waive_lunch_break=waive_lunch_break,
        waive_tea_break=waive_tea_break,
        lunch_break_minutes=lunch_break_minutes,
        tea_break_minutes=tea_break_minutes,
    )
    row = {
        "Date": on_date,
        "E_Code": e_code,
        "Name": name,
        "Status": "P",
        "Daily_Rate_Rs": daily,
        **calc,
    }
    df = get_sheet_df("karigar_attendance")
    if df.empty:
        df = pd.DataFrame([row])
    else:
        mask = (df["Date"].astype(str) == str(on_date)) & (
            df["E_Code"].astype(str).map(clean_key) == clean_key(e_code)
        )
        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("karigar_attendance", df)
    return {"ok": True, "message": "Attendance updated and payroll recalculated.", "row": row}


def calc_salary(in_str: str, out_str: str, daily_rate: float, ot_mult: float = 1.0) -> dict:
    """Backward-compatible wrapper for single in/out punches."""
    pairs = []
    tin = _parse_clock(in_str)
    tout = _parse_clock(out_str)
    if tin is not None:
        pairs.append((tin, tout))
    out = calc_salary_from_punches(pairs, daily_rate, status="P")
    # Policy uses same rate for OT; ignore legacy 1.5x multiplier.
    return {
        "Total_Presence_Hrs": out["Total_Presence_Hrs"],
        "Lunch_Deduction_Hrs": out["Lunch_Deduction_Hrs"],
        "Payable_Hrs": out["Payable_Hrs"],
        "Hourly_Rate_Rs": out["Hourly_Rate_Rs"],
        "Normal_Pay": out["Normal_Pay"],
        "OT_Hours": out["OT_Hours"],
        "OT_Pay": out["OT_Pay"],
        "Total_Pay": out["Total_Pay"],
    }


def _normalize_name(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val or "").strip().lower())


def match_employee_code(raw_code: Any, name: str = "") -> tuple[str, str, float]:
    """Map biometric E. Code to employee_master / karigar_master."""
    code = clean_key(raw_code)
    nm = str(name or "").strip()
    em = get_sheet_df("employee_master")
    km = get_sheet_df("karigar_master")

    if not em.empty and "E_Code" in em.columns:
        hit = em[em["E_Code"].astype(str).map(clean_key) == code]
        if not hit.empty:
            row = hit.iloc[0]
            return str(row["E_Code"]), str(row.get("Name", nm)), float(row.get("Daily_Rate_Rs") or 0)

    if not km.empty and "Karigar_ID" in km.columns:
        hit = km[km["Karigar_ID"].astype(str).map(clean_key) == code]
        if not hit.empty:
            row = hit.iloc[0]
            return str(row["Karigar_ID"]), str(row.get("Name", nm)), float(row.get("Daily_Rate_Rs") or 0)

    if nm and not em.empty and "Name" in em.columns:
        target = _normalize_name(nm)
        for _, row in em.iterrows():
            if _normalize_name(row["Name"]) == target:
                return str(row["E_Code"]), str(row["Name"]), float(row.get("Daily_Rate_Rs") or 0)

    if nm and not km.empty and "Name" in km.columns:
        target = _normalize_name(nm)
        for _, row in km.iterrows():
            if _normalize_name(row["Name"]) == target:
                return str(row["Karigar_ID"]), str(row["Name"]), float(row.get("Daily_Rate_Rs") or 0)

    return code, nm, 0.0


def _extract_report_date(raw: pd.DataFrame) -> str:
    for i in range(min(12, len(raw))):
        for j in range(min(8, raw.shape[1])):
            val = str(raw.iloc[i, j] or "")
            if "date" in val.lower():
                for k in range(j, min(j + 4, raw.shape[1])):
                    cell = str(raw.iloc[i, k] or "")
                    m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", cell)
                    if m:
                        return pd.to_datetime(m.group(1), dayfirst=True).strftime("%Y-%m-%d")
    return str(date.today())


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(20, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if any("e. code" in x or x == "e code" for x in row):
            return i
    return 8


def parse_inout_punch_report(raw: bytes, filename: str = "") -> tuple[str, pd.DataFrame, list[str]]:
    """Parse Daily Attendance IN/OUT Punch Report (.xls / .xlsx)."""
    warnings: list[str] = []
    bio = BytesIO(raw)
    fn = (filename or "").lower()
    if fn.endswith(".xls"):
        try:
            import xlrd  # noqa: F401
        except ImportError as e:
            raise ValueError("Install xlrd on the server to import .xls attendance files.") from e
        preview = pd.read_excel(bio, sheet_name=0, header=None, engine="xlrd")
    else:
        preview = pd.read_excel(bio, sheet_name=0, header=None)
    report_date = _extract_report_date(preview)
    header_row = _find_header_row(preview)
    bio.seek(0)
    if fn.endswith(".xls"):
        df = pd.read_excel(bio, sheet_name=0, header=header_row, engine="xlrd")
    else:
        df = pd.read_excel(bio, sheet_name=0, header=header_row)
    df = df.dropna(how="all")
    return report_date, df, warnings


def import_karigar_attendance_bytes(raw: bytes, filename: str = "") -> dict[str, Any]:
    """Import biometric attendance; upserts by Date + E_Code."""
    report_date, df, warnings = parse_inout_punch_report(raw, filename)
    if df.empty:
        return {"ok": False, "message": "No rows found in attendance file.", "warnings": warnings}

    code_col = next((c for c in df.columns if "code" in str(c).lower()), None)
    name_col = next((c for c in df.columns if str(c).strip().lower() == "name"), None)
    shift_col = next((c for c in df.columns if str(c).strip().lower() == "shift"), None)
    if not code_col:
        return {"ok": False, "message": "Could not find E. Code column.", "warnings": warnings}

    rows: list[dict] = []
    unmatched: list[str] = []
    for _, r in df.iterrows():
        raw_code = r.get(code_col)
        if pd.isna(raw_code) or str(raw_code).strip() == "":
            continue
        emp_name = str(r.get(name_col, "") or "").strip() if name_col else ""
        shift = str(r.get(shift_col, "") or "").strip().upper() if shift_col else ""
        pairs = extract_punch_pairs(r, list(df.columns))
        e_code, name, master_rate = match_employee_code(raw_code, emp_name)
        daily = master_rate or get_daily_rate_for_date(e_code, report_date)
        if daily <= 0 and master_rate <= 0:
            unmatched.append(f"{raw_code} {emp_name}".strip())

        if shift == "NS" and not pairs:
            status = "A"
            calc = calc_salary_from_punches([], daily, on_date=report_date, status=status)
        elif pairs:
            status = "P"
            calc = calc_salary_from_punches(pairs, daily, on_date=report_date, status=status)
        else:
            status = "A"
            calc = calc_salary_from_punches([], daily, on_date=report_date, status=status)

        miss = needs_miss_punch(pairs)
        rows.append(
            {
                "Date": report_date,
                "E_Code": e_code,
                "Raw_E_Code": str(raw_code).strip(),
                "Name": name or emp_name,
                "Shift": shift,
                "Status": status,
                "Daily_Rate_Rs": daily,
                "Needs_Miss_Punch": miss,
                **calc,
            }
        )

    if not rows:
        return {"ok": False, "message": "No attendance rows parsed.", "warnings": warnings}

    existing = get_sheet_df("karigar_attendance")
    if not existing.empty:
        keep = ~(
            (existing["Date"].astype(str) == report_date)
            & (existing["E_Code"].astype(str).map(clean_key).isin([clean_key(x["E_Code"]) for x in rows]))
        )
        existing = existing[keep]
    merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    save_sheet_df("karigar_attendance", merged)

    if unmatched:
        warnings.append(
            f"{len(unmatched)} worker(s) not in master — add E_Code / karigar first: "
            + ", ".join(unmatched[:8])
            + ("…" if len(unmatched) > 8 else "")
        )

    present = sum(1 for x in rows if x.get("Status") == "P")
    return {
        "ok": True,
        "message": f"Imported {len(rows)} attendance row(s) for {report_date} ({present} present).",
        "date": report_date,
        "imported": len(rows),
        "present": present,
        "absent": len(rows) - present,
        "warnings": warnings,
        "unmatched": unmatched[:20],
    }
