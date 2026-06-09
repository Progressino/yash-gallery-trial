"""Karigar attendance policy, punch parsing, and biometric sheet import."""
from __future__ import annotations

import json
import math
import os
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
EARLY_LEAVE_GRACE_MINUTES = 5  # ignore tiny early-outs (e.g. 17:58 vs 18:00)
def _late_arrival_grace_minutes() -> int:
    """Minutes late (after 09:00) with no salary cut — factory rulebook default 17."""
    try:
        return max(0, int((os.environ.get("ATTENDANCE_LATE_GRACE_MIN") or "17").strip()))
    except ValueError:
        return 17
# Late-out grace at shift end and OT cap (e.g. 17:58≈18:00, 20:59≈21:00)
LATE_OUT_GRACE_MINUTES = 5
# If worker leaves early at/before 16:00, apply extra 30-minute deduction.
EARLY_LEAVE_16_PENALTY_MINUTES = 30

# Sunday policy: 09:00–16:00 wall clock, 13:00–14:00 lunch → 6h payable basis.
SUNDAY_WORK_END = time(16, 0)
SUNDAY_LUNCH_START = time(13, 0)
SUNDAY_LUNCH_END = time(14, 0)
SUNDAY_PAYABLE_MINUTES = 6 * 60
SUNDAY_LAST_PRODUCTION_HOUR = "H_15_16"
# work_minutes uses WORK_BLOCKS (already excludes lunch/tea slots); near-full days skip extra lunch cut
NEAR_FULL_BLOCK_MINUTES = STANDARD_PAY_MINUTES - 15

WORK_BLOCKS = (
    (WORK_START, TEA1_START),
    (TEA1_END, LUNCH_START),
    (LUNCH_END, TEA2_START),
    (TEA2_END, WORK_END),
)


def _as_date(on_date: str | date | None) -> date:
    if isinstance(on_date, date):
        return on_date
    return pd.to_datetime(on_date or date.today()).date()


def hourly_rate_from_daily(daily_rate: float, on_date: str | date | None = None) -> float:
    """Hourly rate for payroll/costing: daily ÷ 8 on weekdays, daily ÷ 6 on Sunday."""
    pol = _policy_for_date(_as_date(on_date))
    hours = pol["standard_pay_minutes"] / 60.0
    return round(float(daily_rate or 0) / hours, 4) if hours > 0 else 0.0


def production_hour_cols_for_date(on_date: str | date) -> list[str]:
    """Production hour columns allowed on this date (Sunday ends at 15–16 / 16:00)."""
    from ..db.stitching_db import HOUR_COLS

    base = _as_date(on_date)
    if base.weekday() != 6:
        return list(HOUR_COLS)
    return [h for h in HOUR_COLS if h <= SUNDAY_LAST_PRODUCTION_HOUR or h == "H_13_14"]


def _policy_for_date(base: date) -> dict[str, Any]:
    """Dynamic shift policy. Sunday uses 09:00–16:00 with 13:00–14:00 lunch (6h payable basis)."""
    is_sunday = base.weekday() == 6
    if not is_sunday:
        return {
            "is_sunday": False,
            "work_end": WORK_END,
            "ot_start": OT_START,
            "ot_end": OT_END,
            "lunch_start": LUNCH_START,
            "lunch_end": LUNCH_END,
            "lunch_default_min": float(DEFAULT_LUNCH_BREAK_MINUTES),
            "standard_pay_minutes": float(STANDARD_PAY_MINUTES),
            "near_full_block_minutes": float(NEAR_FULL_BLOCK_MINUTES),
            "work_blocks": list(WORK_BLOCKS),
            "sunday_waive_tea": False,
        }
    blocks = [
        # Sunday basis is 7 hours with only lunch as mid break (manual rulebook).
        (WORK_START, SUNDAY_LUNCH_START),
        (SUNDAY_LUNCH_END, SUNDAY_WORK_END),
    ]
    return {
        "is_sunday": True,
        "work_end": SUNDAY_WORK_END,
        "ot_start": SUNDAY_WORK_END,
        "ot_end": OT_END,
        "lunch_start": SUNDAY_LUNCH_START,
        "lunch_end": SUNDAY_LUNCH_END,
        "lunch_default_min": 60.0,
        "standard_pay_minutes": float(SUNDAY_PAYABLE_MINUTES),
        "near_full_block_minutes": float(SUNDAY_PAYABLE_MINUTES - 15),
        "work_blocks": blocks,
        "sunday_waive_tea": True,
    }


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


def _late_deduction_hrs_display(late_min: float) -> float:
    """Show small lates as hundredths (2 min → 0.02); larger lates as fractional hours."""
    if late_min <= 0:
        return 0.0
    lm = int(round(late_min))
    if lm < 10:
        return round(lm / 100.0, 2)
    return round(late_min / 60.0, 2)


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
    default_out = _policy_for_date(base)["work_end"]
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

    Regular pay uses 8h/day (weekday) or 6h/day (Sunday); hourly = daily ÷ shift hours.
    OT uses the same hourly rate (no multiplier).
    """
    base = _as_date(on_date)
    pol = _policy_for_date(base)
    hourly = hourly_rate_from_daily(daily_rate, base)
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

    intervals = _intervals_from_pairs(punch_pairs, base)
    if not intervals:
        return zero

    raw_first_in = min(s for s, _ in intervals).time()
    late_min_raw = 0.0
    if raw_first_in > WORK_START:
        late_min_raw = (
            datetime.combine(base, raw_first_in) - datetime.combine(base, WORK_START)
        ).total_seconds() / 60.0

    # First punch before 09:00 → 09:00; within late grace → 09:00 for block hours (no pay cut).
    grace = _late_arrival_grace_minutes()
    adjusted: list[tuple[datetime, datetime]] = []
    for idx, (start, end) in enumerate(intervals):
        if idx == 0:
            if start.time() < WORK_START:
                start = _dt_on(base, WORK_START)
            elif start.time() > WORK_START:
                late_in_raw = (
                    datetime.combine(base, start.time()) - datetime.combine(base, WORK_START)
                ).total_seconds() / 60.0
                if late_in_raw <= grace:
                    start = _dt_on(base, WORK_START)
        adjusted.append((start, end))
    intervals = _merge_intervals(adjusted)

    total_presence_min = sum((e - s).total_seconds() / 60.0 for s, e in intervals)

    work_minutes = 0.0
    for block_start, block_end in pol["work_blocks"]:
        bs = _dt_on(base, block_start)
        be = _dt_on(base, block_end)
        for s, e in intervals:
            work_minutes += _overlap_minutes(s, e, bs, be)

    lunch_penalty = 0.0
    lunch_penalty_display = 0.0
    tea_penalty = 0.0
    special_break_penalty = _left_before_lunch_return_at_tea(intervals, base)
    lunch_window = (_dt_on(base, pol["lunch_start"]), _dt_on(base, pol["lunch_end"]))
    tea1_window = (_dt_on(base, TEA1_START), _dt_on(base, TEA1_END))
    tea2_window = (_dt_on(base, TEA2_START), _dt_on(base, TEA2_END))
    # Lunch: deduct if they stayed clocked in through lunch (no break). Tea: deduct if absent from floor.
    worked_through_lunch = _present_during(intervals, *lunch_window)
    tea1_absent = not _present_during(intervals, *tea1_window)
    tea2_absent = not _present_during(intervals, *tea2_window)

    if pol.get("is_sunday"):
        # Sunday rulebook: lunch is a fixed 13:00–14:00 mid break; work blocks already exclude it,
        # so don't deduct it again from payable minutes (but still show it in the sheet).
        lunch_penalty_display = float(pol["lunch_default_min"])
        if waive_lunch_break:
            lunch_penalty = 0.0
        elif lunch_break_minutes is not None:
            # Explicit override, apply as provided.
            lunch_penalty = max(0.0, float(lunch_break_minutes))
        else:
            lunch_penalty = 0.0
    else:
        lunch_penalty_display = 0.0
        if waive_lunch_break:
            lunch_penalty = 0.0
        elif lunch_break_minutes is not None:
            lunch_penalty = max(0.0, float(lunch_break_minutes))
        elif special_break_penalty:
            lunch_penalty = float(pol["lunch_default_min"])
        elif worked_through_lunch:
            lunch_penalty = float(pol["lunch_default_min"])

    if pol.get("sunday_waive_tea"):
        tea_penalty = 0.0
    elif waive_tea_break:
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

    first_in = raw_first_in
    last_out = max(e for _, e in intervals).time()
    late_min = 0.0 if late_min_raw <= grace else late_min_raw
    # Overtime: after 18:00 until actual out (cap 21:00), same hourly rate as regular (daily/8).
    last_out_dt = max(e for _, e in intervals)
    # Late-out grace: treat small differences as exact shift end / OT cap.
    shift_end_dt = _dt_on(base, pol["work_end"])
    if shift_end_dt - timedelta(minutes=LATE_OUT_GRACE_MINUTES) <= last_out_dt <= shift_end_dt:
        last_out_dt = shift_end_dt
    ot_end_dt = _dt_on(base, pol["ot_end"])
    if ot_end_dt - timedelta(minutes=LATE_OUT_GRACE_MINUTES) <= last_out_dt <= ot_end_dt:
        last_out_dt = ot_end_dt

    ot_start = _dt_on(base, pol["ot_start"])
    ot_end_cap = min(last_out_dt, _dt_on(base, pol["ot_end"]))
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
    if last_out < pol["work_end"] and ot_minutes <= 0:
        early_min = (
            datetime.combine(base, pol["work_end"]) - datetime.combine(base, last_out)
        ).total_seconds() / 60.0
    early_min_payable = 0.0 if early_min <= EARLY_LEAVE_GRACE_MINUTES else early_min
    # Rulebook: weekday early-out at/before 16:00 → extra 30m; Sunday shift ends at 16:00 (no penalty).
    early_16_penalty = 0.0
    if not pol.get("is_sunday") and last_out <= time(16, 0):
        early_16_penalty = float(EARLY_LEAVE_16_PENALTY_MINUTES)

    work_net_minutes = max(work_minutes - lunch_penalty - tea_penalty, 0.0)

    # Full shift + OT with only grace-level lateness → master daily rate + OT (manual sheet).
    full_day_with_ot = (
        ot_hrs_bill > 0
        and work_minutes >= pol["near_full_block_minutes"]
        and late_min_raw <= grace
    )

    if full_day_with_ot:
        payable_minutes = float(pol["standard_pay_minutes"])
        normal_pay = round(float(daily_rate), 2)
    elif late_min > 0 or work_minutes < NEAR_FULL_BLOCK_MINUTES:
        shift_net_minutes = max(
            float(pol["standard_pay_minutes"])
            - lunch_penalty
            - tea_penalty
            - late_min
            - early_min_payable
            - early_16_penalty,
            0.0,
        )
        payable_minutes = min(shift_net_minutes, work_net_minutes)
        payable_minutes = min(payable_minutes, float(pol["standard_pay_minutes"]))
        normal_pay = round((payable_minutes / float(pol["standard_pay_minutes"])) * daily_rate, 2)
    else:
        # Near-full day within grace: pay full master daily rate (manual salary sheet).
        near_full_grace_day = (
            work_minutes >= pol["near_full_block_minutes"]
            and late_min_raw <= grace
            and early_min <= EARLY_LEAVE_GRACE_MINUTES
        )
        if near_full_grace_day:
            payable_minutes = float(pol["standard_pay_minutes"])
            normal_pay = round(float(daily_rate), 2)
        else:
            # On-time in: WORK_BLOCKS already exclude scheduled breaks.
            payable_minutes = max(work_minutes - early_min_payable, 0.0)
            payable_minutes = min(payable_minutes, float(pol["standard_pay_minutes"]))
            normal_pay = round((payable_minutes / float(pol["standard_pay_minutes"])) * daily_rate, 2)
    if ot_hrs_bill <= 0:
        normal_pay = min(normal_pay, round(float(daily_rate), 2))
    payable_hrs = round(payable_minutes / 60.0, 2)

    punch_pairs_json = serialize_punch_pairs(punch_pairs)

    late_for_money = 0.0 if late_min_raw <= grace else late_min_raw
    late_deduction_rs = round((late_for_money / 60.0) * hourly, 2)
    early_deduction_rs = round((early_min_payable / 60.0) * hourly, 2)

    return {
        "Status": status,
        "Total_Presence_Hrs": round(total_presence_min / 60.0, 2),
        "Payable_Hrs": payable_hrs,
        "Hourly_Rate_Rs": round(hourly, 2),
        "Normal_Pay": normal_pay,
        "OT_Hours": ot_hrs,
        "OT_Pay": ot_pay,
        "Total_Pay": round(normal_pay + ot_pay, 2),
        "Lunch_Deduction_Hrs": round((lunch_penalty_display or lunch_penalty) / 60.0, 2),
        "Tea_Deduction_Hrs": round(tea_penalty / 60.0, 2),
        "Break_Penalty_Hrs": round((lunch_penalty + tea_penalty) / 60.0, 2),
        "Late_Deduction_Hrs": _late_deduction_hrs_display(late_min_raw),
        "Early_Deduction_Hrs": round(early_min_payable / 60.0, 2),
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


_CLOCK_ONLY_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def _parse_month_day_year_text(text: str) -> str:
    """Jun 08 2026 or Jun 08 2026  To  Jun 08 2026 → YYYY-MM-DD (uses first date)."""
    s = str(text or "").strip()
    if not s:
        return ""
    m = re.search(r"([A-Za-z]{3,})\s+(\d{1,2})\s+(\d{4})", s, re.I)
    if not m:
        return ""
    token = f"{m.group(1)} {int(m.group(2))} {m.group(3)}"
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return pd.to_datetime(token, format=fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return ""


def _cell_to_report_date(cell: Any) -> str:
    """Parse one spreadsheet cell as YYYY-MM-DD when possible."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return ""
    if isinstance(cell, (datetime, pd.Timestamp)):
        try:
            ts = pd.Timestamp(cell)
            # Excel time-only cells can become 1899-12-30 / today's date — reject clock times.
            if ts.hour or ts.minute or ts.second:
                if ts.year < 1900 or (ts.hour, ts.minute) != (0, 0):
                    return ""
            return ts.strftime("%Y-%m-%d")
        except Exception:
            return ""
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "none", "-"):
        return ""
    if _CLOCK_ONLY_RE.match(s):
        return ""
    month_hit = _parse_month_day_year_text(s)
    if month_hit:
        return month_hit
    m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", s, re.I)
    if m:
        try:
            return pd.to_datetime(m.group(1), dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return m.group(1)
    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", s)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        try:
            return pd.to_datetime(f"{d}-{mo}-{y}", dayfirst=True).strftime("%Y-%m-%d")
        except Exception:
            pass
    try:
        parsed = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(parsed):
            return pd.Timestamp(parsed).strftime("%Y-%m-%d")
    except Exception:
        pass
    return ""


_SKIP_DATE_ROW_KEYWORDS = (
    "generated",
    "printed",
    "export",
    "run at",
    "created at",
    "downloaded",
    "prepared by",
    "report time",
    "print time",
)


def _is_attendance_date_label(cell: Any) -> bool:
    """True for header cells that label the attendance day (not arbitrary 'update date' text)."""
    s = str(cell or "").strip().lower()
    if not s:
        return False
    if re.fullmatch(r"date\s*:?", s):
        return True
    return bool(re.match(r"^(report|attendance)\s+date\s*:?", s))


def _extract_report_date(raw: pd.DataFrame) -> str:
    """Read attendance report date from biometric export header (not upload day)."""
    if raw is None or raw.empty:
        return ""
    max_rows = min(25, len(raw))
    max_cols = min(16, raw.shape[1])
    header_row = _find_header_row(raw)
    # Never scan punch rows (IN/OUT times like 08:58 were mis-read as the report date).
    header_scan_end = min(max_rows, max(1, header_row))
    # Pass 1: explicit Date label row (value may be same cell or next columns).
    for i in range(header_scan_end):
        for j in range(max_cols):
            cell = raw.iloc[i, j]
            if not _is_attendance_date_label(cell):
                continue
            found = _cell_to_report_date(cell)
            if found:
                return found
            for k in range(j + 1, min(j + 6, max_cols)):
                found = _cell_to_report_date(raw.iloc[i, k])
                if found:
                    return found
    # Pass 2: title/header block only — skip rows that look like print/export timestamps.
    for i in range(header_scan_end):
        row_text = " ".join(str(raw.iloc[i, j] or "") for j in range(max_cols)).lower()
        if any(kw in row_text for kw in _SKIP_DATE_ROW_KEYWORDS):
            continue
        if "e. code" in row_text or row_text.strip() == "e code":
            continue
        for j in range(max_cols):
            found = _cell_to_report_date(raw.iloc[i, j])
            if found:
                return found
    return ""


def _date_from_filename(filename: str) -> str:
    fn = (filename or "").strip()
    if not fn:
        return ""
    # common: 02-06-2026.xls / 2_6_26.xlsx / attendance 02.06.2026.xls
    m = re.search(r"(\d{1,2})[-/_\.](\d{1,2})[-/_\.](\d{2,4})", fn)
    if not m:
        return ""
    d, mth, y = m.group(1), m.group(2), m.group(3)
    if len(y) == 2:
        y = "20" + y
    try:
        return pd.to_datetime(f"{d}-{mth}-{y}", dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _find_header_row(raw: pd.DataFrame) -> int:
    for i in range(min(20, len(raw))):
        row = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if any("e. code" in x or x == "e code" for x in row):
            return i
    return 8


def _normalize_report_date_override(value: str) -> str:
    """Optional YYYY-MM-DD from upload form — must not default to today."""
    s = str(value or "").strip()
    if not s:
        return ""
    try:
        return pd.Timestamp(pd.to_datetime(s).normalize()).strftime("%Y-%m-%d")
    except Exception:
        return ""


def parse_inout_punch_report(
    raw: bytes,
    filename: str = "",
    report_date_override: str = "",
) -> tuple[str, pd.DataFrame, list[str]]:
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
    override = _normalize_report_date_override(report_date_override)
    if override:
        report_date = override
    else:
        from_header = _extract_report_date(preview)
        from_name = _date_from_filename(filename)
        if from_header and from_name and from_header != from_name:
            # Filename is usually operator-named for the attendance day (e.g. 8-6-26.xlsx).
            report_date = from_name
            warnings.append(
                f"Sheet header implied {from_header} but filename implies {from_name}; "
                f"using {from_name}."
            )
        else:
            report_date = from_header or from_name
    if not report_date:
        raise ValueError(
            "Could not read attendance date from the sheet or filename. "
            "Use a file named like 03-06-2026.xls, ensure the report header shows the attendance date, "
            "or pick the attendance date before uploading."
        )
    header_row = _find_header_row(preview)
    bio.seek(0)
    if fn.endswith(".xls"):
        df = pd.read_excel(bio, sheet_name=0, header=header_row, engine="xlrd")
    else:
        df = pd.read_excel(bio, sheet_name=0, header=header_row)
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return report_date, df, warnings


def recalculate_attendance_for_date(on_date: str) -> dict[str, Any]:
    """Re-run payroll policy on all karigar_attendance rows for one date."""
    df = get_sheet_df("karigar_attendance")
    if df.empty:
        return {"ok": False, "message": f"No attendance rows for {on_date}.", "updated": 0}
    mask = df["Date"].astype(str) == str(on_date)
    if not mask.any():
        return {"ok": False, "message": f"No attendance rows for {on_date}.", "updated": 0}

    updated = 0
    for idx in df.index[mask]:
        row = df.loc[idx]
        status = str(row.get("Status") or "P").strip().upper()
        e_code = str(row.get("E_Code") or "")
        daily = float(row.get("Daily_Rate_Rs") or 0)
        if daily <= 0:
            daily = get_daily_rate_for_date(e_code, on_date)
        pairs = deserialize_punch_pairs(row.get("Punch_Pairs"))
        if not pairs:
            tin = _parse_clock(row.get("In_Punch"))
            tout = _parse_clock(row.get("Out_Punch"))
            if tin is not None:
                pairs = [(tin, tout)]
        calc = calc_salary_from_punches(pairs, daily, on_date=on_date, status=status)
        for key, val in calc.items():
            df.at[idx, key] = val
        df.at[idx, "Daily_Rate_Rs"] = daily
        df.at[idx, "Needs_Miss_Punch"] = needs_miss_punch(pairs)
        updated += 1

    save_sheet_df("karigar_attendance", df)
    return {
        "ok": True,
        "message": f"Recalculated payroll for {updated} attendance row(s) on {on_date}.",
        "date": on_date,
        "updated": updated,
    }


def import_karigar_attendance_bytes(
    raw: bytes,
    filename: str = "",
    report_date_override: str = "",
) -> dict[str, Any]:
    """Import biometric attendance; upserts by Date + E_Code."""
    warnings: list[str] = []
    try:
        report_date, df, warnings = parse_inout_punch_report(
            raw, filename, report_date_override=report_date_override
        )
    except ValueError as exc:
        return {"ok": False, "message": str(exc), "warnings": warnings}
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
        has_in_punch = any(tin is not None for tin, _ in pairs)
        e_code, name, master_rate = match_employee_code(raw_code, emp_name)
        daily = master_rate or get_daily_rate_for_date(e_code, report_date)
        if daily <= 0 and master_rate <= 0:
            unmatched.append(f"{raw_code} {emp_name}".strip())

        if not has_in_punch:
            status = "A"
            pairs = []
            calc = calc_salary_from_punches([], daily, on_date=report_date, status=status)
        else:
            status = "P"
            calc = calc_salary_from_punches(pairs, daily, on_date=report_date, status=status)

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
