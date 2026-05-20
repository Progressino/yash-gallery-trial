"""Stitching Costing business logic (ported from Streamlit v4.3)."""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..db.stitching_db import DATA_KEYS, DEFAULT_SHEETS, HOUR_COLS, get_all_sheets, get_sheet_df, save_sheet_df

IST = timezone(timedelta(hours=5, minutes=30))
HOUR_LBLS = [
    "9-10", "10-11", "11-12", "12-13", "13-14",
    "14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21",
]


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def sticker_in_col(hcol: str) -> str:
    return f"SI_{hcol}"


def sticker_out_col(hcol: str) -> str:
    return f"SO_{hcol}"


def resolve_hour_pieces(entry: dict) -> int:
    """Pieces from sticker in/out (in − out) unless user entered manual pieces."""
    sin = int(entry.get("sticker_in") or 0)
    sout = int(entry.get("sticker_out") or 0)
    manual = bool(entry.get("manual_pieces"))
    explicit = int(entry.get("pieces") or 0)
    if (sin > 0 or sout > 0) and not manual:
        return max(0, sin - sout)
    return explicit


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


def calc_salary(in_str: str, out_str: str, daily_rate: float, ot_mult: float = 1.5) -> dict:
    try:
        fmt = "%H:%M"
        ti = datetime.strptime(in_str.strip(), fmt)
        to_ = datetime.strptime(out_str.strip(), fmt)
        se = datetime.strptime("18:00", fmt)
        ls = datetime.strptime("13:00", fmt)
        le = datetime.strptime("14:00", fmt)
        ph = max(int((to_ - ti).total_seconds()), 0) / 3600
        ld = 1.0 if (ti < le and to_ > ls) else 0.0
        py = max(ph - ld, 0.0)
        hr = daily_rate / 8
        np_ = round(py * hr, 2)
        oh = max(int((to_ - se).total_seconds()), 0) / 3600 if to_ > se else 0.0
        op = round(oh * hr * ot_mult, 2)
        tp = round(np_ + op, 2)
        return {
            "Total_Presence_Hrs": round(ph, 2),
            "Lunch_Deduction_Hrs": round(ld, 2),
            "Payable_Hrs": round(py, 2),
            "Hourly_Rate_Rs": round(hr, 2),
            "Normal_Pay": np_,
            "OT_Hours": round(oh, 2),
            "OT_Pay": op,
            "Total_Pay": tp,
        }
    except Exception:
        return {
            "Total_Presence_Hrs": 0.0,
            "Lunch_Deduction_Hrs": 0.0,
            "Payable_Hrs": 0.0,
            "Hourly_Rate_Rs": 0.0,
            "Normal_Pay": 0.0,
            "OT_Hours": 0.0,
            "OT_Pay": 0.0,
            "Total_Pay": 0.0,
        }


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
    pl["_style"] = pl["Style"].apply(clean_key)
    existing = pl[
        (pl["_date"] == clean_key(date_str))
        & (pl["_kar"] == clean_key(karigar_id))
        & (pl["_challan"] == clean_key(challan_no))
        & (pl["_style"] == clean_key(style))
    ]
    if existing.empty:
        return {"hours": hours, "found": False}

    for _, row in existing.iterrows():
        op_name = str(row["Operation"]).strip()
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
    sm = get_sheet_df("style_master")
    style_ops = sm[sm["Style"] == style] if not sm.empty else pd.DataFrame()
    op_info: dict[str, dict] = {}
    for _, row in style_ops.iterrows():
        op_info[str(row["Operation"])] = {
            "Target": int(row["Target"]),
            "Rate_Rs": float(row["Rate_Rs"]),
            "Hourly_Target": max(1, int(row["Target"])),
        }

    h_vals: dict[str, int] = {}
    si_vals: dict[str, int] = {}
    so_vals: dict[str, int] = {}
    op_vals: dict[str, str | None] = {}
    for e in hour_entries:
        hc = e.get("hour_col") or e.get("hour")
        if hc not in HOUR_COLS:
            continue
        h_vals[hc] = resolve_hour_pieces(e)
        si_vals[hc] = int(e.get("sticker_in") or 0)
        so_vals[hc] = int(e.get("sticker_out") or 0)
        op_vals[hc] = e.get("operation") or None

    if len(op_info) == 1:
        only_op = next(iter(op_info))
        for hc in HOUR_COLS:
            if hc == "H_13_14":
                continue
            if h_vals.get(hc, 0) > 0 and not op_vals.get(hc):
                op_vals[hc] = only_op

    op_totals: dict[str, dict] = defaultdict(lambda: {"pieces": 0, "hours": 0, "value": 0.0})
    for hc in HOUR_COLS:
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
    save_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    rows_added = 0

    for op_name, data in op_totals.items():
        od = op_info[op_name]
        op_eff = round(data["pieces"] / od["Target"] * 100, 1) if od["Target"] > 0 else 0.0
        hour_row = {hcol: (h_vals.get(hcol, 0) if op_vals.get(hcol) == op_name else 0) for hcol in HOUR_COLS}
        sticker_row: dict[str, int] = {}
        for hcol in HOUR_COLS:
            if op_vals.get(hcol) == op_name:
                sticker_row[sticker_in_col(hcol)] = si_vals.get(hcol, 0)
                sticker_row[sticker_out_col(hcol)] = so_vals.get(hcol, 0)
            else:
                sticker_row[sticker_in_col(hcol)] = 0
                sticker_row[sticker_out_col(hcol)] = 0
        budgeted = round(od["Rate_Rs"] * od["Target"], 2)
        actual = round(data["value"], 2)
        new_row = {
            "Date": date_str,
            "Karigar_ID": karigar_id,
            "Karigar_Name": karigar_name,
            "Challan_No": challan_no,
            "Style": style,
            "Operation": op_name,
            **hour_row,
            **sticker_row,
            "Total_Pieces": data["pieces"],
            "Target": od["Target"],
            "Rate_Rs": od["Rate_Rs"],
            "Efficiency_%": op_eff,
            "Piece_Value_Rs": actual,
            "Budgeted_Expense_Rs": budgeted,
            "Actual_Expense_Rs": actual,
            "PL_Rs": round(budgeted - actual, 2),
            "Saved_By": saved_by,
            "Saved_By_Name": saved_by_name,
            "Save_Time": save_time,
        }

        if not log_df.empty:
            log_df = log_df.copy()
            log_df["_ck_date"] = log_df["Date"].apply(clean_key)
            log_df["_ck_kar"] = log_df["Karigar_ID"].apply(clean_key)
            log_df["_ck_challan"] = log_df["Challan_No"].apply(clean_key)
            log_df["_ck_style"] = log_df["Style"].apply(clean_key)
            log_df["_ck_op"] = log_df["Operation"].apply(clean_key)
            keep = ~(
                (log_df["_ck_date"] == clean_key(date_str))
                & (log_df["_ck_kar"] == clean_key(karigar_id))
                & (log_df["_ck_challan"] == clean_key(challan_no))
                & (log_df["_ck_style"] == clean_key(style))
                & (log_df["_ck_op"] == clean_key(op_name))
            )
            log_df = log_df[keep].drop(
                columns=["_ck_date", "_ck_kar", "_ck_challan", "_ck_style", "_ck_op"],
                errors="ignore",
            )

        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        rows_added += 1

    save_sheet_df("production_log", log_df)
    return {"ok": True, "message": f"Saved {rows_added} operation row(s)", "rows_added": rows_added}


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
    att = get_sheet_df("karigar_attendance")
    if att.empty:
        return {"rows": [], "total_payroll": 0}
    ap = att.copy()
    ap["Date_dt"] = pd.to_datetime(ap["Date"])
    ap = ap[(ap["Date_dt"] >= pd.Timestamp(date_from)) & (ap["Date_dt"] <= pd.Timestamp(date_to))]
    if ap.empty:
        return {"rows": [], "total_payroll": 0}
    for c in ["Payable_Hrs", "Normal_Pay", "OT_Hours", "OT_Pay", "Total_Pay"]:
        if c in ap.columns:
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
            Total=("Total_Pay", "sum"),
        )
        .round(2)
        .reset_index()
    )
    return {"rows": pr.to_dict(orient="records"), "total_payroll": float(pr["Total"].sum())}


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


def hour_labels() -> list[dict]:
    return [{"col": c, "label": l} for c, l in zip(HOUR_COLS, HOUR_LBLS)]


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
        vals = safe_num(km.get("Daily_Rate_Rs", 0))
        for kid, val in zip(kids, vals, strict=False):
            if kid:
                rates[str(kid)] = float(val)

    em = get_sheet_df("employee_master")
    if not em.empty and "E_Code" in em.columns:
        kids = em["E_Code"].map(clean_key)
        vals = safe_num(em.get("Daily_Rate_Rs", 0))
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

    hist_cols = [
        c
        for c in [
            "Save_Time",
            "Saved_By_Name",
            "Karigar_Name",
            "Karigar_ID",
            "Challan_No",
            "Style",
            "Operation",
            "Total_Pieces",
            "Actual_Expense_Rs",
            "PL_Rs",
            "Efficiency_%",
        ]
        if c in day_pl.columns
    ]
    history = day_pl.copy()
    if "Save_Time" in history.columns:
        history = history.sort_values("Save_Time", ascending=False)
    history_rows = history[hist_cols].fillna("").to_dict(orient="records") if hist_cols else []

    recent = history_rows[:5] if history_rows else []

    orig_hour_cols = [h for h in HOUR_COLS if h != "H_13_14" and h in day_pl.columns]

    def working_hours(row) -> int:
        return sum(1 for h in orig_hour_cols if safe_num(pd.Series([row.get(h, 0)])).iloc[0] > 0)

    report1_rows = []
    for _, row in day_pl.iterrows():
        wh = working_hours(row)
        hourly_target = float(row.get("Target") or 0)
        rate = float(row.get("Rate_Rs") or 0)
        kid = str(row.get("Karigar_ID", ""))
        daily_salary = _get_daily_salary(kid, date_str)
        hourly_salary = round(daily_salary / 8, 2)
        adj_target = round(hourly_target * wh, 0)
        budgeted_exp = round(rate * hourly_target * wh, 2)
        actual_exp = round(hourly_salary * wh, 2)
        pl_val = round(budgeted_exp - actual_exp, 2)
        total_pcs = float(row.get("Total_Pieces") or 0)
        efficiency = round(total_pcs / adj_target * 100, 1) if adj_target > 0 else 0.0

        hour_vals = {}
        for hcol, hlbl in zip(HOUR_COLS, HOUR_LBLS):
            if hcol in row.index and hcol != "H_13_14":
                v = int(safe_num(pd.Series([row.get(hcol, 0)])).iloc[0])
                if v > 0:
                    hour_vals[hlbl] = v

        report1_rows.append(
            {
                "Karigar_Name": row.get("Karigar_Name", ""),
                "Karigar_ID": kid,
                "Challan_No": row.get("Challan_No", ""),
                "Style": row.get("Style", ""),
                "Operation": row.get("Operation", ""),
                "hours": hour_vals,
                "Working_Hours": wh,
                "Total_Pieces": int(total_pcs),
                "Adj_Target": int(adj_target),
                "Efficiency_%": efficiency,
                "Daily_Salary_Rs": daily_salary,
                "Hourly_Salary_Rs": hourly_salary,
                "Budgeted_Expense_Rs": budgeted_exp,
                "Actual_Expense_Rs": actual_exp,
                "PL_Rs": pl_val,
                "Saved_By_Name": row.get("Saved_By_Name", ""),
                "Save_Time": row.get("Save_Time", ""),
            }
        )

    report2_rows = []
    for _, row in day_pl.iterrows():
        kid = str(row.get("Karigar_ID", ""))
        kar_name = str(row.get("Karigar_Name", ""))
        op_name = str(row.get("Operation", ""))
        rate_rs = float(row.get("Rate_Rs") or 0)
        hourly_target = float(row.get("Target") or 0)

        daily_rate = _get_daily_salary(kid, date_str)
        hourly_sal = round(daily_rate / 8, 2)

        for hcol, hlbl in zip(HOUR_COLS, HOUR_LBLS):
            if hcol == "H_13_14" or hcol not in row.index:
                continue
            pcs = int(safe_num(pd.Series([row.get(hcol, 0)])).iloc[0])
            if pcs <= 0:
                continue
            actual_piece_val = round(pcs * rate_rs, 2)
            target_piece_val = round(hourly_target * rate_rs, 2)
            net_pl = round(actual_piece_val - hourly_sal, 2)
            report2_rows.append(
                {
                    "Karigar": kar_name,
                    "Karigar_ID": kid,
                    "Challan_No": str(row.get("Challan_No", "")),
                    "Style": str(row.get("Style", "")),
                    "Hour": hlbl,
                    "Operation": op_name,
                    "Daily_Salary_Rs": daily_rate,
                    "Hourly_Salary_Rs": hourly_sal,
                    "Pieces_Done": pcs,
                    "Hourly_Target_Pcs": int(hourly_target),
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
                Karigar_ID=("Karigar_ID", "first"),
                Challan_No=("Challan_No", "first"),
                Style=("Style", "first"),
                Hours_Worked=("Hour", "count"),
                Total_Pieces=("Pieces_Done", "sum"),
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

    return {
        "date": date_str,
        "karigar_id": karigar_id or "",
        "history": history_rows,
        "recent_saves": recent,
        "report1": report1_rows,
        "report2_hourly": report2_rows,
        "report2_summary": report2_summary,
        "grand_total": grand_total,
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
    perf["Salary"] = perf["Salary"].round(2)
    perf["Surplus"] = (perf["Piece_Value"] - perf["Salary"]).round(2)
    perf["ROI_%"] = (perf["Piece_Value"] / perf["Salary"].replace(0, 1) * 100).round(1)
    perf["Avg_Eff"] = perf["Avg_Eff"].round(1)
    perf["Grade"] = perf["Avg_Eff"].apply(
        lambda x: "A–Excellent"
        if x >= 100
        else ("B–Good" if x >= 85 else ("C–Average" if x >= 70 else "D–Needs Improvement"))
    )

    summary = {
        "total_piece_value": float(perf["Piece_Value"].sum()),
        "total_salary": float(perf["Salary"].sum()),
        "net_surplus": float(perf["Surplus"].sum()),
    }
    return {
        "ok": True,
        "rows": perf.fillna("").to_dict(orient="records"),
        "summary": summary,
    }


def update_style_operation(style: str, operation: str, *, target: int | None, rate_rs: float | None) -> dict:
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
    save_sheet_df("style_master", df)
    return {"ok": True, "message": "Updated"}


MASTER_KEY_COLS: dict[str, list[str]] = {
    "style_master": ["Style", "Operation"],
    "karigar_master": ["Karigar_ID"],
    "employee_master": ["E_Code"],
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


def add_style_operation_row(style: str, operation: str, target: int, rate_rs: float) -> dict:
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
    df = pd.concat(
        [df, pd.DataFrame([{"Style": style_s, "Operation": op_s, "Target": int(target), "Rate_Rs": float(rate_rs)}])],
        ignore_index=True,
    )
    save_sheet_df("style_master", dedupe_sheet("style_master", df))
    return {"ok": True, "message": "Style operation added"}


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
        return {"ok": True, "message": f"Rate ₹{rate}/day applies from {eff} (synced across app)"}
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
