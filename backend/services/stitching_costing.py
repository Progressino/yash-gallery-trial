"""Stitching Costing business logic (ported from Streamlit v4.3)."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..db.stitching_db import DATA_KEYS, HOUR_COLS, get_all_sheets, get_sheet_df, save_sheet_df

IST = timezone(timedelta(hours=5, minutes=30))
HOUR_LBLS = [
    "9-10", "10-11", "11-12", "12-13", "13-14",
    "14-15", "15-16", "16-17", "17-18", "18-19", "19-20", "20-21",
]


def safe_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


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

    karigar_status = []
    if not km.empty:
        aids = tdpl["Karigar_ID"].astype(str).unique().tolist() if not tdpl.empty else []
        for _, r in km.iterrows():
            kid = str(r["Karigar_ID"])
            karigar_status.append({
                "Karigar_ID": kid,
                "Name": str(r.get("Name", "")),
                "Skill": str(r.get("Skill", "")),
                "Daily_Rate_Rs": float(r.get("Daily_Rate_Rs", 0) or 0),
                "Status": "Working" if kid in aids else "Idle",
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
    hours: dict[str, dict] = {h: {"operation": "", "pieces": 0} for h in HOUR_COLS}
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
            if val > 0:
                hours[hcol] = {"operation": op_name, "pieces": val}
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
    """hour_entries: [{hour_col, operation, pieces}, ...]"""
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
    op_vals: dict[str, str | None] = {}
    for e in hour_entries:
        hc = e.get("hour_col") or e.get("hour")
        if hc not in HOUR_COLS:
            continue
        h_vals[hc] = int(e.get("pieces") or 0)
        op_vals[hc] = e.get("operation") or None

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
            "Total_Pieces": data["pieces"],
            "Target": od["Target"],
            "Rate_Rs": od["Rate_Rs"],
            "Efficiency_%": op_eff,
            "Piece_Value_Rs": actual,
            "Budgeted_Expense_Rs": budgeted,
            "Actual_Expense_Rs": actual,
            "PL_Rs": round(actual - budgeted, 2),
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
    cm_sc["Pending"] = cm_sc["Total_Qty"] - cm_sc.get("Received_Qty", 0)
    cm_sc["Is_Pending"] = cm_sc["Pending"] > 0

    if not sm.empty:
        target_rate = sm.groupby("Style")["Rate_Rs"].sum().reset_index()
        target_rate.columns = ["Style", "Target_Labour_Rate_Pc"]
        cm_sc = cm_sc.merge(target_rate, on="Style", how="left").fillna({"Target_Labour_Rate_Pc": 0})
    else:
        cm_sc["Target_Labour_Rate_Pc"] = 0

    if not pl.empty and "Piece_Value_Rs" in pl.columns:
        pl_sc = pl.copy()
        pl_sc["Piece_Value_Rs"] = safe_num(pl_sc["Piece_Value_Rs"])
        actual_exp = pl_sc.groupby("Challan_No")["Piece_Value_Rs"].sum().reset_index()
        actual_exp.columns = ["Challan_No", "Actual_Labour_Rs"]
        actual_exp["Challan_No"] = actual_exp["Challan_No"].astype(str)
        cm_sc["Challan_No"] = cm_sc["Challan_No"].astype(str)
        cm_sc = cm_sc.merge(actual_exp, on="Challan_No", how="left").fillna({"Actual_Labour_Rs": 0})
    else:
        cm_sc["Actual_Labour_Rs"] = 0

    cm_sc["Target_Labour_Rs"] = (cm_sc["Target_Labour_Rate_Pc"] * cm_sc["Total_Qty"]).round(2)
    cm_sc["Party_Value_Rs"] = (cm_sc["Rate_Per_Pc"] * cm_sc["Total_Qty"]).round(2)
    cm_sc["Total_Expense_Rs"] = (cm_sc["Actual_Labour_Rs"] + cm_sc["Deposit_Rs"]).round(2)
    cm_sc["PL_Rs"] = (cm_sc["Party_Value_Rs"] - cm_sc["Total_Expense_Rs"]).round(2)
    cm_sc["PL_Per_Pc"] = (cm_sc["PL_Rs"] / cm_sc["Total_Qty"].replace(0, 1)).round(2)
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
            Qty=("Total_Qty", "sum"),
            Actual_Labour=("Actual_Labour_Rs", "sum"),
            Party_Value=("Party_Value_Rs", "sum"),
            Total_Expense=("Total_Expense_Rs", "sum"),
            PL=("PL_Rs", "sum"),
            Pending_Challans=("Is_Pending", "sum"),
        )
        .reset_index()
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
            k = nm.replace(".csv", "").strip("/")
            if k in DATA_KEYS:
                df = pd.read_csv(io.StringIO(zf.read(nm).decode()))
                save_sheet_df(k, df)
    return {"ok": True, "message": "All data restored"}


def hour_labels() -> list[dict]:
    return [{"col": c, "label": l} for c, l in zip(HOUR_COLS, HOUR_LBLS)]
