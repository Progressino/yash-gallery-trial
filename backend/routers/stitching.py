"""Stitching Costing API — karigar production, challans, payroll."""
from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..db.stitching_db import (
    DATA_KEYS,
    change_admin_password,
    get_all_sheets,
    get_sheet_df,
    save_sheet_df,
    save_sheet_rows,
    verify_admin_password,
)
from ..services import stitching_costing as svc
from ..services.stitching_gsheet import gsheet_status, sync_from_gsheet, sync_from_gsheet_merge, sync_to_gsheet

router = APIRouter()


class SheetReplaceBody(BaseModel):
    rows: list[dict]


class ProductionEntryBody(BaseModel):
    date: str
    karigar_id: str
    karigar_name: str
    challan_no: str
    style: str
    hour_entries: list[dict] = Field(default_factory=list)
    saved_by: str = "erp"
    saved_by_name: str = ""


class ChallanBody(BaseModel):
    Challan_No: str
    Style: str
    Party: str = ""
    Total_Qty: int
    Received_Qty: int = 0
    Deposit_Rs: float = 0.0
    Rate_Per_Pc: float = 35.0
    Date: str = ""
    Delivery_By: str = ""


class ChallanUpdateBody(BaseModel):
    Total_Qty: Optional[int] = None
    Received_Qty: Optional[int] = None
    Deposit_Rs: Optional[float] = None
    Rate_Per_Pc: Optional[float] = None


class AttendanceBody(BaseModel):
    Date: str
    E_Code: str
    Name: str = ""
    In_Punch: str = "09:00"
    Out_Punch: str = "18:00"
    ot_multiplier: float = 1.0


class PunchPairBody(BaseModel):
    in_time: str
    out_time: str = ""


class AttendancePatchBody(BaseModel):
    Date: str
    E_Code: str
    In_Punch: str = ""
    Out_Punch: str = ""
    punch_pairs: list[PunchPairBody] = Field(default_factory=list)
    Waive_Lunch_Break: bool = False
    Waive_Tea_Break: bool = False
    Lunch_Break_Minutes: Optional[float] = None
    Tea_Break_Minutes: Optional[float] = None


class StyleOpBody(BaseModel):
    Style: str
    Operation: str
    Target: int
    Rate_Rs: float


class KarigarBody(BaseModel):
    Karigar_ID: str
    Name: str
    Skill: str = "Stitching"
    Daily_Rate_Rs: float = 420.0
    Effective_From: str = ""


class KarigarUpdateBody(BaseModel):
    Name: Optional[str] = None
    Skill: Optional[str] = None
    Daily_Rate_Rs: Optional[float] = None
    Effective_From: Optional[str] = None


class MasterDeleteBody(BaseModel):
    sheet: str
    rows: list[dict]


class AdminPasswordBody(BaseModel):
    password: str


class AdminChangePasswordBody(BaseModel):
    current: str
    new_password: str
    confirm: str


class StyleOpUpdateBody(BaseModel):
    Style: str
    Operation: str
    Target: Optional[int] = None
    Rate_Rs: Optional[float] = None
    admin_password: str


class LtlOverrideBody(BaseModel):
    Style: str
    Operation: str
    Karigar_ID: str
    Manual_LTL: Optional[int] = None
    Notes: str = ""
    admin_password: str = ""


class ProductionDeleteBody(BaseModel):
    date: str
    karigar_id: str
    challan_no: str = ""
    style: str = ""
    operation: str = ""
    admin_password: str = ""


class KarigarExpenseBody(BaseModel):
    Date: str
    Karigar_ID: str
    Work_Type: str
    Challan_No: str = ""
    Style: str = ""
    Amount_Rs: float = 0
    Hours: float = 0
    Notes: str = ""
    Expense_ID: str = ""
    admin_password: str = ""


@router.get("/status")
def status():
    sheets = {k: len(get_sheet_df(k)) for k in DATA_KEYS}
    return {"ok": True, "sheets": sheets, "gsheet": gsheet_status(), "hours": svc.hour_labels()}


@router.get("/dashboard")
def dashboard(planning_date: str = ""):
    return svc.dashboard_summary(planning_date or None)


@router.get("/sheets")
def list_sheets():
    return get_all_sheets()


@router.get("/sheets/{key}")
def get_sheet(key: str):
    if key not in DATA_KEYS:
        raise HTTPException(404, f"Unknown sheet {key}")
    df = get_sheet_df(key)
    return {"key": key, "rows": df.fillna("").to_dict(orient="records"), "columns": list(df.columns)}


@router.put("/sheets/{key}")
def put_sheet(key: str, body: SheetReplaceBody):
    if key not in DATA_KEYS:
        raise HTTPException(404, f"Unknown sheet {key}")
    save_sheet_rows(key, body.rows)
    return {"ok": True, "rows": len(body.rows)}


@router.post("/sheets/{key}/import-file")
async def import_sheet_file(key: str, file: UploadFile = File(...), mode: str = "replace"):
    if key not in DATA_KEYS:
        raise HTTPException(404, f"Unknown sheet {key}")
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        raise HTTPException(400, "Use .csv or .xlsx")
    existing = get_sheet_df(key)
    if mode == "merge":
        df = svc.merge_sheet_dataframes(key, existing, df)
    elif mode == "append" and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
        if key in svc.MASTER_KEY_COLS:
            df = svc.dedupe_sheet(key, df)
    elif key in svc.MASTER_KEY_COLS:
        df = svc.dedupe_sheet(key, df)
    if key == "challan_master":
        df = svc.normalize_challan_dataframe(df)
    save_sheet_df(key, df)
    return {"ok": True, "rows": len(df), "mode": mode}


@router.get("/production-entry/load")
def load_entry(date: str, karigar_id: str, challan_no: str, style: str):
    return svc.load_production_entry(date, karigar_id, challan_no, style)


@router.get("/production-entry/reports")
def production_entry_reports(date: str, karigar_id: str = ""):
    return svc.production_entry_reports(date, karigar_id or None)


@router.post("/production-entry")
def save_entry(body: ProductionEntryBody):
    return svc.save_production_entry(
        date_str=body.date,
        karigar_id=body.karigar_id,
        karigar_name=body.karigar_name,
        challan_no=body.challan_no,
        style=body.style,
        hour_entries=body.hour_entries,
        saved_by=body.saved_by,
        saved_by_name=body.saved_by_name,
    )


@router.get("/production-entry/admin/sessions")
def production_admin_sessions(date: str, karigar_id: str = ""):
    return {
        "sessions": svc.list_production_sessions_admin(date, karigar_id or None),
        "work_types": list(svc.KARIGAR_EXPENSE_WORK_TYPES),
    }


@router.post("/production-entry/admin/delete")
def production_admin_delete(body: ProductionDeleteBody):
    if not verify_admin_password(body.admin_password):
        raise HTTPException(403, "Admin password required")
    out = svc.delete_production_entries(
        date_str=body.date,
        karigar_id=body.karigar_id,
        challan_no=body.challan_no,
        style=body.style,
        operation=body.operation,
    )
    if not out.get("ok"):
        raise HTTPException(404, out.get("message", "Delete failed"))
    return out


@router.get("/ltl-setup")
def ltl_setup():
    return svc.get_ltl_setup_table()


@router.get("/expenses")
def list_expenses(date_from: str = "", date_to: str = "", karigar_id: str = ""):
    if not date_from:
        date_from = str(date.today() - timedelta(days=6))
    if not date_to:
        date_to = str(date.today())
    return {
        "rows": svc.list_karigar_expenses(date_from, date_to, karigar_id or None),
        "work_types": list(svc.KARIGAR_EXPENSE_WORK_TYPES),
    }


@router.post("/expenses")
def upsert_expense(body: KarigarExpenseBody):
    if not verify_admin_password(body.admin_password):
        raise HTTPException(403, "Admin password required")
    out = svc.upsert_karigar_expense(
        date_str=body.Date,
        karigar_id=body.Karigar_ID,
        work_type=body.Work_Type,
        challan_no=body.Challan_No,
        style=body.Style,
        amount_rs=body.Amount_Rs,
        hours=body.Hours,
        notes=body.Notes,
        expense_id=body.Expense_ID,
    )
    if not out.get("ok"):
        raise HTTPException(400, out.get("message", "Save failed"))
    return out


@router.delete("/expenses/{expense_id}")
def remove_expense(expense_id: str, admin_password: str = ""):
    if not verify_admin_password(admin_password):
        raise HTTPException(403, "Admin password required")
    out = svc.delete_karigar_expense(expense_id)
    if not out.get("ok"):
        raise HTTPException(404, out.get("message", "Not found"))
    return out


@router.get("/target-control/preview")
def target_control_preview(
    date: str = "",
    style: str = "",
    karigar_id: str = "",
    operation: str = "",
):
    return svc.target_control_preview(
        date or str(date.today()),
        style=style,
        karigar_id=karigar_id,
        operation=operation,
    )


@router.put("/target-control/override")
def put_ltl_override(body: LtlOverrideBody):
    if not verify_admin_password(body.admin_password):
        raise HTTPException(403, "Admin password required to set manual LTL overrides")
    return svc.upsert_ltl_override(
        body.Style,
        body.Operation,
        body.Karigar_ID,
        body.Manual_LTL,
        notes=body.Notes,
    )


@router.get("/style-costing")
def style_costing(month: str = "All", style: str = "All", party: str = "All"):
    return svc.style_costing_report(month=month, style=style, party=party)


@router.get("/efficiency")
def efficiency(date_from: str = "", date_to: str = "", styles: str = ""):
    if not date_from:
        date_from = str(date.today() - timedelta(days=7))
    if not date_to:
        date_to = str(date.today())
    style_list = [s.strip() for s in styles.split(",") if s.strip()] if styles else None
    return svc.efficiency_report(date_from, date_to, style_list)


@router.get("/performance")
def performance(date_from: str = "", date_to: str = ""):
    if not date_from:
        date_from = str(date.today() - timedelta(days=29))
    if not date_to:
        date_to = str(date.today())
    return svc.performance_report(date_from, date_to)


@router.post("/admin/unlock")
def admin_unlock(body: AdminPasswordBody):
    if verify_admin_password(body.password):
        return {"ok": True, "message": "Unlocked"}
    raise HTTPException(403, "Wrong admin password")


@router.post("/admin/change-password")
def admin_change_password(body: AdminChangePasswordBody):
    if body.new_password != body.confirm:
        raise HTTPException(400, "Passwords don't match")
    out = change_admin_password(body.current, body.new_password)
    if not out["ok"]:
        raise HTTPException(400, out["message"])
    return out


@router.get("/master/style-report")
def master_style_report(
    style: str,
    date_from: str = "",
    date_to: str = "",
    view: str = "full",
    include_production_detail: bool = False,
):
    if not style.strip():
        raise HTTPException(400, "Style is required")
    return svc.style_master_report(
        style.strip(),
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
        view=view,
        include_production_detail=include_production_detail,
    )


@router.patch("/master/style-operation")
def patch_style_operation(body: StyleOpUpdateBody):
    if not verify_admin_password(body.admin_password):
        raise HTTPException(403, "Admin password required to edit targets and rates")
    if body.Target is None and body.Rate_Rs is None:
        raise HTTPException(400, "Provide Target and/or Rate_Rs")
    return svc.update_style_operation(
        body.Style, body.Operation, target=body.Target, rate_rs=body.Rate_Rs
    )


@router.get("/payroll")
def payroll(date_from: str = "", date_to: str = ""):
    if not date_from:
        date_from = str(date.today() - timedelta(days=6))
    if not date_to:
        date_to = str(date.today())
    return svc.payroll_report(date_from, date_to)


@router.post("/challans")
def add_challan(body: ChallanBody):
    df = get_sheet_df("challan_master")
    row = body.model_dump()
    if not row.get("Date"):
        row["Date"] = str(date.today())
    if row.get("Style"):
        row["Style"] = str(row["Style"]).strip()
    if row.get("Challan_No"):
        row["Challan_No"] = str(row["Challan_No"]).strip()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = svc.normalize_challan_dataframe(df)
    save_sheet_df("challan_master", df)
    return {"ok": True, "challan_no": body.Challan_No}


@router.delete("/challans/{challan_no}")
def delete_challan(challan_no: str):
    out = svc.delete_challan(challan_no)
    if not out.get("ok"):
        raise HTTPException(404, out.get("message", "Not found"))
    return out


@router.patch("/challans/{challan_no}")
def update_challan(challan_no: str, body: ChallanUpdateBody):
    df = get_sheet_df("challan_master")
    if df.empty:
        raise HTTPException(404, "No challans")
    mask = df["Challan_No"].apply(svc.clean_key) == svc.clean_key(challan_no)
    if not mask.any():
        raise HTTPException(404, f"Challan {challan_no} not found")
    idx = df[mask].index[0]
    for field, val in body.model_dump(exclude_unset=True).items():
        if val is not None:
            df.at[idx, field] = val
    save_sheet_df("challan_master", df)
    return {"ok": True}


@router.post("/attendance/karigar")
def add_karigar_attendance(body: AttendanceBody):
    from ..services import karigar_attendance as att_svc

    em = get_sheet_df("employee_master")
    er = em[em["E_Code"].astype(str) == str(body.E_Code)] if not em.empty else pd.DataFrame()
    daily = svc.get_daily_rate_for_date(str(body.E_Code), body.Date)
    name = body.Name or (str(er["Name"].iloc[0]) if not er.empty else "")
    calc = att_svc.calc_salary(body.In_Punch, body.Out_Punch, daily, body.ot_multiplier)
    row = {
        "Date": body.Date,
        "E_Code": body.E_Code,
        "Name": name,
        "In_Punch": body.In_Punch,
        "Out_Punch": body.Out_Punch,
        "Status": "P",
        "Daily_Rate_Rs": daily,
        **calc,
    }
    df = get_sheet_df("karigar_attendance")
    if not df.empty:
        mask = (df["Date"].astype(str) == str(body.Date)) & (
            df["E_Code"].astype(str).map(svc.clean_key) == svc.clean_key(body.E_Code)
        )
        df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("karigar_attendance", df)
    return {"ok": True, "row": row}


@router.patch("/attendance/karigar")
def patch_karigar_attendance(body: AttendancePatchBody):
    """Fix miss punch / break time after biometric import; recalculates payroll."""
    from ..services import karigar_attendance as att_svc

    try:
        pairs = None
        if body.punch_pairs:
            parsed = att_svc.punch_pairs_from_request(
                [{"in_time": p.in_time, "out_time": p.out_time} for p in body.punch_pairs]
            )
            pairs = parsed or None
        return att_svc.update_karigar_attendance_row(
            on_date=body.Date,
            e_code=body.E_Code,
            in_punch=body.In_Punch,
            out_punch=body.Out_Punch,
            punch_pairs=pairs,
            waive_lunch_break=body.Waive_Lunch_Break,
            waive_tea_break=body.Waive_Tea_Break,
            lunch_break_minutes=body.Lunch_Break_Minutes,
            tea_break_minutes=body.Tea_Break_Minutes,
        )
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@router.post("/attendance/karigar/upload")
async def upload_karigar_attendance(file: UploadFile = File(...)):
    """Import Daily Attendance IN/OUT Punch Report (.xls / .xlsx)."""
    from ..services import karigar_attendance as att_svc

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")
    try:
        out = att_svc.import_karigar_attendance_bytes(raw, file.filename or "attendance.xls")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"Could not parse attendance file: {e}") from e
    if not out.get("ok"):
        raise HTTPException(400, out.get("message", "Import failed"))
    return out


@router.post("/attendance/operating")
def add_operating_attendance(body: AttendanceBody):
    em = get_sheet_df("employee_master")
    er = em[em["E_Code"].astype(str) == str(body.E_Code)] if not em.empty else pd.DataFrame()
    if er.empty:
        raise HTTPException(400, "Employee not found")
    name = body.Name or str(er["Name"].iloc[0])
    hr = float(er["Hourly_Rate_Rs"].iloc[0])
    try:
        fmt = "%H:%M"
        from datetime import datetime as dt

        hrs = round(
            (dt.strptime(body.Out_Punch.strip(), fmt) - dt.strptime(body.In_Punch.strip(), fmt)).total_seconds()
            / 3600,
            2,
        )
    except Exception:
        hrs = 0.0
    tp = round(hrs * hr, 2)
    row = {
        "Date": body.Date,
        "E_Code": body.E_Code,
        "Name": name,
        "In_Punch": body.In_Punch,
        "Out_Punch": body.Out_Punch,
        "Total_Hours": hrs,
        "Hourly_Rate_Rs": hr,
        "Total_Pay": tp,
    }
    df = get_sheet_df("operating_attendance")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("operating_attendance", df)
    return {"ok": True, "row": row}


@router.post("/master/style-operation")
def add_style_operation(body: StyleOpBody):
    out = svc.add_style_operation_row(body.Style, body.Operation, body.Target, body.Rate_Rs)
    if not out.get("ok"):
        raise HTTPException(409, out.get("message", "Duplicate"))
    return out


@router.post("/master/karigar")
def add_karigar(body: KarigarBody):
    out = svc.add_karigar_master(
        body.Karigar_ID,
        body.Name,
        body.Skill,
        body.Daily_Rate_Rs,
        body.Effective_From or None,
    )
    if not out.get("ok"):
        raise HTTPException(409, out.get("message", "Duplicate"))
    return out


@router.patch("/master/karigar/{karigar_id}")
def update_karigar(karigar_id: str, body: KarigarUpdateBody):
    out = svc.update_karigar_master(
        karigar_id,
        name=body.Name,
        skill=body.Skill,
        daily_rate_rs=body.Daily_Rate_Rs,
        effective_from=body.Effective_From,
    )
    if not out.get("ok"):
        raise HTTPException(404, out.get("message", "Not found"))
    return out


@router.post("/master/delete-rows")
def delete_master_rows(body: MasterDeleteBody):
    if body.sheet not in svc.MASTER_EDITABLE_KEYS:
        raise HTTPException(400, f"Sheet {body.sheet} cannot be edited here")
    out = svc.delete_master_rows(body.sheet, body.rows)
    if not out.get("ok"):
        raise HTTPException(400, out.get("message", "Delete failed"))
    return out


@router.get("/master/karigar/{karigar_id}/rate-history")
def karigar_rate_history(karigar_id: str):
    hist = get_sheet_df("karigar_rate_history")
    if hist.empty:
        return {"rows": []}
    kid = karigar_id.strip()
    h = hist[hist["Karigar_ID"].astype(str).str.strip() == kid] if "Karigar_ID" in hist.columns else pd.DataFrame()
    if h.empty:
        return {"rows": []}
    h = h.sort_values("Effective_From", ascending=False)
    return {"rows": h.fillna("").to_dict(orient="records")}


@router.post("/sync/from-gsheet")
def pull_gsheet(replace: bool = False):
    """Pull from Google Sheet. Default merge keeps server rows and adds missing keys."""
    if replace:
        return sync_from_gsheet()
    return sync_from_gsheet_merge()


@router.post("/sync/from-gsheet/merge")
def pull_gsheet_merge():
    return sync_from_gsheet_merge()


@router.post("/sync/to-gsheet")
def push_gsheet():
    return sync_to_gsheet()


@router.get("/export-zip")
def export_zip():
    data = svc.export_all_zip_bytes()
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="stitching_{date.today()}.zip"'},
    )


@router.post("/import-zip")
async def import_zip(file: UploadFile = File(...)):
    raw = await file.read()
    return svc.import_zip_bytes(raw)
