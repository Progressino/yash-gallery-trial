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
from ..services.stitching_gsheet import gsheet_status, sync_from_gsheet, sync_to_gsheet

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
    ot_multiplier: float = 1.5


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
    if mode == "append" and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
    if key in svc.MASTER_KEY_COLS:
        df = svc.dedupe_sheet(key, df)
    save_sheet_df(key, df)
    return {"ok": True, "rows": len(df)}


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
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("challan_master", df)
    return {"ok": True, "challan_no": body.Challan_No}


@router.patch("/challans/{challan_no}")
def update_challan(challan_no: str, body: ChallanUpdateBody):
    df = get_sheet_df("challan_master")
    if df.empty:
        raise HTTPException(404, "No challans")
    mask = df["Challan_No"].astype(str) == challan_no
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
    em = get_sheet_df("employee_master")
    er = em[em["E_Code"].astype(str) == str(body.E_Code)] if not em.empty else pd.DataFrame()
    daily = svc.get_daily_rate_for_date(str(body.E_Code), body.Date)
    name = body.Name or (str(er["Name"].iloc[0]) if not er.empty else "")
    calc = svc.calc_salary(body.In_Punch, body.Out_Punch, daily, body.ot_multiplier)
    row = {
        "Date": body.Date,
        "E_Code": body.E_Code,
        "Name": name,
        "In_Punch": body.In_Punch,
        "Out_Punch": body.Out_Punch,
        "Daily_Rate_Rs": daily,
        **calc,
    }
    df = get_sheet_df("karigar_attendance")
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_sheet_df("karigar_attendance", df)
    return {"ok": True, "row": row}


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
def pull_gsheet():
    return sync_from_gsheet()


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
