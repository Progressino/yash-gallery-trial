"""TNA (Time & Action Calendar) router"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from ..db.tna_db import (
    list_tnas, get_tna, create_tna, update_tna_line, update_tna_status,
    get_tna_stats, TEMPLATES
)

router = APIRouter()

class TNAIn(BaseModel):
    so_number: Optional[str] = ''
    style_name: Optional[str] = ''
    buyer: Optional[str] = ''
    po_number: Optional[str] = ''
    merchandiser: Optional[str] = ''
    season: Optional[str] = ''
    order_qty: Optional[int] = 0
    delivery_date: Optional[str] = ''
    exfactory_date: Optional[str] = ''
    shipment_date: Optional[str] = ''
    priority: Optional[str] = 'Normal'
    template_used: Optional[str] = 'Domestic Order TNA'
    custom_lines: Optional[List[dict]] = None

class TNALineUpdate(BaseModel):
    actual_start: Optional[str] = None
    actual_end: Optional[str] = None
    status: Optional[str] = None
    responsible: Optional[str] = None
    backup_person: Optional[str] = None
    delay_reason: Optional[str] = None
    remarks: Optional[str] = None

class StatusUpdate(BaseModel):
    status: str

@router.get("/stats")
def get_stats():
    return get_tna_stats()

@router.get("/templates")
def get_templates():
    return list(TEMPLATES.keys())

@router.get("")
def get_tnas(status: Optional[str] = None):
    return list_tnas(status)

@router.get("/{tid}")
def get_one_tna(tid: int):
    return get_tna(tid)

@router.post("")
def post_tna(body: TNAIn):
    num = create_tna(body.model_dump())
    return {"tna_number": num}

@router.patch("/lines/{lid}")
def patch_tna_line(lid: int, body: TNALineUpdate):
    update_tna_line(lid, {k: v for k, v in body.model_dump().items() if v is not None})
    return {"ok": True}

@router.patch("/{tid}/status")
def patch_tna_status(tid: int, body: StatusUpdate):
    update_tna_status(tid, body.status)
    return {"ok": True}
