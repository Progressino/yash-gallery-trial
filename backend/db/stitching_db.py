"""Stitching Costing — sheet-backed store (mirrors Google Sheets tabs)."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

def _default_db_path() -> str:
    if os.path.isdir("/data"):
        return "/data/stitching_costing.db"
    return os.path.join(os.path.dirname(__file__), "..", "stitching_costing.db")


_DB = os.environ.get("STITCHING_DB_PATH", _default_db_path())

IST = timezone(timedelta(hours=5, minutes=30))

DATA_KEYS = (
    "style_master",
    "karigar_master",
    "karigar_rate_history",
    "challan_master",
    "production_log",
    "target_ltl_override",
    "ltl_tolerance_bands",
    "employee_master",
    "karigar_attendance",
    "operating_attendance",
    "karigar_expenses",
    "challan_deposit_log",
    "style_cost_finalize",
)

HOUR_COLS = [
    "H_09_10", "H_10_11", "H_11_12", "H_12_13", "H_13_14",
    "H_14_15", "H_15_16", "H_16_17", "H_17_18", "H_18_19", "H_19_20", "H_20_21",
]


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _default_style_master() -> list[dict]:
    rows = []
    for style in ("1894YKDGREEN", "1065YKBLUE"):
        ops = [
            ("Cutting", 120, 2.50),
            ("Stitching Front", 80, 4.00),
            ("Stitching Back", 80, 4.00),
            ("Side Seam", 90, 3.50),
            ("Finishing", 70, 4.50),
        ]
        if style == "1065YKBLUE":
            ops = [
                ("Cutting", 120, 2.50),
                ("Stitching Front", 80, 4.00),
                ("Collar Attach", 60, 5.50),
                ("Side Seam", 90, 3.50),
                ("Finishing", 70, 4.50),
            ]
        for op, tgt, rate in ops:
            rows.append({"Style": style, "Operation": op, "Target": tgt, "Rate_Rs": rate})
    return rows


def _default_karigar_master() -> list[dict]:
    return [
        {"Karigar_ID": "K001", "Name": "Ramesh Kumar", "Skill": "Stitching", "Daily_Rate_Rs": 450},
        {"Karigar_ID": "K002", "Name": "Suresh Singh", "Skill": "Cutting", "Daily_Rate_Rs": 420},
        {"Karigar_ID": "K003", "Name": "Priya Devi", "Skill": "Finishing", "Daily_Rate_Rs": 400},
        {"Karigar_ID": "K004", "Name": "Mohan Lal", "Skill": "Stitching", "Daily_Rate_Rs": 460},
        {"Karigar_ID": "K005", "Name": "Sunita Sharma", "Skill": "Hemming", "Daily_Rate_Rs": 410},
    ]


def _default_employee_master() -> list[dict]:
    return [
        {"E_Code": "K001", "Name": "Ramesh Kumar", "Type": "Karigar", "Daily_Rate_Rs": 450, "Hourly_Rate_Rs": 56.25},
        {"E_Code": "K002", "Name": "Suresh Singh", "Type": "Karigar", "Daily_Rate_Rs": 420, "Hourly_Rate_Rs": 52.50},
        {"E_Code": "K003", "Name": "Priya Devi", "Type": "Karigar", "Daily_Rate_Rs": 400, "Hourly_Rate_Rs": 50.00},
        {"E_Code": "K004", "Name": "Mohan Lal", "Type": "Karigar", "Daily_Rate_Rs": 460, "Hourly_Rate_Rs": 57.50},
        {"E_Code": "K005", "Name": "Sunita Sharma", "Type": "Karigar", "Daily_Rate_Rs": 410, "Hourly_Rate_Rs": 51.25},
        {"E_Code": "E101", "Name": "Amit Sharma", "Type": "Operating", "Daily_Rate_Rs": 600, "Hourly_Rate_Rs": 75.00},
        {"E_Code": "E102", "Name": "Kavita Rao", "Type": "Operating", "Daily_Rate_Rs": 550, "Hourly_Rate_Rs": 68.75},
    ]


def _default_challan_master() -> list[dict]:
    return [
        {
            "Challan_No": "10220-2526",
            "Style": "1894YKDGREEN",
            "Party": "Aashirwad Garments",
            "Total_Qty": 376,
            "Received_Qty": 0,
            "Deposit_Rs": 0.0,
            "Rate_Per_Pc": 35,
            "Date": "2026-02-25",
            "Delivery_By": "2026-03-07",
        },
    ]


DEFAULT_SHEETS: dict[str, list[dict]] = {
    "style_master": _default_style_master(),
    "karigar_master": _default_karigar_master(),
    "karigar_rate_history": [],
    "challan_master": _default_challan_master(),
    "production_log": [],
    "target_ltl_override": [],
    "ltl_tolerance_bands": [
        {"Min_Rs": 200, "Max_Rs": 300, "Tolerance_Pct": 35},
        {"Min_Rs": 300, "Max_Rs": 400, "Tolerance_Pct": 12},
    ],
    "employee_master": _default_employee_master(),
    "karigar_attendance": [],
    "operating_attendance": [],
    "karigar_expenses": [],
    "challan_deposit_log": [],
    "style_cost_finalize": [],
}


def init_db() -> None:
    conn = _connect()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS stitching_sheets (
        sheet_key   TEXT PRIMARY KEY,
        data_json   TEXT NOT NULL DEFAULT '[]',
        updated_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS stitching_meta (
        meta_key    TEXT PRIMARY KEY,
        meta_value  TEXT NOT NULL
    );
    """)
    conn.commit()
    for key, rows in DEFAULT_SHEETS.items():
        if conn.execute("SELECT 1 FROM stitching_sheets WHERE sheet_key=?", (key,)).fetchone() is None:
            save_sheet_rows(key, rows, conn=conn)
    conn.close()


def _now_iso() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def get_sheet_df(key: str) -> pd.DataFrame:
    if key not in DATA_KEYS:
        raise ValueError(f"Unknown sheet: {key}")
    conn = _connect()
    row = conn.execute("SELECT data_json FROM stitching_sheets WHERE sheet_key=?", (key,)).fetchone()
    conn.close()
    if row is None:
        return pd.DataFrame(DEFAULT_SHEETS.get(key, []))
    try:
        data = json.loads(row["data_json"] or "[]")
    except json.JSONDecodeError:
        data = []
    return pd.DataFrame(data) if data else pd.DataFrame()


def save_sheet_df(key: str, df: pd.DataFrame) -> None:
    rows = df.fillna("").to_dict(orient="records") if df is not None and not df.empty else []
    save_sheet_rows(key, rows)


def save_sheet_rows(key: str, rows: list[dict], *, conn: sqlite3.Connection | None = None) -> None:
    if key not in DATA_KEYS:
        raise ValueError(f"Unknown sheet: {key}")
    own = conn is None
    if own:
        conn = _connect()
    conn.execute(
        """INSERT INTO stitching_sheets(sheet_key, data_json, updated_at)
           VALUES(?,?,?)
           ON CONFLICT(sheet_key) DO UPDATE SET data_json=excluded.data_json, updated_at=excluded.updated_at""",
        (key, json.dumps(rows, default=str), _now_iso()),
    )
    conn.commit()
    if own:
        conn.close()


def get_all_sheets() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for key in DATA_KEYS:
        df = get_sheet_df(key)
        out[key] = df.to_dict(orient="records") if not df.empty else []
    return out


def replace_all_sheets(payload: dict[str, list[dict]]) -> None:
    conn = _connect()
    for key in DATA_KEYS:
        if key in payload:
            save_sheet_rows(key, payload[key], conn=conn)
    conn.commit()
    conn.close()


def get_meta(key: str, default: str = "") -> str:
    conn = _connect()
    row = conn.execute("SELECT meta_value FROM stitching_meta WHERE meta_key=?", (key,)).fetchone()
    conn.close()
    return row["meta_value"] if row else default


DEFAULT_ADMIN_PW_HASH = hashlib.sha256(b"admin123").hexdigest()


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def get_admin_pw_hash() -> str:
    h = get_meta("admin_pw_hash", "")
    return h or DEFAULT_ADMIN_PW_HASH


def verify_admin_password(password: str) -> bool:
    return hash_password(password) == get_admin_pw_hash()


def change_admin_password(current: str, new_password: str) -> dict:
    if not verify_admin_password(current):
        return {"ok": False, "message": "Wrong current password"}
    if len(new_password) < 4:
        return {"ok": False, "message": "Min 4 characters"}
    set_meta("admin_pw_hash", hash_password(new_password))
    return {"ok": True, "message": "Password changed"}


def set_meta(key: str, value: str) -> None:
    conn = _connect()
    conn.execute(
        """INSERT INTO stitching_meta(meta_key, meta_value) VALUES(?,?)
           ON CONFLICT(meta_key) DO UPDATE SET meta_value=excluded.meta_value""",
        (key, value),
    )
    conn.commit()
    conn.close()
