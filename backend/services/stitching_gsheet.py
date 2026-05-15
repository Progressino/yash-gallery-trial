"""Optional Google Sheets sync for Stitching Costing."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from ..db.stitching_db import DATA_KEYS, save_sheet_df, get_sheet_df

SHEET_ID = os.environ.get(
    "STITCHING_GSHEET_ID",
    "1_cMCIn5KlvRqXS2yRy7nBidoTmgX8K48gTBaMAqBoFE",
)


def _gsheet_available() -> bool:
    try:
        import gspread  # noqa: F401
        return bool(SHEET_ID)
    except ImportError:
        return False


def _get_client():
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_path = os.environ.get("STITCHING_GCP_CREDENTIALS", "credentials.json")
    if os.environ.get("STITCHING_GCP_SERVICE_ACCOUNT_JSON"):
        info = json.loads(os.environ["STITCHING_GCP_SERVICE_ACCOUNT_JSON"])
        creds = Credentials.from_service_account_info(info, scopes=scopes)
    elif os.path.isfile(creds_path):
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
    else:
        raise FileNotFoundError(
            "Set STITCHING_GCP_CREDENTIALS path or STITCHING_GCP_SERVICE_ACCOUNT_JSON"
        )
    return gspread.authorize(creds).open_by_key(SHEET_ID)


def load_tab(tab_name: str) -> pd.DataFrame:
    from gspread_dataframe import get_as_dataframe

    sh = _get_client()
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        sh.add_worksheet(title=tab_name, rows=1000, cols=50)
        return pd.DataFrame()
    all_values = ws.get_all_values()
    if not all_values or len(all_values) < 2:
        return pd.DataFrame()
    return get_as_dataframe(ws, evaluate_formulas=True).dropna(how="all").reset_index(drop=True)


def save_tab(tab_name: str, df: pd.DataFrame) -> None:
    from gspread_dataframe import set_with_dataframe

    sh = _get_client()
    try:
        ws = sh.worksheet(tab_name)
    except Exception:
        ws = sh.add_worksheet(title=tab_name, rows=1000, cols=50)
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, include_column_header=True)


def sync_from_gsheet(keys: list[str] | None = None) -> dict[str, Any]:
    if not _gsheet_available():
        return {"ok": False, "message": "gspread not installed or STITCHING_GSHEET_ID missing"}
    keys = keys or list(DATA_KEYS)
    loaded = []
    errors = []
    for key in keys:
        if key not in DATA_KEYS:
            continue
        try:
            df = load_tab(key)
            save_sheet_df(key, df)
            loaded.append(key)
        except Exception as e:
            errors.append({key: str(e)})
    return {"ok": len(errors) == 0, "loaded": loaded, "errors": errors}


def sync_to_gsheet(keys: list[str] | None = None) -> dict[str, Any]:
    if not _gsheet_available():
        return {"ok": False, "message": "gspread not installed or STITCHING_GSHEET_ID missing"}
    keys = keys or list(DATA_KEYS)
    saved = []
    errors = []
    for key in keys:
        if key not in DATA_KEYS:
            continue
        try:
            df = get_sheet_df(key)
            save_tab(key, df)
            saved.append(key)
        except Exception as e:
            errors.append({key: str(e)})
    return {"ok": len(errors) == 0, "saved": saved, "errors": errors}


def gsheet_status() -> dict:
    return {
        "available": _gsheet_available(),
        "sheet_id": SHEET_ID if _gsheet_available() else None,
    }
