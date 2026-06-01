"""Stitching report print HTML (PDF via browser)."""
import pandas as pd
import pytest

from backend.db import stitching_db
from backend.db.stitching_db import init_db, save_sheet_df
from backend.services import stitching_costing as svc
from backend.services.stitching_report_print import stitching_reports_print_html


@pytest.fixture(autouse=True)
def isolated_stitching_db(tmp_path, monkeypatch):
    path = tmp_path / "stitch_print.db"
    monkeypatch.setenv("STITCHING_DB_PATH", str(path))
    monkeypatch.setattr(stitching_db, "_DB", str(path))
    init_db()


def test_print_html_contains_main_sections():
    save_sheet_df(
        "style_master",
        pd.DataFrame([{"Style": "SKU-A", "Operation": "Stitch", "Target": 80, "Rate_Rs": 3.0}]),
    )
    save_sheet_df(
        "karigar_master",
        pd.DataFrame([{"Karigar_ID": "K1", "Name": "Test", "Daily_Rate_Rs": 480}]),
    )
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-20",
                    "E_Code": "K1",
                    "Name": "Test",
                    "Payable_Hrs": 8,
                    "Normal_Pay": 480,
                    "OT_Hours": 0,
                    "OT_Pay": 0,
                    "Total_Pay": 480,
                }
            ]
        ),
    )
    svc.save_production_entry(
        date_str="2026-05-20",
        karigar_id="K1",
        karigar_name="Test",
        challan_no="CH-1",
        style="SKU-A",
        hour_entries=[{"hour_col": "H_09_10", "operation": "Stitch", "pieces": 10}],
    )
    hub = svc.stitching_reports_hub("2026-05-20", "2026-05-20")
    html_out = stitching_reports_print_html(hub)
    assert "Karigar profitability" in html_out
    assert "Challan labour" in html_out
    assert "Payroll register" in html_out
    assert "window.print" in html_out
    assert "2026-05-20" in html_out


def test_print_endpoint_returns_html(client, session_for_client):
    _, sess = session_for_client
    save_sheet_df(
        "karigar_attendance",
        pd.DataFrame(
            [
                {
                    "Date": "2026-05-21",
                    "E_Code": "K9",
                    "Name": "X",
                    "Payable_Hrs": 8,
                    "Normal_Pay": 100,
                    "OT_Hours": 0,
                    "OT_Pay": 0,
                    "Total_Pay": 100,
                }
            ]
        ),
    )
    r = client.get(
        "/api/stitching/reports/print",
        params={"date_from": "2026-05-21", "date_to": "2026-05-21"},
    )
    assert r.status_code == 200
    assert "text/html" in (r.headers.get("content-type") or "")
    assert "STITCHING REPORTS" in r.text
