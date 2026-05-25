"""Compact paginated PO result API."""
import pandas as pd

from backend.services.po_calculate_result_api import build_result_page
from backend.services.po_result_spill import clear_spill, has_spill, read_page, spill_df


def test_build_result_page_compact_matrix():
    df = pd.DataFrame(
        {
            "OMS_SKU": ["A", "B", "C"],
            "PO_Qty": [1, 2, 3],
            "Priority": ["HIGH", "LOW", "HIGH"],
        }
    )
    meta = {"sales_through": "2026-05-25", "planning_date": "2026-05-25"}
    page = build_result_page(
        session_id=None,
        po_df=df,
        meta=meta,
        offset=1,
        limit=2,
        compact=True,
    )
    assert page["ok"] is True
    assert page["total"] == 3
    assert page["has_more"] is False
    assert page["columns"] == ["OMS_SKU", "PO_Qty", "Priority"]
    assert len(page["rows_matrix"]) == 2
    assert page["rows_matrix"][0][0] == "B"


def test_spill_and_read_page(tmp_path, monkeypatch):
    monkeypatch.setenv("PO_RESULT_SPILL_DIR", str(tmp_path))
    sid = "test-session-spill"
    clear_spill(sid)
    df = pd.DataFrame({"OMS_SKU": [f"S{i}" for i in range(50)], "PO_Qty": list(range(50))})
    spill_df(sid, df)
    assert has_spill(sid)
    page = read_page(sid, 10, 5)
    assert page is not None
    assert page["total"] == 50
    assert len(page["rows_matrix"]) == 5
    assert page["rows_matrix"][0][0] == "S10"
    clear_spill(sid)
