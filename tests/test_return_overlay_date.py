"""Return overlay reporting date must come from file, not upload day."""
from backend.services.po_return_import import infer_return_overlay_as_of
import pandas as pd


def test_infer_return_as_of_from_filename():
    assert infer_return_overlay_as_of("Return-Data-03-06-2026.rar", None) == "2026-06-03"


def test_infer_return_as_of_from_table_column():
    df = pd.DataFrame([{"OMS_SKU": "SKU1", "Return_Units": 2, "Date": "2026-06-02"}])
    assert infer_return_overlay_as_of("returns.zip", df) == "2026-06-02"


def test_myntra_seller_returns_filename_does_not_use_wrong_iso_substring():
    """2026-06-01 export stamp must not become 2001-06-26 via DD-MM parse."""
    assert (
        infer_return_overlay_as_of(
            "Q1lVo6JU_2026-06-01_Seller_Returns_Report_36841_2026-02-01_2026-02-28.csv",
            None,
        )
        == ""
    )


def test_per_row_return_date_skips_single_as_of():
    df = pd.DataFrame(
        [
            {
                "OMS_SKU": "SKU1",
                "Return_Units": 3,
                "Return_Platform": "myntra",
                "Return_Date": "2026-05-30",
            }
        ]
    )
    assert infer_return_overlay_as_of("returns.zip", df) == ""
