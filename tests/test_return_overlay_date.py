"""Return overlay reporting date must come from file, not upload day."""
from backend.services.po_return_import import infer_return_overlay_as_of
import pandas as pd


def test_infer_return_as_of_from_filename():
    assert infer_return_overlay_as_of("Return-Data-03-06-2026.rar", None) == "2026-06-03"


def test_infer_return_as_of_from_table_column():
    df = pd.DataFrame([{"OMS_SKU": "SKU1", "Return_Units": 2, "Date": "2026-06-02"}])
    assert infer_return_overlay_as_of("returns.zip", df) == "2026-06-02"
