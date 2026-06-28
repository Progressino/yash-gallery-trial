"""Pipeline KPI uses sheet Total row when parsed sum double-counts."""
import pandas as pd

from backend.session import AppSession
from backend.services.existing_po import (
    apply_existing_po_upload_audit,
    pipeline_summary_for_po,
)


def test_pipeline_summary_prefers_sheet_total_row():
    sess = AppSession()
    apply_existing_po_upload_audit(
        sess,
        {
            "totals_match": False,
            "sheet_total_row": {"total_balance_units": 373_193, "pipeline_units": 373_193},
        },
    )
    ep = pd.DataFrame(
        {
            "OMS_SKU": ["A-L", "A-XL", "BAND-L-XL"],
            "PO_Pipeline_Total": [100_000, 200_000, 503_839],
        }
    )
    units, sku_n = pipeline_summary_for_po(sess, ep)
    assert units == 373_193
    assert sku_n == 3


def test_pipeline_summary_uses_parsed_when_no_sheet_total():
    sess = AppSession()
    ep = pd.DataFrame({"OMS_SKU": ["X"], "PO_Pipeline_Total": [42]})
    units, _ = pipeline_summary_for_po(sess, ep)
    assert units == 42
