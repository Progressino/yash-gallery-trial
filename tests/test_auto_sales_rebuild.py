"""Coverage should rebuild sales when platform history exists but sales_df is empty."""
import pandas as pd

from backend.routers.data import _ensure_sales_rebuilt
from backend.session import AppSession


def test_ensure_sales_rebuilt_from_platform_history():
    sess = AppSession()
    sess.sku_mapping = {"SKU1": "SKU1"}
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-01"]),
            "SKU": ["SKU1"],
            "Quantity": [2],
            "Transaction_Type": ["Shipment"],
            "Order_Id": ["O1"],
            "Invoice_Number": [""],
        }
    )
    assert sess.sales_df.empty

    _ensure_sales_rebuilt(sess)

    assert not sess.sales_df.empty


def test_full_coverage_rebuilds_sales(client, session_for_client):
    """Full (non-light) coverage must rebuild sales when platform history is loaded."""
    _, sess = session_for_client
    sess.sku_mapping = {"SKU1": "SKU1"}
    sess.mtr_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-01"]),
            "SKU": ["SKU1"],
            "Quantity": [3],
            "Transaction_Type": ["Shipment"],
            "Order_Id": ["O1"],
            "Invoice_Number": [""],
        }
    )
    sess.sales_df = pd.DataFrame()

    r = client.get("/api/data/coverage")
    assert r.status_code == 200
    body = r.json()
    assert body["sales"] is True
    assert body["sales_rows"] >= 1


def test_light_coverage_skips_restore_while_daily_inventory_parsing(client, session_for_client):
    _, sess = session_for_client
    sess.daily_inventory_upload_status = "running"
    sess.daily_inventory_upload_message = "Parsing…"

    r = client.get("/api/data/coverage")
    assert r.status_code == 200
    body = r.json()
    assert body["daily_inventory_upload_status"] == "running"
