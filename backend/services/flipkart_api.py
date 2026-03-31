"""
Flipkart Marketplace Seller API connector.

Auth:  POST /sellers/oauth-token (client_credentials grant, Basic auth)
Data:  GET  /sellers/v3/orders/list  (date-filtered, paginated)

Output matches the schema produced by flipkart.py parser:
  Date, TxnType, Quantity, Invoice_Amount, OMS_SKU, State, OrderId, Month, Month_Label
"""

import base64
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import requests

from .helpers import map_to_oms_sku

log = logging.getLogger("erp.flipkart_api")

FK_BASE        = "https://api.flipkart.com/sellers"
FK_TOKEN_URL   = f"{FK_BASE}/oauth-token"
FK_ORDERS_URL  = f"{FK_BASE}/v3/orders/list"
_PAGE_SIZE     = 50


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_fk_access_token(app_id: str, app_secret: str) -> str:
    """
    Exchange app_id + app_secret for a Bearer access token via client_credentials grant.
    Token is valid for 60 days (Flipkart renews automatically in the background).
    """
    creds = base64.b64encode(f"{app_id}:{app_secret}".encode()).decode()
    resp = requests.post(
        FK_TOKEN_URL,
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type":  "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials", "scope": "Seller_Api"},
        timeout=30,
    )
    if resp.status_code != 200:
        raise ValueError(f"Flipkart auth failed ({resp.status_code}): {resp.text[:300]}")
    token = resp.json().get("access_token")
    if not token:
        raise ValueError(f"No access_token in Flipkart response: {resp.json()}")
    return token


# ── Orders fetch ──────────────────────────────────────────────────────────────

_FK_STATUS_MAP = {
    # Shipment
    "APPROVED":      "Shipment",
    "PACKED":        "Shipment",
    "SHIPPED":       "Shipment",
    "MANIFESTED":    "Shipment",
    "DELIVERED":     "Shipment",
    "CONFIRMED":     "Shipment",
    # Refund / Return
    "RETURNED":       "Refund",
    "RETURN_INITIATED": "Refund",
    "RETURN_REQUESTED": "Refund",
    "REFUNDED":       "Refund",
    "RETURN_COMPLETE": "Refund",
    # Cancel
    "CANCELLED":      "Cancel",
    "CANCELLATION_INITIATED": "Cancel",
    "FAILED":         "Cancel",
}

def _fk_txn_type(status: str, return_type: Optional[str] = None) -> str:
    if return_type:
        return "Refund"
    return _FK_STATUS_MAP.get(str(status).upper().strip(), "Shipment")


def _fetch_fk_orders_page(
    token: str,
    from_dt: str,
    to_dt: str,
    filter_type: str,
    page: int,
) -> dict:
    """Fetch one page of orders. Returns raw JSON."""
    resp = requests.get(
        FK_ORDERS_URL,
        headers={"Authorization": f"Bearer {token}"},
        params={
            "filter":     filter_type,
            "fromDate":   from_dt,
            "toDate":     to_dt,
            "pageSize":   _PAGE_SIZE,
            "pageNumber": page,
        },
        timeout=30,
    )
    if resp.status_code == 429:
        log.warning("Flipkart rate limit — sleeping 30s")
        time.sleep(30)
        return _fetch_fk_orders_page(token, from_dt, to_dt, filter_type, page)
    if resp.status_code != 200:
        raise ValueError(f"Orders fetch failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


def _orders_to_df(orders: list, sku_mapping: dict) -> pd.DataFrame:
    """Convert a list of order dicts to the flipkart_df schema."""
    rows = []
    for o in orders:
        sku_raw    = str(o.get("skuId") or o.get("productId") or "").strip()
        oms_sku    = map_to_oms_sku(sku_raw, sku_mapping) if sku_mapping else sku_raw
        qty        = float(o.get("quantity", 1) or 1)
        status     = str(o.get("orderStatus") or "")
        return_type = o.get("returnType") or o.get("returnStatus")
        txn_type   = _fk_txn_type(status, return_type)
        date_raw   = o.get("orderDate") or o.get("invoiceDate") or o.get("createdAt", "")
        rev        = float(o.get("invoiceAmount") or o.get("finalSaleAmount") or 0)
        state      = str(o.get("deliveryState") or o.get("city") or "").upper()
        order_id   = str(o.get("orderId") or "")
        rows.append({
            "Date":           pd.to_datetime(date_raw, errors="coerce"),
            "TxnType":        txn_type,
            "Quantity":       qty,
            "Invoice_Amount": rev,
            "OMS_SKU":        oms_sku,
            "State":          state,
            "OrderId":        order_id,
            "RawStatus":      status,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Date"])
    df["Month"]       = df["Date"].dt.to_period("M").astype(str)
    df["Month_Label"] = df["Date"].dt.strftime("%b %Y")
    return df


# ── Full sync ─────────────────────────────────────────────────────────────────

def sync_flipkart_data(
    creds: dict,
    days_back: int = 7,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch Flipkart orders (all types) for the last `days_back` days.
    Returns (df, message).
    """
    app_id     = creds["client_id"]
    app_secret = creds["client_secret"]
    sku_mapping = creds.get("sku_mapping") or {}

    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days_back)
    from_str = start_dt.strftime("%Y-%m-%dT00:00:00.000Z")
    to_str   = end_dt.strftime("%Y-%m-%dT23:59:59.999Z")

    log.info("Flipkart sync: %s → %s", from_str[:10], to_str[:10])

    try:
        token = get_fk_access_token(app_id, app_secret)
    except ValueError as e:
        return pd.DataFrame(), f"Auth failed: {e}"

    all_dfs = []
    notes   = []

    # Fetch forward orders + return orders
    for filter_type in ["dispatch", "return", "cancellation"]:
        page  = 1
        count = 0
        try:
            while True:
                data  = _fetch_fk_orders_page(token, from_str, to_str, filter_type, page)
                items = (
                    data.get("orderItems") or
                    data.get("orders")     or
                    data.get("items")      or
                    []
                )
                if not items:
                    break
                chunk = _orders_to_df(items, sku_mapping)
                if not chunk.empty:
                    all_dfs.append(chunk)
                    count += len(chunk)
                total_pages = data.get("totalPages") or data.get("pageCount") or 1
                if page >= total_pages:
                    break
                page += 1
            notes.append(f"{filter_type}: {count:,} rows")
        except Exception as e:
            notes.append(f"{filter_type}: error — {e}")
            log.warning("Flipkart %s fetch error: %s", filter_type, e)

    if not all_dfs:
        return pd.DataFrame(), "No data. " + " | ".join(notes)

    combined = pd.concat(all_dfs, ignore_index=True)
    msg = f"Synced {len(combined):,} rows. " + " | ".join(notes)
    return combined, msg
