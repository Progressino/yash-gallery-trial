"""
Myntra Merchant Integration Platform (MMIP) API connector.

Auth:  Basic auth (username:password) + X-Api-Key header
Data:  GET  /v4/orders  (date-filtered, paginated)

Output matches the schema produced by myntra.py parser:
  Date, OMS_SKU, TxnType, RawStatus, Quantity, Invoice_Amount,
  State, Payment_Method, Warehouse_Id, OrderId, Month, Month_Label
"""

import base64
import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from .helpers import map_to_oms_sku

log = logging.getLogger("erp.myntra_api")

MYNTRA_BASE  = "https://mmip.myntrainfo.com"
_PAGE_SIZE   = 100


# ── Auth header builder ───────────────────────────────────────────────────────

def _myntra_headers(username: str, password: str, api_key: str) -> dict:
    creds = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "X-Api-Key":     api_key,
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }


def test_myntra_connection(username: str, password: str, api_key: str) -> bool:
    """Quick connectivity check — returns True if credentials appear valid."""
    try:
        resp = requests.get(
            f"{MYNTRA_BASE}/v4/orders",
            headers=_myntra_headers(username, password, api_key),
            params={"limit": 1, "offset": 0},
            timeout=15,
        )
        # 200 or 204 means connected; 401/403 means wrong creds
        return resp.status_code in (200, 204)
    except Exception:
        return False


# ── Status mapping ────────────────────────────────────────────────────────────

def _myntra_txn_type(forward_status: str, reverse_status: Optional[str] = None) -> str:
    """
    Mirrors the classification logic in myntra.py parser:
    - reverse_order_status populated → Refund
    - forward contains return/reverse/rto → Refund
    - cancel/failed → Cancel
    - otherwise → Shipment
    """
    if reverse_status and str(reverse_status).strip().upper() not in ("", "NAN", "NONE", "NULL", "-"):
        return "Refund"
    fs = str(forward_status).upper().strip()
    if any(x in fs for x in ("RETURN", "REVERSE", "RTO", "RTD", "RVP", "REFUND")):
        return "Refund"
    if any(x in fs for x in ("CANCEL", "FAILED", "REJECTED")):
        return "Cancel"
    return "Shipment"


# ── Orders fetch ──────────────────────────────────────────────────────────────

def _fetch_myntra_orders_page(
    headers: dict,
    start_date: str,
    end_date: str,
    offset: int,
) -> dict:
    resp = requests.get(
        f"{MYNTRA_BASE}/v4/orders",
        headers=headers,
        params={
            "startDate": start_date,
            "endDate":   end_date,
            "limit":     _PAGE_SIZE,
            "offset":    offset,
        },
        timeout=30,
    )
    if resp.status_code == 429:
        log.warning("Myntra rate limit — sleeping 20s")
        time.sleep(20)
        return _fetch_myntra_orders_page(headers, start_date, end_date, offset)
    if resp.status_code == 401:
        raise ValueError("Myntra authentication failed — check username, password, and API key")
    if resp.status_code not in (200, 204):
        raise ValueError(f"Myntra orders fetch failed ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code == 204:
        return {"orders": [], "total": 0}
    return resp.json()


def _orders_to_df(orders: list, sku_mapping: dict) -> pd.DataFrame:
    """Convert Myntra order list to myntra_df schema."""
    rows = []
    for o in orders:
        sku_raw  = str(o.get("skuId") or o.get("sku_id") or o.get("sellerSku") or "").strip()
        oms_sku  = map_to_oms_sku(sku_raw, sku_mapping) if sku_mapping else sku_raw
        qty      = float(o.get("quantity", 1) or 1)
        fwd_st   = str(o.get("orderStatus") or o.get("forwardOrderStatus") or o.get("packetStatus") or "")
        rev_st   = o.get("reverseOrderStatus") or o.get("returnStatus") or ""
        txn_type = _myntra_txn_type(fwd_st, rev_st)
        date_raw = (
            o.get("orderCreatedDate") or
            o.get("orderDate")        or
            o.get("createdDate")      or
            o.get("created_on", "")
        )
        rev     = float(o.get("invoiceAmount") or o.get("netAmount") or o.get("shipmentValue") or 0)
        state   = str(o.get("state") or o.get("customerDeliveryStateCode") or "").upper()
        order_id = str(o.get("orderId") or o.get("packerId") or o.get("subOrderId") or "")
        pay_method = str(o.get("paymentMethod") or "")
        rows.append({
            "Date":           pd.to_datetime(date_raw, errors="coerce"),
            "OMS_SKU":        oms_sku,
            "TxnType":        txn_type,
            "RawStatus":      fwd_st,
            "Quantity":       qty,
            "Invoice_Amount": rev,
            "State":          state,
            "Payment_Method": pay_method,
            "Warehouse_Id":   "",
            "OrderId":        order_id,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Date"])
    df["Month"]       = df["Date"].dt.to_period("M").astype(str)
    df["Month_Label"] = df["Date"].dt.strftime("%b %Y")
    return df


# ── Full sync ─────────────────────────────────────────────────────────────────

def sync_myntra_data(
    creds: dict,
    days_back: int = 7,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch Myntra orders for the last `days_back` days.
    Returns (df, message).

    creds keys:
      client_id   → api_key
      client_secret → password  (decrypted)
      refresh_token → username  (decrypted)
      seller_id   → seller_id
    """
    api_key     = creds["client_id"]
    password    = creds["client_secret"]
    username    = creds.get("refresh_token") or creds.get("username", "")
    sku_mapping = creds.get("sku_mapping") or {}

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.isoformat()
    end_str   = end_dt.isoformat()

    log.info("Myntra sync: %s → %s", start_str, end_str)

    headers = _myntra_headers(username, password, api_key)
    all_dfs = []
    offset  = 0

    try:
        while True:
            data   = _fetch_myntra_orders_page(headers, start_str, end_str, offset)
            orders = data.get("orders") or data.get("data") or []
            if not orders:
                break
            chunk = _orders_to_df(orders, sku_mapping)
            if not chunk.empty:
                all_dfs.append(chunk)
            total = data.get("total") or data.get("totalCount") or 0
            offset += len(orders)
            if offset >= total or len(orders) < _PAGE_SIZE:
                break
    except ValueError as e:
        return pd.DataFrame(), f"Sync failed: {e}"
    except Exception as e:
        log.exception("Myntra sync error: %s", e)
        return pd.DataFrame(), f"Sync error: {e}"

    if not all_dfs:
        return pd.DataFrame(), f"No orders found for {start_str} → {end_str}"

    combined = pd.concat(all_dfs, ignore_index=True)
    msg = f"Synced {len(combined):,} rows ({start_str} → {end_str})"
    return combined, msg
