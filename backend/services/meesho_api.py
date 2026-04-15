"""
Meesho Supplier API connector.

Auth:  Custom headers: x-client-id, x-client-secret, x-timestamp, supplier_identifier
Data:  GET  /v1/suppliers/{supplier_id}/orders  (date-filtered, paginated)

Output matches the schema produced by meesho.py parser:
  Date, TxnType, Quantity, Invoice_Amount, State, OrderId, Month, OMS_SKU
"""

import hashlib
import hmac
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from .helpers import clean_line_id_series, map_to_oms_sku
from .meesho import _norm_meesho_size

log = logging.getLogger("erp.meesho_api")

MEESHO_BASE = "https://api.meesho.com"
_PAGE_SIZE  = 50


# ── Auth header builder ───────────────────────────────────────────────────────

def _meesho_headers(client_id: str, client_secret: str, supplier_id: str) -> dict:
    """
    Build Meesho auth headers.
    Uses HMAC-SHA256 signature: sign(client_secret, timestamp).
    """
    ts        = str(int(time.time()))
    signature = hmac.new(
        client_secret.encode(),
        ts.encode(),
        hashlib.sha256,
    ).hexdigest()
    return {
        "x-client-id":        client_id,
        "x-client-secret":    signature,
        "x-timestamp":        ts,
        "supplier_identifier": supplier_id,
        "Content-Type":       "application/json",
        "Accept":             "application/json",
    }


def test_meesho_connection(client_id: str, client_secret: str, supplier_id: str) -> bool:
    """Quick connectivity test — returns True if credentials appear valid."""
    try:
        resp = requests.get(
            f"{MEESHO_BASE}/v1/suppliers/{supplier_id}/orders",
            headers=_meesho_headers(client_id, client_secret, supplier_id),
            params={"page": 1, "limit": 1},
            timeout=15,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False


# ── Status mapping ────────────────────────────────────────────────────────────

def _meesho_txn_type(status: str) -> str:
    s = str(status).upper().strip()
    if any(x in s for x in ("RETURN", "RTO", "REVERSE", "REFUND")):
        return "Refund"
    if "CANCEL" in s:
        return "Cancel"
    return "Shipment"


# ── Orders fetch ──────────────────────────────────────────────────────────────

def _fetch_meesho_orders_page(
    client_id: str,
    client_secret: str,
    supplier_id: str,
    start_date: str,
    end_date: str,
    page: int,
) -> dict:
    resp = requests.get(
        f"{MEESHO_BASE}/v1/suppliers/{supplier_id}/orders",
        headers=_meesho_headers(client_id, client_secret, supplier_id),
        params={
            "start_date": start_date,
            "end_date":   end_date,
            "page":       page,
            "limit":      _PAGE_SIZE,
        },
        timeout=30,
    )
    if resp.status_code == 429:
        log.warning("Meesho rate limit — sleeping 20s")
        time.sleep(20)
        return _fetch_meesho_orders_page(client_id, client_secret, supplier_id, start_date, end_date, page)
    if resp.status_code == 401:
        raise ValueError("Meesho authentication failed — check client_id, client_secret, and supplier_id")
    if resp.status_code not in (200, 204):
        raise ValueError(f"Meesho orders fetch failed ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code == 204:
        return {"orders": [], "total": 0}
    return resp.json()


def _meesho_api_line_sku(o: dict) -> str:
    """Match Order CSV / ZIP: base SKU + size → 1158YKGREEN-XL."""
    sku_raw = str(o.get("sku") or o.get("product_sku") or o.get("skuId") or "").strip()
    if not sku_raw:
        return ""
    size_raw = str(
        o.get("size")
        or o.get("variantSize")
        or o.get("variant_size")
        or o.get("product_size")
        or ""
    ).strip()
    if not size_raw:
        return sku_raw
    z = _norm_meesho_size(size_raw)
    if not z:
        return sku_raw
    suf = f"-{z}"
    if sku_raw.upper().endswith(suf.upper()):
        return sku_raw
    return f"{sku_raw}{suf}"


def _orders_to_df(orders: list, sku_mapping: dict) -> pd.DataFrame:
    """Convert Meesho order list to meesho_df schema."""
    rows = []
    for o in orders:
        sku_raw   = _meesho_api_line_sku(o)
        oms_sku   = map_to_oms_sku(sku_raw, sku_mapping) if sku_mapping else sku_raw
        qty       = float(o.get("quantity", 1) or 1)
        status    = str(o.get("orderStatus") or o.get("order_status") or o.get("status") or "")
        txn_type  = _meesho_txn_type(status)
        date_raw  = (
            o.get("orderDate") or
            o.get("order_date") or
            o.get("createdDate") or
            o.get("order_created_date", "")
        )
        rev      = float(o.get("totalInvoiceValue") or o.get("meeshoPrice") or o.get("supplierDiscountedPrice") or 0)
        state    = str(o.get("endCustomerState") or o.get("end_customer_state") or o.get("state") or "").upper()
        order_id = str(o.get("subOrderNumber") or o.get("sub_order_num") or o.get("orderId") or "")
        sub_key  = str(clean_line_id_series(pd.Series([order_id])).iloc[0])
        fy       = o.get("financialYear")
        mon_num  = o.get("monthNumber")
        if fy and mon_num:
            month_str = f"{int(fy)}-{int(mon_num):02d}"
        else:
            month_str = None
        rows.append({
            "Date":           pd.to_datetime(date_raw, errors="coerce"),
            "TxnType":        txn_type,
            "Quantity":       qty,
            "Invoice_Amount": rev,
            "State":          state,
            "OrderId":        order_id,
            "SKU":            sku_raw,
            "OMS_SKU":        oms_sku,
            "MeeshoSubOrder": sub_key,
            "_month_override": month_str,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["Date"])
    df["Month"] = df.apply(
        lambda r: r["_month_override"] if r["_month_override"] else r["Date"].strftime("%Y-%m"),
        axis=1,
    )
    df = df.drop(columns=["_month_override"])
    return df


# ── Full sync ─────────────────────────────────────────────────────────────────

def sync_meesho_data(
    creds: dict,
    days_back: int = 7,
) -> tuple[pd.DataFrame, str]:
    """
    Fetch Meesho orders for the last `days_back` days.
    Returns (df, message).

    creds keys:
      client_id     → Meesho client_id
      client_secret → Meesho client_secret (decrypted)
      seller_id     → supplier_id
    """
    client_id     = creds["client_id"]
    client_secret = creds["client_secret"]
    supplier_id   = creds["seller_id"]
    sku_mapping   = creds.get("sku_mapping") or {}

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.isoformat()
    end_str   = end_dt.isoformat()

    log.info("Meesho sync: %s → %s", start_str, end_str)

    all_dfs = []
    page    = 1

    try:
        while True:
            data   = _fetch_meesho_orders_page(client_id, client_secret, supplier_id, start_str, end_str, page)
            orders = data.get("orders") or data.get("data") or []
            if not orders:
                break
            chunk = _orders_to_df(orders, sku_mapping)
            if not chunk.empty:
                all_dfs.append(chunk)
            total = data.get("total") or data.get("totalCount") or 0
            if page * _PAGE_SIZE >= total or len(orders) < _PAGE_SIZE:
                break
            page += 1
    except ValueError as e:
        return pd.DataFrame(), f"Sync failed: {e}"
    except Exception as e:
        log.exception("Meesho sync error: %s", e)
        return pd.DataFrame(), f"Sync error: {e}"

    if not all_dfs:
        return pd.DataFrame(), f"No orders found for {start_str} → {end_str}"

    combined = pd.concat(all_dfs, ignore_index=True)
    msg = f"Synced {len(combined):,} rows ({start_str} → {end_str})"
    return combined, msg
