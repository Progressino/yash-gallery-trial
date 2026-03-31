"""
Amazon Selling Partner API (SP-API) integration.
Fetches MTR-equivalent GST reports and parses them using the existing mtr parser.

Flow:
  1. get_access_token()       — LWA OAuth2 → short-lived bearer token (1 hr)
  2. request_mtr_report()     — ask Amazon to generate a report (async, returns report_id)
  3. poll_report_status()     — wait until processingStatus == DONE
  4. get_report_document()    — download the actual CSV bytes
  5. parse with parse_mtr_csv() — reuse existing parser
"""

import gzip
import io
import logging
import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("erp.amazon_sp_api")

# Amazon SP-API endpoints
# India sellers use the EU regional endpoint
SP_API_BASE   = "https://sellingpartnerapi-eu.amazon.com"
LWA_TOKEN_URL = "https://api.amazon.com/auth/o2/token"
INDIA_MKT_ID  = "A21TJRUUN4KGV"

# Report types for India GST MTR
REPORT_TYPE_B2C = "GET_GST_MTR_B2C_CUSTOM"
REPORT_TYPE_B2B = "GET_GST_MTR_B2B_CUSTOM"

# Polling config
_POLL_INTERVAL  = 30   # seconds between status checks
_POLL_TIMEOUT   = 40 * 60  # 40 minutes max


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_access_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """
    Exchange refresh_token for a short-lived LWA access token.
    Raises ValueError on auth failure.
    """
    resp = requests.post(
        LWA_TOKEN_URL,
        data={
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
            "client_id":     client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise ValueError(f"LWA token exchange failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise ValueError(f"No access_token in LWA response: {data}")
    return token


# ── Reports API ───────────────────────────────────────────────────────────────

def _sp_headers(access_token: str) -> dict:
    return {
        "x-amz-access-token": access_token,
        "Content-Type":       "application/json",
    }


def request_mtr_report(
    access_token: str,
    seller_id: str,
    start_date: date,
    end_date: date,
    report_type: str,
    marketplace_id: str = INDIA_MKT_ID,
) -> str:
    """
    Request a report from Amazon. Returns the report_id (string).
    Amazon generates the report asynchronously — poll for status separately.
    """
    payload = {
        "reportType":       report_type,
        "marketplaceIds":   [marketplace_id],
        "dataStartTime":    start_date.isoformat() + "T00:00:00Z",
        "dataEndTime":      end_date.isoformat()   + "T23:59:59Z",
        "reportOptions": {
            "sellerId": seller_id,
        },
    }
    resp = requests.post(
        f"{SP_API_BASE}/reports/2021-06-30/reports",
        headers=_sp_headers(access_token),
        json=payload,
        timeout=30,
    )
    if resp.status_code not in (200, 202):
        raise ValueError(f"Report request failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()["reportId"]


def get_report_status(access_token: str, report_id: str) -> dict:
    """
    Returns dict with keys:
      processingStatus: IN_QUEUE | IN_PROGRESS | DONE | FATAL | CANCELLED
      reportDocumentId: str (only present when DONE)
    """
    resp = requests.get(
        f"{SP_API_BASE}/reports/2021-06-30/reports/{report_id}",
        headers=_sp_headers(access_token),
        timeout=30,
    )
    if resp.status_code != 200:
        raise ValueError(f"Report status check failed ({resp.status_code}): {resp.text[:200]}")
    return resp.json()


def get_report_document(access_token: str, report_document_id: str) -> bytes:
    """
    Fetch the report document metadata (download URL), then download the file.
    Amazon may gzip-compress the content — this function decompresses if needed.
    """
    # Step 1: get the document URL
    resp = requests.get(
        f"{SP_API_BASE}/reports/2021-06-30/documents/{report_document_id}",
        headers=_sp_headers(access_token),
        timeout=30,
    )
    if resp.status_code != 200:
        raise ValueError(f"Document metadata fetch failed ({resp.status_code}): {resp.text[:200]}")
    meta = resp.json()
    url         = meta["url"]
    compression = meta.get("compressionAlgorithm", "")

    # Step 2: download the actual file
    dl = requests.get(url, timeout=300)
    dl.raise_for_status()
    content = dl.content

    # Step 3: decompress if needed
    if compression == "GZIP":
        content = gzip.decompress(content)

    return content


def _poll_until_done(
    access_token: str,
    report_id: str,
    label: str = "",
) -> Optional[str]:
    """
    Poll report status until DONE or terminal state.
    Returns report_document_id on success, None on FATAL/CANCELLED.
    Raises TimeoutError if max wait exceeded.
    """
    deadline = time.time() + _POLL_TIMEOUT
    while time.time() < deadline:
        status = get_report_status(access_token, report_id)
        proc = status.get("processingStatus", "")
        log.debug("Report %s [%s] status: %s", label, report_id[:12], proc)

        if proc == "DONE":
            return status.get("reportDocumentId")
        if proc in ("FATAL", "CANCELLED"):
            log.warning("Report %s ended with status %s", report_id[:12], proc)
            return None

        time.sleep(_POLL_INTERVAL)

    raise TimeoutError(f"Report {report_id[:12]} did not complete within {_POLL_TIMEOUT // 60} min")


# ── Full sync ─────────────────────────────────────────────────────────────────

def sync_amazon_data(
    creds: dict,
    days_back: int = 7,
) -> tuple[pd.DataFrame, str]:
    """
    High-level function: fetches B2C + B2B MTR reports for the last `days_back` days.
    Returns (combined_df, status_message).

    Reuses existing parse_mtr_csv() — the SP-API MTR CSV format is identical
    to the manually uploaded MTR files.
    """
    from .mtr import parse_mtr_csv

    client_id      = creds["client_id"]
    client_secret  = creds["client_secret"]
    refresh_token  = creds["refresh_token"]
    seller_id      = creds["seller_id"]
    marketplace_id = creds.get("marketplace_id", INDIA_MKT_ID)

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days_back)

    log.info("Amazon SP-API sync: %s to %s (days_back=%d)", start_dt, end_dt, days_back)

    try:
        access_token = get_access_token(client_id, client_secret, refresh_token)
    except ValueError as e:
        return pd.DataFrame(), f"Auth failed: {e}"

    dfs   = []
    notes = []

    for report_type, label in [
        (REPORT_TYPE_B2C, "B2C MTR"),
        (REPORT_TYPE_B2B, "B2B MTR"),
    ]:
        try:
            report_id = request_mtr_report(
                access_token, seller_id, start_dt, end_dt, report_type, marketplace_id
            )
            log.info("Requested %s report: %s", label, report_id[:12])

            doc_id = _poll_until_done(access_token, report_id, label)
            if doc_id is None:
                notes.append(f"{label}: report generation failed")
                continue

            csv_bytes = get_report_document(access_token, doc_id)
            if not csv_bytes:
                notes.append(f"{label}: empty document")
                continue

            df, msg = parse_mtr_csv(csv_bytes, f"sp-api-{label.lower()}-{end_dt}.csv")
            if df.empty:
                notes.append(f"{label}: parsed 0 rows ({msg})")
            else:
                dfs.append(df)
                notes.append(f"{label}: {len(df):,} rows")
                log.info("%s parsed: %d rows", label, len(df))

        except TimeoutError as e:
            notes.append(f"{label}: timeout — {e}")
            log.warning("SP-API sync timeout: %s", e)
        except Exception as e:
            notes.append(f"{label}: error — {e}")
            log.exception("SP-API sync error for %s: %s", label, e)

    if not dfs:
        return pd.DataFrame(), "Sync completed — no new data. " + " | ".join(notes)

    combined = pd.concat(dfs, ignore_index=True)
    msg = f"Synced {len(combined):,} rows ({start_dt} → {end_dt}). " + " | ".join(notes)
    return combined, msg
