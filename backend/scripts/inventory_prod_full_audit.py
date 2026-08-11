"""Full production inventory audit (self-hosted / VPS with /root/app/.env).

Usage (on runner):
  cd /root/app && set -a && . ./.env && set +a && python3 backend/scripts/inventory_prod_full_audit.py
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from collections import Counter

import requests

BASE = os.environ.get("AUDIT_BASE", "http://127.0.0.1:8000").rstrip("/")
EXAMPLE_SKUS = (
    "AK-228BLACK-3XL",
    "AK-228BLACK-5XL",
    "1294YKGREEN-S",
)


def login(session: requests.Session) -> None:
    user = os.environ.get("AUTH_USERNAME") or ""
    pw = os.environ.get("AUTH_PASSWORD") or ""
    if not user or not pw:
        raise SystemExit("AUTH_USERNAME/PASSWORD missing")
    headers = {"X-Device-Id": "gha-inv-full-audit"}
    r = session.post(
        f"{BASE}/api/auth/login",
        json={"username": user, "password": pw},
        headers=headers,
        timeout=60,
    )
    print("LOGIN", r.status_code, (r.text or "")[:180])
    r.raise_for_status()
    body = r.json()
    if body.get("requires_otp"):
        raise SystemExit("login requires OTP")
    session.headers.update(headers)


def main() -> int:
    report: dict = {"ok": True, "checks": {}, "issues": []}
    s = requests.Session()

    h = s.get(f"{BASE}/api/health", timeout=30)
    h.raise_for_status()
    health = h.json()
    report["health"] = {
        k: health.get(k)
        for k in ("status", "git_sha", "label", "built_at", "warm_cache", "warm_cache_loaded_at")
    }
    print("HEALTH", json.dumps(report["health"]))
    if health.get("status") != "ok":
        report["ok"] = False
        report["issues"].append("health not ok")

    login(s)

    inv0 = s.get(f"{BASE}/api/data/inventory", params={"limit": 1, "offset": 0}, timeout=120)
    print("inventory_status", inv0.status_code)
    inv0.raise_for_status()
    meta = inv0.json()
    report["loaded"] = bool(meta.get("loaded"))
    report["snapshot"] = {
        "snapshot_date": meta.get("snapshot_date"),
        "snapshot_date_label": meta.get("snapshot_date_label"),
        "total_rows": meta.get("total_rows"),
        "columns": meta.get("columns") or list((meta.get("rows") or [{}])[0].keys()) if meta.get("rows") else [],
    }
    totals = meta.get("totals") or {}
    report["api_totals"] = {k: totals.get(k) for k in sorted(totals.keys()) if not str(k).startswith("_")}
    print("API_TOTALS", json.dumps(report["api_totals"], default=str))
    if not meta.get("loaded"):
        report["ok"] = False
        report["issues"].append("inventory not loaded")
        print(json.dumps(report, indent=2, default=str))
        return 1

    # Paginate all rows
    total_rows = int(meta.get("total_rows") or 0)
    sum_cols = Counter()
    sku_examples: dict = {}
    zero_oms_pos_amz = 0
    neg_any = 0
    offset = 0
    n_seen = 0
    while offset < max(total_rows, 1):
        page = s.get(
            f"{BASE}/api/data/inventory",
            params={"limit": 5000, "offset": offset},
            timeout=180,
        )
        page.raise_for_status()
        batch = page.json().get("rows") or []
        if not batch:
            break
        for row in batch:
            n_seen += 1
            for col, val in row.items():
                if col in ("OMS_SKU", "SKU", "sku") or not isinstance(val, (int, float)):
                    continue
                # only sum numeric inventory-ish columns
                cl = str(col).lower()
                if "inventory" in cl or col in (
                    "Marketplace_Total",
                    "Buffer_Stock",
                    "Total_Inventory",
                ):
                    try:
                        sum_cols[col] += float(val or 0)
                        if float(val or 0) < 0:
                            neg_any += 1
                    except (TypeError, ValueError):
                        pass
            sku = str(row.get("OMS_SKU") or row.get("SKU") or "").upper()
            o = float(row.get("OMS_Inventory") or 0)
            a = float(row.get("Amazon_Inventory") or 0)
            if o == 0 and a > 0:
                zero_oms_pos_amz += 1
            if sku in EXAMPLE_SKUS:
                sku_examples[sku] = {
                    k: row.get(k)
                    for k in (
                        "OMS_Inventory",
                        "Amazon_Inventory",
                        "Marketplace_Total",
                        "Total_Inventory",
                        "Buffer_Stock",
                        "Flipkart_Inventory",
                        "Myntra_Other_Inventory",
                    )
                    if k in row or True
                }
        offset += len(batch)
        if len(batch) < 5000:
            break

    report["row_counts"] = {"reported_total_rows": total_rows, "rows_fetched": n_seen}
    report["row_sums"] = {k: int(round(v)) for k, v in sum_cols.most_common()}
    report["sku_examples"] = sku_examples
    report["stats"] = {
        "skus_oms0_amazon_gt0": zero_oms_pos_amz,
        "negative_numeric_cells": neg_any,
    }
    print("ROW_SUMS", json.dumps(report["row_sums"]))
    print("SKU_EXAMPLES", json.dumps(sku_examples, default=str))
    print("STATS", json.dumps(report["stats"]))

    # Match API totals to row sums for key cols
    for col in ("OMS_Inventory", "Amazon_Inventory", "Total_Inventory", "Marketplace_Total"):
        api_v = totals.get(col)
        row_v = report["row_sums"].get(col)
        if api_v is None or row_v is None:
            continue
        match = int(round(float(api_v))) == int(round(float(row_v)))
        report["checks"][f"totals_vs_rows_{col}"] = {
            "api": int(round(float(api_v))),
            "rows": int(round(float(row_v))),
            "match": match,
        }
        if not match:
            report["ok"] = False
            report["issues"].append(f"{col}: api {api_v} != rows {row_v}")
        print(f"CHECK totals_vs_rows {col}: {match} api={api_v} rows={row_v}")

    # Server export
    exp = s.get(f"{BASE}/api/data/inventory/export.csv", timeout=180)
    print("export_status", exp.status_code, "bytes", len(exp.content))
    exp.raise_for_status()
    rows = list(csv.reader(io.StringIO(exp.text)))
    hdr, tot = rows[0], rows[1]
    assert tot[0] == "__TOTALS__", tot[:3]
    export_oms = int(float(tot[hdr.index("OMS_Inventory")]))
    export_total = int(float(tot[hdr.index("Total_Inventory")])) if "Total_Inventory" in hdr else None
    api_oms = int(round(float(totals.get("OMS_Inventory") or 0)))
    report["export"] = {
        "bytes": len(exp.content),
        "oms": export_oms,
        "total": export_total,
        "header": hdr[:20],
        "note_row": rows[2][:4] if len(rows) > 2 else None,
    }
    report["checks"]["export_oms_matches_api"] = export_oms == api_oms
    if export_oms != api_oms:
        report["ok"] = False
        report["issues"].append(f"export OMS {export_oms} != api {api_oms}")
    print("EXPORT", report["export"])

    # History — brief 7d oms + combined
    for channel, days in (("oms", 7), ("combined", 7)):
        start = s.post(
            f"{BASE}/api/po/daily-inventory-history/matrix-export",
            params={"days": days, "channel": channel},
            timeout=60,
        )
        print("hist_start", channel, start.status_code, (start.text or "")[:200])
        key = f"history_async_{channel}_{days}d"
        if start.status_code == 404:
            report["ok"] = False
            report["issues"].append(f"{key}: 404")
            report["checks"][key] = {"ok": False, "error": "404"}
            continue
        if not start.ok:
            report["ok"] = False
            report["issues"].append(f"{key}: HTTP {start.status_code}")
            report["checks"][key] = {"ok": False, "status": start.status_code}
            continue
        job_id = (start.json() or {}).get("job_id")
        ready = False
        dl_bytes = 0
        deadline = time.time() + 600
        last = {}
        while time.time() < deadline:
            st = s.get(
                f"{BASE}/api/po/daily-inventory-history/matrix-export/{job_id}",
                timeout=60,
            )
            st.raise_for_status()
            last = st.json()
            if last.get("status") == "error":
                report["ok"] = False
                report["issues"].append(f"{key}: {last.get('error')}")
                break
            if last.get("status") == "ready" or last.get("ready"):
                dl = s.get(
                    f"{BASE}/api/po/daily-inventory-history/matrix-export/{job_id}/download",
                    timeout=120,
                )
                dl.raise_for_status()
                dl_bytes = len(dl.content)
                ready = dl_bytes > 50
                break
            time.sleep(2)
        report["checks"][key] = {
            "ok": ready,
            "job_id": job_id,
            "download_bytes": dl_bytes,
            "last_status": last.get("status"),
        }
        if not ready:
            report["ok"] = False
            report["issues"].append(f"{key} not ready")
        print("HIST", key, report["checks"][key])

    # Sync history path (expect may 522 / timeout — document only)
    try:
        sync = s.get(
            f"{BASE}/api/po/daily-inventory-history/matrix.csv",
            params={"days": 7, "channel": "oms"},
            timeout=25,
        )
        report["checks"]["history_sync_matrix_csv"] = {
            "status": sync.status_code,
            "bytes": len(sync.content),
            "note": "prefer async export; sync may 522 via CDN",
        }
        print("SYNC_HIST", report["checks"]["history_sync_matrix_csv"])
    except requests.RequestException as e:
        report["checks"]["history_sync_matrix_csv"] = {
            "status": "error",
            "error": str(e)[:200],
            "note": "sync path timed out/failed — async is the intended path",
        }
        print("SYNC_HIST_FAIL", e)

    # Interpretation block
    oms = int(round(float(totals.get("OMS_Inventory") or 0)))
    amz = int(round(float(totals.get("Amazon_Inventory") or 0)))
    mkt = int(round(float(totals.get("Marketplace_Total") or 0)))
    tot = int(round(float(totals.get("Total_Inventory") or 0)))
    report["analysis"] = {
        "daily_total_formula": "Total_Inventory = OMS + all marketplace inventory columns (sum, not max)",
        "oms_is_warehouse_only": True,
        "amazon_is_fba": True,
        "recon_sheet_202803_vs_oms_201315": {
            "delta": 202803 - oms if oms else None,
            "note": (
                "A user spreadsheet total of 202,803 is NOT the app OMS total. "
                "App OMS is 201,315 on the Aug-11 snapshot; difference usually from "
                "double-counting Total_Inventory or mixing History Combined with Daily."
            ),
        },
        "components_approx": {
            "OMS": oms,
            "Amazon_FBA": amz,
            "Marketplace_Total": mkt,
            "Total_Inventory": tot,
            "oms_plus_marketplace_total": oms + mkt if mkt else None,
            "oms_plus_amazon_only": oms + amz,
        },
        "sku_mismatch_explanation": (
            "SKUs with OMS=0 and Amazon>0 are correct when stock sits in FBA only. "
            "Do not compare app Total_Inventory to an OMS-only sheet row."
        ),
    }
    print("ANALYSIS", json.dumps(report["analysis"], indent=2))

    report["verdict"] = "PASS" if report["ok"] else "FAIL"
    print("VERDICT", report["verdict"])
    if report["issues"]:
        print("ISSUES", report["issues"])
    print("REPORT_JSON_BEGIN")
    print(json.dumps(report, indent=2, default=str))
    print("REPORT_JSON_END")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
