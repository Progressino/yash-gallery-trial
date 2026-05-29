#!/usr/bin/env python3
"""Run restore-full integration checks and print a human-readable report."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


def _seed_warm():
    import backend.main as main

    days = pd.date_range("2025-06-01", periods=40, freq="D")
    n = len(days)
    sku = "REST-SKU"
    sales = pd.DataFrame(
        {
            "TxnDate": days.tolist() * 4,
            "OMS_SKU": [sku] * (n * 4),
            "Qty": [2] * (n * 4),
            "Source": ["Amazon"] * n + ["Myntra"] * n + ["Meesho"] * n + ["Flipkart"] * n,
        }
    )
    main._warm_cache = {
        "sku_mapping": {sku: sku},
        "mtr_df": pd.DataFrame(
            {
                "Date": days,
                "SKU": [sku] * n,
                "Transaction_Type": ["Shipment"] * n,
                "Quantity": [3.0] * n,
            }
        ),
        "myntra_df": pd.DataFrame({"TxnDate": days, "OMS_SKU": [sku] * n, "Qty": [2] * n}),
        "meesho_df": pd.DataFrame({"TxnDate": days, "OMS_SKU": [sku] * n, "Qty": [2] * n}),
        "flipkart_df": pd.DataFrame({"TxnDate": days, "OMS_SKU": [sku] * n, "Qty": [2] * n}),
        "snapdeal_df": pd.DataFrame(),
        "sales_df": sales,
        "inventory_df_variant": pd.DataFrame({"OMS_SKU": [sku], "Total_Inventory": [120]}),
        "inventory_df_parent": pd.DataFrame(),
    }
    main._warm_cache_generation = 99


def main() -> int:
    from backend.main import app
    from backend.session import store, wipe_app_session

    def _decode(token: str | None):
        if token == "verify-token":
            return {"sub": "verify", "role": "Admin", "permissions": []}
        return None

    import backend.main as main_mod
    import backend.routers.auth as auth_router

    main_mod.decode_token = _decode  # type: ignore[method-assign]
    auth_router.decode_token = _decode  # type: ignore[method-assign]
    auth_router.verify_token = lambda t: "verify" if t == "verify-token" else None  # type: ignore[method-assign]

    _seed_warm()
    client = TestClient(app)
    client.cookies.set("auth_token", "verify-token")
    client.get("/api/health")
    sid = client.cookies.get("session_id")
    sess = store.get(sid)
    assert sess is not None
    wipe_app_session(sess)

    checks: list[tuple[str, bool, str]] = []

    r = client.post("/api/data/restore-full")
    ok = r.status_code == 200 and r.json().get("ok")
    b = r.json() if r.status_code == 200 else {}
    checks.append(
        (
            "POST /api/data/restore-full",
            ok,
            f"status={r.status_code} mtr={b.get('mtr_rows')} sales={b.get('sales_rows')}",
        )
    )

    endpoints = [
        ("GET /api/data/coverage", "/api/data/coverage", {}),
        ("GET sales-summary", "/api/data/sales-summary", {"months": "12"}),
        ("GET platform-summary", "/api/data/platform-summary", {}),
        ("GET sales-by-source", "/api/data/sales-by-source", {}),
        ("GET inventory", "/api/data/inventory", {"limit": "5"}),
        ("GET mtr-analytics", "/api/data/mtr-analytics", {"months": "3"}),
        ("GET myntra-analytics", "/api/data/myntra-analytics", {"months": "3"}),
        ("GET meesho-analytics", "/api/data/meesho-analytics", {"months": "3"}),
        ("GET flipkart-analytics", "/api/data/flipkart-analytics", {"months": "3"}),
        ("GET daily-dsr", "/api/data/daily-dsr", {"days": "7"}),
    ]
    for label, path, params in endpoints:
        er = client.get(path, params=params)
        checks.append((label, er.status_code == 200, f"HTTP {er.status_code}"))

    print("Restore-full verification")
    print("-" * 60)
    failed = 0
    for label, passed, detail in checks:
        mark = "OK" if passed else "FAIL"
        print(f"[{mark}] {label} — {detail}")
        if not passed:
            failed += 1
    print("-" * 60)
    if failed:
        print(f"{failed} check(s) failed")
        return 1
    print("All checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
