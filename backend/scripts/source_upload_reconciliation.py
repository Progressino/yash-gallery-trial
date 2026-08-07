#!/usr/bin/env python3
"""
Source-file ERP reconciliation (no mock data).

Re-parses Inventory / Sales RAR(s) with production loaders and fails on
structural integrity regressions:

  * inventory RAR → non-empty consolidated frame
  * OMS raw Inventory + Combo sum tracks OMS_Inventory channel (±tolerance)
  * POWDERBLUE and POWERBLUE never both present after coalesce for same style+size
  * sales platform files all parse (0 silent total loss)
  * every non-blank uploaded sales SKU resolves to a non-empty canonical key
  * PO engine joins POWER stock with POWDER sales under the master map

Usage:
  python -m backend.scripts.source_upload_reconciliation \\
    --inventory /path/Inventory\\ 6-Aug-26.rar \\
    --sales /path/Sales\\ 5-Aug-26.rar \\
    --out /tmp/recon_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# repo root on path when run as module
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _parse_sales_dir(sales_dir: Path, mapping: dict) -> tuple[list[dict], pd.DataFrame]:
    from backend.services.flipkart import _parse_flipkart_xlsx
    from backend.services.meesho import parse_meesho_csv
    from backend.services.mtr import parse_mtr_csv
    from backend.services.myntra import _parse_myntra_csv
    from backend.services.po_engine import canonical_oms_key

    audits: list[dict] = []
    frames: list[pd.DataFrame] = []
    for p in sorted(sales_dir.rglob("*")):
        if not p.is_file():
            continue
        fn = p.name
        low = fn.lower()
        blob = p.read_bytes()
        try:
            if "amazon" in low:
                df, src = parse_mtr_csv(blob, fn)
                plat = "amazon"
            elif "meesho" in low:
                df, src = parse_meesho_csv(blob)
                plat = "meesho"
            elif "myntra" in low:
                df, src = _parse_myntra_csv(blob, fn, mapping)
                plat = "myntra"
            elif "flipkart" in low:
                df = _parse_flipkart_xlsx(blob, fn, mapping)
                src = "ok"
                plat = "flipkart"
            else:
                continue
        except Exception as exc:
            audits.append(
                {
                    "file": fn,
                    "platform": "error",
                    "rows": 0,
                    "ok": False,
                    "detail": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        n = 0 if df is None else int(len(df))
        audits.append(
            {
                "file": fn,
                "platform": plat,
                "rows": n,
                "ok": n > 0 or "empty" in str(src).lower(),
                "detail": str(src)[:120],
            }
        )
        if df is None or df.empty:
            continue
        work = df.copy()
        work["_platform"] = plat
        work["_file"] = fn
        sku_col = next(
            (c for c in ("OMS_SKU", "Sku", "SKU", "sku") if c in work.columns),
            None,
        )
        qty_col = next(
            (c for c in ("Quantity", "Qty", "quantity") if c in work.columns),
            None,
        )
        if not sku_col or not qty_col:
            continue
        out = pd.DataFrame(
            {
                "Sku_raw": work[sku_col].astype(str),
                "Quantity": pd.to_numeric(work[qty_col], errors="coerce").fillna(0),
                "Platform": plat,
                "SourceFile": fn,
            }
        )
        out["OMS_SKU"] = out["Sku_raw"].map(lambda s: canonical_oms_key(s, mapping))
        frames.append(out)

    if frames:
        return audits, pd.concat(frames, ignore_index=True)
    return audits, pd.DataFrame(
        columns=["Sku_raw", "Quantity", "Platform", "SourceFile", "OMS_SKU"]
    )


def run_reconciliation(
    inventory_rar: Path,
    sales_rar: Path | None,
    *,
    oms_tolerance: float = 50.0,
) -> dict[str, Any]:
    from backend.services.inventory import (
        _parse_oms_or_combo,
        coalesce_inventory_by_sku_mapping,
        load_inventory_consolidated,
    )
    from backend.services.po_engine import calculate_po_base, inventory_oms_key
    from backend.services.sku_mapping import (
        clear_bundled_sku_mapping_cache,
        load_bundled_sku_mapping,
    )

    clear_bundled_sku_mapping_cache()
    mapping = load_bundled_sku_mapping()
    failures: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "inventory_rar": str(inventory_rar),
        "sales_rar": str(sales_rar) if sales_rar else None,
        "failures": failures,
        "checks": [],
    }

    def check(cid: str, ok: bool, detail: str, **data: Any) -> None:
        row = {"id": cid, "ok": bool(ok), "detail": detail, **data}
        report["checks"].append(row)
        if not ok:
            failures.append(row)

    inv_bytes = inventory_rar.read_bytes()
    report["inventory_sha16"] = _sha16(inv_bytes)
    inv, dbg = load_inventory_consolidated(
        None, None, None, inv_bytes, mapping, return_debug=True
    )
    check(
        "inv:parse_nonempty",
        inv is not None and not inv.empty,
        f"rows={0 if inv is None else len(inv)} debug={dbg.get('rar_files')}",
    )
    inv = coalesce_inventory_by_sku_mapping(inv, mapping)

    totals = {
        c: float(pd.to_numeric(inv[c], errors="coerce").fillna(0).sum())
        for c in inv.columns
        if c != "OMS_SKU" and pd.api.types.is_numeric_dtype(inv[c])
    }
    report["inventory_totals"] = totals
    check(
        "inv:no_negative_total",
        totals.get("Total_Inventory", 0) >= 0
        and int((pd.to_numeric(inv.get("Total_Inventory"), errors="coerce").fillna(0) < 0).sum())
        == 0,
        f"Total_Inventory={totals.get('Total_Inventory')}",
    )

    # OMS channel ≈ raw OMS Inventory + combo stock (intended: combo packs land on OMS).
    from backend.services.inventory import _extract_all_from_rar

    extracted, _manifest = _extract_all_from_rar(inv_bytes)
    oms_raw_sum = 0.0
    combo_sum = 0.0
    for blob in extracted.get("oms_csvs") or []:
        part = _parse_oms_or_combo(blob)
        if not part.empty and "OMS_Inventory" in part.columns:
            oms_raw_sum += float(pd.to_numeric(part["OMS_Inventory"], errors="coerce").fillna(0).sum())
    for blob in extracted.get("combo_csvs") or []:
        part = _parse_oms_or_combo(blob)
        if not part.empty and "OMS_Inventory" in part.columns:
            combo_sum += float(pd.to_numeric(part["OMS_Inventory"], errors="coerce").fillna(0).sum())
    expected_oms = oms_raw_sum + combo_sum
    got_oms = float(totals.get("OMS_Inventory", 0))
    oms_delta = abs(got_oms - expected_oms)
    check(
        "inv:oms_equals_raw_plus_combo",
        oms_delta <= oms_tolerance,
        f"OMS_Inventory={got_oms} raw_oms={oms_raw_sum} combo={combo_sum} "
        f"expected={expected_oms} delta={oms_delta}",
        delta=oms_delta,
        raw_oms=oms_raw_sum,
        combo=combo_sum,
    )

    # Twin-spelling residual after coalesce
    skus = set(inv["OMS_SKU"].astype(str).str.upper())
    powder = {s for s in skus if "POWDERBLUE" in s}
    power = {s for s in skus if "POWERBLUE" in s and "POWDERBLUE" not in s}
    twin_hits = []
    for p in powder:
        alt = p.replace("POWDERBLUE", "POWERBLUE")
        if alt in power:
            twin_hits.append((p, alt))
    check(
        "sku:no_powder_power_twins",
        len(twin_hits) == 0,
        f"powder_rows={len(powder)} power_rows={len(power)} twins={twin_hits[:5]}",
    )
    bottle_hits = [
        s
        for s in skus
        if "BOTTLEGREEN" in s
        and "BOTTELGREEN" not in s
        and s.replace("BOTTLEGREEN", "BOTTELGREEN") in skus
    ]
    check(
        "sku:no_bottle_bottel_twins",
        len(bottle_hits) == 0,
        f"residual={bottle_hits[:5]}",
    )

    # Warehouse source spelling for 5038 is POWDER; must canonicalize to POWER only.
    for size in ("3XL", "L", "M", "S", "XL"):
        raw = f"5038YKPOWDERBLUE-{size}"
        key = inventory_oms_key(raw)
        check(
            f"sku:5038_powder_canonical_{size}",
            key == f"5038YKPOWERBLUE-{size}",
            f"{raw} → {key}",
        )

    sales_frame = pd.DataFrame()
    if sales_rar and sales_rar.is_file():
        import tempfile
        import subprocess

        sales_bytes = sales_rar.read_bytes()
        report["sales_sha16"] = _sha16(sales_bytes)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            subprocess.run(
                ["bsdtar", "-xf", str(sales_rar), "-C", str(tdp)],
                check=True,
                capture_output=True,
            )
            audits, sales_frame = _parse_sales_dir(tdp, mapping)
        report["sales_file_audit"] = audits
        failed_files = [a for a in audits if not a.get("ok")]
        check(
            "sales:all_platform_files_parse",
            len(failed_files) == 0,
            f"failed={len(failed_files)} of {len(audits)}",
            failed=failed_files[:5],
        )
        if not sales_frame.empty:
            blank = int((sales_frame["OMS_SKU"].astype(str).str.len() == 0).sum())
            check(
                "sales:every_sku_canonical",
                blank == 0,
                f"blank_canon_rows={blank} total_rows={len(sales_frame)}",
            )
            powder_after = int(
                sales_frame["OMS_SKU"].astype(str).str.contains("POWDERBLUE", na=False).sum()
            )
            check(
                "sales:no_powder_canon_residual",
                powder_after == 0,
                f"powder_canon_rows={powder_after}",
            )
            report["sales_qty_by_platform"] = (
                sales_frame.groupby("Platform")["Quantity"].sum().astype(float).to_dict()
            )

            # PO join smoke: inject POWDER sales against POWER inventory (source shape).
            inv_5038 = inv[inv["OMS_SKU"].astype(str).str.contains("5038YKPOWERBLUE", na=False)]
            if not inv_5038.empty:
                test_sales = pd.DataFrame(
                    {
                        "Sku": ["5038YKPOWDERBLUE-3XL"] * 6
                        + ["5038YKPOWDERBLUE-L"] * 11,
                        "TxnDate": pd.date_range("2026-07-01", periods=17, freq="D"),
                        "Transaction Type": ["Shipment"] * 17,
                        "Quantity": [1] * 17,
                        "Units_Effective": [1] * 17,
                    }
                )
                po = calculate_po_base(
                    test_sales,
                    inv_5038,
                    period_days=30,
                    lead_time=45,
                    target_days=45,
                    sku_mapping=mapping,
                )
                sub = po[po["OMS_SKU"].astype(str).str.contains("5038YK", na=False)]
                has_powder = sub["OMS_SKU"].astype(str).str.contains("POWDERBLUE").any()
                row_3xl = sub[sub["OMS_SKU"] == "5038YKPOWERBLUE-3XL"]
                sold_ok = (
                    not row_3xl.empty
                    and float(row_3xl.iloc[0]["Sold_Units"]) == 6.0
                    and float(row_3xl.iloc[0]["Total_Inventory"]) > 0
                )
                check(
                    "po:5038_power_stock_plus_powder_sales",
                    (not has_powder) and sold_ok,
                    f"powder_residual={bool(has_powder)} 3xl={row_3xl[['OMS_SKU','Total_Inventory','Sold_Units']].to_dict('records') if len(row_3xl) else []}",
                )

    report["ok"] = len(failures) == 0
    report["failure_count"] = len(failures)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--sales", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--oms-tolerance", type=float, default=50.0)
    args = ap.parse_args(argv)

    if not args.inventory.is_file():
        print(f"inventory RAR not found: {args.inventory}", file=sys.stderr)
        return 2

    report = run_reconciliation(
        args.inventory,
        args.sales if args.sales and args.sales.is_file() else None,
        oms_tolerance=args.oms_tolerance,
    )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
