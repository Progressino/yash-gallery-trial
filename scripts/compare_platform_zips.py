#!/usr/bin/env python3
"""Compare local Myntra/Flipkart ZIP archives vs production Tier-3 data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.flipkart import load_flipkart_from_zip
from backend.services.myntra import load_myntra_from_zip
from backend.services.sku_mapping import load_bundled_sku_mapping

LOCAL_ZIPS = {
    "myntra_ppmp": Path("/Users/samraisinghani/Downloads/Myntra PPMP Jan-2024 to Jan-2026.zip"),
    "myntra_sjit": Path("/Users/samraisinghani/Downloads/Myntra Sjit Jan-2024 to Jan-2026.zip"),
    "flipkart_akiko": Path("/Users/samraisinghani/Downloads/Flipkart Akiko Jan-2024 To Dec-2025.zip"),
    "flipkart_ag": Path("/Users/samraisinghani/Downloads/Flipkart AG Jan-2024 To Dec-2025.zip"),
}

PROD_JSON = Path("/tmp/prod_platform_compare_full.json")


def monthly_shipments(df: pd.DataFrame) -> dict[str, int]:
    if df is None or df.empty:
        return {}
    d = pd.to_datetime(df["Date"], errors="coerce")
    work = df.loc[d.notna()].copy()
    if work.empty:
        return {}
    work["_m"] = d.loc[work.index].dt.to_period("M").astype(str)
    ship = work[work["TxnType"].astype(str).str.lower().isin(["shipment", "sale"])]
    return {k: int(v) for k, v in ship.groupby("_m")["Quantity"].sum().items()}


def parse_zip(key: str, path: Path, mapping: dict) -> tuple[pd.DataFrame, list[str]]:
    if key.startswith("myntra"):
        raw = path.read_bytes()
        df, _n_files, skipped = load_myntra_from_zip(raw, mapping, source_filename=path.name)
        return df, skipped
    df, _n_files, skipped = load_flipkart_from_zip(path, mapping, source_filename=path.name)
    return df, skipped


def pct_diff(local: int, prod: int) -> float | None:
    if prod == 0:
        return None if local == 0 else float("inf")
    return round(100.0 * (local - prod) / prod, 1)


def compare_months(local: dict[str, int], prod: dict[str, int]) -> list[dict]:
    months = sorted(set(local) | set(prod))
    rows: list[dict] = []
    for m in months:
        l = int(local.get(m, 0))
        p = int(prod.get(m, 0))
        if l == 0 and p == 0:
            continue
        status = "ok"
        if p == 0 and l > 0:
            status = "missing_on_app"
        elif l == 0 and p > 0:
            status = "missing_in_zip"
        elif abs(l - p) > max(50, 0.05 * max(l, p)):
            status = "mismatch"
        rows.append(
            {
                "month": m,
                "local_shipments": l,
                "app_shipments": p,
                "delta": l - p,
                "pct_diff": pct_diff(l, p),
                "status": status,
            }
        )
    return rows


def main() -> int:
    mapping = load_bundled_sku_mapping()
    print(f"SKU mapping keys: {len(mapping)}")

    parsed: dict[str, pd.DataFrame] = {}
    skipped_all: dict[str, list[str]] = {}
    for key, path in LOCAL_ZIPS.items():
        if not path.is_file():
            print(f"MISSING FILE: {path}")
            return 1
        print(f"Parsing {key} …")
        df, skipped = parse_zip(key, path, mapping)
        parsed[key] = df
        skipped_all[key] = skipped
        d = pd.to_datetime(df["Date"], errors="coerce") if not df.empty else pd.Series(dtype="datetime64[ns]")
        print(
            f"  {key}: rows={len(df):,} skus={df['OMS_SKU'].nunique() if not df.empty else 0} "
            f"range={str(d.min())[:10] if len(d) else '—'} → {str(d.max())[:10] if len(d) else '—'}"
        )
        if skipped:
            print(f"  skipped/warnings: {len(skipped)}")
            for s in skipped[:5]:
                print(f"    - {s}")

    myntra_local = pd.concat([parsed["myntra_ppmp"], parsed["myntra_sjit"]], ignore_index=True)
    from backend.services.daily_store import _dedup_platform_df

    myntra_local = _dedup_platform_df(myntra_local, "myntra")
    flipkart_local = pd.concat([parsed["flipkart_akiko"], parsed["flipkart_ag"]], ignore_index=True)
    flipkart_local = _dedup_platform_df(flipkart_local, "flipkart")

    local_monthly = {
        "myntra": monthly_shipments(myntra_local),
        "flipkart": monthly_shipments(flipkart_local),
        "myntra_ppmp": monthly_shipments(parsed["myntra_ppmp"]),
        "myntra_sjit": monthly_shipments(parsed["myntra_sjit"]),
        "flipkart_akiko": monthly_shipments(parsed["flipkart_akiko"]),
        "flipkart_ag": monthly_shipments(parsed["flipkart_ag"]),
    }

    prod = json.loads(PROD_JSON.read_text()) if PROD_JSON.is_file() else {}
    prod_monthly = {
        "myntra": (prod.get("myntra") or {}).get("monthly_shipments") or {},
        "flipkart": (prod.get("flipkart") or {}).get("monthly_shipments") or {},
    }

    report = {
        "local_zip_stats": {
            k: {
                "rows": int(len(v)),
                "months": len(local_monthly[k.replace("_local", "")]) if k.endswith("_local") else len(local_monthly.get(k, {})),
            }
            for k, v in {
                "myntra_combined": myntra_local,
                "flipkart_combined": flipkart_local,
                **parsed,
            }.items()
        },
        "prod_summary": prod.get("summary", {}),
        "comparisons": {},
        "skipped": skipped_all,
    }

    for plat in ("myntra", "flipkart"):
        cmp = compare_months(local_monthly[plat], prod_monthly[plat])
        report["comparisons"][plat] = {
            "months_compared": len(cmp),
            "missing_on_app": [r for r in cmp if r["status"] == "missing_on_app"],
            "missing_in_zip": [r for r in cmp if r["status"] == "missing_in_zip"],
            "mismatch": [r for r in cmp if r["status"] == "mismatch"],
            "ok": sum(1 for r in cmp if r["status"] == "ok"),
            "all_months": cmp,
        }

    out = ROOT / "reports" / "platform_zip_compare_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ROOT / "reports" / "platform_zip_compare_report.md"
    lines = [
        "# Myntra & Flipkart ZIP vs App Comparison",
        "",
        f"SKU mapping keys used: **{len(mapping):,}**",
        "",
        "## Local ZIP parse summary",
        "",
        "| Source | Rows | Shipment months |",
        "|--------|------|-----------------|",
    ]
    for label, df in [
        ("Myntra PPMP", parsed["myntra_ppmp"]),
        ("Myntra Sjit", parsed["myntra_sjit"]),
        ("Myntra combined (deduped)", myntra_local),
        ("Flipkart Akiko (PE)", parsed["flipkart_akiko"]),
        ("Flipkart AG", parsed["flipkart_ag"]),
        ("Flipkart combined (deduped)", flipkart_local),
    ]:
        ms = monthly_shipments(df)
        lines.append(f"| {label} | {len(df):,} | {len(ms)} |")

    lines += ["", "## Production Tier-3 summary", ""]
    for plat in ("myntra", "flipkart"):
        s = (prod.get("summary") or {}).get(plat) or {}
        lines.append(
            f"- **{plat.title()}**: {s.get('file_count', '?')} files, "
            f"{int(s.get('total_rows') or 0):,} rows, "
            f"{s.get('min_date', '?')} → {s.get('max_date', '?')}"
        )

    for plat in ("myntra", "flipkart"):
        c = report["comparisons"][plat]
        lines += [
            "",
            f"## {plat.title()} month-by-month (shipment units)",
            "",
            f"- OK: **{c['ok']}** months",
            f"- Missing on app: **{len(c['missing_on_app'])}** months",
            f"- Mismatch (>5% or >50 units): **{len(c['mismatch'])}** months",
            f"- On app but not in ZIPs: **{len(c['missing_in_zip'])}** months",
            "",
        ]
        if c["missing_on_app"]:
            lines.append("### Missing on app")
            lines.append("")
            lines.append("| Month | ZIP shipments | App shipments |")
            lines.append("|-------|---------------|---------------|")
            for r in c["missing_on_app"]:
                lines.append(f"| {r['month']} | {r['local_shipments']:,} | {r['app_shipments']:,} |")
            lines.append("")
        if c["mismatch"]:
            lines.append("### Quantity mismatch")
            lines.append("")
            lines.append("| Month | ZIP | App | Delta | % diff |")
            lines.append("|-------|-----|-----|-------|--------|")
            for r in sorted(c["mismatch"], key=lambda x: abs(x["delta"]), reverse=True)[:24]:
                pct = r["pct_diff"]
                pct_s = f"{pct:+.1f}%" if pct is not None and pct != float("inf") else "n/a"
                lines.append(
                    f"| {r['month']} | {r['local_shipments']:,} | {r['app_shipments']:,} | "
                    f"{r['delta']:+,} | {pct_s} |"
                )
            lines.append("")

    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"Wrote {md}")

    for plat in ("myntra", "flipkart"):
        c = report["comparisons"][plat]
        print(
            f"{plat}: ok={c['ok']} missing_on_app={len(c['missing_on_app'])} "
            f"mismatch={len(c['mismatch'])} missing_in_zip={len(c['missing_in_zip'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
