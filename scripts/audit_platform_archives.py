"""Audit local platform archives before ingestion.

Parses each archive with the real backend loaders and reports row counts,
date coverage, pre-2025 volume, SKU coverage and transaction-type mix so we
can decide what is safe to ingest (per-SKU sales) vs what must be rejected
(returns / settlement / foreign-marketplace exports).

Usage:
    python scripts/audit_platform_archives.py <spec.json>

spec.json = [{"slug","platform","path","kind"}...] where kind is one of:
    mtr_zip        master zip containing MTR CSV(s) / nested monthly zips
    mtr_dir        directory of extracted MTR month zips / CSVs
    meesho_zip     Meesho master zip
    flipkart_zip   Flipkart master zip
    raw_returns    Amazon FBA returns CSVs directory (diagnostic only)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CUT = pd.Timestamp("2025-01-01")


def _mapping() -> dict:
    from backend.services.sku_mapping import load_bundled_sku_mapping

    try:
        return load_bundled_sku_mapping()
    except Exception:
        return {}


def _date_series(df: pd.DataFrame) -> pd.Series:
    for col in ("Date", "TxnDate", "_Date"):
        if col in df.columns:
            return pd.to_datetime(df[col], errors="coerce")
    return pd.Series([pd.NaT] * len(df))


def _sku_col(df: pd.DataFrame) -> str | None:
    for col in ("OMS_SKU", "SKU", "sku"):
        if col in df.columns:
            return col
    return None


def _report(slug: str, df: pd.DataFrame, skipped: list[str]) -> dict:
    if df is None or df.empty:
        return {"slug": slug, "rows": 0, "note": "EMPTY / unparseable", "skipped": len(skipped or [])}
    d = _date_series(df)
    sc = _sku_col(df)
    sku_good = 0
    sku_total = 0
    if sc:
        s = df[sc].astype(str).str.strip().str.upper()
        sku_total = len(s)
        sku_good = int((~(s.eq("") | s.isin(["NAN", "NONE", "UNKNOWN"]))).sum())
    txn = {}
    for c in ("Transaction_Type", "Txn_Type", "DSR_Segment"):
        if c in df.columns:
            txn = df[c].astype(str).value_counts().head(6).to_dict()
            break
    return {
        "slug": slug,
        "rows": int(len(df)),
        "date_min": str(d.min())[:10] if d.notna().any() else "NA",
        "date_max": str(d.max())[:10] if d.notna().any() else "NA",
        "pre2025": int((d < CUT).sum()),
        "rows_2025plus": int((d >= CUT).sum()),
        "sku_col": sc,
        "sku_usable": f"{sku_good:,}/{sku_total:,}",
        "txn_types": txn,
        "skipped": len(skipped or []),
        "columns": list(df.columns)[:12],
    }


def parse(entry: dict, mapping: dict) -> dict:
    slug = entry["slug"]
    kind = entry["kind"]
    path = Path(entry["path"])

    if kind == "mtr_zip":
        from backend.services.mtr import load_mtr_from_zip

        df, _n, skipped = load_mtr_from_zip(path.read_bytes())
        return _report(slug, df, skipped)

    if kind == "mtr_dir":
        from backend.services.mtr import load_mtr_from_extracted_files

        files = [(p.name, p.read_bytes()) for p in sorted(path.rglob("*")) if p.is_file()]
        df, _n, skipped = load_mtr_from_extracted_files(files)
        return _report(slug, df, skipped)

    if kind == "meesho_zip":
        from backend.services.meesho import load_meesho_from_zip

        df, _n, skipped = load_meesho_from_zip(path.read_bytes(), source_filename=path.name)
        return _report(slug, df, skipped)

    if kind == "flipkart_zip":
        from backend.services.flipkart import load_flipkart_from_zip

        df, _n, skipped = load_flipkart_from_zip(path, mapping, source_filename=path.name)
        return _report(slug, df, skipped)

    if kind == "raw_returns":
        frames = []
        for p in sorted(path.rglob("*.csv")):
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        cols = list(raw.columns)
        is_return = "return-date" in cols and "detailed-disposition" in cols
        has_amount = any(c.lower() in ("invoice_amount", "item-price", "amount") for c in cols)
        return {
            "slug": slug,
            "rows": int(len(raw)),
            "note": "FBA RETURNS report (return-date + disposition)" if is_return else "unknown raw csv",
            "has_sales_amount": has_amount,
            "columns": cols[:14],
        }

    return {"slug": slug, "note": f"unknown kind {kind}"}


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    mapping = _mapping()
    print(f"[audit] SKU mapping loaded: {len(mapping):,} keys")
    out = []
    for entry in spec:
        print(f"\n===== {entry['slug']}  ({entry['kind']}) =====")
        try:
            rep = parse(entry, mapping)
        except Exception as e:
            import traceback

            traceback.print_exc()
            rep = {"slug": entry["slug"], "error": str(e)}
        out.append(rep)
        print(json.dumps(rep, indent=2, default=str))
    Path("_audit_report.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[audit] wrote _audit_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
