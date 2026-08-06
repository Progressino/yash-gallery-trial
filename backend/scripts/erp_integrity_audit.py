#!/usr/bin/env python3
"""
End-to-end ERP integrity audit against production warm-cache frames.

Uses real uploaded data only (parquet / meta under /data/warm_cache).
Does not invent mock sales or inventory.

  docker compose -p progressino -f docker-compose.prod.yml exec -T backend \\
    python -m backend.scripts.erp_integrity_audit
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


def _cache() -> Path:
    return Path(os.environ.get("WARM_CACHE_DIR", "/data/warm_cache"))


def _read_parquet(name: str) -> pd.DataFrame | None:
    p = _cache() / name
    if not p.is_file():
        return None
    try:
        return pd.read_parquet(p)
    except Exception as exc:
        return pd.DataFrame({"__error__": [str(exc)]})


def _read_json(name: str) -> dict:
    p = _cache() / name
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _check(
    cid: str,
    area: str,
    title: str,
    ok: bool,
    detail: str,
    *,
    severity: str = "fail",
    data: dict | None = None,
) -> dict:
    return {
        "id": cid,
        "area": area,
        "title": title,
        "ok": bool(ok),
        "severity": "ok" if ok else severity,
        "detail": detail,
        "data": data or {},
    }


def phase_upload_presence() -> list[dict]:
    out: list[dict] = []
    files = {
        "sales_df.parquet": "Unified sales frame",
        "inventory_df_variant.parquet": "Current inventory snapshot",
        "daily_inventory_history_df.parquet": "Daily inventory history",
        "mtr_df.parquet": "Amazon MTR",
        "myntra_df.parquet": "Myntra sales",
        "meesho_df.parquet": "Meesho sales",
        "flipkart_df.parquet": "Flipkart sales",
        "snapdeal_df.parquet": "Snapdeal sales",
        "sku_status_lead_df.parquet": "SKU status / lead times",
        "existing_po_df.parquet": "Existing PO / pipeline",
        "manual_intransit_overlay_df.parquet": "Manual in-transit overlay",
    }
    present = []
    missing = []
    for fn, label in files.items():
        p = _cache() / fn
        if p.is_file():
            present.append(fn)
            sz = p.stat().st_size
            out.append(
                _check(
                    f"file:{fn}",
                    "upload",
                    f"{label} present",
                    True,
                    f"{fn} ({sz:,} bytes)",
                    data={"path": str(p), "bytes": sz},
                )
            )
        else:
            missing.append(fn)
    out.append(
        _check(
            "upload:manifest",
            "upload",
            "Core frames present",
            "sales_df.parquet" in present and "inventory_df_variant.parquet" in present,
            f"present={len(present)} missing={missing[:8]}",
            severity="fail",
            data={"present": present, "missing": missing},
        )
    )
    return out


def phase_sku_canonicalisation(sales: pd.DataFrame | None, inv: pd.DataFrame | None) -> list[dict]:
    out: list[dict] = []
    try:
        from backend.services.inventory import coalesce_inventory_by_sku_mapping, _inventory_alias_oms_key
        from backend.services.po_engine import inventory_oms_key, canonical_oms_key
        from backend.services.sku_mapping import load_sku_mapping_from_disk, load_bundled_sku_mapping
    except Exception as exc:
        return [_check("sku_canon_import", "sku", "Import SKU helpers", False, str(exc))]

    mapping = {}
    try:
        mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
    except Exception:
        pass

    # Case + alias self-consistency
    samples = [
        ("7114YKPOWDERBLUE-F", "7114YKPOWERBLUE-F"),
        ("7100YKYEAL-L-XL", "7100YKTEAL-L-XL"),
        ("5041YKBOTTLEGREEN-XL", "5041YKBOTTELGREEN-XL"),
        ("1130YKPURPLE-6xl", "1130YKPURPLE-6XL"),
        ("1415YKBALCK-6XL", "1415YKBLACK-6XL"),
    ]
    alias_ok = True
    details = []
    for a, b in samples:
        ka = inventory_oms_key(a)
        kb = inventory_oms_key(b)
        aa = _inventory_alias_oms_key(a, mapping)
        ab = _inventory_alias_oms_key(b, mapping)
        match = aa == ab
        if not match:
            alias_ok = False
        details.append({"a": a, "b": b, "key_a": aa, "key_b": ab, "match": match})
    out.append(
        _check(
            "sku:builtin_aliases",
            "sku",
            "Built-in twin spellings collapse",
            alias_ok,
            f"{sum(1 for d in details if d['match'])}/{len(details)} twin pairs collapse",
            data={"pairs": details},
        )
    )

    if inv is not None and not inv.empty and "OMS_SKU" in inv.columns:
        raw_n = inv["OMS_SKU"].astype(str).nunique()
        co = coalesce_inventory_by_sku_mapping(inv, mapping)
        co_n = co["OMS_SKU"].nunique() if co is not None and not co.empty else 0
        # Casefold twins should collapse
        upper = inv["OMS_SKU"].astype(str).str.strip().str.upper()
        case_twins = int(upper.nunique() < inv["OMS_SKU"].astype(str).nunique())
        out.append(
            _check(
                "sku:inv_coalesce",
                "sku",
                "Inventory coalesce reduces spelling twins",
                co_n <= raw_n,
                f"raw_skus={raw_n} coalesced_skus={co_n} casefold_duplicates={bool(case_twins)}",
                severity="warn",
                data={"raw": raw_n, "coalesced": co_n},
            )
        )

        # Detect residual BOTTLE + BOTTEL both present (should not after coalesce)
        skus = set(co["OMS_SKU"].astype(str).str.upper()) if co is not None else set()
        bottle_both = any(
            s.replace("BOTTLE", "BOTTEL") in skus and "BOTTLE" in s and "BOTTEL" not in s
            for s in skus
        )
        out.append(
            _check(
                "sku:bottle_residual",
                "sku",
                "No residual BOTTLE twin alongside BOTTEL",
                not bottle_both,
                "BOTTLE and BOTTEL both present after coalesce" if bottle_both else "OK",
            )
        )

    if sales is not None and not sales.empty:
        sku_col = next(
            (c for c in ("OMS_SKU", "Sku", "SKU", "sku") if c in sales.columns),
            None,
        )
        if sku_col:
            s = sales[sku_col].astype(str)
            mixed = int((s != s.str.upper()).sum())
            out.append(
                _check(
                    "sku:sales_case",
                    "sku",
                    "Sales SKUs mostly uppercased after parse",
                    mixed < max(100, int(len(s) * 0.01)),
                    f"mixed_case_rows={mixed}/{len(s)}",
                    severity="warn",
                    data={"mixed_case_rows": mixed},
                )
            )
    return out


def phase_inventory_history(
    inv: pd.DataFrame | None,
    hist: pd.DataFrame | None,
    inv_meta: dict,
) -> list[dict]:
    out: list[dict] = []
    if hist is None or hist.empty:
        return [_check("hist:empty", "history", "Inventory history present", False, "No history parquet")]

    try:
        from backend.services.daily_inventory_history import (
            coalesce_inventory_history_sku_aliases,
            filter_inventory_history_channel,
            inventory_history_wide_matrix,
            align_history_day_to_variant,
            scrub_absurd_inventory_history_rows,
        )
        from backend.services.inventory import coalesce_inventory_by_sku_mapping
    except Exception as exc:
        return [_check("hist:import", "history", "Import history helpers", False, str(exc))]

    work = scrub_absurd_inventory_history_rows(hist)
    dates = pd.to_datetime(work["Date"], errors="coerce").dt.normalize()
    max_d = dates.max()
    min_d = dates.min()
    out.append(
        _check(
            "hist:range",
            "history",
            "History date range",
            pd.notna(max_d),
            f"{min_d.date() if pd.notna(min_d) else '?'} → {max_d.date() if pd.notna(max_d) else '?'}"
            f" days={dates.nunique()} rows={len(work)} skus={work['OMS_SKU'].nunique()}",
            data={
                "min": str(min_d.date()) if pd.notna(min_d) else "",
                "max": str(max_d.date()) if pd.notna(max_d) else "",
                "days": int(dates.nunique()),
                "rows": int(len(work)),
            },
        )
    )

    # Channel view totals for max day
    if pd.notna(max_d):
        day = work[dates == max_d]
        for ch in ("oms", "amazon", "combined"):
            try:
                view = filter_inventory_history_channel(work, ch)
                vd = view[pd.to_datetime(view["Date"], errors="coerce").dt.normalize() == max_d]
                if not vd.empty:
                    vd = vd.groupby(vd["OMS_SKU"].astype(str).str.upper(), as_index=False)["Qty"].max()
                tot = float(pd.to_numeric(vd["Qty"], errors="coerce").fillna(0).sum()) if len(vd) else 0.0
                out.append(
                    _check(
                        f"hist:day_total_{ch}",
                        "history",
                        f"Day {max_d.date()} {ch} total",
                        tot >= 0,
                        f"total={tot:,.0f} skus={len(vd)}",
                        severity="ok",
                        data={"date": str(max_d.date()), "channel": ch, "total": tot},
                    )
                )
            except Exception as exc:
                out.append(
                    _check(
                        f"hist:day_total_{ch}",
                        "history",
                        f"Day {max_d.date()} {ch} total",
                        False,
                        str(exc),
                    )
                )

    # Align latest snapshot to variant when both exist
    if inv is not None and not inv.empty and "OMS_Inventory" in inv.columns and pd.notna(max_d):
        inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), {})
        oms_sum = float(pd.to_numeric(inv_c["OMS_Inventory"], errors="coerce").fillna(0).sum())
        snap = str(inv_meta.get("inventory_snapshot_date") or "")[:10]
        align_date = snap if len(snap) == 10 else str(max_d.date())
        try:
            aligned = align_history_day_to_variant(work, inv_c, align_date)
            oms_view = filter_inventory_history_channel(aligned, "oms")
            day_rows = oms_view[
                pd.to_datetime(oms_view["Date"], errors="coerce").dt.normalize()
                == pd.Timestamp(align_date)
            ]
            if not day_rows.empty:
                day_rows = day_rows.groupby(
                    day_rows["OMS_SKU"].astype(str).str.upper(), as_index=False
                )["Qty"].max()
            hist_oms = float(pd.to_numeric(day_rows["Qty"], errors="coerce").fillna(0).sum())
            delta = abs(hist_oms - oms_sum)
            # Allow small float/rounding noise
            ok = delta <= max(50.0, oms_sum * 0.002)
            out.append(
                _check(
                    "hist:vs_actual_oms",
                    "history",
                    "History OMS day matches Inventory OMS_Inventory",
                    ok,
                    (
                        f"date={align_date} history_oms={hist_oms:,.0f} "
                        f"inventory_oms={oms_sum:,.0f} Δ={hist_oms - oms_sum:+,.0f}"
                    ),
                    severity="fail" if not ok else "ok",
                    data={
                        "date": align_date,
                        "history_oms": hist_oms,
                        "inventory_oms": oms_sum,
                        "delta": hist_oms - oms_sum,
                    },
                )
            )
            # Also compute pre-align for diagnosis
            pre = filter_inventory_history_channel(work, "oms")
            pre_day = pre[
                pd.to_datetime(pre["Date"], errors="coerce").dt.normalize()
                == pd.Timestamp(align_date)
            ]
            if not pre_day.empty:
                pre_day = pre_day.groupby(
                    pre_day["OMS_SKU"].astype(str).str.upper(), as_index=False
                )["Qty"].max()
            pre_tot = float(pd.to_numeric(pre_day["Qty"], errors="coerce").fillna(0).sum())
            out.append(
                _check(
                    "hist:pre_align_oms",
                    "history",
                    "History OMS before align (diagnostic)",
                    True,
                    f"pre_align={pre_tot:,.0f} inventory={oms_sum:,.0f} Δ={pre_tot - oms_sum:+,.0f}",
                    data={"pre_align": pre_tot, "inventory_oms": oms_sum},
                )
            )
        except Exception as exc:
            out.append(
                _check(
                    "hist:vs_actual_oms",
                    "history",
                    "History OMS day matches Inventory OMS_Inventory",
                    False,
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-800:]}",
                )
            )

        # Component vs total
        if "Total_Inventory" in inv_c.columns:
            total = float(pd.to_numeric(inv_c["Total_Inventory"], errors="coerce").fillna(0).sum())
            # rebuild expected from source columns if possible
            from backend.services.inventory import inventory_source_columns, recompute_inventory_totals

            rebuilt = recompute_inventory_totals(inv_c.copy())
            t2 = float(pd.to_numeric(rebuilt["Total_Inventory"], errors="coerce").fillna(0).sum())
            out.append(
                _check(
                    "inv:total_recompute",
                    "inventory",
                    "Total_Inventory matches recompute",
                    abs(total - t2) <= 1.0,
                    f"stored_total={total:,.0f} recomputed={t2:,.0f}",
                    data={"stored": total, "recomputed": t2},
                )
            )
            # OMS should not exceed Total by large amount (possible double)
            out.append(
                _check(
                    "inv:oms_lte_total_plus_noise",
                    "inventory",
                    "OMS not absurdly above Total",
                    oms_sum <= total + 1000 or total == 0,
                    f"oms={oms_sum:,.0f} total={total:,.0f}",
                    severity="warn",
                )
            )

    # Matrix date_totals sanity
    try:
        wide = inventory_history_wide_matrix(
            work, days=7, end_date=str(max_d.date()) if pd.notna(max_d) else None, channel="oms"
        )
        dt = wide.get("date_totals") or []
        last = float(dt[-1]) if dt else 0.0
        out.append(
            _check(
                "hist:matrix_oms_total",
                "history",
                "Wide matrix last OMS date total",
                last > 0,
                f"last_total={last:,.0f} dates={wide.get('dates', [])[-3:]}",
                severity="warn",
                data={"last_total": last, "date_totals_tail": dt[-3:] if dt else []},
            )
        )
    except Exception as exc:
        out.append(
            _check(
                "hist:matrix_oms_total",
                "history",
                "Wide matrix last OMS date total",
                False,
                str(exc),
            )
        )

    # Alias coalesce should not explode row qty on dual census
    try:
        c = coalesce_inventory_history_sku_aliases(work, {})
        # bottle day max qty per sku after coalesce vs naive upper-sum on one sample day
        if pd.notna(max_d):
            day = work[dates == max_d].copy()
            day["U"] = day["OMS_SKU"].astype(str).str.upper()
            bottle = day[day["U"].str.contains("BOTTEL|BOTTLE", na=False)]
            if not bottle.empty:
                raw_sum = float(pd.to_numeric(bottle["Qty"], errors="coerce").fillna(0).sum())
                cday = c[pd.to_datetime(c["Date"], errors="coerce").dt.normalize() == max_d]
                cbot = cday[cday["OMS_SKU"].astype(str).str.contains("BOTTEL|BOTTLE", na=False)]
                c_sum = float(pd.to_numeric(cbot["Qty"], errors="coerce").fillna(0).sum())
                # After coalesce BOTTLE rows gone; should not be >> raw
                out.append(
                    _check(
                        "hist:bottle_coalesce",
                        "history",
                        "BOTTLE family not double-counted after coalesce",
                        c_sum <= raw_sum * 1.05 + 100,
                        f"raw_bottle_family={raw_sum:,.0f} coalesced={c_sum:,.0f}",
                        severity="warn",
                        data={"raw": raw_sum, "coalesced": c_sum},
                    )
                )
    except Exception as exc:
        out.append(
            _check("hist:bottle_coalesce", "history", "BOTTLE coalesce", False, str(exc), severity="warn")
        )

    return out


def phase_sales_inventory_overlap(
    sales: pd.DataFrame | None,
    inv: pd.DataFrame | None,
) -> list[dict]:
    out: list[dict] = []
    if sales is None or sales.empty or inv is None or inv.empty:
        return [
            _check(
                "sales_inv:frames",
                "sales",
                "Sales and inventory frames loadable",
                False,
                f"sales={sales is not None and not getattr(sales,'empty',True)} "
                f"inv={inv is not None and not getattr(inv,'empty',True)}",
            )
        ]

    try:
        from backend.services.inventory import coalesce_inventory_by_sku_mapping
        from backend.services.po_engine import canonical_oms_key
        from backend.services.sku_mapping import load_sku_mapping_from_disk, load_bundled_sku_mapping

        mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
    except Exception as exc:
        return [_check("sales_inv:import", "sales", "Import for overlap", False, str(exc))]

    inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), mapping)
    inv_skus = set(inv_c["OMS_SKU"].astype(str).str.upper())

    sku_col = next((c for c in ("OMS_SKU", "Sku", "SKU") if c in sales.columns), None)
    qty_col = next(
        (c for c in ("Units_Effective", "Quantity", "Qty", "Units") if c in sales.columns),
        None,
    )
    date_col = next(
        (c for c in ("TxnDate", "Date", "Order_Date", "order_date") if c in sales.columns),
        None,
    )
    out.append(
        _check(
            "sales:columns",
            "sales",
            "Sales frame has SKU + qty columns",
            bool(sku_col and qty_col),
            f"sku_col={sku_col} qty_col={qty_col} date_col={date_col} rows={len(sales)} cols={list(sales.columns)[:20]}",
            data={"cols": list(sales.columns)[:40]},
        )
    )
    if not sku_col or not qty_col:
        return out

    work = sales[[sku_col, qty_col] + ([date_col] if date_col else [])].copy()
    work["_sku"] = work[sku_col].map(lambda v: canonical_oms_key(v, mapping))
    work["_qty"] = pd.to_numeric(work[qty_col], errors="coerce").fillna(0)
    total_units = float(work["_qty"].clip(lower=0).sum())
    neg = int((work["_qty"] < 0).sum())
    out.append(
        _check(
            "sales:units",
            "sales",
            "Sales units aggregate",
            total_units > 0,
            f"positive_units={total_units:,.0f} negative_rows={neg} unique_skus={work['_sku'].nunique()}",
            severity="warn" if total_units <= 0 else "ok",
            data={"units": total_units, "neg_rows": neg},
        )
    )

    # Last 30d if dates exist
    if date_col:
        work["_d"] = pd.to_datetime(work[date_col], errors="coerce")
        mx = work["_d"].max()
        if pd.notna(mx):
            win = work[work["_d"] >= (pd.Timestamp(mx).normalize() - pd.Timedelta(days=29))]
            last30 = float(win["_qty"].clip(lower=0).sum())
            out.append(
                _check(
                    "sales:last30",
                    "sales",
                    "Sales last 30 days ending last txn",
                    last30 >= 0,
                    f"end={mx.date() if hasattr(mx,'date') else mx} units_30d={last30:,.0f}",
                    data={"end": str(mx)[:10], "units_30d": last30},
                )
            )

    sales_skus = set(work["_sku"].astype(str).str.upper()) - {""}
    only_sales = len(sales_skus - inv_skus)
    only_inv = len(inv_skus - sales_skus)
    both = len(sales_skus & inv_skus)
    out.append(
        _check(
            "sales:inv_sku_overlap",
            "sales",
            "Sales vs inventory SKU overlap",
            both > 0,
            f"both={both} sales_only={only_sales} inv_only={only_inv}",
            severity="warn",
            data={"both": both, "sales_only": only_sales, "inv_only": only_inv},
        )
    )
    return out


def phase_po_cross_check(
    sales: pd.DataFrame | None,
    inv: pd.DataFrame | None,
    hist: pd.DataFrame | None,
    status: pd.DataFrame | None,
    existing_po: pd.DataFrame | None,
) -> list[dict]:
    out: list[dict] = []
    try:
        from backend.session import AppSession
        from backend.services.inventory import coalesce_inventory_by_sku_mapping
        from backend.services.po_engine import calculate_po_base
        from backend.services.sku_mapping import load_sku_mapping_from_disk, load_bundled_sku_mapping
    except Exception as exc:
        return [_check("po:import", "po", "Import PO engine", False, str(exc))]

    if inv is None or inv.empty or sales is None or sales.empty:
        return [_check("po:frames", "po", "Frames for PO", False, "Need sales + inventory")]

    mapping = {}
    try:
        mapping = {**(load_bundled_sku_mapping() or {}), **(load_sku_mapping_from_disk() or {})}
    except Exception:
        pass

    sess = AppSession()
    inv_c = coalesce_inventory_by_sku_mapping(inv.copy(), mapping)
    sess.inventory_df_variant = inv_c
    sess.sales_df = sales
    if hist is not None and not hist.empty:
        sess.daily_inventory_history_df = hist
    if status is not None and not status.empty:
        sess.sku_status_lead_df = status
    if existing_po is not None and not existing_po.empty:
        sess.existing_po_df = existing_po
    sess.sku_mapping = mapping

    body = {
        "period_days": 30,
        "lead_time": 45,
        "target_days": 45,
        "use_seasonality": False,
        "group_by_parent": False,
        "use_oms_inventory_only": False,
        "inventory_history_channel": "oms",
    }
    try:
        # Ensure sales has TxnDate expected by the engine
        sales_work = sales
        if sales is not None and "TxnDate" not in sales.columns:
            for c in ("Date", "Order_Date", "order_date", "Txn_Date"):
                if c in sales.columns:
                    sales_work = sales.copy()
                    sales_work["TxnDate"] = pd.to_datetime(sales[c], errors="coerce")
                    break
        if sales_work is None or "TxnDate" not in sales_work.columns:
            out.append(
                _check(
                    "po:sales_txndate",
                    "po",
                    "Sales has TxnDate for PO engine",
                    False,
                    f"cols={list(sales.columns)[:30] if sales is not None else None}",
                )
            )
            return out
        po_df = calculate_po_base(
            sales_df=sales_work,
            inv_df=inv_c,
            period_days=int(body["period_days"]),
            lead_time=int(body["lead_time"]),
            target_days=int(body["target_days"]),
            use_seasonality=bool(body["use_seasonality"]),
            group_by_parent=bool(body["group_by_parent"]),
            existing_po_df=existing_po if existing_po is not None else pd.DataFrame(),
            sku_status_df=status if status is not None else pd.DataFrame(),
            inventory_history_df=hist,
            sku_mapping=mapping,
            use_oms_inventory_only=bool(body["use_oms_inventory_only"]),
            inventory_history_channel=str(body["inventory_history_channel"]),
        )
        if po_df is None or getattr(po_df, "empty", True):
            out.append(
                _check(
                    "po:run",
                    "po",
                    "PO base calculated",
                    False,
                    "calculate_po_base returned empty — often incomplete sales schema",
                )
            )
            return out
        out.append(
            _check(
                "po:run",
                "po",
                "PO base calculated",
                True,
                f"rows={len(po_df)} cols={list(po_df.columns)[:25]}",
            )
        )

        # Inventory consistency: sample top inventory SKUs
        inv_col = "Total_Inventory" if "Total_Inventory" in po_df.columns else None
        if inv_col is None and "OMS_Inventory" in po_df.columns:
            inv_col = "OMS_Inventory"
        sku_col = "OMS_SKU" if "OMS_SKU" in po_df.columns else po_df.columns[0]
        inv_idx = inv_c.set_index(inv_c["OMS_SKU"].astype(str).str.upper())
        mismatches = []
        sample = po_df.nlargest(min(80, len(po_df)), inv_col) if inv_col else po_df.head(80)
        for _, row in sample.iterrows():
            sku = str(row[sku_col]).strip().upper()
            if sku not in inv_idx.index:
                continue
            po_inv = float(pd.to_numeric(row.get(inv_col), errors="coerce") or 0)
            src_raw = inv_idx.loc[sku]
            if isinstance(src_raw, pd.DataFrame):
                src_raw = src_raw.iloc[0]
            src_col = inv_col if inv_col in inv_idx.columns else "Total_Inventory"
            src_inv = float(pd.to_numeric(src_raw.get(src_col), errors="coerce") or 0)
            if abs(po_inv - src_inv) > 0.51:
                mismatches.append(
                    {"sku": sku, "po": po_inv, "inventory": src_inv, "delta": po_inv - src_inv}
                )
        out.append(
            _check(
                "po:inv_match_sample",
                "po",
                "PO inventory matches snapshot (top inventory SKUs)",
                len(mismatches) == 0,
                f"mismatches={len(mismatches)} of {len(sample)} sampled",
                severity="fail" if mismatches else "ok",
                data={"mismatches": mismatches[:25]},
            )
        )

        if "Eff_Days" in po_df.columns:
            eff = pd.to_numeric(po_df["Eff_Days"], errors="coerce").fillna(0)
            out.append(
                _check(
                    "po:eff_days_range",
                    "po",
                    "Eff_Days within ADS window",
                    bool((eff >= 0).all() and (eff <= 90).all()),
                    f"min={float(eff.min())} max={float(eff.max())} mean={float(eff.mean()):.1f}",
                    severity="warn",
                )
            )
        for col in ("Final_PO_Qty", "PO_Qty", "New_PO_Qty", "Raise_Qty"):
            if col in po_df.columns:
                q = pd.to_numeric(po_df[col], errors="coerce").fillna(0)
                out.append(
                    _check(
                        f"po:qty_{col}",
                        "po",
                        f"{col} non-negative",
                        bool((q >= -0.01).all()),
                        f"sum={float(q.clip(lower=0).sum()):,.0f} negatives={(q < -0.01).sum()}",
                    )
                )
                break
    except Exception as exc:
        out.append(
            _check(
                "po:run",
                "po",
                "PO base calculated",
                False,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}",
            )
        )
    return out


def phase_data_health_builtin() -> list[dict]:
    try:
        from backend.services.data_health import run_data_health_checks

        result = run_data_health_checks()
        checks = list(result.get("checks") or [])
        # Prefix area so report is clear
        for c in checks:
            c["id"] = f"builtin:{c.get('id')}"
        return checks
    except Exception as exc:
        return [
            _check(
                "builtin:data_health",
                "health",
                "Built-in data_health suite",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        ]


def main() -> int:
    report: dict[str, Any] = {
        "ok": True,
        "cache_dir": str(_cache()),
        "phases": {},
        "checks": [],
        "root_causes": [],
        "summary": {},
    }
    try:
        meta = _read_json("inventory_session_meta.json")
        sales = _read_parquet("sales_df.parquet")
        inv = _read_parquet("inventory_df_variant.parquet")
        hist = _read_parquet("daily_inventory_history_df.parquet")
        status = _read_parquet("sku_status_lead_df.parquet")
        existing_po = _read_parquet("existing_po_df.parquet")

        checks: list[dict] = []
        checks.extend(phase_upload_presence())
        checks.extend(phase_sku_canonicalisation(sales, inv if inv is not None and "__error__" not in (inv.columns if inv is not None else []) else None))
        # Clean error frames
        if inv is not None and "__error__" in inv.columns:
            checks.append(_check("inv:load", "inventory", "Load inventory", False, str(inv["__error__"].iloc[0])))
            inv = None
        if sales is not None and "__error__" in sales.columns:
            checks.append(_check("sales:load", "sales", "Load sales", False, str(sales["__error__"].iloc[0])))
            sales = None
        if hist is not None and "__error__" in hist.columns:
            checks.append(_check("hist:load", "history", "Load history", False, str(hist["__error__"].iloc[0])))
            hist = None

        checks.extend(phase_inventory_history(inv, hist, meta))
        checks.extend(phase_sales_inventory_overlap(sales, inv))
        checks.extend(phase_po_cross_check(sales, inv, hist, status, existing_po))
        checks.extend(phase_data_health_builtin())

        fails = [c for c in checks if not c.get("ok") and c.get("severity") == "fail"]
        warns = [c for c in checks if not c.get("ok") and c.get("severity") == "warn"]
        report["checks"] = checks
        report["ok"] = len(fails) == 0
        report["summary"] = {
            "total_checks": len(checks),
            "fail_count": len(fails),
            "warn_count": len(warns),
            "pass_count": sum(1 for c in checks if c.get("ok")),
            "inventory_snapshot_date": meta.get("inventory_snapshot_date"),
            "inventory_uploaded_at": meta.get("inventory_snapshot_uploaded_at"),
        }
        # Synthesize root causes from fails
        for c in fails:
            report["root_causes"].append(
                {
                    "id": c.get("id"),
                    "area": c.get("area"),
                    "title": c.get("title"),
                    "detail": c.get("detail"),
                    "severity": "high",
                    "data": c.get("data"),
                }
            )
        for c in warns:
            report["root_causes"].append(
                {
                    "id": c.get("id"),
                    "area": c.get("area"),
                    "title": c.get("title"),
                    "detail": c.get("detail"),
                    "severity": "medium",
                    "data": c.get("data"),
                }
            )
    except Exception as exc:
        report["ok"] = False
        report["fatal"] = f"{type(exc).__name__}: {exc}"
        report["trace"] = traceback.format_exc()[-3000:]

    print(json.dumps(report, default=str, indent=2))
    # Compact human footer on stderr
    s = report.get("summary") or {}
    print(
        f"\n# SUMMARY ok={report.get('ok')} checks={s.get('total_checks')} "
        f"fail={s.get('fail_count')} warn={s.get('warn_count')} "
        f"snap={s.get('inventory_snapshot_date')}",
        file=sys.stderr,
    )
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
