#!/usr/bin/env python3
"""Lightweight PO column audit on a SKU subset (avoids full-catalog OOM on VPS)."""
from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd


SAMPLE_SKUS = [
    "1072YKBLACK-4XL",
    "1057YKBLUE-M",
    "DPT21MULTI",
    "1488YKWHITE-XL-XXL",
    "1488YKWHITE-M-L",
    "1057YKBLUE-L",
]


def _canon(s: str) -> str:
    return str(s or "").strip().upper()


def _pack_round(qty: float) -> int:
    if qty <= 0:
        return 0
    if qty < 10:
        return int(math.ceil(qty / 5.0) * 5)
    return int(math.ceil(qty / 10.0) * 10)


def main() -> int:
    from backend.session import AppSession
    from backend.services.po_engine import calculate_po_base, round_po_pack
    from backend.services.daily_inventory_history import (
        effective_days_from_history,
        inventory_history_max_date,
        combine_inventory_channels,
        trim_inventory_history_for_po,
        should_skip_inventory_history_extend,
        extend_history_with_sales,
        coverage_days_within,
    )
    from backend.services.po_session_hydrate import ensure_inventory_history_authoritative_for_read
    import backend.main as main_mod

    sess = AppSession()
    ok, loaded = main_mod._load_warm_cache_from_disk()
    if not ok or not loaded:
        print("FAIL: warm cache not loaded", flush=True)
        return 2
    for k, v in loaded.items():
        setattr(sess, k, v)
        if not main_mod._warm_cache:
            main_mod._warm_cache = {}
        main_mod._warm_cache[k] = v
    ensure_inventory_history_authoritative_for_read(sess)

    sales = getattr(sess, "sales_df", None)
    inv = getattr(sess, "inventory_df_variant", None)
    ih = getattr(sess, "daily_inventory_history_df", None)
    epo = getattr(sess, "existing_po_df", None)
    status = getattr(sess, "sku_status_lead_df", None)

    # Expand sample with a few high-stock and zero-stock SKUs from live inventory
    inv2 = inv.copy()
    inv2["OMS_SKU"] = inv2["OMS_SKU"].astype(str).str.strip().str.upper()
    hi = inv2.nlargest(3, "OMS_Inventory")["OMS_SKU"].tolist()
    zero = inv2[pd.to_numeric(inv2["OMS_Inventory"], errors="coerce").fillna(0) <= 0]["OMS_SKU"].head(3).tolist()
    skus = list(dict.fromkeys([_canon(s) for s in SAMPLE_SKUS + hi + zero]))
    print("SKU_SET", len(skus), skus[:12], flush=True)

    sales_f = sales.copy()
    sku_col = "OMS_SKU" if "OMS_SKU" in sales_f.columns else ("Sku" if "Sku" in sales_f.columns else None)
    if not sku_col:
        print("FAIL: sales has no Sku/OMS_SKU", list(sales_f.columns)[:20], flush=True)
        return 2
    sales_f["_SKU"] = sales_f[sku_col].astype(str).str.strip().str.upper()
    sales_f = sales_f[sales_f["_SKU"].isin(skus)].drop(columns=["_SKU"]).copy()
    inv_f = inv2[inv2["OMS_SKU"].isin(skus)].copy()
    ih_f = ih.copy()
    ih_f["OMS_SKU"] = ih_f["OMS_SKU"].astype(str).str.strip().str.upper()
    ih_f = ih_f[ih_f["OMS_SKU"].isin(skus)].copy()
    epo_f = None
    if epo is not None and not getattr(epo, "empty", True):
        epo_f = epo.copy()
        if "OMS_SKU" in epo_f.columns:
            epo_f["OMS_SKU"] = epo_f["OMS_SKU"].astype(str).str.strip().str.upper()
            epo_f = epo_f[epo_f["OMS_SKU"].isin(skus)].copy()
    status_f = None
    if status is not None and not getattr(status, "empty", True):
        status_f = status.copy()
        for c in ("OMS_SKU", "Sku", "SKU"):
            if c in status_f.columns:
                status_f[c] = status_f[c].astype(str).str.strip().str.upper()
                status_f = status_f[status_f[c].isin(skus)].copy()
                break

    period = 30
    lead = 45
    target = 180
    print("CALC_SUBSET sales", len(sales_f), "inv", len(inv_f), "hist", len(ih_f), flush=True)
    po_df = calculate_po_base(
        sales_f,
        inv_f,
        period_days=period,
        lead_time=lead,
        target_days=target,
        demand_basis="Sold",
        use_seasonality=False,
        seasonal_weight=0.5,
        existing_po_df=epo_f,
        sku_status_df=status_f,
        inventory_history_df=ih_f,
        enforce_two_size_minimum=False,  # isolate formula math
        enforce_lead_time_release_gate=True,
        grace_days=0,
        safety_pct=0.0,
        group_by_parent=False,
        sku_mapping=getattr(sess, "sku_mapping", None) or main_mod._warm_cache.get("sku_mapping"),
    )
    print("PO_ROWS", len(po_df), "cols", [c for c in po_df.columns if "Eff" in c or c in ("ADS", "PO_Qty", "Days_Left", "Projected_Running_Days", "Recent_ADS", "Sold_Units")], flush=True)

    # Independent Eff_Days using same window logic as engine
    sales_win = sales_f.copy()
    sales_win["TxnDate"] = pd.to_datetime(sales_win["TxnDate"], errors="coerce")
    max_date = sales_win["TxnDate"].max()
    max_date = max_date.normalize() if pd.notna(max_date) else None
    if max_date is None:
        max_date = inventory_history_max_date(ih_f) or pd.Timestamp("2026-07-23")
    inv_window_end = pd.Timestamp(max_date).normalize()
    hmax = inventory_history_max_date(ih_f)
    if hmax is not None:
        inv_window_end = max(inv_window_end, pd.Timestamp(hmax).normalize())
    inv_window_start = inv_window_end - timedelta(days=period - 1)
    print("WINDOW", inv_window_start.date(), "->", inv_window_end.date(), "sales_max", max_date.date(), flush=True)

    # Sales for roll-forward needs OMS_SKU
    sales_rf = sales_win.copy()
    if "OMS_SKU" not in sales_rf.columns:
        sales_rf["OMS_SKU"] = sales_rf[sku_col].astype(str).str.strip().str.upper()

    ih_trim = trim_inventory_history_for_po(ih_f, inv_window_start, inv_window_end, po_skus=set(skus))
    sheet_max = pd.to_datetime(ih_trim["Date"], errors="coerce").max() if len(ih_trim) else None
    cov = coverage_days_within(ih_trim, inv_window_start, inv_window_end)
    skip = should_skip_inventory_history_extend(sheet_max, inv_window_end, cov, ads_window=period)
    if skip:
        ih_work = ih_trim
    else:
        ih_work = extend_history_with_sales(ih_trim, sales_df=sales_rf, cap_date=inv_window_end)
    indep = effective_days_from_history(ih_work, inv_window_start, inv_window_end).rename(
        columns={"Eff_Days_Inventory": "Indep_Eff"}
    )

    merged = po_df.merge(indep, on="OMS_SKU", how="left")
    merged["Indep_Eff"] = merged["Indep_Eff"].fillna(0).astype(int)

    sold_col = next((c for c in ("Sold_Units", "ADS_Sold_Units") if c in merged.columns), None)
    fails = 0
    print("\n=== PER-SKU ===", flush=True)
    for _, r in merged.iterrows():
        sku = str(r["OMS_SKU"])
        e_days = int(pd.to_numeric(r.get("Eff_Days"), errors="coerce") or 0)
        e_inv = int(pd.to_numeric(r.get("Eff_Days_Inventory"), errors="coerce") or 0)
        indep_e = int(pd.to_numeric(r.get("Indep_Eff"), errors="coerce") or 0)
        sold = float(pd.to_numeric(r.get(sold_col), errors="coerce") or 0) if sold_col else 0.0
        recent = float(pd.to_numeric(r.get("Recent_ADS"), errors="coerce") or 0)
        ads = float(pd.to_numeric(r.get("ADS"), errors="coerce") or 0)
        inv_oms = float(pd.to_numeric(r.get("OMS_Inventory"), errors="coerce") or 0)
        inv_tot = float(pd.to_numeric(r.get("Total_Inventory"), errors="coerce") or 0)
        pipe = float(pd.to_numeric(r.get("PO_Pipeline_Effective", r.get("PO_Pipeline_Total", 0)), errors="coerce") or 0)
        days_left = float(pd.to_numeric(r.get("Days_Left"), errors="coerce") or 0)
        proj = float(pd.to_numeric(r.get("Projected_Running_Days"), errors="coerce") or 0)
        po_qty = float(pd.to_numeric(r.get("PO_Qty"), errors="coerce") or 0)
        lead_r = float(pd.to_numeric(r.get("Lead_Time_Days"), errors="coerce") or lead)

        # Inventory used for cover (Total by default)
        inv_for = inv_tot
        exp_recent = (sold / e_days) if e_days > 0 else 0.0
        exp_dl = (inv_for / ads) if ads > 0 else 999.0
        exp_proj = ((inv_for + pipe) / ads) if ads > 0 else 999.0
        need = ads > 0 and exp_proj < lead_r
        raw = ads * max(target - exp_proj, 0) if need else 0.0
        exp_po = round_po_pack(raw)

        notes = []
        status = "PASS"
        if e_inv != indep_e:
            status = "FAIL"
            notes.append(f"Eff_Inv {e_inv}!=indep {indep_e}")
        if e_inv > 0 and e_days != e_inv:
            status = "FAIL"
            notes.append(f"Eff_Days {e_days}!=Inv {e_inv}")
        if e_inv > 0 and abs(recent - exp_recent) > max(0.02, abs(exp_recent) * 0.02 + 1e-6):
            status = "FAIL"
            notes.append(f"Recent {recent:.4f}!=sold/eff {exp_recent:.4f}")
        if ads > 0 and days_left < 998 and abs(days_left - exp_dl) > 0.51:
            # try OMS
            if abs(days_left - (inv_oms / ads)) > 0.51:
                status = "FAIL"
                notes.append(f"Days_Left {days_left:.2f}!=tot {exp_dl:.2f}")
        if ads > 0 and proj < 998 and abs(proj - exp_proj) > 0.51:
            if abs(proj - ((inv_oms + pipe) / ads)) > 0.51:
                status = "FAIL"
                notes.append(f"Proj {proj:.2f}!=tot {exp_proj:.2f}")
        if need and abs(po_qty - exp_po) > max(5, exp_po * 0.05):
            # status/closed can zero PO — only fail if both non-zero diverge
            if po_qty > 0 or exp_po > 0:
                sheet = str(r.get("SKU_Sheet_Status") or "")
                if sheet.upper() in ("CLOSED", "DOUBT"):
                    notes.append(f"PO gated by status={sheet}")
                elif po_qty == 0 and exp_po > 0:
                    notes.append(f"PO zeroed (exp~{exp_po}) — check gates")
                else:
                    status = "FAIL"
                    notes.append(f"PO {po_qty}!=~{exp_po}")
        elif (not need) and po_qty > 0:
            status = "FAIL"
            notes.append(f"PO {po_qty} but lead-gate should block (proj={exp_proj:.1f}<{lead_r}? {need})")

        if status == "FAIL":
            fails += 1

        # Daily stock trail for Eff_Days transparency (last 8 days combined)
        hist_c = combine_inventory_channels(ih_f[ih_f["OMS_SKU"] == sku])
        trail = ""
        if hist_c is not None and not hist_c.empty:
            hist_c = hist_c.copy()
            hist_c["Date"] = pd.to_datetime(hist_c["Date"]).dt.normalize()
            last = hist_c.sort_values("Date").tail(8)
            trail = " ".join(f"{d.strftime('%m/%d')}={int(q)}" for d, q in zip(last["Date"], last["Qty"]))

        print(
            f"{status} {sku}: Eff={e_days} InvEff={e_inv} indep={indep_e} "
            f"sold={sold:.0f} Recent={recent:.3f} ADS={ads:.3f} "
            f"OMS={inv_oms:.0f} Tot={inv_tot:.0f} pipe={pipe:.0f} "
            f"DL={days_left:.1f} Proj={proj:.1f} PO={po_qty:.0f} "
            f"| {'; '.join(notes) or 'ok'}",
            flush=True,
        )
        if trail:
            print(f"   hist_tail: {trail}", flush=True)

    # Special check: zero current stock must not claim full-window Eff_Days from phantom roll-forward
    print("\n=== PHANTOM EFF CHECK ===", flush=True)
    for sku in skus:
        row = merged[merged["OMS_SKU"] == sku]
        if row.empty:
            continue
        r = row.iloc[0]
        oms = float(pd.to_numeric(r.get("OMS_Inventory"), errors="coerce") or 0)
        tot = float(pd.to_numeric(r.get("Total_Inventory"), errors="coerce") or 0)
        e_days = int(pd.to_numeric(r.get("Eff_Days"), errors="coerce") or 0)
        if oms <= 0 and tot <= 0 and e_days >= 25:
            # Inspect last history qty
            hc = combine_inventory_channels(ih_f[ih_f["OMS_SKU"] == sku])
            if hc is not None and not hc.empty:
                hc = hc.copy()
                hc["Date"] = pd.to_datetime(hc["Date"]).dt.normalize()
                last_q = float(hc.sort_values("Date").iloc[-1]["Qty"])
                print(f"WARN {sku}: stock=0 Eff={e_days} last_hist_qty={last_q}", flush=True)
                if last_q <= 0 and e_days >= 25:
                    fails += 1
                    print(f"FAIL {sku}: phantom Eff_Days on zero history", flush=True)
            else:
                print(f"WARN {sku}: stock=0 Eff={e_days} no hist rows", flush=True)

    print("AUDIT_FAILS", fails, flush=True)
    print("RESULT", "PASS" if fails == 0 else "FAIL", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
