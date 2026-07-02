# Meesho & Snapdeal Data Audit Report

**Date:** 2026-07-02  
**Scope:** Test five archives before upload; apply new rules — **store historical in Tier‑1**, and **exclude 2024** (PO uses the last 8 quarters).

## Result: none of the five archives are usable — wrong export type

Every file is a **tax / settlement report with no SKU column**, so sales cannot be attributed to products. They were **not uploaded**.

| Archive | Files | Detected format | SKU present? | Verdict |
|---------|-------|-----------------|--------------|---------|
| Meesho PE.zip | 11 | Meesho **TCS / GST** (`tcs_sales.xlsx`, `tcs_sales_return.xlsx`) | **No** (OMS_SKU all blank) | Reject |
| Meesho AG.zip | 10 | Meesho **TCS / GST** | **No** | Reject |
| Snapdeal PE.zip | 26 | Snapdeal **payment/settlement** (Summary, Total_Suborders, Returns, Commission, Payments, TCS) | **No** | Reject |
| Snapdeal AG.zip | 26 | Snapdeal **payment/settlement** | **No** | Reject |
| Snapdeal YG.zip | 26 | Snapdeal **payment/settlement** | **No** | Reject |

### Why they fail

- **Meesho PE/AG** — each monthly `.zip` contains `tcs_sales.xlsx` / `tcs_sales_return.xlsx`. These are **Tax-Collected-at-Source (GST) filings**. The parser extracts date / quantity / order id / state, but the **Seller SKU field does not exist**, so `OMS_SKU` is empty on 100% of rows (Meesho AG: 12,165 rows, all blank SKU; Meesho PE: 4,134 rows, all blank).
- **Snapdeal PE/AG/YG** — every workbook is a **settlement/payment report**. Sheets are `Summary, Total_Suboders, Returns, Commission and other charges, Non Order Transactions, Payments, ClosingBalance, TCS`. None of the nine sheets contain a SKU / supplier-SKU / product column. The Snapdeal parser correctly rejects each file:
  > "Snapdeal payment/settlement XLSX (no Seller SKU column). Use the Snapdeal seller order / OMS export for per-SKU sales."

Uploading these would add SKU-less order/refund rows that pollute the platform frame without helping per-SKU PO — so ingestion is blocked by a new safety guard (below).

## What is needed instead

Please re-export the **per-SKU sales / order reports** (the same type used for the existing Meesho and Flipkart history):

- **Meesho** → *Order export* / *Outward orders* report that includes the **SKU / Supplier SKU** column (not the GST/TCS filing).
- **Snapdeal** → *Seller order report* / *OMS export* with the **Seller SKU (SUPC/Supplier SKU)** column (not the payment/settlement statement).

Once those arrive, ingestion is one command per archive (see below).

---

## New rules implemented (going forward)

Both directives are now built into `backend/scripts/ingest_platform_zip_disk.py`:

### 1. Historical data → Tier‑1 (default)

- **Tier‑1** = the durable warm-cache platform frames (`meesho_df`, `snapdeal_df`, `myntra_df`, `flipkart_df`, `mtr_df`) persisted as parquet in `WARM_CACHE_DIR` (`/data/warm_cache`). This is the **primary source for quarterly PO** and survives restarts/redeploys (Docker volume).
- **Tier‑3** = capped SQLite `daily_uploads` (max ~200 files/platform, auto-trimmed) — for incremental daily uploads only.
- The ingest tool now writes **Tier‑1 by default**; `--tier3` is opt-in for dailies.

### 2. Drop 2024 (last 8 quarters)

- `--min-date 2025-01-01` filters out all pre-2025 rows before merging.

### 3. SKU safety guard

- The tool **aborts** (exit code 2) when parsed rows have no usable `OMS_SKU` — exactly what caught all five of these tax/settlement files. Override with `--allow-empty-sku` only if intentional.

### Usage (for the correct exports)

```bash
docker compose -p progressino -f docker-compose.prod.yml exec -T backend \
  python -m backend.scripts.ingest_platform_zip_disk meesho "/tmp/Meesho Orders.zip" \
  --min-date 2025-01-01
```

```bash
docker compose -p progressino -f docker-compose.prod.yml exec -T backend \
  python -m backend.scripts.ingest_platform_zip_disk snapdeal "/tmp/Snapdeal Seller Orders.zip" \
  --min-date 2025-01-01
```

---

## Note on the earlier Myntra/Flipkart upload

Those were written to **Tier‑3** and included 2024. Going forward the correct exports will land in **Tier‑1 with 2024 excluded**. If you want, I can also **re-home the existing Myntra/Flipkart history into Tier‑1 and trim 2024** in a follow-up so all platforms follow the same rule.

---

## Files in repo

- `scripts/check_meesho_snapdeal_zips.py` — local parse/QA for these archives
- `backend/scripts/ingest_platform_zip_disk.py` — Tier‑1 (default) / Tier‑3 ingest with `--min-date` filter and SKU guard
