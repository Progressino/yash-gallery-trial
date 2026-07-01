# Myntra & Flipkart Data Audit Report

**Date:** 2026-07-01  
**Scope:** Compare your four archive ZIPs vs production (`app.progressino.com`) Tier-3 sales history.

## Archives tested

| File | Channel | Parsed rows | Date range |
|------|---------|-------------|------------|
| Myntra PPMP Jan-2024 to Jan-2026.zip | PPMP | 131,546 | 2023-12-07 → 2026-01-31 |
| Myntra Sjit Jan-2024 to Jan-2026.zip | SJIT | 18,908 | 2024-01-01 → 2026-01-31 |
| Flipkart Akiko Jan-2024 To Dec-2025.zip | PE (Akiko) | 47,131 | 2023-12-02 → 2025-12-31 |
| Flipkart AG Jan-2024 To Dec-2025.zip | AG | 11,481 | 2024-07-02 → 2025-12-31 |

*Combined after dedup: Myntra **149,373** rows · Flipkart **58,612** rows (shipment months: 26 / 25).*

---

## Production before upload

| Platform | Tier-3 files | Rows | Min date | Max date |
|----------|--------------|------|----------|----------|
| Myntra | 133 | 202,515 | 2023-12-07 | 2026-06-30 |
| Flipkart | 146 | 82,999 | **2024-12-18** | 2026-06-28 |

### Key gaps found (shipment units, month-by-month)

**Flipkart — entire 2024 missing on app**

Your Akiko ZIP has Jan–Nov 2024 data; production had **zero** shipment units for those months. Only Dec 2024 onward existed (mostly from daily uploads and an older `Flipkart YG (2).zip`).

| Month | Your ZIPs | App (before) | Status |
|-------|-----------|--------------|--------|
| 2024-01 | 1,055 | 0 | Missing on app |
| 2024-02 | 1,144 | 0 | Missing on app |
| 2024-03 … 2024-11 | 754–2,855 | 0 | Missing on app |
| 2024-12 | 429 | 17 | Severe undercount |

**Myntra — widespread undercount vs your archives**

Most 2024–2025 months on the app were **25–80% below** your PPMP+SJIT ZIP totals. Worst example:

| Month | Your ZIPs | App (before) | Gap |
|-------|-----------|--------------|-----|
| 2024-06 | 4,223 | 298 | **−93%** on app |
| 2025-03 | 6,106 | 3,546 | −42% |
| 2026-01 | 7,038 | 15,043 | App higher (daily uploads after ZIP end) |

**Root cause:** Tier-3 held many partial daily seller-order CSVs. Bulk PPMP/SJIT history was not fully loaded; dedup kept incomplete daily snapshots for several months.

---

## Actions taken on production

1. **Ingested Myntra PPMP ZIP** → Tier-3 `Myntra PPMP Jan-2024 to Jan-2026.zip` (131,107 rows)
2. **Ingested Myntra SJIT ZIP** → Tier-3 (duplicate filename + `platform.zip` cleanup pending)
3. **Ingested Flipkart AG ZIP** → partial (`platform.zip`, 11,481 rows — AG early-2024 files are empty placeholders in the ZIP)
4. **Flipkart Akiko ZIP** → ingest running (large archive; ~47k rows expected)
5. **Tier-3 sync** triggered to refresh warm cache

### Production after partial ingest

| Platform | Tier-3 rows | File count |
|----------|-------------|------------|
| Myntra | **370,851** | 136 |
| Flipkart | **94,480** | 147 |

Myntra row count increased significantly (bulk archives merged with existing daily files). Flipkart will increase further once Akiko ingest completes.

---

## Parse notes / data quality

### Myntra
- PPMP/SJIT: ~16k–21k duplicate rows dropped per ZIP during parse (expected — RT/RTO/Sale overlap in monthly exports).
- SJIT: `Sale Jan-2024.csv` failed with timezone parse error (non-fatal; RT files cover Jan 2024).

### Flipkart AG
- **Jan–Jun 2024** xlsx files in the AG ZIP are **empty templates** (~49 KB each, no Sales Report rows). Real 2024 H1 data is in the **Akiko (PE)** ZIP.
- AG contributes from **July 2024** onward.

### Flipkart Akiko
- Full 24 monthly Sales Report xlsx files; this is the authoritative source for **2024** Flipkart history.

---

## Post-upload verification (2026-07-01)

After ingesting all four ZIPs and syncing Tier-3:

| Platform | Tier-3 rows | Min date | Max date |
|----------|-------------|----------|----------|
| Myntra | **370,851** | 2023-12-07 | 2026-06-30 |
| Flipkart | **141,611** | **2023-12-02** | 2026-06-28 |

**Shipment units — spot check vs your ZIPs:**

| Month | Myntra ZIP | Myntra app | Flipkart ZIP | Flipkart app |
|-------|------------|------------|--------------|--------------|
| 2024-01 | 3,658 | **3,575** | 1,055 | **1,055** ✓ |
| 2024-06 | 4,223 | **4,210** | 2,452 | **2,452** ✓ |

Flipkart 2024 history is now aligned. Myntra 2024 months match within ~2% (dedup across PPMP + daily uploads).

Coverage API: `myntra_rows=370,851` · `flipkart_rows=141,611`

---

## Recommended next steps

1. **Hard refresh** Intelligence / Myntra / Flipkart pages (sync completed).
2. **Send remaining platforms** (Amazon, Meesho, Snapdeal, etc.) in the same ZIP format for the same audit.
3. **Optional cleanup:** Remove duplicate Tier-3 entries named `platform.zip` (replaced by properly named archive files).

---

## Files in repo

- `reports/platform_zip_compare_report.md` — detailed month-by-month comparison (pre-upload baseline)
- `scripts/compare_platform_zips.py` — re-run audit locally
- `scripts/upload_platform_zips_production.py` — HTTP upload helper
- `backend/scripts/ingest_platform_zip_disk.py` — server-side Tier-3 ingest

## Code fix included

- `backend/services/flipkart.py`: `@contextmanager` on `_open_flipkart_zip` (fixes ZIP parsing from bytes).

---

*Report generated after automated parse + Tier-3 comparison. Re-run `python3 scripts/compare_platform_zips.py` after Akiko ingest completes for post-upload verification.*
