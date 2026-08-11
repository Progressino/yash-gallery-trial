# Inventory Aug 11, 2026 — recon notes (local re-parse)

## Uploaded RAR contents
- `OMS 11-Aug-26.csv` (primary warehouse)
- `1609-combo_skus.csv`
- Amazon ledgers (2 seller files)
- Flipkart Current Inventory (3)
- Myntra Seller Inventory Report (2)

## File-level OMS
| Metric | Value |
|--------|------:|
| Raw OMS `Inventory` sum | 196,053 |
| Combo `Combo Qty Stock` sum | 5,249 |
| OMS + combo | 201,302 |
| App after alias/coalesce `OMS_Inventory` | **201,315** |

App OMS matches the stated UI/OMS figure (**201,315**).

## Consolidated (load_inventory_consolidated)
| Column | Total |
|--------|------:|
| OMS_Inventory | 201,315 |
| Amazon_Inventory (FBA sellable, non-ZNNE/TWWR) | 49,511 |
| Flipkart_Inventory | 488 |
| Myntra_Other_Inventory | 3,154 |
| Marketplace_Total | 53,153 |
| **Total_Inventory** (OMS + marketplaces) | **254,468** |
| Buffer_Stock (informational only) | 26,941 |

## Business rules
- **Inventory Daily Total Units** = OMS + all marketplace columns (sum). Not max.
- **Inventory History Combined channel** = per SKU-day **max(OMS, Amazon)**. Not the Daily Total.
- **OMS file** “Inventory” ≠ FBA. SKUs can show OMS=0 with Amazon>0.

## Example SKUs
| SKU | OMS file | App OMS | App Amazon | App Total |
|-----|---------:|--------:|-----------:|----------:|
| AK-228BLACK-3XL | 0 | 0 | 15 | 15 |
| AK-228BLACK-5XL | 0 | 0 | 11 | 11 |

If the UI showed **20** / **3**, that was not OMS warehouse for this re-parse — check Amazon_Inventory / Total_Inventory columns or a stale snapshot.

## UI vs export 201,315 vs 202,803
Likely causes addressed:
1. **Stale `inventory_api_totals`** refreshed only when column *keys* changed, so cards could show an old OMS total while row export summed a newer frame (or the reverse).
2. **Export summed mixed columns** (client XLSX had no official TOTALS row; summing all columns double-counts Total_Inventory).
3. Metric confusion (History Combined “Total inv.” vs Daily OMS).

Fixes: live totals validation on GET; server `/data/inventory/export.csv` with `__TOTALS__` row from the same totals function; clearer UI labels.

## Inventory History 522
Cloudflare/origin timeout on long synchronous `matrix.csv`. Fix: async job POST + poll + download.
