# Material Issue Notes (MIN) control + Fabric Warehouse — Team Notes

## 1. Cutting / Production MIN (Job Order Material Issue Note, `IN-####`)

Where: **Production → open the Job Order → Material Issue Note panel** (Cutting, Stitching, Finishing).

- A Draft MIN is created automatically from the BOM when the JO is created. **BOM Required** is frozen at that time.
- **Edit Issue Qty**: change the qty per material line while the note is Draft. The screen shows Diff and Variance % against BOM.
- **Reason is mandatory** whenever Issue Qty differs from BOM. Choices: Mini CAD Average Change, Wastage, Roll/Bundle Adjustment, Rework, Stock Shortage, Other. **Other requires remarks.**
- **Tolerance: ±10% of BOM** (server setting `MIN_ISSUE_TOLERANCE_PCT`, default 10).
  - Within tolerance: the issuer can Confirm directly.
  - Over tolerance (above or below): note becomes **Pending Approval**. Only **Manager / Admin / Super Admin / Sir** can **Approve**, and only then can it be Confirmed.
- **Confirm Issue is the stock event.** Stock is deducted at the **actual Issue Qty** (not BOM) from Item Master and Checked printed-fabric stock. Nothing moves while the note is Draft or Pending Approval. Negative stock is blocked.
- **Print / Download**: print view shows JO, SO, SKU, materials, BOM / Issue / Diff / %, dates, issued-by, received-by, remarks.
- **Additional Issue**: if more material is needed later against the same JO, create an additional Draft MIN (shows BOM vs cumulative issued). It follows the same tolerance and approval rules.
- **Post-confirm Correct**: a Confirmed MIN is read-only. Manager / Admin can post a correction (delta qty + reason). Stock is adjusted by the delta and an audit entry is written.
- Cancel is available on unconfirmed notes.

## 2. Purchase MIN (Job Work Order MIN)

Where: **Purchase → MIN tab**.

- Same tolerance (±10%, `MIN_ISSUE_TOLERANCE_PCT`) and the same Manager approval rule when over tolerance. The tolerance % is shown in the adjust screen when the limit is exceeded.
- Existing behaviour is unchanged: Draft → Confirm posts Item Master stock + grey ledger; revise/correct and print stay as before; existing Purchase reason list is kept.

## 3. Fabric Warehouse — `/fabric-warehouse`

Where: left menu **Fabric Warehouse** (next to Grey; same access as the Grey module).

- **KPI strip**: Opening, Today IN, Today OUT, Closing, SO Allocated, Advance Reserved, Free, Returns/Adjustments, Pending Unchecked. Filters: date range, warehouse (Checked / Unchecked), P-Code, type, txn type. Click a KPI to filter the ledger.
- **Stock by P-Code**: physical, checked, unchecked, SO allocated, advance reserved, free, location.
- **Free vs Reserved vs physical**:
  - **Physical** = what is actually on the shelf (checked + unchecked).
  - **Reserved** (SO Allocated / Advance Reserved) = a hold on checked stock. It is **not** an OUT and never reduces physical stock.
  - **Free** = Checked available − SO allocated − Advance reserved. Free is what can still be reserved or issued.
  - Physical OUT happens only on **Confirmed MIN** (Purchase or Production) or a confirmed JO fabric issue.
- **Ledger**: date-wise register (DateTime, P-Code, Type, Ref, From, To, IN, OUT, Running Bal, SO/JO, User, Remarks). It is read-only and built from existing records (GRN, QC approve, opening, MIN, JO issue/return, adjustments, reservations). It does not keep its own stock.
- **Export**: Excel (xlsx) or CSV from the ledger toolbar, using the current filters. Print view is available.
- Click a Ref to jump to the GRN / MIN / JO / Grey / Item Master record.

## 4. Not in Phase 3 yet (do not rely on these)

- Full reconciliation report (Opening + IN − OUT ± Adj = Closing vs Item Master vs Checked stock). The dashboard shows live figures but does not reconcile them.
- BOM vs Issued vs Returned variance reports.
- Blocking or auditing backdated ledger edits; full audit UI.
- Dual confirmation on returns (receiving warehouse accepting before IN).
- Deprecating free-form "Issue Fabric". It still works when no open MIN covers the material; prefer MIN Confirm.
- Ledger reference links are plain deep links to the owning module. They are not yet document-level pop-ups, and not every link opens the exact record.
- Changing BOM master data. Mini CAD average is only a reason on the MIN.
- Replacing the Grey Fabric QC / Reserve screens.
