/** Shared PO table columns, labels, and CSV export (used by PO Fresh + legacy PO Engine). */

export const PO_DISPLAY_COLS = [
  'Priority',
  'OMS_SKU',
  'Bundle_Size',
  'SKU_Sheet_Status',
  'Lead_Time_Days',
  'Total_Inventory',
  'Days_Left',
  'Sold_Units',
  'Return_Units',
  'Return_Overlay_Units',
  'Net_Units',
  'Ship_Units_150d',
  'Eff_Days',
  'Eff_Days_Inventory',
  'Recent_ADS',
  'LY_ADS',
  'Seasonal_Month_ADS',
  'Flat30_ADS',
  'ADS',
  'Cutting_Ratio',
  'Gross_PO_Qty',
  'PO_Qty_Ordered',
  'Pending_Cutting',
  'Balance_to_Dispatch',
  'Finishing_Balance',
  'Finishing_Received',
  'Finishing_Issue_No',
  'Finishing_Iss_Date',
  'PO_Pipeline_Total',
  'PO_Raised_On_View_Date',
  'PO_Last_Raised_Qty',
  'PO_Last_Raised_Date',
  'PO_Raised_Yesterday',
  'PO_Raised_Today',
  'PO_Confirmed_Raise_Pipeline',
  'PO_Pipeline_Effective',
  'Projected_Running_Days',
  'Post_PO_Cover_Days_Capped',
  'PO_Qty',
  'PO_Block_Reason',
  'Suggest_Close_SKU',
] as const

export const COL_LABEL: Record<string, string> = {
  Bundle_Size: '📦 Bundle size',
  Sold_Units: '📦 Sold Units',
  Return_Units: '↩️ Returns (sales)',
  Return_Overlay_Units: '↩️ Returns (upload)',
  Net_Units: '📦 Net sold',
  Eff_Days: '📅 Eff. Days (active)',
  Eff_Days_Inventory: '📦 In-stock days (history)',
  Recent_ADS: '📉 Recent ADS',
  LY_ADS: '📈 LY ADS',
  Seasonal_Month_ADS: '🌗 Season ADS (mo+1)',
  Flat30_ADS: '📅 Sheet FREQ (÷30)',
  ADS: '⚡ ADS (Used)',
  PO_Pipeline_Total: '🏭 Total Pipeline',
  PO_Raised_On_View_Date: '📌 Raised on raise date',
  PO_Last_Raised_Qty: '📌 Last raised qty',
  PO_Last_Raised_Date: '📅 Last raised date',
  PO_Raised_Yesterday: '📌 Raised qty (yesterday)',
  PO_Raised_Today: '📌 Raised qty (today)',
  PO_Confirmed_Raise_Pipeline: '📌 Confirmed raises (window)',
  PO_Pipeline_Effective: '🏭 Eff. pipeline (sheet + raises)',
  PO_Qty_Ordered: '📋 PO Ordered (on sheet)',
  Pending_Cutting: '✂️ Pend. Cutting',
  Balance_to_Dispatch: '📦 Bal. Dispatch',
  Finishing_Balance: '✨ Finishing left',
  Finishing_Received: '✅ Finishing received',
  Finishing_Issue_No: '🔖 Issue No',
  Finishing_Iss_Date: '📅 Issue date',
  Projected_Running_Days: '📅 Proj. run (Tot inv + pipe) / ADS',
  Post_PO_Cover_Days_Capped: '📏 Post-PO cover (actual)',
  Cutting_Ratio: '✂️ Cut Ratio',
  Lead_Time_Days: '📅 Lead (d)',
  SKU_Sheet_Status: '📋 Sheet status',
  Ship_Units_150d: '📉 Ship ~5mo',
  PO_Block_Reason: '🚫 Block reason',
  Suggest_Close_SKU: '💡 Close hint',
}

export const COL_HELP: Record<string, string> = {
  Priority:
    'Operational priority bucket used for sorting. Derived from cover, lead-time risk, stockout flags, and sheet status rules.',
  OMS_SKU:
    'The OMS SKU identifier after canonicalization/mapping (used as the join key across inventory, sales, pipelines, and ledgers).',
  Total_Inventory:
    'Current inventory units from the inventory upload (includes marketplace stock when present). Used for cover metrics; the PO formula prefers OMS_Inventory when available.',
  Sold_Units:
    'Total shipped units in the ADS window (period_days). Computed from shipment rows only.',
  Return_Units:
    'Returns inside the ADS window from platform data. Used to compute Net_Units and ADS_Net_Units.',
  Return_Overlay_Units:
    'Extra return deduction uploaded by the operator (overlay). Net PO is reduced by this amount, then pack-rounded.',
  Net_Units:
    'Net shipped units in the ADS window (period_days): Sold_Units − Return_Units − Return_Overlay_Units (floored at 0).',
  Eff_Days:
    'Effective demand days used as ADS denominator in sparse/intermittent cases. Uses active-span (first→last demand day) and may collapse to distinct sale-days when sales are sparse, so ADS is not diluted by long quiet stretches.',
  Recent_ADS:
    'Recent average daily sales in the ADS window: Sold_Units (or Net_Units) ÷ Eff_Days.',
  LY_ADS:
    'Last-year same-window ADS: (LY demand units) ÷ period_days. Lifts ADS when recent is quiet (if LY fallback is on).',
  Seasonal_Month_ADS:
    'Seasonal uplift floor: same calendar month + next month in prior years.',
  Flat30_ADS:
    'Sheet-style FREQ floor: max(rolling 30-day rate, month-to-date rate) ÷ 30.',
  ADS:
    'Final ADS used by the engine: max(primary signal, Seasonal_Month_ADS, Flat30_ADS). Primary = max(Recent_ADS, LY_ADS) with non-sparse burst cap at sold÷period_days.',
  Days_Left:
    'Inventory cover in days before new PO: Days_Left = Inventory / ADS (999 when ADS=0).',
  PO_Pipeline_Total:
    'Total open pipeline units from the uploaded Existing PO sheet (sum of PO / cutting / dispatch / finishing columns, normalized).',
  PO_Confirmed_Raise_Pipeline:
    'Confirmed raises inside the configured lookback window (from the PO raise ledger). Used to bridge gaps when the existing PO sheet is not yet updated.',
  PO_Pipeline_Effective:
    'Effective pipeline used for cover + PO math: max(PO_Pipeline_Total, PO_Confirmed_Raise_Pipeline). This avoids double-counting when the sheet already includes the same orders.',
  Projected_Running_Days:
    'Projected cover before new PO: (Inventory + PO_Pipeline_Effective) / ADS (999 when ADS=0).',
  Gross_PO_Qty:
    'Gross PO before overlays/gates: when lead-gate is OFF (current mode), Gross_PO_Qty = pack_round(max(ADS * (target_cover_days − Projected_Running_Days), 0)).',
  PO_Qty:
    'New PO Qty (net recommendation): starts from Gross_PO_Qty, then subtracts Return_Overlay_Units (if present), floors at 0, and pack-rounds again. Existing-PO pipeline + confirmed raises reduce Projected_Running_Days, which reduces PO need.',
  Post_PO_Cover_Days_Capped:
    'Post-PO cover after adding new PO: (Inventory + PO_Pipeline_Effective + PO_Qty) / ADS (999 when ADS=0).',
  Lead_Time_Days:
    'Factory lead time in days from the status/lead sheet when available; otherwise defaults to the Lead time parameter.',
  PO_Block_Reason:
    'Why New PO Qty was blocked to 0 (e.g., sheet status excluded/closed, lead-time gate rules, or two-size minimum).',
  Cutting_Ratio:
    'Cutting Planner ratio used to split a parent style’s PO across sizes. Uses PO_Qty share when present; otherwise ADS share when 2+ sizes have gross need.',
}

export const PRIORITY_ORDER: Record<string, number> = {
  '🔴 URGENT': 0,
  '🟡 HIGH': 1,
  '🟢 MEDIUM': 2,
  '🔄 In Pipeline': 3,
  '⚪ OK': 4,
}

export const QUARTER_COL_RE = /^(Apr[-–]Jun|Jul[-–]Sep|Oct[-–]Dec|Jan[-–]Mar)\s+\d{4}$/i
export const EXPECTED_QUARTER_COLS = 8

export function poColHeaderLabel(col: string): string {
  if (col === 'PO_Qty') return 'New PO Qty'
  const lbl = COL_LABEL[col]
  return lbl ?? col.replace(/_/g, ' ')
}

export function poColHelpText(col: string): string | undefined {
  if (COL_HELP[col]) return COL_HELP[col]
  if (col.includes('ADS')) return COL_HELP.ADS
  if (col.includes('Pipeline')) return COL_HELP.PO_Pipeline_Effective
  if (col.includes('Cover') || col.includes('Days')) return undefined
  return undefined
}

export function poSkuKey(sku: string | undefined | null): string {
  return String(sku ?? '').trim().toUpperCase()
}

export function countQuarterColumns(columns: string[] | undefined): number {
  return (columns ?? []).filter(c => QUARTER_COL_RE.test(c.trim())).length
}

export function quarterColumnsFromApi(columns: string[] | undefined): string[] {
  return [...(columns ?? []).filter(c => QUARTER_COL_RE.test(c.trim()))].reverse()
}

export function formatPoCell(col: string, v: unknown): string {
  if (v == null || v === '') return '—'
  if (typeof v === 'number') {
    if (col.includes('Days') || col.includes('ADS') || col === 'Cutting_Ratio') {
      return Number(v).toFixed(col === 'Cutting_Ratio' ? 4 : 1)
    }
    return Number.isInteger(v) ? String(v) : Number(v).toFixed(2)
  }
  return String(v)
}

export function poCellClass(col: string, v: unknown): string {
  if (col === 'PO_Qty' && Number(v) > 0) return 'font-semibold text-orange-700'
  if (col === 'Priority' && String(v).includes('URGENT')) return 'text-red-700 font-semibold'
  return ''
}

export function downloadCsvBlob(csv: string, filename: string) {
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
}

export function finalPoQtyForSku(
  row: Record<string, unknown>,
  editedQty: Record<string, number> | undefined,
): number {
  const sku = String(row.OMS_SKU ?? '')
  if (editedQty && editedQty[sku] !== undefined) return editedQty[sku]
  const n = Number(row.PO_Qty ?? 0)
  return Number.isFinite(n) ? n : 0
}

export function exportPoCsv(
  rows: Record<string, unknown>[],
  poCols: string[],
  quarterCols: string[],
  quarterMap: Record<string, Record<string, number | string>>,
  filename: string,
  editedQty?: Record<string, number>,
) {
  const all = [...poCols, ...quarterCols]
  const header = all.join(',')
  const body = rows
    .map(r => {
      const sku = poSkuKey(String(r.OMS_SKU ?? ''))
      const qRow = quarterMap[sku] ?? {}
      return all
        .map(c => {
          if (c === 'PO_Qty') return JSON.stringify(finalPoQtyForSku(r, editedQty))
          if (quarterCols.includes(c)) return JSON.stringify(qRow[c] ?? 0)
          return JSON.stringify(r[c] ?? '')
        })
        .join(',')
    })
    .join('\n')
  downloadCsvBlob(header + '\n' + body, filename)
}

export function visiblePoColumns(apiColumns: string[] | undefined, includeQuarterly: string[]): string[] {
  const fromApi = apiColumns ?? []
  const po = PO_DISPLAY_COLS.filter(c => fromApi.length === 0 || fromApi.includes(c))
  return [...po, ...includeQuarterly]
}
