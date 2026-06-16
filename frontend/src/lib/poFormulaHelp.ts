/** PO column / parameter formula copy — mirrors backend ``po_engine.py``. */

export interface POFormulaContext {
  periodDays: number
  leadTime: number
  targetDays: number
  graceDays: number
  demandBasis: string
  useSeasonality: boolean
  seasonalWeight: number
  enforceLeadGate: boolean
  safetyPct: number
  finalPoQty?: number
  raiseViewDate?: string
}

export interface POFormulaDef {
  title: string
  summary: string
  formula: string
  steps?: string[]
  sources?: string[]
}

export interface PORowBreakdownLine {
  label: string
  expression?: string
  value: string
  highlight?: boolean
}

const fmt = (n: number, dp = 3) =>
  Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: dp }) : '—'

const num = (row: Record<string, unknown>, key: string) => Number(row[key] ?? 0)

export const PO_COLUMN_FORMULAS: Record<string, POFormulaDef> = {
  Priority: {
    title: 'Priority',
    summary: 'Urgency band for this SKU row.',
    formula: 'Based on Days_Left vs lead time and whether PO / pipeline > 0',
    steps: [
      '🔴 URGENT — PO > 0 and inventory cover (Days_Left) is below factory lead time',
      '🟡 HIGH — PO > 0 and Days_Left is below lead + grace days',
      '🟢 MEDIUM — PO > 0 but cover is still comfortable',
      '🔄 In Pipeline — no new PO, but effective pipeline > 0',
      '⚪ OK — no PO needed and no active pipeline',
    ],
  },
  OMS_SKU: {
    title: 'OMS SKU',
    summary: 'Canonical inventory / sales identifier after SKU mapping.',
    formula: 'Normalized token from sales + inventory uploads',
    steps: [
      'Strips marketplace suffixes and duplicate size tokens',
      'SKU mapping can collapse listing variants to one OMS key',
    ],
  },
  Bundle_Size: {
    title: 'Bundle size',
    summary: 'Pack / bundle multiplier from SKU mapping when the listing sells multi-size packs.',
    formula: 'From SKU mapping sheet (default 1)',
    steps: ['Used when unbundling pipeline from bundled PO rows'],
  },
  SKU_Sheet_Status: {
    title: 'Sheet status',
    summary: 'Status from the SKU status / lead upload.',
    formula: 'Direct match on SKU status sheet',
    steps: [
      'Closed / Doubt / sales-after-closed rows block new PO',
      'Open / Active rows can receive PO when other gates pass',
    ],
    sources: ['Upload → History & setup → SKU status / lead'],
  },
  Lead_Time_Days: {
    title: 'Lead time (days)',
    summary: 'Factory lead time used for urgency and planning context.',
    formula: 'Per-SKU lead from status sheet, else global Lead Time parameter',
    steps: [
      'Resolved per SKU from the status / lead sheet when uploaded',
      'Parent-level lead rolls down to size variants',
      'Falls back to the global Lead Time parameter when no sheet row matches',
    ],
    sources: ['SKU status / lead sheet', 'Parameters → Lead Time (days)'],
  },
  Total_Inventory: {
    title: 'Total inventory',
    summary: 'On-hand units across warehouse + marketplace channels.',
    formula: 'Sum from latest inventory snapshot',
    steps: [
      'Includes OMS warehouse, FBA, and other channel columns when present',
      'Used for Days_Left and projected cover display',
    ],
    sources: ['Inventory upload'],
  },
  Days_Left: {
    title: 'Days left (inventory only)',
    summary: 'How many days current stock lasts at ADS — pipeline is not included.',
    formula: 'Days_Left = Total_Inventory ÷ ADS',
    steps: ['Rounded to 1 decimal', 'Shows 999 when ADS = 0'],
  },
  Sold_Units: {
    title: 'Sold units',
    summary: 'Shipment quantity in the ADS window (Period days).',
    formula: `SUM(shipments) over last Period days`,
    steps: ['Uses platform sales merged for PO (Amazon, Flipkart, Meesho, Myntra, Snapdeal)'],
  },
  Return_Units: {
    title: 'Returns (sales)',
    summary: 'Return / refund units detected in the sales feed for the ADS window.',
    formula: 'SUM(return transactions) in ADS window',
    steps: ['Informational — Net demand uses Units_Effective when Demand basis = Net'],
  },
  Return_Overlay_Units: {
    title: 'Returns (upload)',
    summary: 'Extra return units from an optional returns import overlay.',
    formula: 'From returns CSV upload',
    steps: ['Subtracted from net demand when overlay is loaded'],
  },
  Net_Units: {
    title: 'Net sold',
    summary: 'Net demand units in the ADS window.',
    formula: 'Net_Units = effective demand after returns (and overlay when loaded)',
    steps: ['When Demand basis = Net, this drives ADS instead of Sold_Units'],
  },
  Ship_Units_150d: {
    title: 'Ship ~5mo',
    summary: 'Shipment units in the last ~150 calendar days.',
    formula: 'SUM(shipments) over 150-day lookback',
    steps: [
      'Used as context for very low-volume SKUs',
      'Can extend Eff_Days span when ADS-window sales are sparse',
    ],
  },
  Eff_Days: {
    title: 'Effective days (active)',
    summary: 'Denominator for Recent ADS — active selling span, not full calendar period.',
    formula: 'last sale day − first sale day + 1 (in ADS window), or distinct sale days when intermittent',
    steps: [
      'Counts only days with qualifying transactions in the Period window',
      'Sparse intermittent sellers (low units / long gap) use distinct sale days',
      'Very low volume (≤5 units) keeps calendar span to avoid ADS spikes',
      'Daily inventory history can shorten this to in-stock days only',
      'Click a cell value to open the day-by-day inventory drill-down',
    ],
  },
  Eff_Days_Inventory: {
    title: 'In-stock days (history)',
    summary: 'Days with on-hand qty ≥ 1 in the daily inventory matrix.',
    formula: 'COUNT(days where snapshot qty ≥ 1) in ADS window',
    steps: [
      'From wide daily inventory history upload',
      'Engine may use min(active span, in-stock days) when history is available',
      'Click a cell value to see per-day snapshots',
    ],
    sources: ['Upload → Daily inventory history'],
  },
  Recent_ADS: {
    title: 'Recent ADS',
    summary: 'Demand rate from the current Period window using effective days.',
    formula: 'Recent_ADS = demand_units ÷ Eff_Days',
    steps: [
      'demand_units = Sold_Units when Demand basis = Sold, else Net_Units',
      'OOS-restock SKUs may get an imputed recent rate from parent / siblings',
    ],
  },
  LY_ADS: {
    title: 'LY ADS',
    summary: 'Same calendar window one year ago, spread over Period days.',
    formula: 'LY_ADS = LY_demand_units ÷ Period_days',
    steps: ['Example: run date Apr 3 → LY window is Apr 3 last year for Period days'],
  },
  Seasonal_Month_ADS: {
    title: 'Season ADS (month+1)',
    summary: 'Historical ADS for this calendar month and the next month (prior years).',
    formula: 'AVG(monthly rate) from same month pair in lookback years',
    steps: ['Floors final ADS when seasonality is enabled'],
  },
  Flat30_ADS: {
    title: 'Sheet FREQ (÷30)',
    summary: 'Spreadsheet-style flat rate: max(rolling 30d, month-to-date) ÷ 30.',
    formula: 'Flat30_ADS = max(last_30d_units, MTD_units) ÷ 30',
    steps: ['Matches operator Req.xlsx FREQ column', 'Floors final ADS'],
  },
  ADS: {
    title: 'ADS (used)',
    summary: 'Final average daily sales used for all cover and PO math.',
    formula: 'ADS = max(blended_recent, Seasonal_Month_ADS, Flat30_ADS)',
    steps: [
      'Blended recent = Recent_ADS, or weighted with LY_ADS when YoY seasonality is on',
      'Without seasonality: max(Recent_ADS, LY_ADS) before seasonal / flat floors',
      'Non-sparse sellers cap inflated short-span Recent_ADS at sold÷Period',
    ],
  },
  Cutting_Ratio: {
    title: 'Cut ratio',
    summary: 'Share of parent PO qty allocated to this size in the Cutting Planner.',
    formula: 'PO_Qty ÷ parent_PO_sum  OR  ADS ÷ parent_ADS_sum when spreading gross need',
    steps: [
      'When parent has net PO, split by each size’s PO_Qty share',
      'When only gross need across ≥2 sizes, split by ADS (or gross share)',
      'Single-size parents keep 100% on that size',
    ],
  },
  Gross_PO_Qty: {
    title: 'Gross PO qty',
    summary: 'Recommended order before return overlay adjustments.',
    formula: 'See PO_Qty — Gross matches PO before return overlay in most cases',
    steps: ['Rounded up to pack size (5 or 10)'],
  },
  PO_Qty_Ordered: {
    title: 'PO ordered (on sheet)',
    summary: 'Units already on the existing PO upload (ordered column).',
    formula: 'From existing PO sheet',
    steps: ['Part of pipeline breakdown'],
    sources: ['Existing PO upload'],
  },
  Pending_Cutting: {
    title: 'Pending cutting',
    summary: 'Units in cutting stage on the existing PO sheet.',
    formula: 'From existing PO sheet',
    sources: ['Existing PO upload'],
  },
  Balance_to_Dispatch: {
    title: 'Balance to dispatch',
    summary: 'Finished goods waiting dispatch on the existing PO sheet.',
    formula: 'From existing PO sheet',
    sources: ['Existing PO upload'],
  },
  Finishing_Balance: {
    title: 'Finishing left',
    summary: 'Units still in finishing from the existing PO sheet.',
    formula: 'From existing PO sheet finishing columns',
    sources: ['Existing PO upload'],
  },
  Finishing_Received: {
    title: 'Finishing received',
    summary: 'Units received into finishing.',
    formula: 'From existing PO sheet',
    sources: ['Existing PO upload'],
  },
  Finishing_Issue_No: {
    title: 'Finishing issue no',
    summary: 'Reference issue number from finishing tracker on PO sheet.',
    formula: 'Text field from existing PO upload',
  },
  Finishing_Iss_Date: {
    title: 'Finishing issue date',
    summary: 'Issue date from finishing tracker on PO sheet.',
    formula: 'Date field from existing PO upload',
  },
  PO_Pipeline_Total: {
    title: 'Total pipeline',
    summary: 'All in-production units from the uploaded existing PO sheet.',
    formula: 'PO_Qty_Ordered + Pending_Cutting + Balance_to_Dispatch (+ finishing when present)',
    steps: [
      'Bundled vs per-size rows are merged so pipeline is not double-counted',
      'Per-size pipeline rolls up to bundled inventory rows when needed',
    ],
    sources: ['Existing PO upload'],
  },
  PO_Raised_On_View_Date: {
    title: 'Raised on view date',
    summary: 'Qty confirmed in the raise ledger on the selected raise date.',
    formula: 'SUM(ledger qty) for SKU on raise date',
    sources: ['Raise ledger'],
  },
  PO_Last_Raised_Qty: {
    title: 'Last raised qty',
    summary: 'Most recent confirmed raise quantity for this SKU.',
    formula: 'Latest ledger entry',
    sources: ['Raise ledger'],
  },
  PO_Last_Raised_Date: {
    title: 'Last raised date',
    summary: 'Date of the most recent confirmed raise.',
    formula: 'Latest ledger entry date',
    sources: ['Raise ledger'],
  },
  PO_Raised_Yesterday: {
    title: 'Raised yesterday',
    summary: 'Qty confirmed in the ledger yesterday (IST).',
    formula: 'SUM(ledger qty) for yesterday',
    sources: ['Raise ledger'],
  },
  PO_Raised_Today: {
    title: 'Raised today',
    summary: 'Qty confirmed in the ledger today (IST).',
    formula: 'SUM(ledger qty) for today',
    sources: ['Raise ledger'],
  },
  PO_Confirmed_Raise_Pipeline: {
    title: 'Confirmed raises (window)',
    summary: 'Raises confirmed in-app that are not yet on the uploaded PO sheet.',
    formula: 'SUM(recent ledger raises)',
    steps: ['Bridged into effective pipeline to avoid double-releasing the same units'],
    sources: ['Raise ledger'],
  },
  PO_Pipeline_Effective: {
    title: 'Effective pipeline',
    summary: 'Pipeline used in cover and PO formulas.',
    formula: 'PO_Pipeline_Effective = max(PO_Pipeline_Total, PO_Confirmed_Raise_Pipeline)',
    steps: ['Avoids double-counting once the sheet already reflects an order'],
  },
  Projected_Running_Days: {
    title: 'Projected running days',
    summary: 'Stock cover before any new PO — inventory plus effective pipeline.',
    formula: 'Projected_Running_Days = (Total_Inventory + PO_Pipeline_Effective) ÷ ADS',
    steps: ['Rounded to 1 decimal', 'Compared against lead time when the gate is on'],
  },
  Post_PO_Cover_Days_Capped: {
    title: 'Post-PO cover',
    summary: 'Expected cover after the new PO quantity (uses your edited qty when changed).',
    formula: 'Post_PO_Cover = (Total_Inventory + PO_Pipeline_Effective + PO_Qty) ÷ ADS',
    steps: ['Updates live when you edit New PO Qty'],
  },
  PO_Qty: {
    title: 'New PO qty',
    summary: 'Net units to raise today — editable before export.',
    formula: 'Balance toward post-PO target cover',
    steps: [
      'PO = pack_round(ADS × (target_days + grace − Projected_Running_Days))',
      'May be zeroed by status sheet, two-size rule, or block reasons',
      'Return overlay can reduce final PO_Qty',
    ],
  },
  PO_Block_Reason: {
    title: 'Block reason',
    summary: 'Why PO was blocked or reduced after all rules.',
    formula: 'Concatenated rule messages',
    steps: [
      'Closed / missing lead on status sheet',
      'Lead-time gate (cover already ≥ lead)',
      'Two-size minimum rule',
      'Single-size parent recommendation',
    ],
  },
  Suggest_Close_SKU: {
    title: 'Close hint',
    summary: 'Suggestion to close or alter SKU based on slow / dead demand patterns.',
    formula: 'Heuristic from ADS, sales history, and status sheet',
    steps: ['Informational — does not change PO_Qty by itself'],
  },
}

export const PO_PARAM_FORMULAS: Record<string, POFormulaDef> = {
  period_days: {
    title: 'Period (days)',
    summary: 'Length of the ADS / demand window.',
    formula: 'ADS window = last Period days of sales',
    steps: ['Sold_Units, Net_Units, Eff_Days, and LY window all use this length'],
  },
  lead_time: {
    title: 'Lead time (days)',
    summary: 'Default factory lead when the status sheet has no per-SKU lead.',
    formula: 'Fallback Lead_Time_Days',
    steps: ['Per-SKU sheet lead overrides this when present'],
  },
  target_days: {
    title: 'Post-PO running days',
    summary: 'Target stock cover after PO.',
    formula: 'target_cover = target_days + grace_days',
    steps: ['PO fills toward this post-PO horizon'],
  },
  demand_basis: {
    title: 'Demand basis',
    summary: 'Whether ADS uses gross shipments or net of returns.',
    formula: 'Sold → shipment qty | Net → Units_Effective',
    steps: ['Affects Sold_Units vs Net_Units and all ADS variants'],
  },
  grace_days: {
    title: 'Grace days',
    summary: 'Extra urgency buffer added to lead time for priority bands and target cover.',
    formula: 'HIGH priority when Days_Left < Lead_Time + grace_days',
    steps: ['Also added to target_days when lead gate is off'],
  },
  safety_pct: {
    title: 'Safety stock %',
    summary: 'Reserved for future safety-stock uplift (currently 0% in engine).',
    formula: 'Not applied in current PO release formula',
    steps: ['Displayed for operator tuning — check release notes when enabled'],
  },
  urgent_all_sizes_days: {
    title: 'All sizes below (days)',
    summary: 'When any size of a parent runs below this projected cover, ghost sibling rows appear.',
    formula: 'If any size Projected_Running_Days < threshold → show all sizes',
    steps: ['Helps raise PO for every size in a style even when some sizes lack sales'],
  },
  use_seasonality: {
    title: 'YoY seasonality',
    summary: 'Blend Recent ADS with last-year same-window ADS.',
    formula: 'blend = Recent × (1 − weight) + LY × weight',
    steps: ['Still floored by Seasonal_Month_ADS and Flat30_ADS'],
  },
  seasonal_weight: {
    title: 'Seasonal weight',
    summary: 'How much last-year ADS influences the blend when seasonality is on.',
    formula: 'weight ∈ [0, 1]',
  },
  group_by_parent: {
    title: 'Group by parent SKU',
    summary: 'Roll size variants up to parent-level demand and pipeline keys.',
    formula: 'Parent_SKU = style token without size suffix',
    steps: ['Changes table grouping and some ADS / Eff_Days rollups'],
  },
  enforce_two_size_minimum: {
    title: 'Require ≥2 sizes to place PO',
    summary: 'Blocks PO when only one size in a parent has demand.',
    formula: 'If only 1 size has Gross_PO_Qty > 0 → PO_Qty = 0',
    steps: ['Surfaces recommendation to alter SKU sizing mix'],
  },
  enforce_lead_time_release_gate: {
    title: 'Lead-time gate (disabled)',
    summary: 'Deprecated in app-only PO mode.',
    formula: 'Not used in current release',
    steps: ['PO calculations use target-cover formula from app data only.'],
  },
}

const QUARTER_COL_RE = /^(Apr[-–]Jun|Jul[-–]Sep|Oct[-–]Dec|Jan[-–]Mar)\s+\d{4}$/i

const QUARTERLY_FORMULA: POFormulaDef = {
  title: 'Quarterly history',
  summary: 'Historical shipment units by calendar quarter.',
  formula: 'SUM(shipments) per quarter from platform sales history',
  steps: ['Avg/Mo = average monthly units across loaded quarters'],
}

export function getPOFormulaDef(key: string): POFormulaDef | null {
  if (PO_COLUMN_FORMULAS[key]) return PO_COLUMN_FORMULAS[key]
  if (PO_PARAM_FORMULAS[key]) return PO_PARAM_FORMULAS[key]
  if (key === 'Avg_Monthly' || key === 'Avg/Mo' || QUARTER_COL_RE.test(key)) return QUARTERLY_FORMULA
  if (key.startsWith('──')) return null
  return null
}

export function poColumnHasFormula(col: string): boolean {
  return getPOFormulaDef(col) != null
}

function packRound(qty: number): number {
  const q = Math.max(0, qty)
  if (q <= 0) return 0
  const pack = q >= 10 ? 10 : 5
  return Math.floor(Math.ceil(q / pack) * pack)
}

/** Row-specific numbers shown when the user clicks a cell value. */
export function buildPORowBreakdown(
  col: string,
  row: Record<string, unknown>,
  ctx: POFormulaContext,
): PORowBreakdownLine[] {
  const ads = num(row, 'ADS')
  const inv = num(row, 'Total_Inventory')
  const pipe = num(row, 'PO_Pipeline_Effective') || num(row, 'PO_Pipeline_Total')
  const lead = num(row, 'Lead_Time_Days') || ctx.leadTime
  const sold = num(row, 'Sold_Units')
  const net = num(row, 'Net_Units')
  const eff = num(row, 'Eff_Days')
  const recent = num(row, 'Recent_ADS')
  const ly = num(row, 'LY_ADS')
  const seasonal = num(row, 'Seasonal_Month_ADS')
  const flat = num(row, 'Flat30_ADS')
  const projected = ads > 0 ? (inv + pipe) / ads : 999
  const poQty = ctx.finalPoQty ?? num(row, 'PO_Qty')
  const targetCover = ctx.targetDays + ctx.graceDays

  const lines: PORowBreakdownLine[] = [
    { label: 'SKU', value: String(row['OMS_SKU'] ?? ''), highlight: true },
  ]

  switch (col) {
    case 'ADS':
    case 'Recent_ADS':
    case 'LY_ADS':
    case 'Seasonal_Month_ADS':
    case 'Flat30_ADS':
      lines.push(
        { label: 'Demand basis', value: ctx.demandBasis },
        { label: 'Sold units (window)', value: fmt(sold, 0) },
        { label: 'Net units (window)', value: fmt(net, 0) },
        { label: 'Eff. days (active)', value: fmt(eff, 1) },
        {
          label: 'Recent ADS',
          expression: `${fmt(ctx.demandBasis === 'Net' ? net : sold, 0)} ÷ ${fmt(eff, 1)}`,
          value: fmt(recent),
        },
        {
          label: 'LY ADS',
          expression: `LY units ÷ ${ctx.periodDays}`,
          value: fmt(ly),
        },
        { label: 'Season ADS', value: fmt(seasonal) },
        { label: 'Flat30 ADS', value: fmt(flat) },
        {
          label: 'ADS (used)',
          expression: 'max(blend, seasonal, flat30)',
          value: fmt(ads),
          highlight: true,
        },
      )
      break
    case 'Eff_Days':
    case 'Eff_Days_Inventory':
      lines.push(
        { label: 'Sold / net in window', value: `${fmt(sold, 0)} / ${fmt(net, 0)}` },
        { label: 'Eff. days (active)', value: fmt(eff, 1), highlight: col === 'Eff_Days' },
        { label: 'In-stock days (history)', value: fmt(num(row, 'Eff_Days_Inventory'), 0), highlight: col === 'Eff_Days_Inventory' },
        { label: 'Tip', value: 'Click 🔍 on the cell to open daily inventory snapshots' },
      )
      break
    case 'Days_Left':
      lines.push(
        { label: 'Total inventory', value: fmt(inv, 0) },
        { label: 'ADS', value: fmt(ads) },
        {
          label: 'Days left',
          expression: `${fmt(inv, 0)} ÷ ${fmt(ads)}`,
          value: fmt(num(row, 'Days_Left'), 1),
          highlight: true,
        },
      )
      break
    case 'Projected_Running_Days':
      lines.push(
        { label: 'Total inventory', value: fmt(inv, 0) },
        { label: 'Eff. pipeline', value: fmt(pipe, 0) },
        { label: 'ADS', value: fmt(ads) },
        {
          label: 'Projected days',
          expression: `(${fmt(inv, 0)} + ${fmt(pipe, 0)}) ÷ ${fmt(ads)}`,
          value: fmt(projected, 1),
          highlight: true,
        },
        { label: 'Lead time', value: `${fmt(lead, 0)} d` },
      )
      break
    case 'Post_PO_Cover_Days_Capped':
      lines.push(
        { label: 'Total inventory', value: fmt(inv, 0) },
        { label: 'Eff. pipeline', value: fmt(pipe, 0) },
        { label: 'New PO qty', value: fmt(poQty, 0) },
        { label: 'ADS', value: fmt(ads) },
        {
          label: 'Post-PO cover',
          expression: `(${fmt(inv, 0)} + ${fmt(pipe, 0)} + ${fmt(poQty, 0)}) ÷ ${fmt(ads)}`,
          value: fmt(ads > 0 ? (inv + pipe + poQty) / ads : 999, 1),
          highlight: true,
        },
      )
      break
    case 'PO_Qty':
    case 'Gross_PO_Qty': {
      const raw = ads * (targetCover - projected)
      lines.push(
        { label: 'Projected days', value: fmt(projected, 1) },
        { label: 'Target cover', value: `${fmt(targetCover, 0)} d` },
        {
          label: 'Raw units',
          expression: `${fmt(ads)} × (${fmt(targetCover, 0)} − ${fmt(projected, 1)})`,
          value: fmt(Math.max(0, raw), 1),
        },
        {
          label: 'After pack round',
          value: fmt(packRound(Math.max(0, raw)), 0),
        },
        {
          label: 'PO Qty (final)',
          value: fmt(poQty, 0),
          highlight: true,
        },
      )
      const block = String(row['PO_Block_Reason'] ?? '').trim()
      if (block && block !== 'nan') {
        lines.push({ label: 'Block reason', value: block })
      }
      break
    }
    case 'PO_Pipeline_Effective':
      lines.push(
        { label: 'Sheet pipeline', value: fmt(num(row, 'PO_Pipeline_Total'), 0) },
        { label: 'Confirmed raises', value: fmt(num(row, 'PO_Confirmed_Raise_Pipeline'), 0) },
        {
          label: 'Effective',
          expression: 'max(sheet, raises)',
          value: fmt(pipe, 0),
          highlight: true,
        },
      )
      break
    case 'PO_Pipeline_Total':
      lines.push(
        { label: 'PO ordered', value: fmt(num(row, 'PO_Qty_Ordered'), 0) },
        { label: 'Pending cutting', value: fmt(num(row, 'Pending_Cutting'), 0) },
        { label: 'Balance dispatch', value: fmt(num(row, 'Balance_to_Dispatch'), 0) },
        { label: 'Total pipeline', value: fmt(num(row, 'PO_Pipeline_Total'), 0), highlight: true },
      )
      break
    case 'Cutting_Ratio': {
      const ratio = num(row, 'Cutting_Ratio')
      lines.push(
        { label: 'This size PO Qty', value: fmt(poQty, 0) },
        { label: 'Cut ratio', value: `${(ratio * 100).toFixed(1)}%`, highlight: true },
      )
      break
    }
    case 'Priority':
      lines.push(
        { label: 'Days left', value: fmt(num(row, 'Days_Left'), 1) },
        { label: 'Lead time', value: fmt(lead, 0) },
        { label: 'PO Qty', value: fmt(poQty, 0) },
        { label: 'Eff. pipeline', value: fmt(pipe, 0) },
        { label: 'Priority', value: String(row['Priority'] ?? ''), highlight: true },
      )
      break
    default: {
      const v = row[col]
      if (v !== undefined && v !== null && v !== '') {
        lines.push({
          label: col.replace(/_/g, ' '),
          value: typeof v === 'number' ? fmt(v, 3) : String(v),
          highlight: true,
        })
      }
      break
    }
  }

  return lines
}
