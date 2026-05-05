import { useState, useMemo, useCallback, memo, useRef } from 'react'
import { api, getCoverage } from '../api/client'
import { useSession } from '../store/session'
import { usePOStore } from '../store/po'

interface PORow {
  OMS_SKU: string
  Total_Inventory?: number
  Sold_Units?: number
  Return_Units?: number
  Net_Units?: number
  Recent_ADS?: number
  ADS?: number
  LY_ADS?: number
  Seasonal_Month_ADS?: number
  Days_Left?: number
  Gross_PO_Qty?: number
  PO_Pipeline_Total?: number
  PO_Qty?: number
  Priority?: string
  Stockout_Flag?: string
  Parent_SKU?: string
  [key: string]: string | number | undefined
}

interface ParentGroup {
  parentSku: string
  variants: Array<PORow & { finalQty: number }>
  worstPriority: string
  totalInventory: number
  worstDaysLeft: number
  worstProjectedDays: number
  totalSoldUnits: number
  totalADS: number
  totalGrossQty: number
  totalPOOrdered: number
  totalPendingCutting: number
  totalBalanceDispatch: number
  totalPipeline: number
  totalFinalQty: number
  quarterTotals: Record<string, number>
  avgMonthly: number
  worstStatus: string
}

interface POResult {
  ok: boolean
  message?: string
  rows?: PORow[]
  columns?: string[]
}

interface QuarterlyRow {
  OMS_SKU: string
  Avg_Monthly?: number
  ADS?: number
  Units_90d?: number
  Units_30d?: number
  Freq_30d?: number
  Status?: string
  [key: string]: string | number | undefined
}

interface QuarterlyResult {
  loaded: boolean
  columns?: string[]
  rows?: QuarterlyRow[]
}

const PO_DISPLAY_COLS = [
  'Priority', 'OMS_SKU', 'SKU_Sheet_Status', 'Lead_Time_Days', 'Total_Inventory', 'Days_Left',
  'Sold_Units', 'Ship_Units_150d', 'Eff_Days', 'Recent_ADS', 'LY_ADS', 'Seasonal_Month_ADS', 'Flat30_ADS', 'ADS',
  'Cutting_Ratio', 'Gross_PO_Qty',
  'PO_Qty_Ordered', 'Pending_Cutting', 'Balance_to_Dispatch',
  'PO_Pipeline_Total', 'Projected_Running_Days', 'PO_Qty',
  'PO_Block_Reason', 'Suggest_Close_SKU',
]

const COL_LABEL: Record<string, string> = {
  'Sold_Units':               '📦 Sold Units',
  'Eff_Days':                 '📅 Eff. Days (active)',
  'Recent_ADS':               '📉 Recent ADS',
  'LY_ADS':                   '📈 LY ADS',
  'Seasonal_Month_ADS':       '🌗 Season ADS (mo+1)',
  'Flat30_ADS':               '📅 Sheet FREQ (÷30)',
  'ADS':                      '⚡ ADS (Used)',
  'PO_Pipeline_Total':        '🏭 Total Pipeline',
  'PO_Qty_Ordered':           '📋 PO Ordered',
  'Pending_Cutting':          '✂️ Pend. Cutting',
  'Balance_to_Dispatch':      '📦 Bal. Dispatch',
  'Projected_Running_Days':   '📅 Proj. run (Tot inv + pipe) / ADS',
  'Cutting_Ratio':            '✂️ Cut Ratio',
  'Lead_Time_Days':          '📅 Lead (d)',
  'SKU_Sheet_Status':        '📋 Sheet status',
  'Ship_Units_150d':         '📉 Ship ~5mo',
  'PO_Block_Reason':         '🚫 Block reason',
  'Suggest_Close_SKU':       '💡 Close hint',
}

const PRIORITY_ORDER: Record<string, number> = {
  '🔴 URGENT': 0, '🟡 HIGH': 1, '🟢 MEDIUM': 2, '🔄 In Pipeline': 3, '⚪ OK': 4,
}

const STATUS_COLORS: Record<string, string> = {
  'Fast Moving':  'text-green-700 bg-green-50',
  'Moderate':     'text-blue-700 bg-blue-50',
  'Slow Selling': 'text-yellow-700 bg-yellow-50',
  'Not Moving':   'text-red-600 bg-red-50',
}

type Tab = 'po' | 'quarterly' | 'shipment'
const QUARTER_COL_RE = /^(Apr[-–]Jun|Jul[-–]Sep|Oct[-–]Dec|Jan[-–]Mar)\s+\d{4}$/i
type Marketplace = 'amazon' | 'flipkart' | 'myntra' | 'meesho'

/** Scroll region + max height so thead ``sticky top-0`` has a real scrollport (Layout uses ``main`` scroll otherwise). */
const PO_TABLE_SCROLL_CLASS =
  'max-h-[min(72vh,calc(100dvh-19rem))] overflow-auto overscroll-contain shadow-sm'

interface ShipmentParams {
  marketplace: Marketplace
  period_days: number
  lead_time: number
  target_days: number
  demand_basis: string
  min_denominator: number
  safety_pct: number
  round_to: number
  cap_to_oms_inventory: boolean
}

export default function POEngine() {
  const setCoverage = useSession(s => s.setCoverage)
  const skuStatusLoaded = useSession(s => s.sku_status_lead ?? false)
  const skuStatusRows = useSession(s => s.sku_status_lead_rows ?? 0)
  const skuFileRef = useRef<HTMLInputElement>(null)
  /** Bumps on each Calculate / quarterly load so stale async responses are ignored. */
  const poRunSeqRef = useRef(0)
  const [skuUploadBusy, setSkuUploadBusy] = useState(false)
  const [skuUploadMsg, setSkuUploadMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  const {
    activeTab, setActiveTab,
    params, setParams,
    result: _storeResult, setResult,
    quarterly: _storeQuarterly, setQuarterly,
    search, setSearch,
    sortByPriority, setSortByPriority,
    editedQty, setEditedQty,
    selected, setSelected,
    qSearch, setQSearch,
    groupedView, setGroupedView,
    collapsedParents, setCollapsedParents,
  } = usePOStore()

  // Cast to local interfaces so downstream code keeps its named-field types
  const result   = _storeResult   as POResult | null
  const quarterly = _storeQuarterly as QuarterlyResult | null

  // ephemeral UI state (no need to persist across navigation)
  const [loading, setLoading] = useState(false)
  /** Quarterly pivot loads after PO (or alone on Quarterly tab); was bundled with PO and blocked the UI. */
  const [quarterlyLoading, setQuarterlyLoading] = useState(false)
  const [shipLoading, setShipLoading] = useState(false)
  const [raiseModal, setRaiseModal] = useState(false)
  const [debugInfo, setDebugInfo]   = useState<Record<string, unknown> | null>(null)
  const [shipment, setShipment] = useState<POResult | null>(null)
  const [shipSearch, setShipSearch] = useState('')
  const [shipGroupedView, setShipGroupedView] = useState(true)
  const [shipCollapsedParents, setShipCollapsedParents] = useState<Set<string>>(new Set())
  const [shipParams, setShipParams] = useState<ShipmentParams>({
    marketplace: 'amazon',
    period_days: 30,
    lead_time: 7,
    target_days: 14,
    demand_basis: 'Sold',
    min_denominator: 7,
    safety_pct: 10,
    round_to: 5,
    cap_to_oms_inventory: true,
  })
  // Cutting planner: parentSku → total pieces of material available
  const [materialQty, setMaterialQty] = useState<Record<string, number>>({})

  const loadQuarterlyForRun = async (seq: number) => {
    setQuarterlyLoading(true)
    try {
      const { data } = await api.get<QuarterlyResult>('/po/quarterly', {
        params: { group_by_parent: params.group_by_parent, n_quarters: 8 },
      })
      if (seq !== poRunSeqRef.current) return
      setQuarterly(data)
    } catch (e: unknown) {
      if (seq === poRunSeqRef.current) {
        console.warn('[PO] quarterly fetch failed:', e)
      }
    } finally {
      if (seq === poRunSeqRef.current) setQuarterlyLoading(false)
    }
  }

  /** Quarterly tab only — does not run PO math (fast). */
  const runQuarterlyOnly = async () => {
    const seq = ++poRunSeqRef.current
    setQuarterlyLoading(true)
    try {
      const { data } = await api.get<QuarterlyResult>('/po/quarterly', {
        params: { group_by_parent: params.group_by_parent, n_quarters: 8 },
      })
      if (seq !== poRunSeqRef.current) return
      setQuarterly(data)
    } catch (e: unknown) {
      setQuarterly({ loaded: false, rows: [], columns: [] })
      console.warn('[PO] quarterly-only load failed:', e)
    } finally {
      if (seq === poRunSeqRef.current) setQuarterlyLoading(false)
    }
  }

  const onSkuStatusFile = async (files: FileList | null) => {
    const f = files?.[0]
    if (!f) return
    setSkuUploadBusy(true)
    setSkuUploadMsg(null)
    try {
      const fd = new FormData()
      fd.append('file', f)
      const { data } = await api.post<{ ok: boolean; message?: string; rows?: number }>(
        '/po/sku-status-lead',
        fd,
        { headers: { 'Content-Type': 'multipart/form-data' } },
      )
      if (data.ok) {
        setSkuUploadMsg({ type: 'ok', text: data.message || `Loaded ${data.rows ?? 0} rows.` })
        const c = await getCoverage()
        setCoverage(c)
      } else {
        setSkuUploadMsg({ type: 'err', text: data.message || 'Upload failed' })
      }
    } catch (e: unknown) {
      setSkuUploadMsg({ type: 'err', text: e instanceof Error ? e.message : 'Upload failed' })
    } finally {
      setSkuUploadBusy(false)
      if (skuFileRef.current) skuFileRef.current.value = ''
    }
  }

  const run = async () => {
    const seq = ++poRunSeqRef.current
    setLoading(true)
    setEditedQty({})
    setSelected(new Set())
    let poRes: POResult | null = null
    try {
      const res = await api.post<POResult>('/po/calculate', params)
      if (seq !== poRunSeqRef.current) return
      poRes = res.data
      setResult(poRes)
    } catch (e: unknown) {
      if (seq === poRunSeqRef.current) {
        setResult({ ok: false, message: e instanceof Error ? e.message : 'Error' })
      }
    } finally {
      if (seq === poRunSeqRef.current) setLoading(false)
    }

    if (seq !== poRunSeqRef.current || !poRes?.ok) return
    void loadQuarterlyForRun(seq)
  }

  const runShipment = async () => {
    setShipLoading(true)
    try {
      const res = await api.post<POResult>('/shipment/calculate', shipParams)
      setShipment(res.data)
    } catch (e: unknown) {
      setShipment({ ok: false, message: e instanceof Error ? e.message : 'Error' })
    } finally {
      setShipLoading(false)
    }
  }

  // ── Quarterly columns (non-metadata) — newest first so recent data is immediately visible ──
  const qCols = quarterly?.columns ?? []
  const quarterCols = useMemo(() =>
    [...qCols.filter(c => QUARTER_COL_RE.test(c.trim()))].reverse(),
    [qCols]
  )

  // ── Quarter lookup map ──
  const quarterMap = useMemo(() => {
    const map: Record<string, Record<string, number | string>> = {}
    const rows = quarterly?.rows ?? []
    for (const row of rows) {
      const omsSku = String(row.OMS_SKU).toUpperCase()
      map[omsSku] = {}
      for (const c of quarterCols) {
        map[omsSku][c] = Number(row[c] ?? 0)
      }
      map[omsSku]['Status'] = row['Status'] as string
      map[omsSku]['Avg_Monthly'] = Number(row['Avg_Monthly'] ?? 0)
    }
    return map
  }, [quarterly, quarterCols])

  // ── PO tab rows ──
  const allRows = result?.rows ?? []

  const filtered = useMemo(() =>
    !search
      ? allRows
      : allRows.filter(r => String(r['OMS_SKU'] ?? '').toLowerCase().includes(search.toLowerCase())),
    [allRows, search]
  )

  const rows = useMemo(() =>
    sortByPriority
      ? [...filtered].sort((a, b) =>
          (PRIORITY_ORDER[a['Priority'] as string] ?? 9) -
          (PRIORITY_ORDER[b['Priority'] as string] ?? 9)
        )
      : filtered,
    [filtered, sortByPriority]
  )

  const { urgent, high, medium, pipeline, totalPOUnits } = useMemo(() => ({
    urgent:       allRows.filter(r => r['Priority'] === '🔴 URGENT').length,
    high:         allRows.filter(r => r['Priority'] === '🟡 HIGH').length,
    medium:       allRows.filter(r => r['Priority'] === '🟢 MEDIUM').length,
    pipeline:     allRows.filter(r => r['Priority'] === '🔄 In Pipeline').length,
    totalPOUnits: allRows.reduce((s, r) => {
      const sku = String(r['OMS_SKU'])
      return s + (editedQty[sku] !== undefined ? editedQty[sku] : Number(r['PO_Qty'] || 0))
    }, 0),
  }), [allRows, editedQty])

  // ── Selection helpers ──
  const visibleSkus = useMemo(() => rows.map(r => String(r['OMS_SKU'])), [rows])
  const allVisibleSelected = useMemo(
    () => visibleSkus.length > 0 && visibleSkus.every(s => selected.has(s)),
    [visibleSkus, selected]
  )
  const someSelected = selected.size > 0

  const toggleRow = useCallback((sku: string) => {
    const next = new Set(selected)
    next.has(sku) ? next.delete(sku) : next.add(sku)
    setSelected(next)
  }, [selected, setSelected])

  const toggleAll = useCallback(() => {
    const n = new Set(selected)
    if (visibleSkus.length > 0 && visibleSkus.every(s => n.has(s))) {
      visibleSkus.forEach(s => n.delete(s))
    } else {
      visibleSkus.forEach(s => n.add(s))
    }
    setSelected(n)
  }, [selected, setSelected, visibleSkus])

  // ── Selected rows for raise PO ──
  const selectedRows = useMemo(() =>
    allRows
      .filter(r => selected.has(String(r['OMS_SKU'])))
      .map(r => {
        const sku = String(r['OMS_SKU'])
        const finalQty = editedQty[sku] !== undefined ? editedQty[sku] : Number(r['PO_Qty'] || 0)
        return { ...r, Final_PO_Qty: finalQty }
      })
      .filter(r => r.Final_PO_Qty > 0),
    [allRows, selected, editedQty]
  )

  const totalRaiseUnits = useMemo(
    () => selectedRows.reduce((s, r) => s + r.Final_PO_Qty, 0),
    [selectedRows]
  )

  // ── Parent groups (for size-grouped view) ──
  const parentGroups = useMemo((): ParentGroup[] => {
    const groupMap = new Map<string, ParentGroup>()
    for (const row of rows) {
      const parentSku = String(row['Parent_SKU'] || row['OMS_SKU'])
      const sku       = String(row['OMS_SKU'])
      const finalQty  = editedQty[sku] !== undefined ? editedQty[sku] : Number(row['PO_Qty'] ?? 0)
      if (!groupMap.has(parentSku)) {
        groupMap.set(parentSku, {
          parentSku, variants: [],
          worstPriority: '⚪ OK', totalInventory: 0, worstDaysLeft: 999, worstProjectedDays: 999,
          totalSoldUnits: 0, totalADS: 0, totalGrossQty: 0,
          totalPOOrdered: 0, totalPendingCutting: 0, totalBalanceDispatch: 0,
          totalPipeline: 0, totalFinalQty: 0, quarterTotals: {}, avgMonthly: 0, worstStatus: '',
        })
      }
      const g = groupMap.get(parentSku)!
      g.variants.push({ ...row, finalQty })
      g.totalInventory      += Number(row['Total_Inventory'] ?? 0)
      g.worstDaysLeft        = Math.min(g.worstDaysLeft, Number(row['Days_Left'] ?? 999))
      g.worstProjectedDays   = Math.min(g.worstProjectedDays, Number(row['Projected_Running_Days'] ?? 999))
      g.totalSoldUnits      += Number(row['Sold_Units'] ?? 0)
      g.totalADS            += Number(row['ADS'] ?? 0)
      g.totalGrossQty       += Number(row['Gross_PO_Qty'] ?? 0)
      g.totalPOOrdered      += Number(row['PO_Qty_Ordered'] ?? 0)
      g.totalPendingCutting += Number(row['Pending_Cutting'] ?? 0)
      g.totalBalanceDispatch += Number(row['Balance_to_Dispatch'] ?? 0)
      g.totalPipeline       += Number(row['PO_Pipeline_Total'] ?? 0)
      g.totalFinalQty       += finalQty
      const p = String(row['Priority'] ?? '')
      if ((PRIORITY_ORDER[p] ?? 9) < (PRIORITY_ORDER[g.worstPriority] ?? 9)) g.worstPriority = p
      const qRow = quarterMap[poSkuKey(sku)] ?? {}
      for (const c of quarterCols) {
        g.quarterTotals[c] = (g.quarterTotals[c] ?? 0) + Number(qRow[c] ?? 0)
      }
      g.avgMonthly += Number(qRow['Avg_Monthly'] ?? 0)
      // Track worst status (Fast Moving > Moderate > Slow Selling > Not Moving)
      const sOrder: Record<string, number> = { 'Fast Moving': 0, 'Moderate': 1, 'Slow Selling': 2, 'Not Moving': 3 }
      const rs = String(qRow['Status'] ?? '')
      if (rs && (g.worstStatus === '' || (sOrder[rs] ?? 9) > (sOrder[g.worstStatus] ?? 9))) g.worstStatus = rs
    }
    return Array.from(groupMap.values())
  }, [rows, editedQty, quarterMap, quarterCols])

  const toggleCollapse = useCallback((parentSku: string) => {
    const next = new Set(collapsedParents)
    next.has(parentSku) ? next.delete(parentSku) : next.add(parentSku)
    setCollapsedParents(next)
  }, [collapsedParents, setCollapsedParents])

  const toggleParentSelect = useCallback((group: ParentGroup) => {
    const childSkus = group.variants.map(r => String(r['OMS_SKU']))
    const allChecked = childSkus.every(s => selected.has(s))
    const next = new Set(selected)
    allChecked ? childSkus.forEach(s => next.delete(s)) : childSkus.forEach(s => next.add(s))
    setSelected(next)
  }, [selected, setSelected])

  // ── Grouped table visible slice (cap to prevent DOM overload) ──
  const GROUPED_CAP = 400
  const visibleGroups = useMemo(() => parentGroups.slice(0, GROUPED_CAP), [parentGroups])

  // ── Quarterly tab rows ──
  const qAllRows  = quarterly?.rows ?? []
  const qFiltered = useMemo(() =>
    !qSearch
      ? qAllRows
      : qAllRows.filter(r => String(r['OMS_SKU'] ?? '').toLowerCase().includes(qSearch.toLowerCase())),
    [qAllRows, qSearch]
  )

  const shipmentColumns = shipment?.columns ?? []
  const shipmentRows = useMemo(() =>
    (shipment?.rows ?? []).filter(r =>
      !shipSearch || String(r['OMS_SKU'] ?? '').toLowerCase().includes(shipSearch.toLowerCase())
    ),
    [shipment?.rows, shipSearch]
  )
  const shipNumericCols = useMemo(
    () => shipmentColumns.filter(c => !['OMS_SKU', 'Priority'].includes(c)),
    [shipmentColumns]
  )
  const shipmentParentGroups = useMemo(() => {
    const groups = new Map<string, { parentSku: string; worstPriority: string; totals: Record<string, number>; rows: PORow[] }>()
    for (const row of shipmentRows) {
      const sku = String(row['OMS_SKU'] ?? '')
      const parentSku = sku ? sku.split('-').slice(0, -1).join('-') || sku : ''
      if (!groups.has(parentSku)) {
        groups.set(parentSku, { parentSku, worstPriority: '⚪ OK', totals: {}, rows: [] })
      }
      const g = groups.get(parentSku)!
      g.rows.push(row)
      const p = String(row['Priority'] ?? '')
      if ((PRIORITY_ORDER[p] ?? 9) < (PRIORITY_ORDER[g.worstPriority] ?? 9)) g.worstPriority = p
      for (const c of shipNumericCols) {
        g.totals[c] = (g.totals[c] ?? 0) + Number(row[c] ?? 0)
      }
    }
    return Array.from(groups.values())
  }, [shipmentRows, shipNumericCols])

  return (
    <div className="p-6 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🎯 PO Engine</h2>
        <p className="text-gray-400 text-sm mt-1">Calculate purchase orders with quarterly history inline.</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {(['po', 'quarterly', 'shipment'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-5 py-2.5 text-sm font-semibold rounded-t-lg border-b-2 transition-colors
              ${activeTab === t
                ? 'border-[#002B5B] text-[#002B5B] bg-white'
                : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            {t === 'po' ? '🎯 PO Recommendation' : t === 'quarterly' ? '📊 Quarterly History' : '🚚 Shipment Engine'}
          </button>
        ))}
      </div>

      {/* ── PO Recommendation Tab ── */}
      {activeTab === 'po' && (
        <>
          {/* Parameters */}
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Param label="Period (days)" type="number"
                value={params.period_days} onChange={v => setParams({ ...params, period_days: +v })} />
              <Param label="Lead Time (days)" type="number"
                value={params.lead_time} onChange={v => setParams({ ...params, lead_time: +v })} />
              <Param label="Target Cover (days)" type="number"
                value={params.target_days} onChange={v => setParams({ ...params, target_days: +v })} />
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">Demand Basis</label>
                <select
                  value={params.demand_basis}
                  onChange={e => setParams({ ...params, demand_basis: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="Sold">Sold</option>
                  <option value="Net">Net</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <Param label="Grace Days (urgency buffer)" type="number"
                value={params.grace_days} onChange={v => setParams({ ...params, grace_days: +v })} />
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">
                  Safety Stock % ({params.safety_pct}%)
                </label>
                <input
                  type="range" min={0} max={100} step={5}
                  value={params.safety_pct}
                  onChange={e => setParams({ ...params, safety_pct: +e.target.value })}
                  className="w-full accent-[#002B5B]"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                  <span>0%</span><span>50%</span><span>100%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-6 mt-4 flex-wrap">
              <Toggle label="YoY Seasonality" checked={params.use_seasonality}
                onChange={v => setParams({ ...params, use_seasonality: v })} />
              {params.use_seasonality && (
                <Param label={`Seasonal Weight (${Math.round(params.seasonal_weight * 100)}%)`} type="range"
                  value={params.seasonal_weight} min={0} max={1} step={0.05}
                  onChange={v => setParams({ ...params, seasonal_weight: +v })} />
              )}
              <Toggle label="Group by Parent SKU" checked={params.group_by_parent}
                onChange={v => setParams({ ...params, group_by_parent: v })} />
              <Toggle
                label="Require ≥2 sizes to place PO"
                checked={params.enforce_two_size_minimum}
                onChange={v => setParams({ ...params, enforce_two_size_minimum: v })}
              />
            </div>

            <div className="mt-5 p-4 rounded-lg border border-dashed border-gray-300 bg-gray-50/80">
              <h4 className="text-sm font-semibold text-gray-700 mb-1">SKU status &amp; lead time (for PO)</h4>
              <p className="text-xs text-gray-500 mb-3">
                Upload Excel/CSV with <strong>SKU</strong>, <strong>Status</strong>, and <strong>Lead time</strong> columns.
                Status (including closed) is shown in the PO table for reference only; it does not change PO quantities.
                A positive <strong>Lead time</strong> per SKU overrides the default lead above for that SKU&apos;s PO math only.
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <input ref={skuFileRef} type="file" accept=".xlsx,.xls,.csv" className="hidden" onChange={e => void onSkuStatusFile(e.target.files)} />
                <button
                  type="button"
                  disabled={skuUploadBusy}
                  onClick={() => skuFileRef.current?.click()}
                  className="text-xs px-4 py-2 rounded-lg border border-[#002B5B] text-[#002B5B] font-semibold hover:bg-blue-50 disabled:opacity-50"
                >
                  {skuUploadBusy ? 'Uploading…' : '📤 Upload sheet'}
                </button>
                <span className="text-xs text-gray-600">
                  {skuStatusLoaded ? <span className="text-green-700 font-medium">✓ {skuStatusRows} SKU rows loaded</span> : <span>No sheet loaded (optional)</span>}
                </span>
              </div>
              {skuUploadMsg && (
                <p className={`mt-2 text-xs rounded px-2 py-1 ${skuUploadMsg.type === 'ok' ? 'bg-green-50 text-green-800' : 'bg-red-50 text-red-700'}`}>
                  {skuUploadMsg.text}
                </p>
              )}
            </div>

            <div className="mt-5 space-y-1">
              <button
                type="button"
                onClick={() => void run()}
                disabled={loading}
                className="px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
              >
                {loading ? '⏳ Running PO…' : '🎯 Calculate PO'}
              </button>
              {loading && (
                <p className="text-xs text-gray-500">
                  Computing recommendations from sales and inventory (usually the quick step).
                </p>
              )}
            </div>

            {result && !result.ok && (
              <p className="mt-3 text-sm text-red-600 bg-red-50 rounded p-2">{result.message}</p>
            )}
          </div>

          {result?.ok && allRows.length > 0 && (
            <>
              {quarterlyLoading && (
                <div
                  role="status"
                  className="flex items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900"
                >
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-amber-600 border-t-transparent shrink-0" aria-hidden />
                  <span>
                    <strong className="font-semibold">Loading quarterly history…</strong>
                    {' '}
                    <span className="text-amber-800/90">
                      PO numbers are ready; quarter columns appear when this finishes (large sales files can take a bit).
                    </span>
                  </span>
                </div>
              )}
              {/* KPI Strip */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <KpiCard label="🔴 URGENT"      value={urgent}       accent="border-l-red-500" />
                <KpiCard label="🟡 HIGH"        value={high}         accent="border-l-yellow-400" />
                <KpiCard label="🟢 MEDIUM"      value={medium}       accent="border-l-green-500" />
                <KpiCard label="🔄 In Pipeline" value={pipeline}     accent="border-l-blue-400" />
                <KpiCard label="Total PO Units" value={totalPOUnits} accent="border-l-[#002B5B]" />
              </div>

              {/* Toolbar */}
              <div className="flex items-center gap-3 flex-wrap">
                <input
                  value={search} onChange={e => setSearch(e.target.value)}
                  placeholder="Search SKU…"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <Toggle label="Sort by Priority" checked={sortByPriority} onChange={setSortByPriority} />
                <button
                  onClick={() => setGroupedView(!groupedView)}
                  className={`text-xs px-3 py-1.5 rounded border font-medium transition-colors ${
                    groupedView ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'border-gray-300 text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  📐 Size Families
                </button>
                <span className="text-xs text-gray-400">{rows.length} SKUs</span>
                {quarterlyLoading ? (
                  <span className="text-xs text-amber-800 font-medium bg-amber-50 px-2 py-0.5 rounded-full border border-amber-200">
                    … loading quarters
                  </span>
                ) : quarterCols.length > 0 ? (
                  <span className="text-xs text-green-600 font-medium bg-green-50 px-2 py-0.5 rounded-full">
                    ✓ {quarterCols.length} quarters loaded
                  </span>
                ) : null}
                {result?.ok && (
                  <button
                    onClick={async () => {
                      try {
                        const r = await api.get('/po/quarterly-debug')
                        setDebugInfo(r.data)
                      } catch { setDebugInfo({ error: 'fetch failed' }) }
                    }}
                    className="text-xs px-2 py-1 rounded border border-gray-200 text-gray-400 hover:bg-gray-50"
                    title="Show quarterly data diagnostic"
                  >
                    🔍 Diag
                  </button>
                )}
                <div className="ml-auto flex gap-2">
                  {someSelected && (
                    <button
                      onClick={() => setRaiseModal(true)}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700 shadow-sm"
                    >
                      🚀 Raise PO ({selected.size} SKU{selected.size > 1 ? 's' : ''}, {totalRaiseUnits.toLocaleString()} units)
                    </button>
                  )}
                  <button
                    onClick={() => exportPOCsv(rows, editedQty, quarterCols, quarterMap)}
                    className="text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                  >
                    ⬇ Export CSV
                  </button>
                </div>
              </div>

              {/* Diagnostic panel */}
              {debugInfo && (
                <div className="text-xs bg-yellow-50 border border-yellow-200 rounded-lg p-3 font-mono overflow-auto max-h-64">
                  <div className="flex justify-between mb-1">
                    <span className="font-bold text-yellow-800">🔍 Quarterly Diagnostic</span>
                    <button onClick={() => setDebugInfo(null)} className="text-yellow-600 hover:text-yellow-900">✕</button>
                  </div>
                  <pre className="text-yellow-900 whitespace-pre-wrap">{JSON.stringify(debugInfo, null, 2)}</pre>
                </div>
              )}

              {/* Pipeline info banner */}
              <div className="flex items-center gap-2 text-xs text-gray-500 bg-blue-50 border border-blue-100 rounded-lg px-4 py-2">
                <span>💡</span>
                <span>
                  <strong className="text-blue-700">🏭 In Production</strong> = units already ordered (from your PO sheet). {' '}
                  <strong>Gross PO Qty</strong> − <strong>In Production</strong> = <strong className="text-orange-600">PO Qty</strong> (net new order). {' '}
                  Edit <strong className="text-orange-600">PO Qty</strong> cells directly, then select SKUs and click <strong className="text-green-700">Raise PO</strong>.
                </span>
              </div>

              {/* ── Flat Table ── */}
              {!groupedView && (
              <div className={`bg-white rounded-xl border border-gray-200 ${PO_TABLE_SCROLL_CLASS}`}>
                <table className="w-full text-sm border-separate border-spacing-0">
                  <thead>
                    <tr className="bg-gray-50">
                      {/* Checkbox */}
                      <th className="px-3 py-3 text-left sticky left-0 top-0 z-[45] bg-gray-50 border-b border-gray-200 shadow-[1px_0_0_0_rgb(229,231,235)]">
                        <input
                          type="checkbox"
                          checked={allVisibleSelected}
                          onChange={toggleAll}
                          className="rounded cursor-pointer accent-[#002B5B]"
                          title="Select all visible"
                        />
                      </th>
                      {/* PO columns */}
                      {PO_DISPLAY_COLS.map(c => (
                        <th key={c}
                          className={`text-left px-4 py-3 font-semibold whitespace-nowrap border-b border-gray-200 bg-gray-50
                            ${c === 'OMS_SKU'
                              ? 'sticky left-9 top-0 z-[45] shadow-[1px_0_0_0_rgb(229,231,235)] text-gray-600'
                              : 'sticky top-0 z-30 text-gray-600'}
                            ${c === 'PO_Qty' ? 'text-orange-600' : ''}
                            ${c === 'PO_Qty_Ordered' ? 'text-slate-600' : ''}
                            ${c === 'Pending_Cutting' ? 'text-purple-600' : ''}
                            ${c === 'Balance_to_Dispatch' ? 'text-teal-600' : ''}`}
                        >
                          {c === 'PO_Qty'
                            ? <span>PO Qty ✏️</span>
                            : COL_LABEL[c]
                              ? <span>{COL_LABEL[c]}</span>
                              : c.replace(/_/g, ' ')}
                        </th>
                      ))}
                      {/* Quarter history divider + columns */}
                      {quarterCols.length > 0 && (
                        <>
                          <th className="px-2 py-3 sticky top-0 z-30 bg-indigo-50 text-indigo-400 text-xs font-bold whitespace-nowrap text-center border-b border-gray-200 border-l border-r border-indigo-100">
                            ── QUARTERLY HISTORY ──
                          </th>
                          {quarterCols.map(c => (
                            <th key={c} className="text-right px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200 border-r border-indigo-100">
                              {c}
                            </th>
                          ))}
                          <th className="text-right px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200 border-r border-indigo-100">
                            Avg/Mo
                          </th>
                          <th className="text-left px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200">
                            Status
                          </th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.slice(0, 500).map((row, i) => {
                      const sku      = String(row['OMS_SKU'])
                      const priority = String(row['Priority'] ?? '')
                      const isSelected = selected.has(sku)
                      const rowBg =
                        isSelected
                          ? 'bg-blue-50'
                          : priority === '🔴 URGENT'      ? 'bg-red-50'
                          : priority === '🟡 HIGH'        ? 'bg-yellow-50'
                          : priority === '🟢 MEDIUM'      ? 'bg-amber-50'
                          : priority === '🔄 In Pipeline' ? 'bg-blue-50/40'
                          : ''

                      const computedQty  = Number(row['PO_Qty'] ?? 0)
                      const finalQty     = editedQty[sku] !== undefined ? editedQty[sku] : computedQty
                      const qRow         = quarterMap[poSkuKey(sku)] ?? {}
                      const status       = String(qRow['Status'] ?? '')
                      const statusClass  = STATUS_COLORS[status] ?? 'text-gray-400 bg-gray-50'

                      return (
                        <tr key={i} className={`hover:brightness-[0.97] transition-colors ${rowBg} [&>td]:border-b [&>td]:border-gray-100`}>
                          {/* Checkbox */}
                          <td className={`px-3 py-2 sticky left-0 z-10 ${isSelected ? 'bg-blue-50' : 'bg-white'}`}>
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => toggleRow(sku)}
                              className="rounded cursor-pointer accent-[#002B5B]"
                            />
                          </td>

                          {/* PO Columns */}
                          {PO_DISPLAY_COLS.map(col => (
                            <td key={col}
                              className={`px-4 py-2 whitespace-nowrap text-gray-700
                                ${col === 'OMS_SKU' ? 'sticky left-9 z-10 font-medium text-gray-900 ' + (isSelected ? 'bg-blue-50' : 'bg-white') + ' shadow-sm' : ''}`}
                            >
                              {col === 'Priority'
                                ? <PriorityBadge priority={priority} />
                                : col === 'OMS_SKU'
                                  ? <span className="font-medium">{sku}</span>
                                  : col === 'PO_Pipeline_Total'
                                    ? Number(row[col] ?? 0) > 0
                                      ? <span className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full">
                                          🏭 {Number(row[col]).toLocaleString()}
                                        </span>
                                      : <span className="text-gray-300">—</span>
                                    : col === 'PO_Qty_Ordered'
                                      ? Number(row[col] ?? 0) > 0
                                        ? <span className="text-xs font-semibold text-slate-700">{Number(row[col]).toLocaleString()}</span>
                                        : <span className="text-gray-300">—</span>
                                      : col === 'Pending_Cutting'
                                        ? Number(row[col] ?? 0) > 0
                                          ? <span className="text-xs font-semibold text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded">{Number(row[col]).toLocaleString()}</span>
                                          : <span className="text-gray-300">—</span>
                                        : col === 'Balance_to_Dispatch'
                                          ? Number(row[col] ?? 0) > 0
                                            ? <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-1.5 py-0.5 rounded">{Number(row[col]).toLocaleString()}</span>
                                            : <span className="text-gray-300">—</span>
                                          : col === 'Cutting_Ratio'
                                            ? Number(row[col] ?? 0) > 0
                                              ? <span className="text-xs font-semibold text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded">
                                                  {(Number(row[col]) * 100).toFixed(1)}%
                                                </span>
                                              : <span className="text-gray-300">—</span>
                                          : col === 'Projected_Running_Days'
                                            ? <DaysLeftBadge days={Number(row[col] ?? 999)} />
                                          : col === 'PO_Qty'
                                            ? <QtyInput
                                                value={finalQty}
                                                computed={computedQty}
                                                onChange={v => setEditedQty({ ...editedQty, [sku]: v })}
                                                onReset={() => { const n = {...editedQty}; delete n[sku]; setEditedQty(n) }}
                                              />
                                            : col === 'Days_Left'
                                              ? <DaysLeftBadge days={Number(row[col] ?? 999)} />
                                              : typeof row[col] === 'number'
                                                ? Number(row[col]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                                : row[col] ?? '—'
                              }
                            </td>
                          ))}

                          {/* Quarterly columns */}
                          {quarterCols.length > 0 && (
                            <>
                              <td className="px-2 py-2 bg-indigo-50 border-l border-r border-indigo-100" />
                              {quarterCols.map(c => {
                                const v = Number(qRow[c] ?? 0)
                                return (
                                  <td key={c} className="px-3 py-2 text-right whitespace-nowrap bg-indigo-50/50 border-r border-indigo-100/60">
                                    {v > 0
                                      ? <span className="font-medium text-indigo-700">{v.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>}
                                  </td>
                                )
                              })}
                              <td className="px-3 py-2 text-right whitespace-nowrap bg-indigo-50/50 border-r border-indigo-100/60 font-semibold text-indigo-800 text-xs">
                                {qRow['Avg_Monthly'] ? Number(qRow['Avg_Monthly']).toFixed(1) : '—'}
                              </td>
                              <td className="px-3 py-2 whitespace-nowrap bg-indigo-50/50">
                                {status
                                  ? <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusClass}`}>{status}</span>
                                  : <span className="text-gray-300">—</span>}
                              </td>
                            </>
                          )}
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                {rows.length > 500 && (
                  <p className="text-xs text-gray-400 text-center py-2">Showing 500 of {rows.length}</p>
                )}
              </div>
              )}

              {/* ── Grouped (Size Families) Table ── */}
              {groupedView && (
              <div className={`bg-white rounded-xl border border-gray-200 ${PO_TABLE_SCROLL_CLASS}`}>
                <table className="w-full text-sm border-separate border-spacing-0">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="px-3 py-3 sticky left-0 top-0 z-[45] bg-gray-50 border-b border-gray-200 w-14 shadow-[1px_0_0_0_rgb(229,231,235)]" />
                      {PO_DISPLAY_COLS.map(c => (
                        <th key={c}
                          className={`text-left px-4 py-3 font-semibold whitespace-nowrap border-b border-gray-200 bg-gray-50
                            ${c === 'OMS_SKU'
                              ? 'sticky left-14 top-0 z-[45] shadow-[1px_0_0_0_rgb(229,231,235)] text-gray-600'
                              : 'sticky top-0 z-30 text-gray-600'}
                            ${c === 'PO_Qty' ? 'text-orange-600' : ''}
                            ${c === 'PO_Qty_Ordered' ? 'text-slate-600' : ''}
                            ${c === 'Pending_Cutting' ? 'text-purple-600' : ''}
                            ${c === 'Balance_to_Dispatch' ? 'text-teal-600' : ''}`}
                        >
                          {c === 'PO_Qty'
                            ? <span>PO Qty ✏️</span>
                            : COL_LABEL[c]
                              ? <span>{COL_LABEL[c]}</span>
                              : c.replace(/_/g, ' ')}
                        </th>
                      ))}
                      {quarterCols.length > 0 && (
                        <>
                          <th className="px-2 py-3 sticky top-0 z-30 bg-indigo-50 text-indigo-400 text-xs font-bold whitespace-nowrap text-center border-b border-gray-200 border-l border-r border-indigo-100">
                            ── QUARTERLY HISTORY ──
                          </th>
                          {quarterCols.map(c => (
                            <th key={c} className="text-right px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200 border-r border-indigo-100">
                              {c}
                            </th>
                          ))}
                          <th className="text-right px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200 border-r border-indigo-100">Avg/Mo</th>
                          <th className="text-left px-3 py-3 font-semibold text-indigo-600 whitespace-nowrap text-xs sticky top-0 z-30 bg-indigo-50 border-b border-gray-200">Status</th>
                        </>
                      )}
                      {/* Cutting planner columns */}
                      <th className="px-2 py-3 sticky top-0 z-30 bg-amber-50 text-amber-400 text-xs font-bold whitespace-nowrap text-center border-b border-gray-200 border-l border-r border-amber-100">
                        ── CUTTING PLANNER ──
                      </th>
                      <th className="text-right px-3 py-3 font-semibold text-amber-700 whitespace-nowrap text-xs sticky top-0 z-30 bg-amber-50 border-b border-gray-200 border-r border-amber-100">Cut Ratio</th>
                      <th className="text-right px-3 py-3 font-semibold text-amber-700 whitespace-nowrap text-xs sticky top-0 z-30 bg-amber-50 border-b border-gray-200 border-r border-amber-100">PO Qty</th>
                      <th className="text-center px-3 py-3 font-semibold text-amber-700 whitespace-nowrap text-xs sticky top-0 z-30 bg-amber-50 border-b border-gray-200 border-r border-amber-100">🧵 Material Avail.</th>
                      <th className="text-right px-3 py-3 font-semibold text-amber-700 whitespace-nowrap text-xs sticky top-0 z-30 bg-amber-50 border-b border-gray-200 border-r border-amber-100">✂️ Sug. Cut</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleGroups.flatMap(group => {
                      const isCollapsed = collapsedParents.has(group.parentSku)
                      const childSkus   = group.variants.map(r => String(r['OMS_SKU']))
                      const allChecked  = childSkus.every(s => selected.has(s))
                      const someChecked = childSkus.some(s => selected.has(s))
                      const gSClass     = STATUS_COLORS[group.worstStatus] ?? 'text-gray-400 bg-gray-50'

                      const parentRow = (
                        <tr key={group.parentSku + '-hdr'} className="bg-slate-100 hover:bg-slate-200/60 transition-colors [&>td]:border-b [&>td]:border-slate-200">
                          <td className="px-3 py-2.5 sticky left-0 bg-slate-100 z-10 w-14">
                            <div className="flex items-center gap-1.5">
                              <button
                                onClick={() => toggleCollapse(group.parentSku)}
                                className="text-gray-500 text-[10px] w-3 text-center font-mono leading-none"
                              >
                                {isCollapsed ? '▶' : '▼'}
                              </button>
                              <input
                                type="checkbox"
                                checked={allChecked}
                                onChange={() => toggleParentSelect(group)}
                                className={`rounded cursor-pointer accent-[#002B5B] ${someChecked && !allChecked ? 'opacity-60' : ''}`}
                              />
                            </div>
                          </td>
                          {PO_DISPLAY_COLS.map(c => (
                            <td key={c}
                              className={`px-4 py-2.5 whitespace-nowrap font-semibold text-gray-800
                                ${c === 'OMS_SKU' ? 'sticky left-14 bg-slate-100 z-10 shadow-sm' : ''}`}
                            >
                              {c === 'Priority'
                                ? <PriorityBadge priority={group.worstPriority} />
                                : c === 'OMS_SKU'
                                  ? <span className="font-bold text-[#002B5B]">
                                      {group.parentSku}
                                      <span className="ml-1.5 text-xs text-gray-400 font-normal">
                                        ({group.variants.length} size{group.variants.length > 1 ? 's' : ''})
                                      </span>
                                    </span>
                                  : c === 'Total_Inventory'  ? group.totalInventory.toLocaleString()
                                  : c === 'Days_Left'        ? <DaysLeftBadge days={group.worstDaysLeft} />
                                  : c === 'Sold_Units'       ? group.totalSoldUnits.toLocaleString()
                                  : c === 'ADS'              ? group.totalADS.toFixed(3)
                                  : c === 'Gross_PO_Qty'     ? group.totalGrossQty.toLocaleString()
                                  : c === 'PO_Qty_Ordered'
                                    ? group.totalPOOrdered > 0
                                      ? <span className="text-xs font-semibold text-slate-700">{group.totalPOOrdered.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'Pending_Cutting'
                                    ? group.totalPendingCutting > 0
                                      ? <span className="text-xs font-semibold text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded">{group.totalPendingCutting.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'Balance_to_Dispatch'
                                    ? group.totalBalanceDispatch > 0
                                      ? <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-1.5 py-0.5 rounded">{group.totalBalanceDispatch.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'PO_Pipeline_Total'
                                    ? group.totalPipeline > 0
                                      ? <span className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 bg-blue-100 border border-blue-200 px-2 py-0.5 rounded-full">
                                          🏭 {group.totalPipeline.toLocaleString()}
                                        </span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'Projected_Running_Days'
                                    ? <DaysLeftBadge days={group.worstProjectedDays} />
                                  : c === 'PO_Qty'
                                    ? <span className={`font-bold text-base ${group.totalFinalQty > 0 ? 'text-orange-600' : 'text-gray-400'}`}>
                                        {group.totalFinalQty.toLocaleString()}
                                      </span>
                                  : '—'
                              }
                            </td>
                          ))}
                          {quarterCols.length > 0 && (
                            <>
                              <td className="px-2 py-2.5 bg-indigo-50 border-l border-r border-indigo-100" />
                              {quarterCols.map(c => {
                                const v = group.quarterTotals[c] ?? 0
                                return (
                                  <td key={c} className="px-3 py-2.5 text-right whitespace-nowrap bg-indigo-50/70 border-r border-indigo-100/60 font-semibold text-indigo-800">
                                    {v > 0 ? v.toLocaleString() : <span className="text-gray-300">—</span>}
                                  </td>
                                )
                              })}
                              <td className="px-3 py-2.5 text-right whitespace-nowrap bg-indigo-50/70 border-r border-indigo-100/60 font-bold text-indigo-800">
                                {group.avgMonthly > 0 ? group.avgMonthly.toFixed(1) : '—'}
                              </td>
                              <td className="px-3 py-2.5 whitespace-nowrap bg-indigo-50/70">
                                {group.worstStatus
                                  ? <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${gSClass}`}>{group.worstStatus}</span>
                                  : <span className="text-gray-300">—</span>}
                              </td>
                            </>
                          )}
                          {/* Cutting planner — parent row: auto-fills from total PO qty, overridable */}
                          {(() => {
                            const autoQty = group.totalFinalQty  // default = total PO qty
                            const matQty  = materialQty[group.parentSku] !== undefined
                              ? materialQty[group.parentSku]
                              : autoQty
                            return (
                              <>
                                <td className="px-2 py-2.5 bg-amber-50 border-l border-r border-amber-100" />
                                <td className="px-3 py-2.5 bg-amber-50/50 border-r border-amber-100 text-right text-xs text-amber-600 font-semibold">100%</td>
                                <td className="px-3 py-2.5 bg-amber-50/50 border-r border-amber-100 text-right font-bold text-amber-700 text-sm">
                                  {group.totalFinalQty > 0
                                    ? group.totalFinalQty.toLocaleString()
                                    : <span className="text-amber-200 font-normal text-xs">—</span>}
                                </td>
                                <td className="px-3 py-2.5 bg-amber-50/50 border-r border-amber-100 text-center">
                                  <input
                                    type="number" min={0} step={5}
                                    placeholder={autoQty > 0 ? String(autoQty) : 'qty'}
                                    value={materialQty[group.parentSku] !== undefined ? materialQty[group.parentSku] : ''}
                                    onChange={e => {
                                      const v = parseInt(e.target.value, 10)
                                      setMaterialQty(prev => ({ ...prev, [group.parentSku]: isNaN(v) ? 0 : v }))
                                    }}
                                    onBlur={e => { if (!e.target.value) setMaterialQty(prev => { const n={...prev}; delete n[group.parentSku]; return n }) }}
                                    onClick={e => e.stopPropagation()}
                                    className="w-24 border border-amber-300 rounded px-2 py-1 text-xs text-center focus:outline-none focus:ring-1 focus:ring-amber-400 bg-white"
                                  />
                                  {materialQty[group.parentSku] === undefined && autoQty > 0 && (
                                    <div className="text-[10px] text-amber-400 mt-0.5">auto (PO qty)</div>
                                  )}
                                </td>
                                <td className="px-3 py-2.5 bg-amber-50/50 border-r border-amber-100 text-right font-bold text-amber-700 text-sm">
                                  {matQty > 0
                                    ? matQty.toLocaleString()
                                    : <span className="text-amber-200 font-normal text-xs">—</span>}
                                </td>
                              </>
                            )
                          })()}
                        </tr>
                      )

                      const childRows = isCollapsed ? [] : group.variants.map((variant, vi) => {
                        const sku         = String(variant['OMS_SKU'])
                        const sizeLabel   = sku.startsWith(group.parentSku)
                          ? sku.slice(group.parentSku.length).replace(/^[-_]/, '').trim()
                          : sku
                        const isSelected  = selected.has(sku)
                        const priority    = String(variant['Priority'] ?? '')
                        const qRow        = quarterMap[poSkuKey(sku)] ?? {}
                        const vstatus     = String(qRow['Status'] ?? '')
                        const vSClass     = STATUS_COLORS[vstatus] ?? 'text-gray-400 bg-gray-50'
                        const computedQty = Number(variant['PO_Qty'] ?? 0)
                        const finalQty    = variant.finalQty

                        return (
                          <tr key={sku + '-' + vi}
                            className={`hover:brightness-[0.97] transition-colors ${isSelected ? 'bg-blue-50' : 'bg-white'} [&>td]:border-b [&>td]:border-gray-100`}
                          >
                            <td className={`px-3 py-2 sticky left-0 z-10 w-14 ${isSelected ? 'bg-blue-50' : 'bg-white'}`}>
                              <div className="pl-5 flex items-center">
                                <input type="checkbox" checked={isSelected} onChange={() => toggleRow(sku)}
                                  className="rounded cursor-pointer accent-[#002B5B]" />
                              </div>
                            </td>
                            {PO_DISPLAY_COLS.map(col => (
                              <td key={col}
                                className={`px-4 py-2 whitespace-nowrap text-gray-700
                                  ${col === 'OMS_SKU' ? 'sticky left-14 z-10 ' + (isSelected ? 'bg-blue-50' : 'bg-white') + ' shadow-sm' : ''}`}
                              >
                                {col === 'Priority' ? <PriorityBadge priority={priority} />
                                  : col === 'OMS_SKU'
                                    ? <span>
                                        <span className="inline-block w-12 text-xs font-bold text-gray-600 uppercase mr-1">{sizeLabel || sku}</span>
                                        <span className="text-gray-400 text-xs">{sku}</span>
                                      </span>
                                  : col === 'PO_Pipeline_Total'
                                    ? Number(variant[col] ?? 0) > 0
                                      ? <span className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700 bg-blue-50 border border-blue-200 px-2 py-0.5 rounded-full">
                                          🏭 {Number(variant[col]).toLocaleString()}
                                        </span>
                                      : <span className="text-gray-300">—</span>
                                  : col === 'PO_Qty_Ordered'
                                    ? Number(variant[col] ?? 0) > 0
                                      ? <span className="text-xs font-semibold text-slate-700">{Number(variant[col]).toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : col === 'Pending_Cutting'
                                    ? Number(variant[col] ?? 0) > 0
                                      ? <span className="text-xs font-semibold text-purple-700 bg-purple-50 px-1.5 py-0.5 rounded">{Number(variant[col]).toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : col === 'Balance_to_Dispatch'
                                    ? Number(variant[col] ?? 0) > 0
                                      ? <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-1.5 py-0.5 rounded">{Number(variant[col]).toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : col === 'Projected_Running_Days'
                                    ? <DaysLeftBadge days={Number(variant[col] ?? 999)} />
                                  : col === 'PO_Qty'
                                    ? <QtyInput value={finalQty} computed={computedQty}
                                        onChange={v => setEditedQty({ ...editedQty, [sku]: v })}
                                        onReset={() => { const n = {...editedQty}; delete n[sku]; setEditedQty(n) }} />
                                  : col === 'Days_Left' ? <DaysLeftBadge days={Number(variant[col] ?? 999)} />
                                  : typeof variant[col] === 'number'
                                    ? Number(variant[col]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                    : variant[col] ?? '—'
                                }
                              </td>
                            ))}
                            {quarterCols.length > 0 && (
                              <>
                                <td className="px-2 py-2 bg-indigo-50 border-l border-r border-indigo-100" />
                                {quarterCols.map(c => {
                                  const v = Number(qRow[c] ?? 0)
                                  return (
                                    <td key={c} className="px-3 py-2 text-right whitespace-nowrap bg-indigo-50/50 border-r border-indigo-100/60">
                                      {v > 0 ? <span className="font-medium text-indigo-700">{v.toLocaleString()}</span> : <span className="text-gray-300">—</span>}
                                    </td>
                                  )
                                })}
                                <td className="px-3 py-2 text-right whitespace-nowrap bg-indigo-50/50 border-r border-indigo-100/60 font-semibold text-indigo-800 text-xs">
                                  {qRow['Avg_Monthly'] ? Number(qRow['Avg_Monthly']).toFixed(1) : '—'}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap bg-indigo-50/50">
                                  {vstatus ? <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${vSClass}`}>{vstatus}</span> : <span className="text-gray-300">—</span>}
                                </td>
                              </>
                            )}
                            {/* Cutting planner — variant row */}
                            {(() => {
                              const ratio = Number(variant['Cutting_Ratio'] ?? 0)
                              const autoQty = group.totalFinalQty
                              const matQty = materialQty[group.parentSku] !== undefined
                                ? materialQty[group.parentSku]
                                : autoQty
                              const sugCut = matQty > 0 ? Math.round((matQty * ratio) / 5) * 5 : 0
                              const pct = ratio > 0 ? (ratio * 100).toFixed(1) + '%' : '—'
                              return (
                                <>
                                  <td className="px-2 py-2 bg-amber-50/30 border-l border-r border-amber-100" />
                                  <td className="px-3 py-2 text-right whitespace-nowrap bg-amber-50/30 border-r border-amber-100">
                                    <span className="text-xs font-semibold text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded">{pct}</span>
                                  </td>
                                  <td className="px-3 py-2 text-right whitespace-nowrap bg-amber-50/30 border-r border-amber-100">
                                    {finalQty > 0
                                      ? <span className="text-sm font-bold text-amber-800">{finalQty.toLocaleString()}</span>
                                      : <span className="text-gray-300 text-xs">—</span>}
                                  </td>
                                  <td className="px-3 py-2 bg-amber-50/30 border-r border-amber-100" />
                                  <td className="px-3 py-2 text-right whitespace-nowrap bg-amber-50/30 border-r border-amber-100">
                                    {sugCut > 0
                                      ? <span className="text-sm font-bold text-amber-800">{sugCut.toLocaleString()}</span>
                                      : <span className="text-gray-300 text-xs">—</span>}
                                  </td>
                                </>
                              )
                            })()}
                          </tr>
                        )
                      })

                      return [parentRow, ...childRows]
                    })}
                  </tbody>
                </table>
                {parentGroups.length === 0 && (
                  <p className="text-xs text-gray-400 text-center py-4">No data</p>
                )}
                {parentGroups.length > GROUPED_CAP && (
                  <p className="text-xs text-gray-400 text-center py-2">
                    Showing {GROUPED_CAP} of {parentGroups.length} parent SKUs — use search to filter
                  </p>
                )}
              </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Quarterly History Tab ── */}
      {activeTab === 'quarterly' && (
        <>
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <div className="flex items-center gap-4 flex-wrap">
              <Toggle label="Group by Parent SKU" checked={params.group_by_parent}
                onChange={v => setParams({ ...params, group_by_parent: v })} />
              <button
                type="button"
                onClick={() => void runQuarterlyOnly()}
                disabled={quarterlyLoading}
                className="px-5 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
              >
                {quarterlyLoading ? '⏳ Loading history…' : '📊 Load Quarterly History'}
              </button>
              {quarterly && !quarterly.loaded && !quarterlyLoading && (
                <span className="text-sm text-red-500">No data — build Sales first.</span>
              )}
            </div>
          </div>

          {quarterly?.loaded && qAllRows.length > 0 && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <KpiCard label="Total SKUs" value={qAllRows.length} />
                <KpiCard label="Fast Moving"
                  value={qAllRows.filter(r => r['Status'] === 'Fast Moving').length}
                  accent="border-l-green-500" />
                <KpiCard label="Moderate"
                  value={qAllRows.filter(r => r['Status'] === 'Moderate').length}
                  accent="border-l-blue-400" />
                <KpiCard label="Not Moving"
                  value={qAllRows.filter(r => r['Status'] === 'Not Moving').length}
                  accent="border-l-red-400" />
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <input
                  value={qSearch} onChange={e => setQSearch(e.target.value)}
                  placeholder="Search SKU…"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <span className="text-xs text-gray-400">{qFiltered.length} SKUs</span>
                <button
                  onClick={() => downloadQCsv(quarterly.rows ?? [], quarterly.columns ?? [])}
                  className="ml-auto text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                >
                  ⬇ Export CSV
                </button>
              </div>

              <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 z-30">
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap sticky left-0 bg-gray-50 z-40 shadow-sm">
                        SKU
                      </th>
                      {quarterCols.map(c => (
                        <th key={c} className="text-right px-3 py-3 font-semibold text-gray-500 whitespace-nowrap text-xs">
                          {c}
                        </th>
                      ))}
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Avg/Month</th>
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Daily Avg</th>
                      <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Last 30d</th>
                      <th className="text-left px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {qFiltered.slice(0, 500).map((row, i) => {
                      const status = String(row['Status'] ?? '')
                      const statusClass = STATUS_COLORS[status] ?? ''
                      return (
                        <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                          <td className="px-4 py-2 font-medium text-gray-900 whitespace-nowrap sticky left-0 bg-white z-10 shadow-sm">
                            {row['OMS_SKU']}
                          </td>
                          {quarterCols.map(c => (
                            <td key={c} className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                              {Number(row[c] ?? 0) > 0
                                ? <span className="font-medium">{Number(row[c]).toLocaleString()}</span>
                                : <span className="text-gray-300">—</span>}
                            </td>
                          ))}
                          <td className="px-3 py-2 text-right whitespace-nowrap font-semibold text-gray-800">
                            {Number(row['Avg_Monthly'] ?? 0).toFixed(1)}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                            {Number(row['ADS'] ?? 0).toFixed(3)}
                          </td>
                          <td className="px-3 py-2 text-right whitespace-nowrap text-gray-700">
                            {Number(row['Units_30d'] ?? 0).toLocaleString()}
                          </td>
                          <td className="px-3 py-2 whitespace-nowrap">
                            <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${statusClass}`}>
                              {status || '—'}
                            </span>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
                {qFiltered.length > 500 && (
                  <p className="text-xs text-gray-400 text-center py-2">Showing 500 of {qFiltered.length}</p>
                )}
              </div>
              <p className="text-xs text-gray-400">
                Quarterly totals = forward shipments only. Avg/Month = last 4 quarters ÷ 3. Daily Avg = last 90d ÷ 90.
              </p>
            </>
          )}
        </>
      )}

      {/* ── Shipment Engine Tab ── */}
      {activeTab === 'shipment' && (
        <>
          <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
            <h3 className="font-semibold text-[#002B5B] mb-4">Shipment Parameters</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">Marketplace</label>
                <select
                  value={shipParams.marketplace}
                  onChange={e => setShipParams({ ...shipParams, marketplace: e.target.value as Marketplace })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="amazon">Amazon</option>
                  <option value="flipkart">Flipkart</option>
                  <option value="myntra">Myntra</option>
                  <option value="meesho">Meesho</option>
                </select>
              </div>
              <Param label="Period (days)" type="number"
                value={shipParams.period_days} onChange={v => setShipParams({ ...shipParams, period_days: +v })} />
              <Param label="Lead Time (days)" type="number"
                value={shipParams.lead_time} onChange={v => setShipParams({ ...shipParams, lead_time: +v })} />
              <Param label="Target Cover (days)" type="number"
                value={shipParams.target_days} onChange={v => setShipParams({ ...shipParams, target_days: +v })} />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <Param label="Safety %" type="number"
                value={shipParams.safety_pct} onChange={v => setShipParams({ ...shipParams, safety_pct: +v })} />
              <Param label="Round To" type="number"
                value={shipParams.round_to} onChange={v => setShipParams({ ...shipParams, round_to: +v })} />
              <div>
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">Demand Basis</label>
                <select
                  value={shipParams.demand_basis}
                  onChange={e => setShipParams({ ...shipParams, demand_basis: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="Sold">Sold</option>
                  <option value="Net">Net</option>
                </select>
              </div>
              <div className="flex items-end">
                <Toggle
                  label="Cap by OMS Inventory"
                  checked={shipParams.cap_to_oms_inventory}
                  onChange={v => setShipParams({ ...shipParams, cap_to_oms_inventory: v })}
                />
              </div>
            </div>

            <button
              onClick={runShipment}
              disabled={shipLoading}
              className="mt-5 px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
            >
              {shipLoading ? '⏳ Calculating…' : '🚚 Calculate Shipment'}
            </button>

            {shipment && !shipment.ok && (
              <p className="mt-3 text-sm text-red-600 bg-red-50 rounded p-2">{shipment.message}</p>
            )}
          </div>

          {shipment?.ok && (shipment.rows?.length ?? 0) > 0 && (
            <>
              <div className="flex items-center gap-3 flex-wrap">
                <input
                  value={shipSearch}
                  onChange={e => setShipSearch(e.target.value)}
                  placeholder="Search SKU…"
                  className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
                />
                <button
                  onClick={() => setShipGroupedView(!shipGroupedView)}
                  className={`text-xs px-3 py-1.5 rounded border font-medium transition-colors ${
                    shipGroupedView ? 'bg-[#002B5B] text-white border-[#002B5B]' : 'border-gray-300 text-gray-600 hover:bg-gray-50'
                  }`}
                >
                  📐 Size Families
                </button>
                <span className="text-xs text-gray-400">{shipmentRows.length} SKUs</span>
                <button
                  onClick={() => downloadShipmentCsv(shipment?.rows ?? [], shipmentColumns)}
                  className="ml-auto text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                >
                  ⬇ Export CSV
                </button>
              </div>
              {!shipGroupedView && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead className="sticky top-0 z-30">
                      <tr className="bg-gray-50 border-b border-gray-200">
                        {shipmentColumns.map(c => (
                          <th key={c} className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap bg-gray-50">
                            {c.replace(/_/g, ' ')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {shipmentRows.slice(0, 500).map((row, i) => (
                        <tr key={i} className="border-b border-gray-100 hover:bg-gray-50">
                          {shipmentColumns.map(c => (
                            <td key={c} className="px-4 py-2 whitespace-nowrap text-gray-700">
                              {c === 'Priority'
                                ? <PriorityBadge priority={String(row[c] ?? '')} />
                                : typeof row[c] === 'number'
                                  ? Number(row[c]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                  : row[c] ?? '—'}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {shipGroupedView && (
                <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead className="sticky top-0 z-30">
                      <tr className="bg-gray-50 border-b border-gray-200">
                        <th className="px-3 py-3 w-12 bg-gray-50" />
                        {shipmentColumns.map(c => (
                          <th key={c} className="text-left px-4 py-3 font-semibold text-gray-600 whitespace-nowrap bg-gray-50">
                            {c.replace(/_/g, ' ')}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {shipmentParentGroups.flatMap((group) => {
                        const collapsed = shipCollapsedParents.has(group.parentSku)
                        const parentRow = (
                          <tr key={group.parentSku + '-p'} className="bg-slate-100 border-b border-slate-200">
                            <td className="px-3 py-2">
                              <button
                                onClick={() => {
                                  const next = new Set(shipCollapsedParents)
                                  next.has(group.parentSku) ? next.delete(group.parentSku) : next.add(group.parentSku)
                                  setShipCollapsedParents(next)
                                }}
                                className="text-gray-500 text-[10px] w-3 text-center font-mono leading-none"
                              >
                                {collapsed ? '▶' : '▼'}
                              </button>
                            </td>
                            {shipmentColumns.map(c => (
                              <td key={c} className="px-4 py-2.5 whitespace-nowrap font-semibold text-gray-800">
                                {c === 'Priority'
                                  ? <PriorityBadge priority={group.worstPriority} />
                                  : c === 'OMS_SKU'
                                    ? <span className="font-bold text-[#002B5B]">{group.parentSku}</span>
                                    : Number(group.totals[c] ?? 0).toLocaleString(undefined, { maximumFractionDigits: 3 })}
                              </td>
                            ))}
                          </tr>
                        )
                        const children = collapsed ? [] : group.rows.map((row, i) => (
                          <tr key={group.parentSku + '-c-' + i} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="px-3 py-2" />
                            {shipmentColumns.map(c => (
                              <td key={c} className="px-4 py-2 whitespace-nowrap text-gray-700">
                                {c === 'Priority'
                                  ? <PriorityBadge priority={String(row[c] ?? '')} />
                                  : typeof row[c] === 'number'
                                    ? Number(row[c]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                    : row[c] ?? '—'}
                              </td>
                            ))}
                          </tr>
                        ))
                        return [parentRow, ...children]
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </>
      )}

      {/* ── Raise PO Modal ── */}
      {raiseModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col">
            <div className="p-5 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-[#002B5B]">🚀 Raise Purchase Order</h3>
                <p className="text-sm text-gray-500 mt-0.5">
                  {selectedRows.length} SKU{selectedRows.length !== 1 ? 's' : ''} · {totalRaiseUnits.toLocaleString()} total units
                </p>
              </div>
              <button onClick={() => setRaiseModal(false)} className="text-gray-400 hover:text-gray-600 text-2xl leading-none">×</button>
            </div>

            <div className="overflow-auto flex-1 p-4">
              {selectedRows.length === 0 ? (
                <div className="text-center text-gray-400 py-8">
                  No SKUs with PO Qty &gt; 0 in selection.
                  <br /><span className="text-sm">Adjust quantities or include SKUs that need ordering.</span>
                </div>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-3 py-2 font-semibold text-gray-600">SKU</th>
                      <th className="text-left px-3 py-2 font-semibold text-gray-600">Priority</th>
                      <th className="text-right px-3 py-2 font-semibold text-gray-600">Days Left</th>
                      <th className="text-right px-3 py-2 font-semibold text-gray-600">ADS</th>
                      <th className="text-right px-3 py-2 font-semibold text-orange-600">PO Qty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedRows.map((r, i) => (
                      <tr key={i} className="border-b border-gray-100">
                        <td className="px-3 py-2 font-medium text-gray-900">{r['OMS_SKU']}</td>
                        <td className="px-3 py-2"><PriorityBadge priority={String(r['Priority'] ?? '')} /></td>
                        <td className="px-3 py-2 text-right"><DaysLeftBadge days={Number(r['Days_Left'] ?? 999)} /></td>
                        <td className="px-3 py-2 text-right text-gray-600">{Number(r['ADS'] ?? 0).toFixed(3)}</td>
                        <td className="px-3 py-2 text-right font-bold text-orange-600">{r.Final_PO_Qty.toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="bg-gray-50 border-t-2 border-gray-300">
                      <td colSpan={4} className="px-3 py-2 font-semibold text-gray-700">Total</td>
                      <td className="px-3 py-2 text-right font-bold text-orange-600 text-base">{totalRaiseUnits.toLocaleString()}</td>
                    </tr>
                  </tfoot>
                </table>
              )}
            </div>

            <div className="p-4 border-t border-gray-200 flex gap-3 justify-end">
              <button
                onClick={() => setRaiseModal(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 hover:bg-gray-50"
              >
                Cancel
              </button>
              {selectedRows.length > 0 && (
                <button
                  onClick={() => { exportRaisePO(selectedRows); setRaiseModal(false) }}
                  className="px-5 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700"
                >
                  ⬇ Export & Confirm PO
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Sub-components ──────────────────────────────────────────────

const PriorityBadge = memo(function PriorityBadge({ priority }: { priority: string }) {
  return <span className="font-semibold text-xs whitespace-nowrap">{priority || '⚪ OK'}</span>
})

const DaysLeftBadge = memo(function DaysLeftBadge({ days }: { days: number }) {
  const safe = Number.isFinite(days) ? days : 999
  if (safe >= 999) return <span className="text-gray-400">999+</span>
  const color = safe < 14 ? 'text-red-600 font-bold' : safe < 30 ? 'text-yellow-600 font-semibold' : 'text-gray-700'
  return <span className={color}>{Math.round(safe)}</span>
})

const QtyInput = memo(function QtyInput({
  value, computed, onChange, onReset,
}: {
  value: number
  computed: number
  onChange: (v: number) => void
  onReset: () => void
}) {
  const isEdited = value !== computed
  return (
    <div className="flex items-center gap-1 min-w-[90px]">
      <input
        type="number"
        value={value}
        min={0}
        step={5}
        onChange={e => onChange(Math.max(0, Math.round(+e.target.value / 5) * 5))}
        className={`w-20 border rounded px-2 py-1 text-xs text-right font-bold focus:outline-none focus:ring-1
          ${isEdited
            ? 'border-orange-400 bg-orange-50 text-orange-700 ring-orange-300 focus:ring-orange-400'
            : 'border-gray-300 bg-white text-orange-600 focus:ring-blue-300'
          }`}
      />
      {isEdited && (
        <button
          onClick={onReset}
          title={`Reset to ${computed}`}
          className="text-gray-300 hover:text-gray-500 text-sm leading-none"
        >
          ↩
        </button>
      )}
    </div>
  )
})

const KpiCard = memo(function KpiCard({ label, value, accent }: { label: string; value: number; accent?: string }) {
  return (
    <div className={`bg-white rounded-xl border border-gray-200 p-4 shadow-sm border-l-4 ${accent ?? 'border-l-[#002B5B]'}`}>
      <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-[#002B5B] mt-1">{value.toLocaleString()}</p>
    </div>
  )
})

function Param({
  label, type, value, min, max, step, onChange,
}: {
  label: string; type: string; value: number
  min?: number; max?: number; step?: number
  onChange: (v: string) => void
}) {
  return (
    <div>
      <label className="text-xs font-semibold text-gray-500 uppercase block mb-1">{label}</label>
      <input
        type={type} value={value} min={min} max={max} step={step}
        onChange={e => onChange(e.target.value)}
        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
      />
    </div>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 cursor-pointer text-sm text-gray-700">
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} className="rounded" />
      {label}
    </label>
  )
}

// ── Export helpers ──────────────────────────────────────────────

function exportPOCsv(
  rows: PORow[],
  editedQty: Record<string, number>,
  quarterCols: string[],
  quarterMap: Record<string, Record<string, number | string>>,
) {
  const base = [...PO_DISPLAY_COLS]
  const all  = [...base, ...quarterCols]
  const header = all.join(',')
  const body = rows.map(r => {
    const sku = String(r['OMS_SKU'])
    const qRow = quarterMap[poSkuKey(sku)] ?? {}
    return all.map(c => {
      if (c === 'PO_Qty') {
        const v = editedQty[sku] !== undefined ? editedQty[sku] : Number(r[c] ?? 0)
        return JSON.stringify(v)
      }
      if (quarterCols.includes(c)) return JSON.stringify(qRow[c] ?? 0)
      return JSON.stringify(r[c] ?? '')
    }).join(',')
  }).join('\n')
  trigger(header + '\n' + body, 'po_recommendation.csv')
}

function exportRaisePO(rows: Array<PORow & { Final_PO_Qty: number }>) {
  const cols  = ['OMS_SKU', 'Priority', 'Days_Left', 'ADS', 'Gross_PO_Qty', 'PO_Pipeline_Total', 'Final_PO_Qty']
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => JSON.stringify(r[c as keyof typeof r] ?? '')).join(',')).join('\n')
  trigger(header + '\n' + body, 'raise_po_' + new Date().toISOString().slice(0, 10) + '.csv')
}

function downloadQCsv(rows: QuarterlyRow[], columns: string[]) {
  const header = columns.join(',')
  const body   = rows.map(r => columns.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  trigger(header + '\n' + body, 'quarterly_history.csv')
}

function downloadShipmentCsv(rows: PORow[], columns: string[]) {
  const header = columns.join(',')
  const body   = rows.map(r => columns.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  trigger(header + '\n' + body, 'shipment_' + new Date().toISOString().slice(0, 10) + '.csv')
}

/** Match quarterly pivot keys (built uppercase) to inventory / PO SKU casing. */
function poSkuKey(sku: string | undefined | null) {
  return String(sku ?? '').trim().toUpperCase()
}

function trigger(csv: string, filename: string) {
  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
}
