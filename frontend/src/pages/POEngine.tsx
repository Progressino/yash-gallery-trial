import { useState, useMemo, useCallback, memo, useRef, useLayoutEffect, useEffect, type ReactNode } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams, Link } from 'react-router-dom'
import axios from 'axios'
import { api, getCoverage, getPoCalculateStatus, startPoCalculate, waitForPoCalculate } from '../api/client'
import { useSession } from '../store/session'
import { useAuth, mayResetSharedData } from '../store/auth'
import { usePOStore, type Tab } from '../store/po'
import { PODashboardPanel } from '../components/PODashboardPanel'
import { PageLoadingStripe } from '../components/LoadingProgressBar'
import { calendarDateIST, yesterdayIST } from '../lib/dates'
import { archivePoExportOnServer } from '../lib/archivePoExport'
import { looksLikePoExportCsv, pickPoExportCsvFromDownloads } from '../lib/pickPoExportCsv'

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
  worstPostPoCover: number
  totalSoldUnits: number
  totalADS: number
  totalGrossQty: number
  totalPOOrdered: number
  totalPendingCutting: number
  totalBalanceDispatch: number
  totalPipeline: number
  totalPipelineEffective: number
  totalConfirmedRaise: number
  totalRaisedToday: number
  totalRaisedYesterday: number
  totalFinalQty: number
  quarterTotals: Record<string, number>
  avgMonthly: number
  worstStatus: string
}

interface POSummary {
  new_po_qty_sum?: number
  new_po_sku_count?: number
  pipeline_qty_sum?: number
  pipeline_sku_count?: number
  sheet_po_ordered_sum?: number
  existing_po_applied?: boolean
  existing_po_generation?: number
  existing_po_filename?: string
}

interface POResult {
  ok: boolean
  message?: string
  rows?: PORow[]
  columns?: string[]
  sales_through?: string
  planning_date?: string
  raise_ledger_rows?: number
  ledger_auto_import?: string | null
  summary?: POSummary
}
interface PORiskRow extends PORow {
  risk_reasons: string
  risk_score: number
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
  status?: 'warming' | 'error'
  progress?: number
  message?: string
  columns?: string[]
  rows?: QuarterlyRow[]
}

const PO_DISPLAY_COLS = [
  'Priority', 'OMS_SKU', 'Bundle_Size', 'SKU_Sheet_Status', 'Lead_Time_Days', 'Total_Inventory', 'Days_Left',
  'Sold_Units', 'Return_Units', 'Return_Overlay_Units', 'Net_Units',
  'Ship_Units_150d', 'Eff_Days', 'Eff_Days_Inventory',
  'Recent_ADS', 'LY_ADS', 'Seasonal_Month_ADS', 'Flat30_ADS', 'ADS',
  'Cutting_Ratio', 'Gross_PO_Qty',
  'PO_Qty_Ordered', 'Pending_Cutting', 'Balance_to_Dispatch',
  'PO_Pipeline_Total',
  'PO_Raised_On_View_Date', 'PO_Last_Raised_Qty', 'PO_Last_Raised_Date',
  'PO_Raised_Yesterday', 'PO_Raised_Today', 'PO_Confirmed_Raise_Pipeline', 'PO_Pipeline_Effective',
  'Projected_Running_Days', 'Post_PO_Cover_Days_Capped', 'PO_Qty',
  'PO_Block_Reason', 'Suggest_Close_SKU',
]

const COL_LABEL: Record<string, string> = {
  'Bundle_Size':              '📦 Bundle size',
  'Sold_Units':               '📦 Sold Units',
  'Return_Units':             '↩️ Returns (sales)',
  'Return_Overlay_Units':     '↩️ Returns (upload)',
  'Net_Units':                '📦 Net sold',
  'Eff_Days':                 '📅 Eff. Days (active)',
  'Eff_Days_Inventory':       '📦 In-stock days (history)',
  'Recent_ADS':               '📉 Recent ADS',
  'LY_ADS':                   '📈 LY ADS',
  'Seasonal_Month_ADS':       '🌗 Season ADS (mo+1)',
  'Flat30_ADS':               '📅 Sheet FREQ (÷30)',
  'ADS':                      '⚡ ADS (Used)',
  'PO_Pipeline_Total':        '🏭 Total Pipeline',
  'PO_Raised_On_View_Date':  '📌 Raised on raise date',
  'PO_Last_Raised_Qty':      '📌 Last raised qty',
  'PO_Last_Raised_Date':     '📅 Last raised date',
  'PO_Raised_Yesterday':     '📌 Raised qty (yesterday)',
  'PO_Raised_Today':         '📌 Raised qty (today)',
  'PO_Confirmed_Raise_Pipeline': '📌 Confirmed raises (window)',
  'PO_Pipeline_Effective':   '🏭 Eff. pipeline (sheet + raises)',
  'PO_Qty_Ordered':           '📋 PO Ordered (on sheet)',
  'Pending_Cutting':          '✂️ Pend. Cutting',
  'Balance_to_Dispatch':      '📦 Bal. Dispatch',
  'Projected_Running_Days':   '📅 Proj. run (Tot inv + pipe) / ADS',
  'Post_PO_Cover_Days_Capped':'📏 Post-PO cover (actual)',
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

function poColHeaderLabel(col: string, raiseViewDate: string): ReactNode {
  if (col === 'PO_Qty') return <span title="New units to raise today (after pipeline deduction)">New PO Qty ✏️</span>
  if (col === 'PO_Raised_On_View_Date') {
    return (
      <span className="text-sky-800" title="Qty raised on the Raise date picker (recalculate after changing the date)">
        Raised {raiseViewDate}
      </span>
    )
  }
  const lbl = COL_LABEL[col]
  if (lbl) return <span>{lbl}</span>
  return col.replace(/_/g, ' ')
}

function renderRaiseLedgerCell(col: string, row: PORow): ReactNode {
  const raiseQtyCols = new Set([
    'PO_Raised_On_View_Date',
    'PO_Raised_Yesterday',
    'PO_Raised_Today',
    'PO_Confirmed_Raise_Pipeline',
    'PO_Pipeline_Effective',
  ])
  if (raiseQtyCols.has(col)) {
    const n = Number(row[col] ?? 0)
    if (col === 'PO_Pipeline_Effective') {
      const base = Number(row['PO_Pipeline_Total'] ?? 0)
      return (
        <span className="text-xs font-semibold text-teal-800">
          {n.toLocaleString()}
          {base > 0 && n > base ? (
            <span className="block text-[10px] text-gray-500 font-normal">sheet {base.toLocaleString()} + raises</span>
          ) : null}
        </span>
      )
    }
    return n > 0
      ? <span className="text-xs font-bold text-sky-900 bg-sky-50 border border-sky-200 px-2 py-0.5 rounded">{n.toLocaleString()}</span>
      : <span className="text-gray-300">—</span>
  }
  if (col === 'PO_Last_Raised_Qty') {
    const n = Number(row[col] ?? 0)
    const d = String(row['PO_Last_Raised_Date'] ?? '').trim()
    if (n <= 0 && !d) return <span className="text-gray-300">—</span>
    return (
      <span className="text-xs">
        <span className="font-semibold text-violet-900">{n > 0 ? n.toLocaleString() : '—'}</span>
        {d ? <span className="block text-[10px] text-gray-500 mt-0.5">{d}</span> : null}
      </span>
    )
  }
  if (col === 'PO_Last_Raised_Date') {
    const d = String(row[col] ?? '').trim()
    return d ? <span className="text-xs font-mono text-gray-600">{d}</span> : <span className="text-gray-300">—</span>
  }
  return null
}

const STATUS_COLORS: Record<string, string> = {
  'Fast Moving':  'text-green-700 bg-green-50',
  'Moderate':     'text-blue-700 bg-blue-50',
  'Slow Selling': 'text-yellow-700 bg-yellow-50',
  'Not Moving':   'text-red-600 bg-red-50',
}

const TAB_ORDER: Tab[] = ['po', 'dashboard', 'quarterly', 'shipment']

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

/** Match backend ``(Total_Inventory + PO_Pipeline_Effective + PO_Qty) / ADS`` (1 dp). */
function postPoCoverDays(row: PORow, finalPoQty: number): number {
  const ads = Number(row['ADS'] ?? 0)
  if (!Number.isFinite(ads) || ads <= 0) return 999
  const inv = Number(row['Total_Inventory'] ?? 0)
  const pipeEff = Number(row['PO_Pipeline_Effective'] ?? row['PO_Pipeline_Total'] ?? 0)
  const q = Math.max(0, Math.floor(finalPoQty))
  const v = (inv + pipeEff + q) / ads
  return Math.round(v * 10) / 10
}

export default function POEngine() {
  const queryClient = useQueryClient()
  const authUser = useAuth(s => s.user)
  const canDeleteRaiseSkus = mayResetSharedData(authUser)
  const setCoverage = useSession(s => s.setCoverage)
  const skuStatusLoaded = useSession(s => s.sku_status_lead ?? false)
  const skuStatusRows = useSession(s => s.sku_status_lead_rows ?? 0)
  const dailyInvLoaded = useSession(s => s.daily_inventory_history ?? false)
  const dailyInvRows = useSession(s => s.daily_inventory_history_rows ?? 0)
  const dailyInvSkus = useSession(s => s.daily_inventory_history_skus ?? 0)
  const raiseLedgerRows = useSession(s => s.po_raise_ledger_rows ?? 0)
  const existingPoLoaded = useSession(s => s.existing_po)
  const existingPoFilename = useSession(s => s.existing_po_filename)
  const existingPoUploadedAt = useSession(s => s.existing_po_uploaded_at)
  const existingPoNeedsRecalc = useSession(s => s.existing_po_needs_recalc ?? false)
  const existingPoRows = useSession(s => s.existing_po_rows ?? 0)
  const existingPoPerSizeSkus = useSession(s => s.existing_po_per_size_skus ?? 0)
  const existingPoLooksAggregated = useSession(s => s.existing_po_looks_aggregated ?? false)
  const [appBuildLabel, setAppBuildLabel] = useState<string | null>(null)

  const PO_MERGE_VERSION_KEY = 'po-merge-version-seen'

  useEffect(() => {
    api
      .get<{ git_sha?: string; label?: string; built_at?: string; po_merge_version?: number }>('/health')
      .then(r => {
        const sha = r.data.git_sha || r.data.label
        if (!sha) return
        const built = r.data.built_at ? ` · ${r.data.built_at.slice(0, 10)}` : ''
        setAppBuildLabel(`${sha}${built}`)
      })
      .catch(() => {})
  }, [])

  const { data: ledgerDatesResp } = useQuery({
    queryKey: ['po-raise-ledger-dates', raiseLedgerRows],
    queryFn: () => api.get<{ ok: boolean; dates: { date: string; sku_count: number; total_units: number }[] }>('/po/raise-ledger/dates').then(r => r.data),
    enabled: raiseLedgerRows > 0,
  })
  const ledgerDates = ledgerDatesResp?.dates ?? []
  const ledgerCsvRef = useRef<HTMLInputElement>(null)
  const returnFileRef = useRef<HTMLInputElement>(null)
  /** Bumps on each Calculate / quarterly load so stale async responses are ignored. */
  const poRunSeqRef = useRef(0)
  /** True once we've auto-retried a stuck/stale PO calculate, so we only retry once. */
  const poStaleRetryRef = useRef(false)
  // Per-SKU inventory history drill-down (lets the user verify the "Eff Days" math).
  const [effInvSku, setEffInvSku] = useState<string | null>(null)
  const [effInvLoading, setEffInvLoading] = useState(false)
  const [dailyInvMaxDate, setDailyInvMaxDate] = useState<string | null>(null)
  const [effInvData, setEffInvData] = useState<{
    sku: string
    canonical_sku?: string
    parent_used?: boolean
    window_days: number
    window_start: string
    window_end: string
    covered_days?: number
    uploaded_days?: number
    derived_days?: number
    in_stock_days: number
    out_of_stock_days: number
    in_stock_min_qty?: number
    rows: { date: string; qty: number; in_stock: boolean; source?: string }[]
  } | null>(null)

  const openEffInvDrawer = async (sku: string, windowDays: number) => {
    setEffInvSku(sku)
    setEffInvLoading(true)
    setEffInvData(null)
    try {
      const { data } = await api.get('/po/daily-inventory-history/sku', {
        params: {
          sku,
          window_days: Math.max(7, Math.min(365, windowDays || 30)),
          end_date: dailyInvMaxDate || calendarDateIST(),
        },
      })
      if (data?.ok) setEffInvData(data)
    } catch {
      setEffInvData(null)
    } finally {
      setEffInvLoading(false)
    }
  }
  const closeEffInvDrawer = () => { setEffInvSku(null); setEffInvData(null) }

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
    skipSharedCacheOnce, setSkipSharedCacheOnce,
  } = usePOStore()

  const [searchParams, setSearchParams] = useSearchParams()

  const syncTabFromUrl = useCallback(() => {
    const tab = searchParams.get('tab')
    if (tab === 'dashboard' || tab === 'quarterly' || tab === 'shipment') {
      setActiveTab(tab)
    }
  }, [searchParams, setActiveTab])

  useLayoutEffect(() => {
    syncTabFromUrl()
  }, [syncTabFromUrl])

  useEffect(() => {
    return usePOStore.persist.onFinishHydration(() => {
      syncTabFromUrl()
    })
  }, [syncTabFromUrl])

  const refreshPoCoverage = useCallback(async () => {
    try {
      const [cov, skuRes, invRes] = await Promise.all([
        getCoverage(),
        api.get<{ ok?: boolean; loaded?: boolean; rows?: unknown[] }>('/po/sku-status-lead'),
        api.get<{
          ok?: boolean
          loaded?: boolean
          rows?: number
          skus?: number
          max_date?: string
        }>('/po/daily-inventory-history'),
      ])
      const merged = { ...cov }
      if (skuRes.data?.loaded) {
        merged.sku_status_lead = true
        merged.sku_status_lead_rows = Array.isArray(skuRes.data.rows)
          ? skuRes.data.rows.length
          : merged.sku_status_lead_rows
      }
      if (invRes.data?.loaded) {
        merged.daily_inventory_history = true
        merged.daily_inventory_history_rows = invRes.data.rows ?? merged.daily_inventory_history_rows
        merged.daily_inventory_history_skus = invRes.data.skus ?? merged.daily_inventory_history_skus
        if (invRes.data.max_date) setDailyInvMaxDate(invRes.data.max_date)
      }
      setCoverage(merged)
    } catch {
      /* warm cache may still be loading — retry on interval */
    }
  }, [setCoverage])

  const selectTab = useCallback(
    (t: Tab) => {
      setActiveTab(t)
      if (t === 'po') {
        setSearchParams({}, { replace: true })
      } else {
        setSearchParams({ tab: t }, { replace: true })
      }
    },
    [setActiveTab, setSearchParams]
  )

  // Cast to local interfaces so downstream code keeps its named-field types
  const result   = _storeResult   as POResult | null
  const quarterly = _storeQuarterly as QuarterlyResult | null

  // ephemeral UI state (no need to persist across navigation)
  const [loading, setLoading] = useState(false)
  const [poProgress, setPoProgress] = useState('')
  const [poProgressPct, setPoProgressPct] = useState<number | null>(null)

  useEffect(() => {
    if (loading) return
    void refreshPoCoverage()
    const id = window.setInterval(() => {
      if (!loading) void refreshPoCoverage()
    }, 12_000)
    return () => window.clearInterval(id)
  }, [refreshPoCoverage, loading])

  /** Auto-resume if the server is still calculating (e.g. after gateway blip or page refresh). */
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const st = await getPoCalculateStatus()
        if (cancelled || st.status !== 'running') return
        const seq = ++poRunSeqRef.current
        setLoading(true)
        setPoProgress(st.message || 'Resuming PO calculation…')
        setPoProgressPct(typeof st.progress === 'number' ? st.progress : 10)
        const poRes = await waitForPoCalculate((msg, pct) => {
          if (cancelled || seq !== poRunSeqRef.current) return
          setPoProgress(msg)
          if (pct != null && Number.isFinite(pct)) setPoProgressPct(pct)
        })
        if (cancelled || seq !== poRunSeqRef.current || !poRes) return
        setResult(poRes as POResult)
        if (poRes.ledger_auto_import) {
          setLedgerImportMsg({ type: 'ok', text: poRes.ledger_auto_import })
        }
        void loadQuarterlyForRun(seq)
      } catch (e: unknown) {
        if (!cancelled) {
          const msg =
            e instanceof Error
              ? e.message
              : 'PO calculation failed. Wait a minute and try Calculate PO again.'
          if (!poStaleRetryRef.current && /stopped responding|timed out/i.test(msg)) {
            poStaleRetryRef.current = true
            setLoading(false)
            setPoProgress('')
            setPoProgressPct(null)
            void run()
            return
          }
          setResult({ ok: false, message: msg })
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
          setPoProgress('')
          setPoProgressPct(null)
        }
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- resume once on mount only
  }, [])
  /** Quarterly pivot loads after PO (or alone on Quarterly tab); was bundled with PO and blocked the UI. */
  const [quarterlyLoading, setQuarterlyLoading] = useState(false)
  const [quarterlyProgress, setQuarterlyProgress] = useState<number | null>(null)
  const [quarterlyLoadMessage, setQuarterlyLoadMessage] = useState<string | undefined>()
  const [shipLoading, setShipLoading] = useState(false)
  const [raiseModal, setRaiseModal] = useState(false)
  const [raiseConfirmBusy, setRaiseConfirmBusy] = useState(false)
  const [raiseConfirmErr, setRaiseConfirmErr] = useState<string | null>(null)
  const [clearLedgerBusy, setClearLedgerBusy] = useState(false)
  const [ledgerImportBusy, setLedgerImportBusy] = useState(false)
  const [ledgerImportMsg, setLedgerImportMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [returnImportBusy, setReturnImportBusy] = useState(false)
  const [returnImportMsg, setReturnImportMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)
  const [ledgerImportDate, setLedgerImportDate] = useState(() => yesterdayIST())
  const planningDate = calendarDateIST()
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

  const fetchQuarterlyWithPoll = async (seq: number): Promise<void> => {
    const paramsQ = { group_by_parent: params.group_by_parent, n_quarters: 8 }
    const maxPolls = 90
    for (let poll = 0; poll < maxPolls; poll++) {
      if (seq !== poRunSeqRef.current) return
      try {
        const { data } = await api.get<QuarterlyResult>('/po/quarterly', {
          params: paramsQ,
          timeout: 90_000,
        })
        if (seq !== poRunSeqRef.current) return
        if (data.status === 'warming' || (!data.loaded && !data.rows?.length && poll < maxPolls - 1)) {
          setQuarterlyProgress(
            typeof data.progress === 'number' ? Math.min(99, data.progress) : null,
          )
          setQuarterlyLoadMessage(data.message || 'Loading quarterly history…')
          await new Promise(r => setTimeout(r, 3000))
          continue
        }
        if (data.status === 'error') {
          setQuarterly({ loaded: false, rows: [], columns: [], message: data.message })
          return
        }
        setQuarterly(data)
        setQuarterlyProgress(null)
        setQuarterlyLoadMessage(undefined)
        return
      } catch (e: unknown) {
        const retry =
          axios.isAxiosError(e) &&
          (e.response?.status === 502 || e.code === 'ECONNABORTED') &&
          poll < maxPolls - 1
        if (retry) {
          await new Promise(r => setTimeout(r, 3000 + poll * 500))
          continue
        }
        console.warn('[PO] quarterly fetch failed:', e)
        setQuarterly({ loaded: false, rows: [], columns: [] })
        return
      }
    }
  }

  const loadQuarterlyForRun = async (seq: number) => {
    setQuarterlyLoading(true)
    setQuarterlyProgress(8)
    setQuarterlyLoadMessage('Loading quarterly history…')
    try {
      await fetchQuarterlyWithPoll(seq)
    } finally {
      if (seq === poRunSeqRef.current) {
        setQuarterlyLoading(false)
        setQuarterlyProgress(null)
        setQuarterlyLoadMessage(undefined)
      }
    }
  }

  /** Quarterly tab — polls until server cache is ready (avoids gateway timeouts). */
  const runQuarterlyOnly = async () => {
    const seq = ++poRunSeqRef.current
    setQuarterlyLoading(true)
    setQuarterlyProgress(8)
    setQuarterlyLoadMessage('Loading quarterly history…')
    try {
      await fetchQuarterlyWithPoll(seq)
    } finally {
      if (seq === poRunSeqRef.current) {
        setQuarterlyLoading(false)
        setQuarterlyProgress(null)
        setQuarterlyLoadMessage(undefined)
      }
    }
  }

  const quarterlyAutoFetch = useRef(false)
  useEffect(() => {
    if (activeTab !== 'quarterly') return
    if (quarterly?.loaded && (quarterly.rows?.length ?? 0) > 0) return
    if (quarterlyLoading || quarterlyAutoFetch.current) return
    quarterlyAutoFetch.current = true
    void runQuarterlyOnly()
  }, [activeTab, quarterly?.loaded, quarterly?.rows?.length, quarterlyLoading])

  const markPoTableStaleAfterLedgerChange = useCallback((serverMessage?: string) => {
    if (result?.ok && (result.rows?.length ?? 0) > 0) {
      setResult({
        ok: false,
        message:
          serverMessage ??
          'Raise ledger changed. Click Calculate PO to refresh PO Qty, pipeline, and In Production columns.',
      })
    }
    setEditedQty({})
    setSelected(new Set())
    setLedgerImportMsg({
      type: 'ok',
      text:
        serverMessage ??
        'Raise ledger updated. Click Calculate PO — the table still shows the previous run until you recalculate.',
    })
  }, [result, setResult, setEditedQty, setSelected])

  useEffect(() => {
    if (!existingPoNeedsRecalc && !existingPoLooksAggregated) return
    setSkipSharedCacheOnce(true)
    if (
      existingPoLooksAggregated &&
      result?.ok &&
      (result.rows?.length ?? 0) > 0
    ) {
      setResult({
        ok: false,
        message:
          'Existing PO looks bundled-only. Re-upload the full export on Upload, then Calculate PO.',
      })
    }
  }, [existingPoNeedsRecalc, existingPoLooksAggregated, result, setResult, setSkipSharedCacheOnce])

  const refreshRaiseLedger = useCallback(
    async (serverMessage?: string) => {
      const c = await getCoverage()
      setCoverage(c)
      void queryClient.invalidateQueries({ queryKey: ['po-raise-ledger-summary'] })
      void queryClient.invalidateQueries({ queryKey: ['po-raise-ledger-dates'] })
      markPoTableStaleAfterLedgerChange(serverMessage)
    },
    [queryClient, setCoverage, markPoTableStaleAfterLedgerChange, result],
  )

  const run = async (isRetry = false) => {
    const seq = ++poRunSeqRef.current
    if (!isRetry) poStaleRetryRef.current = false
    setLoading(true)
    setPoProgress(isRetry ? 'PO calculation stalled — restarting automatically…' : 'Starting PO calculation…')
    setPoProgressPct(2)
    setEditedQty({})
    setSelected(new Set())
    let poRes: POResult | null = null
    let staleRetry = false
    try {
      const useSharedCache = !skipSharedCacheOnce && !existingPoNeedsRecalc && !isRetry
      if (skipSharedCacheOnce) setSkipSharedCacheOnce(false)
      poRes = (await startPoCalculate(
        {
          ...params,
          planning_date: planningDate,
          raise_view_date: ledgerImportDate,
          raise_ledger_lookback_days: 14,
          use_shared_cache: useSharedCache,
        },
        (msg, pct) => {
          if (seq !== poRunSeqRef.current) return
          setPoProgress(msg)
          if (pct != null && Number.isFinite(pct)) setPoProgressPct(pct)
        },
      )) as POResult
      if (seq !== poRunSeqRef.current) return
      setResult(poRes)
    } catch (e: unknown) {
      if (seq === poRunSeqRef.current) {
        const msg =
          e instanceof Error
            ? e.message
            : 'PO calculation failed. If you saw 502, wait a minute and try again — the job may still finish on the server.'
        if (!poStaleRetryRef.current && /stopped responding|timed out/i.test(msg)) {
          poStaleRetryRef.current = true
          staleRetry = true
        } else {
          setResult({ ok: false, message: msg })
        }
      }
    } finally {
      if (seq === poRunSeqRef.current) {
        setLoading(false)
        setPoProgress('')
        setPoProgressPct(null)
      }
    }

    if (staleRetry && seq === poRunSeqRef.current) {
      await new Promise(r => setTimeout(r, 1500))
      if (seq === poRunSeqRef.current) void run(true)
      return
    }

    if (seq === poRunSeqRef.current && poRes?.ledger_auto_import) {
      setLedgerImportMsg({ type: 'ok', text: poRes.ledger_auto_import })
    }

    if (seq !== poRunSeqRef.current || !poRes?.ok) return
    void refreshPoCoverage()
    void loadQuarterlyForRun(seq)
  }

  /** After a PO-engine deploy, auto-load today's shared cache (replaces stale session tables). */
  useEffect(() => {
    if (activeTab !== 'po' || loading || existingPoNeedsRecalc) return
    let cancelled = false
    ;(async () => {
      try {
        const { data: health } = await api.get<{ po_merge_version?: number }>('/health')
        const ver = health.po_merge_version
        if (!ver || cancelled) return
        const seen = Number(sessionStorage.getItem(PO_MERGE_VERSION_KEY) || 0)
        if (seen >= ver) return
        const seq = ++poRunSeqRef.current
        const hasTable = Boolean(result?.ok && (result.rows?.length ?? 0) > 0)
        if (!hasTable) {
          setLoading(true)
          setPoProgress('PO engine updated — loading shared calculation…')
          setPoProgressPct(5)
        }
        const poRes = await startPoCalculate(
          {
            ...params,
            planning_date: planningDate,
            raise_view_date: ledgerImportDate,
            raise_ledger_lookback_days: 14,
            use_shared_cache: true,
          },
          (msg, pct) => {
            if (cancelled || seq !== poRunSeqRef.current) return
            if (!hasTable) {
              setPoProgress(msg)
              if (pct != null && Number.isFinite(pct)) setPoProgressPct(pct)
            }
          },
        )
        if (cancelled || seq !== poRunSeqRef.current) return
        if (poRes?.ok) {
          sessionStorage.setItem(PO_MERGE_VERSION_KEY, String(ver))
          setResult(poRes as POResult)
          void loadQuarterlyForRun(seq)
        }
      } catch {
        /* shared cache optional — operator can still click Calculate PO */
      } finally {
        if (!cancelled) {
          setLoading(false)
          setPoProgress('')
          setPoProgressPct(null)
        }
      }
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refresh once per po_merge_version bump
  }, [activeTab, existingPoNeedsRecalc, loading])

  const confirmRaiseAndExport = async (rows: Array<PORow & { Final_PO_Qty: number }>) => {
    setRaiseConfirmErr(null)
    setRaiseConfirmBusy(true)
    const payloadRows = rows
      .map(r => ({
        oms_sku: String(r['OMS_SKU'] ?? ''),
        qty: Math.max(0, Math.floor(Number(r.Final_PO_Qty ?? 0))),
      }))
      .filter(r => r.oms_sku.length > 0 && r.qty > 0)
    if (payloadRows.length === 0) {
      setRaiseConfirmBusy(false)
      setRaiseConfirmErr('No positive quantities to record.')
      return
    }
    try {
      const { data } = await api.post<{
        ok?: boolean
        message?: string
        po_number?: string
        raised_date?: string
        total_qty?: number
      }>('/po/raise-confirm', {
        rows: payloadRows,
        raised_date: planningDate,
        group_by_parent: params.group_by_parent,
      })
      if (!data?.ok) {
        setRaiseConfirmErr(data?.message || 'Could not save raise ledger.')
        return
      }
      const poNumber = data.po_number || `PO-${planningDate}`
      exportRaisePO(rows, poNumber)
      const { buildPoRaiseReportHtml, downloadPoRaiseReport } = await import('../utils/poRaiseReport')
      downloadPoRaiseReport(
        buildPoRaiseReportHtml({
          poNumber,
          raisedDate: data.raised_date || planningDate,
          rows,
          totalQty: data.total_qty ?? rows.reduce((s, r) => s + r.Final_PO_Qty, 0),
        }),
        poNumber,
      )
      setRaiseModal(false)
      const c = await getCoverage()
      setCoverage(c)
      await run()
    } catch (e: unknown) {
      setRaiseConfirmErr(e instanceof Error ? e.message : 'Raise ledger save failed')
    } finally {
      setRaiseConfirmBusy(false)
    }
  }

  const clearRaiseLedger = async () => {
    if (clearLedgerBusy) return
    const ok = window.confirm(
      `Clear the entire PO raise ledger (${raiseLedgerRows.toLocaleString()} SKU-day row${raiseLedgerRows === 1 ? '' : 's'})?\n\n` +
      'This removes confirmed raises from the app and the server database. The next Calculate PO ' +
      'will no longer treat those quantities as extra pipeline and may re-recommend them.',
    )
    if (!ok) return
    setClearLedgerBusy(true)
    try {
      const { data } = await api.delete<{ ok?: boolean; message?: string }>('/po/raise-ledger')
      await refreshRaiseLedger(data?.message)
    } catch (e: unknown) {
      console.warn('[PO] clear raise ledger failed:', e)
    } finally {
      setClearLedgerBusy(false)
    }
  }

  const importLedgerCsvFile = async (file: File, raisedDate: string) => {
    setLedgerImportBusy(true)
    setLedgerImportMsg(null)
    try {
      if (!looksLikePoExportCsv(file.name)) {
        const ok = window.confirm(
          `"${file.name}" does not look like a PO export CSV (expected e.g. po_recommendation.csv). Import anyway?`,
        )
        if (!ok) return
      }
      const fd = new FormData()
      fd.append('file', file)
      fd.append('raised_date', raisedDate)
      fd.append('group_by_parent', params.group_by_parent ? 'true' : 'false')
      fd.append('replace_day', 'true')
      const { data } = await api.post<{ ok?: boolean; message?: string }>(
        '/po/raise-ledger/import-file',
        fd,
        {
          timeout: 120_000,
          headers: { 'Content-Type': 'multipart/form-data' },
        },
      )
      if (data?.ok) {
        setLedgerImportMsg({ type: 'ok', text: data.message || 'Imported into raise ledger.' })
        const c = await getCoverage()
        setCoverage(c)
        await run()
      } else {
        setLedgerImportMsg({ type: 'err', text: data?.message || 'Import failed.' })
      }
    } catch (e: unknown) {
      setLedgerImportMsg({
        type: 'err',
        text: e instanceof Error ? e.message : 'Import failed.',
      })
    } finally {
      setLedgerImportBusy(false)
      if (ledgerCsvRef.current) ledgerCsvRef.current.value = ''
    }
  }

  const onLedgerCsvImport = async (files: FileList | null) => {
    const f = files?.[0]
    if (!f) return
    await importLedgerCsvFile(f, ledgerImportDate)
  }

  const importReturnFile = async (file: File) => {
    setReturnImportBusy(true)
    setReturnImportMsg(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('group_by_parent', params.group_by_parent ? 'true' : 'false')
      fd.append('replace', 'true')
      const { data } = await api.post<{ ok?: boolean; message?: string; sales_rebuild?: string }>(
        '/po/returns/import-file',
        fd,
        { timeout: 900_000, headers: { 'Content-Type': 'multipart/form-data' } },
      )
      if (data?.ok) {
        setReturnImportMsg({ type: 'ok', text: data.message || 'Returns imported.' })
        await run()
      } else {
        setReturnImportMsg({ type: 'err', text: data?.message || 'Import failed.' })
      }
    } catch (e: unknown) {
      setReturnImportMsg({ type: 'err', text: e instanceof Error ? e.message : 'Import failed.' })
    } finally {
      setReturnImportBusy(false)
      if (returnFileRef.current) returnFileRef.current.value = ''
    }
  }

  /** Opens Downloads (Chrome/Edge) or file picker with raise date = yesterday (IST). */
  const importYesterdayExportFromDownloads = async () => {
    const yday = yesterdayIST()
    setLedgerImportDate(yday)
    try {
      const picked = await pickPoExportCsvFromDownloads()
      if (picked) {
        await importLedgerCsvFile(picked, yday)
        return
      }
      setLedgerImportMsg({
        type: 'ok',
        text: `Raise date set to ${yday}. Choose yesterday's po_recommendation.csv (usually in Downloads).`,
      })
      ledgerCsvRef.current?.click()
    } catch (e: unknown) {
      setLedgerImportMsg({
        type: 'err',
        text: e instanceof Error ? e.message : 'Could not open file picker.',
      })
    }
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

  // After PO results load, fetch quarterly history if cache is empty (auto-retry inside loader).
  useEffect(() => {
    if (!result?.ok || quarterlyLoading || loading) return
    const rows = quarterly?.rows ?? []
    const hasValues =
      rows.length > 0 &&
      rows.some(r => quarterCols.some(c => Number((r as QuarterlyRow)[c] ?? 0) > 0))
    if (hasValues) return
    void loadQuarterlyForRun(poRunSeqRef.current)
  }, [result?.ok, loading, quarterly?.rows, quarterCols.length])

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

  const riskReviewRows = useMemo((): PORiskRow[] => {
    const src = rows
    if (!src.length) return []
    const invVals = src.map(r => Number(r['Total_Inventory'] ?? 0))
    const poVals = src.map(r => Number(r['PO_Qty'] ?? 0))
    const invSorted = [...invVals].sort((a, b) => a - b)
    const poSorted = [...poVals].sort((a, b) => a - b)
    const q = (arr: number[], p: number) => {
      if (!arr.length) return 0
      const idx = Math.min(arr.length - 1, Math.max(0, Math.floor(p * (arr.length - 1))))
      return arr[idx]
    }
    const highInventory = Math.max(150, q(invSorted, 0.9))
    const veryHighPO = Math.max(300, q(poSorted, 0.99))
    const po95 = Math.max(1, q(poSorted, 0.95))
    const inv95 = Math.max(1, q(invSorted, 0.95))

    const out: PORiskRow[] = []
    for (const r of src) {
      const reasons: string[] = []
      const inv = Number(r['Total_Inventory'] ?? 0)
      const ads = Number(r['ADS'] ?? 0)
      const po = Number(r['PO_Qty'] ?? 0)
      const pipelineQty = Number(r['PO_Pipeline_Total'] ?? 0)
      const daysLeft = Number(r['Days_Left'] ?? 999)
      const pr = String(r['Priority'] ?? '')
      const signals = [
        Number(r['Recent_ADS'] ?? 0),
        Number(r['LY_ADS'] ?? 0),
        Number(r['Seasonal_Month_ADS'] ?? 0),
        Number(r['Flat30_ADS'] ?? 0),
      ]
      const demandSignalsPos = signals.filter(v => Number.isFinite(v) && v > 0).length

      if (inv >= highInventory && ads <= 0) reasons.push('HIGH_INVENTORY_ZERO_ADS')
      if (pr === '🔴 URGENT' && po > 0 && demandSignalsPos <= 1) reasons.push('URGENT_LOW_CONFIDENCE_DEMAND')
      if (po >= veryHighPO) reasons.push('VERY_HIGH_PO_OUTLIER')
      if (pipelineQty > 0 && po > 0 && daysLeft >= 999) reasons.push('PIPELINE_PLUS_NEW_ORDER_WITH_999_DAYS')
      if (!reasons.length) continue

      let score = 0
      if (reasons.includes('HIGH_INVENTORY_ZERO_ADS')) score += 5
      if (reasons.includes('URGENT_LOW_CONFIDENCE_DEMAND')) score += 4
      if (reasons.includes('VERY_HIGH_PO_OUTLIER')) score += 3
      if (reasons.includes('PIPELINE_PLUS_NEW_ORDER_WITH_999_DAYS')) score += 2
      score += Math.min(5, po / po95)
      score += Math.min(3, inv / inv95)
      out.push({ ...r, risk_reasons: reasons.join(';'), risk_score: Number(score.toFixed(2)) })
    }
    return out.sort((a, b) => b.risk_score - a.risk_score)
  }, [rows])

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
          worstPostPoCover: 999,
          totalSoldUnits: 0, totalADS: 0, totalGrossQty: 0,
          totalPOOrdered: 0, totalPendingCutting: 0, totalBalanceDispatch: 0,
          totalPipeline: 0, totalPipelineEffective: 0, totalConfirmedRaise: 0,
          totalRaisedToday: 0, totalRaisedYesterday: 0,
          totalFinalQty: 0, quarterTotals: {}, avgMonthly: 0, worstStatus: '',
        })
      }
      const g = groupMap.get(parentSku)!
      g.variants.push({ ...row, finalQty })
      g.totalInventory      += Number(row['Total_Inventory'] ?? 0)
      g.worstDaysLeft        = Math.min(g.worstDaysLeft, Number(row['Days_Left'] ?? 999))
      g.worstProjectedDays   = Math.min(g.worstProjectedDays, Number(row['Projected_Running_Days'] ?? 999))
      g.worstPostPoCover     = Math.min(g.worstPostPoCover, postPoCoverDays(row, finalQty))
      g.totalSoldUnits      += Number(row['Sold_Units'] ?? 0)
      g.totalADS            += Number(row['ADS'] ?? 0)
      g.totalGrossQty       += Number(row['Gross_PO_Qty'] ?? 0)
      g.totalPOOrdered      += Number(row['PO_Qty_Ordered'] ?? 0)
      g.totalPendingCutting += Number(row['Pending_Cutting'] ?? 0)
      g.totalBalanceDispatch += Number(row['Balance_to_Dispatch'] ?? 0)
      g.totalPipeline       += Number(row['PO_Pipeline_Total'] ?? 0)
      g.totalPipelineEffective += Number(row['PO_Pipeline_Effective'] ?? row['PO_Pipeline_Total'] ?? 0)
      g.totalConfirmedRaise += Number(row['PO_Confirmed_Raise_Pipeline'] ?? 0)
      g.totalRaisedToday    += Number(row['PO_Raised_Today'] ?? 0)
      g.totalRaisedYesterday += Number(row['PO_Raised_Yesterday'] ?? 0)
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

  const poPageBusy =
    loading ||
    quarterlyLoading ||
    shipLoading ||
    raiseConfirmBusy ||
    clearLedgerBusy ||
    ledgerImportBusy ||
    effInvLoading

  const poPageBusyLabel = useMemo(() => {
    if (loading) return 'Calculating PO recommendations…'
    if (quarterlyLoading) return 'Loading quarterly history…'
    if (shipLoading) return 'Running shipment engine…'
    if (raiseConfirmBusy) return 'Confirming raises…'
    if (clearLedgerBusy) return 'Clearing raise ledger…'
    if (ledgerImportBusy) return 'Importing ledger CSV…'
    if (effInvLoading) return 'Refreshing effective inventory…'
    return undefined
  }, [
    loading,
    quarterlyLoading,
    shipLoading,
    raiseConfirmBusy,
    clearLedgerBusy,
    ledgerImportBusy,
    effInvLoading,
  ])

  return (
    <div className="p-6 space-y-6">
      <PageLoadingStripe
        active={poPageBusy}
        label={loading ? poProgress || poPageBusyLabel : poPageBusyLabel}
        percent={loading ? poProgressPct : null}
        className="sticky top-0 z-30 -mt-2 mb-2"
      />
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">🎯 PO Engine</h2>
        <p className="text-gray-400 text-sm mt-1">Calculate purchase orders with quarterly history inline.</p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200 overflow-x-auto flex-wrap sm:flex-nowrap">
        {TAB_ORDER.map(t => (
          <button
            key={t}
            type="button"
            onClick={() => selectTab(t)}
            className={`px-5 py-2.5 text-sm font-semibold rounded-t-lg border-b-2 transition-colors shrink-0
              ${activeTab === t
                ? 'border-[#002B5B] text-[#002B5B] bg-white'
                : 'border-transparent text-gray-500 hover:text-gray-700'}`}
          >
            {t === 'po'
              ? '🎯 PO Recommendation'
              : t === 'dashboard'
                ? '📋 PO Dashboard'
                : t === 'quarterly'
                  ? '📊 Quarterly History'
                  : '🚚 Shipment Engine'}
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
              <div>
                <label
                  className="text-xs font-semibold text-gray-500 uppercase block mb-1"
                  title="Target stock cover after PO release — editable; use 180+ for longer planning horizons"
                >
                  Post-PO Running Days
                </label>
                <input
                  type="number"
                  min={30}
                  step={1}
                  value={params.target_days}
                  onChange={e => setParams({
                    ...params,
                    target_days: Math.max(30, Math.round(+e.target.value || 30)),
                  })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-semibold"
                />
                <div className="flex flex-wrap gap-1 mt-1">
                  {[90, 120, 135, 150, 180, 210].map(days => (
                    <button
                      key={days}
                      type="button"
                      onClick={() => setParams({ ...params, target_days: days })}
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold border transition-colors ${
                        params.target_days === days
                          ? 'bg-[#002B5B] text-white border-[#002B5B]'
                          : 'bg-white text-gray-600 border-gray-300 hover:border-[#002B5B]'
                      }`}
                    >
                      {days}d
                    </button>
                  ))}
                </div>
              </div>
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
                <label className="text-xs font-semibold text-gray-500 uppercase block mb-1" title="When any size of a parent has running days below this threshold, all sibling sizes are included in the PO output">
                  All sizes below (days)
                </label>
                <input
                  type="number" min={0} step={5}
                  value={params.urgent_all_sizes_days}
                  onChange={e => setParams({ ...params, urgent_all_sizes_days: Math.max(0, +e.target.value) })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                  title="Show all sizes when any variant's running days is below this threshold"
                />
                <p className="text-[10px] text-gray-400 mt-0.5">Show all sizes when any variant runs low</p>
              </div>
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
              <Toggle
                label="Lead-time gate (hold PO while cover > lead time)"
                checked={!!params.enforce_lead_time_release_gate}
                onChange={v => setParams({ ...params, enforce_lead_time_release_gate: v })}
              />
            </div>

            <div className="mt-5 p-4 rounded-lg border border-slate-200 bg-slate-50/90 text-xs text-slate-800 space-y-2">
              <p className="font-semibold text-slate-900">PO baseline sheets</p>
              <p className="text-slate-700 leading-relaxed">
                SKU status / lead time and the wide <strong>daily inventory history</strong> matrix are uploaded on the{' '}
                <Link to="/upload" className="text-[#002B5B] font-semibold underline hover:text-blue-800">
                  Upload Data
                </Link>{' '}
                page → <strong>History &amp; setup</strong> → <em>PO Engine — baseline sheets</em> (Admin when the org lock is on).
              </p>
              <p className="text-slate-600">
                Status — SKU status:{' '}
                {skuStatusLoaded ? (
                  <span className="text-emerald-700 font-medium">✓ {skuStatusRows} rows</span>
                ) : (
                  <span>optional / not loaded</span>
                )}
                . Inventory matrix:{' '}
                {dailyInvLoaded ? (
                  <span className="text-emerald-700 font-medium">
                    ✓ {dailyInvRows.toLocaleString()} rows · {dailyInvSkus.toLocaleString()} SKUs
                    {dailyInvMaxDate ? (
                      <span className="text-slate-600 font-normal">
                        {' '}
                        · latest <strong>{dailyInvMaxDate}</strong>
                      </span>
                    ) : null}
                  </span>
                ) : (
                  <span>optional / not loaded</span>
                )}
                .
              </p>
            </div>

            <div className="mt-5 space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => void run()}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
                >
                  {loading ? '⏳ Running PO…' : '🎯 Calculate PO'}
                </button>
                {appBuildLabel ? (
                  <span
                    className="text-[11px] font-mono text-slate-600 bg-white border border-slate-200 rounded px-2 py-1"
                    title={`Server build ${appBuildLabel}`}
                  >
                    Build {appBuildLabel.length > 24 ? `${appBuildLabel.slice(0, 7)}…` : appBuildLabel}
                  </span>
                ) : null}
              </div>
              {existingPoLooksAggregated ? (
                <p className="text-[11px] text-red-900 bg-red-50 border border-red-300 rounded px-2 py-1.5 font-medium">
                  Existing PO in this session looks <strong>bundled-only</strong> ({existingPoPerSizeSkus.toLocaleString()} per-size SKUs vs{' '}
                  {existingPoRows.toLocaleString()} total). Re-upload the full export on{' '}
                  <Link to="/upload" className="underline font-semibold">Upload</Link>, then <strong>Calculate PO</strong> — otherwise only combined sizes (L-XL, S-M) appear with summed pipeline.
                </p>
              ) : null}
              {existingPoNeedsRecalc ? (
                <p className="text-[11px] text-amber-900 bg-amber-50 border border-amber-300 rounded px-2 py-1.5 font-medium">
                  Existing PO saved
                  {existingPoFilename ? (
                    <>
                      {' '}
                      (<strong>{existingPoFilename}</strong>
                      {existingPoUploadedAt ? (
                        <span className="font-normal text-amber-800">
                          {' '}
                          · {new Date(existingPoUploadedAt).toLocaleString()}
                        </span>
                      ) : null}
                      )
                    </>
                  ) : null}
                  {existingPoRows > 0 ? (
                    <span className="font-normal text-amber-800">
                      {' '}
                      · {existingPoRows.toLocaleString()} SKUs
                    </span>
                  ) : null}
                  . Click <strong>Calculate PO</strong> to refresh pipeline columns —{' '}
                  <strong>no re-upload needed</strong> until you load a newer file on Upload.
                </p>
              ) : existingPoLoaded ? (
                <p className="text-[11px] text-emerald-900 bg-emerald-50 border border-emerald-200 rounded px-2 py-1.5 font-medium">
                  Existing PO saved
                  {existingPoFilename ? (
                    <>
                      {' '}
                      — <strong>{existingPoFilename}</strong>
                    </>
                  ) : null}
                  {existingPoRows > 0 ? (
                    <span className="font-normal text-emerald-800">
                      {' '}
                      · {existingPoRows.toLocaleString()} SKUs
                    </span>
                  ) : null}
                  {existingPoUploadedAt ? (
                    <span className="font-normal text-emerald-700">
                      {' '}
                      ({new Date(existingPoUploadedAt).toLocaleString()})
                    </span>
                  ) : null}
                  . Stored on the server — no re-upload needed until you load a newer file on Upload.
                  {existingPoRows > 0 && existingPoRows < 5000 ? (
                    <span className="text-amber-900 font-medium">
                      {' '}
                      Sheet looks partial; upload the full export if counts look too low.
                    </span>
                  ) : null}
                </p>
              ) : (
                <p className="text-[11px] text-amber-800 bg-amber-50 border border-amber-200 rounded px-2 py-1">
                  No Existing PO sheet in this session — upload on <Link to="/upload" className="underline font-semibold">Upload → History &amp; setup</Link> first.
                </p>
              )}
              {loading && (
                <div className="mt-2 space-y-1 max-w-md">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                    <div
                      className="h-full rounded-full bg-[#002B5B] transition-[width] duration-300"
                      style={{ width: `${poProgressPct ?? 5}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 flex justify-between gap-2">
                    <span>
                      {poProgress || 'Computing recommendations from sales and inventory (large catalogs may take 5–15 minutes).'}
                    </span>
                    {poProgressPct != null ? (
                      <span className="tabular-nums text-slate-600 shrink-0">{poProgressPct}%</span>
                    ) : null}
                  </p>
                </div>
              )}
            </div>

            {result && !result.ok && (
              <p className="mt-3 text-sm text-red-600 bg-red-50 rounded p-2">{result.message}</p>
            )}
            {result?.ok && (result as { from_shared_cache?: boolean }).from_shared_cache && (
              <p className="mt-3 text-sm text-sky-800 bg-sky-50 border border-sky-200 rounded-lg px-3 py-2">
                Loaded a <strong>shared PO run</strong> from earlier today on this server (same planning date and
                settings). Click <strong>Calculate PO</strong> again with{' '}
                <code className="text-xs">use_shared_cache: false</code> only if you need a fresh recompute after
                uploading new data.
              </p>
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
                <KpiCard
                  label="New PO to raise"
                  value={totalPOUnits}
                  accent="border-l-[#002B5B]"
                  title="Fresh units to raise today (PO Qty). Already-ordered / pipeline units from your Existing PO sheet are separate."
                />
              </div>
              {result?.summary?.existing_po_applied ? (
                <p className="text-[11px] text-slate-700 bg-slate-50 border border-slate-200 rounded px-2 py-1.5">
                  <strong>{result.summary.existing_po_filename || existingPoFilename || 'Existing PO'}</strong>
                  {' '}is applied in this run:{' '}
                  <strong>{(result.summary.pipeline_sku_count ?? 0).toLocaleString()}</strong> SKUs with pipeline
                  ({(result.summary.pipeline_qty_sum ?? 0).toLocaleString()} units on sheet),{' '}
                  <strong>{(result.summary.sheet_po_ordered_sum ?? 0).toLocaleString()}</strong> units marked PO Ordered.
                  {' '}Engine recommends <strong>{(result.summary.new_po_qty_sum ?? totalPOUnits).toLocaleString()}</strong> additional units
                  ({(result.summary.new_po_sku_count ?? 0).toLocaleString()} SKUs) to reach your target cover.
                </p>
              ) : null}

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
                {riskReviewRows.length > 0 && (
                  <span className="text-xs text-red-700 font-medium bg-red-50 px-2 py-0.5 rounded-full border border-red-200">
                    ⚠ {riskReviewRows.length} flagged
                  </span>
                )}
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
                      onClick={() => { setRaiseConfirmErr(null); setRaiseModal(true) }}
                      className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700 shadow-sm"
                    >
                      🚀 Raise PO ({selected.size} SKU{selected.size > 1 ? 's' : ''}, {totalRaiseUnits.toLocaleString()} units)
                    </button>
                  )}
                  <input
                    ref={ledgerCsvRef}
                    type="file"
                    accept=".csv,.xlsx,.xls,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    className="hidden"
                    onChange={e => void onLedgerCsvImport(e.target.files)}
                  />
                  <label className="flex flex-col gap-0.5 text-xs text-gray-600">
                    <span className="flex items-center gap-1.5">
                      <span className="whitespace-nowrap">Raise date</span>
                      <input
                        type="date"
                        value={ledgerImportDate}
                        onChange={e => setLedgerImportDate(e.target.value)}
                        className="border border-gray-300 rounded px-2 py-1 text-xs w-[9.5rem]"
                      />
                    </span>
                    <span className="text-[10px] text-gray-400">Column &quot;Raised {ledgerImportDate}&quot; updates after Calculate PO</span>
                  </label>
                  <button
                    type="button"
                    disabled={ledgerImportBusy}
                    onClick={() => void importYesterdayExportFromDownloads()}
                    title="Sets raise date to yesterday (IST), opens your Downloads folder when the browser allows, then imports po_recommendation.csv into the raise ledger and recalculates PO."
                    className="text-xs px-3 py-1.5 rounded border border-violet-400 bg-violet-50 text-violet-900 font-semibold hover:bg-violet-100 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {ledgerImportBusy ? '…' : `📂 Import yesterday (${yesterdayIST()})`}
                  </button>
                  <button
                    type="button"
                    disabled={ledgerImportBusy}
                    onClick={() => ledgerCsvRef.current?.click()}
                    title="Record SKUs from a saved PO export (CSV or Excel). Export CSV also records the ledger for the Raise date."
                    className="text-xs px-3 py-1.5 rounded border border-sky-300 text-sky-800 hover:bg-sky-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {ledgerImportBusy ? '…' : '📥 Import raises (CSV / Excel)'}
                  </button>
                  <input
                    ref={returnFileRef}
                    type="file"
                    accept=".csv,.xlsx,.xls,.rar,.zip,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/x-rar-compressed,application/zip"
                    className="hidden"
                    onChange={e => {
                      const f = e.target.files?.[0]
                      if (f) void importReturnFile(f)
                    }}
                  />
                  <button
                    type="button"
                    disabled={returnImportBusy}
                    onClick={() => returnFileRef.current?.click()}
                    title="Upload Return Data.rar or CSV/Excel — merges Amazon, Myntra, Meesho, Flipkart returns; updates dashboard net sales."
                    className="text-xs px-3 py-1.5 rounded border border-orange-300 text-orange-900 hover:bg-orange-50 disabled:opacity-50"
                  >
                    {returnImportBusy ? '…' : '↩ Import returns'}
                  </button>
                  <button
                    onClick={() => exportPOCsv(rows, editedQty, quarterCols, quarterMap, ledgerImportDate)}
                    className="text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
                    title="Downloads CSV and records quantities in the raise ledger for the Raise date (uses PO Qty column)."
                  >
                    ⬇ Export CSV
                  </button>
                  {raiseLedgerRows > 0 && (
                    <button
                      type="button"
                      onClick={() => void clearRaiseLedger()}
                      disabled={clearLedgerBusy}
                      title="Clear all confirmed PO raises (session + server database). Use PO Dashboard → Delete day for one date only."
                      className="text-xs px-3 py-1.5 rounded border border-amber-300 text-amber-700 hover:bg-amber-50 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {clearLedgerBusy
                        ? 'Clearing…'
                        : `🧹 Clear raise ledger (${raiseLedgerRows.toLocaleString()})`}
                    </button>
                  )}
                  {riskReviewRows.length > 0 && (
                    <button
                      onClick={() => exportRiskReviewCsv(riskReviewRows)}
                      className="text-xs px-3 py-1.5 rounded border border-red-300 text-red-700 hover:bg-red-50"
                    >
                      ⚠ Export Review CSV
                    </button>
                  )}
                </div>
              </div>

              {ledgerImportMsg && (
                <div className={`text-xs rounded-lg px-3 py-2 border ${ledgerImportMsg.type === 'ok' ? 'bg-emerald-50 text-emerald-900 border-emerald-200' : 'bg-red-50 text-red-800 border-red-200'}`}>
                  {ledgerImportMsg.text}
                </div>
              )}
              {returnImportMsg && (
                <div
                  className={`text-xs rounded-lg px-3 py-2 border ${
                    returnImportMsg.type === 'ok'
                      ? 'bg-orange-50 text-orange-900 border-orange-200'
                      : 'bg-red-50 text-red-800 border-red-200'
                  }`}
                >
                  {returnImportMsg.text}
                  <span className="block text-[10px] text-gray-500 mt-1">
                    You can upload <strong>Return Data.rar</strong> directly (all marketplace files inside are merged).
                  </span>
                </div>
              )}

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

              {result?.ok && (result.sales_through || result.planning_date) && (
                <p className="text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2">
                  <strong>Data anchor:</strong> sales through <strong>{result.sales_through ?? '—'}</strong>
                  {result.planning_date ? (
                    <> · planning day <strong>{result.planning_date}</strong> (IST)</>
                  ) : null}
                  {result.sales_through && result.planning_date && result.sales_through < result.planning_date ? (
                    <> — no sales uploaded for the planning day yet; PO uses the last sales date, not calendar &quot;today&quot;.</>
                  ) : null}
                </p>
              )}

              {result?.ok && raiseLedgerRows === 0 && (
                <p className="text-xs text-rose-800 bg-rose-50 border border-rose-200 rounded-lg px-4 py-2">
                  <strong>Raise ledger is empty.</strong> Import yesterday&apos;s PO file (e.g.{' '}
                  <strong>Po Requirement 14-May-26.xlsx</strong>) via{' '}
                  <strong>Import raises (CSV / Excel)</strong> for the raise date (e.g. Saturday&apos;s file), then{' '}
                  <strong>Calculate PO</strong>. New exports via <strong>Export CSV</strong> or <strong>Export &amp; Confirm</strong> are recorded automatically.
                </p>
              )}

              {raiseLedgerRows > 0 && ledgerDates.length > 0 && (
                <div className="text-xs text-sky-900 bg-sky-50 border border-sky-200 rounded-lg px-4 py-2 space-y-2">
                  <p>
                    <strong>Raise ledger ({raiseLedgerRows.toLocaleString()} rows)</strong> — quantities on dates:{' '}
                    {ledgerDates.map(d => d.date).join(', ')}. Set <strong>Raise date</strong> to match, then{' '}
                    <strong>Calculate PO</strong> so column &quot;Raised {ledgerImportDate}&quot; fills in.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {ledgerDates.slice(0, 8).map(d => (
                      <button
                        key={d.date}
                        type="button"
                        onClick={() => {
                          setLedgerImportDate(d.date)
                          void run()
                        }}
                        className={`px-2 py-1 rounded border text-[11px] ${
                          d.date === ledgerImportDate
                            ? 'bg-sky-700 text-white border-sky-700'
                            : 'bg-white border-sky-300 hover:bg-sky-100'
                        }`}
                      >
                        {d.date} ({d.total_units.toLocaleString()} u)
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Pipeline info banner */}
              <div className="flex items-center gap-2 text-xs text-gray-500 bg-blue-50 border border-blue-100 rounded-lg px-4 py-2">
                <span>💡</span>
                <span>
                  <strong className="text-blue-700">🏭 In Production</strong> = units already ordered (from your PO sheet). {' '}
                  <strong>Gross PO Qty</strong> − <strong>In Production</strong> = <strong className="text-orange-600">PO Qty</strong> (net new order). {' '}
                  Edit <strong className="text-orange-600">PO Qty</strong> cells directly; <strong className="text-sky-800">Post-PO cover</strong> updates with your edited qty. Select SKUs and use <strong className="text-green-700">Raise PO</strong> → <strong>Export & Confirm</strong> to record raises in the ledger.
                  {' '}Plain <strong>Export CSV</strong> does not — use <strong>Import raises (CSV)</strong> for an older file.{' '}
                  <strong className="text-sky-800">Raise ledger:</strong> {raiseLedgerRows.toLocaleString()} SKU-day row(s) — confirmed qty feeds <strong>eff. pipeline</strong> and the PO Dashboard so SKUs are not double-released.
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
                          {poColHeaderLabel(c, ledgerImportDate)}
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
                                          : col === 'Post_PO_Cover_Days_Capped'
                                            ? <DaysLeftBadge days={postPoCoverDays(row, finalQty)} />
                                          : renderRaiseLedgerCell(col, row) ?? (
                                          col === 'PO_Qty'
                                            ? <QtyInput
                                                value={finalQty}
                                                computed={computedQty}
                                                onChange={v => setEditedQty({ ...editedQty, [sku]: v })}
                                                onReset={() => { const n = {...editedQty}; delete n[sku]; setEditedQty(n) }}
                                              />
                                            : col === 'Days_Left'
                                              ? <DaysLeftBadge days={Number(row[col] ?? 999)} />
                                            : (col === 'Eff_Days_Inventory' || col === 'Eff_Days')
                                              ? (
                                                  <button
                                                    type="button"
                                                    onClick={() => openEffInvDrawer(sku, Math.max(7, Math.round(Number(row['Eff_Days'] ?? 30))))}
                                                    title="Click to see the daily inventory history used for this Effective-Days calculation"
                                                    className={
                                                      'text-xs font-semibold px-2 py-0.5 rounded border ' +
                                                      (col === 'Eff_Days_Inventory'
                                                        ? (dailyInvLoaded
                                                            ? 'text-emerald-700 bg-emerald-50 border-emerald-200 hover:bg-emerald-100'
                                                            : 'text-gray-400 bg-gray-50 border-gray-200 hover:bg-gray-100')
                                                        : 'text-slate-700 bg-slate-50 border-slate-200 hover:bg-slate-100')
                                                    }
                                                  >
                                                    {typeof row[col] === 'number'
                                                      ? Number(row[col]).toLocaleString(undefined, { maximumFractionDigits: 1 })
                                                      : (row[col] ?? '—')}
                                                    <span className="ml-1 opacity-60">🔍</span>
                                                  </button>
                                                )
                                              : typeof row[col] === 'number'
                                                ? Number(row[col]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                                : row[col] ?? '—'
                                          )}
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
                          {poColHeaderLabel(c, ledgerImportDate)}
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
                                  : c === 'PO_Qty_Ordered' || c === 'Pending_Cutting' || c === 'Balance_to_Dispatch' || c === 'PO_Pipeline_Total' || c === 'PO_Pipeline_Effective'
                                    ? group.variants.length > 1
                                      ? <span className="text-[10px] text-gray-400 font-normal" title="Pipeline is per size — expand the group">per size ▼</span>
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
                                              : group.totalPipelineEffective > 0
                                                ? <span className="text-xs font-semibold text-teal-800">{group.totalPipelineEffective.toLocaleString()}</span>
                                                : <span className="text-gray-300">—</span>
                                  : c === 'PO_Confirmed_Raise_Pipeline'
                                    ? group.totalConfirmedRaise > 0
                                      ? <span className="text-xs font-bold text-sky-900">{group.totalConfirmedRaise.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'PO_Raised_Today'
                                    ? group.totalRaisedToday > 0
                                      ? <span className="text-xs font-bold text-sky-900">{group.totalRaisedToday.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'PO_Raised_Yesterday'
                                    ? group.totalRaisedYesterday > 0
                                      ? <span className="text-xs font-semibold text-sky-800">{group.totalRaisedYesterday.toLocaleString()}</span>
                                      : <span className="text-gray-300">—</span>
                                  : c === 'PO_Raised_On_View_Date' || c === 'PO_Last_Raised_Qty' || c === 'PO_Last_Raised_Date'
                                    ? '—'
                                  : c === 'Projected_Running_Days'
                                    ? <DaysLeftBadge days={group.worstProjectedDays} />
                                  : c === 'Post_PO_Cover_Days_Capped'
                                    ? <DaysLeftBadge days={group.worstPostPoCover} />
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
                                  : col === 'Post_PO_Cover_Days_Capped'
                                    ? <DaysLeftBadge days={postPoCoverDays(variant, finalQty)} />
                                  : renderRaiseLedgerCell(col, variant) ?? (
                                  col === 'PO_Qty'
                                    ? <QtyInput value={finalQty} computed={computedQty}
                                        onChange={v => setEditedQty({ ...editedQty, [sku]: v })}
                                        onReset={() => { const n = {...editedQty}; delete n[sku]; setEditedQty(n) }} />
                                  : col === 'Days_Left' ? <DaysLeftBadge days={Number(variant[col] ?? 999)} />
                                  : typeof variant[col] === 'number'
                                    ? Number(variant[col]).toLocaleString(undefined, { maximumFractionDigits: 3 })
                                    : variant[col] ?? '—'
                                  )}
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

      {/* ── PO Dashboard Tab ── */}
      {activeTab === 'dashboard' && (
        <PODashboardPanel
          embedded
          canDeleteRaiseSkus={canDeleteRaiseSkus}
          onRaiseLedgerChanged={(msg) => refreshRaiseLedger(msg)}
        />
      )}

      {/* ── Quarterly History Tab ── */}
      {activeTab === 'quarterly' && (
        <>
          <PageLoadingStripe
            active={quarterlyLoading}
            label={quarterlyLoadMessage ?? 'Loading quarterly history…'}
            percent={quarterlyProgress}
            className="mb-3"
          />
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
                {quarterlyLoading ? '⏳ Loading history…' : '📊 Reload Quarterly History'}
              </button>
              {quarterly && !quarterly.loaded && !quarterlyLoading && (
                <span className="text-sm text-red-500">
                  {quarterly.message || 'No data — build Sales first.'}
                </span>
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
                <p className="text-[11px] text-sky-900 bg-sky-50 border border-sky-200 rounded px-2 py-1.5 mt-2 leading-snug">
                  <strong>Export &amp; Confirm</strong> downloads the CSV and records these quantities in the session <strong>raise ledger</strong>.
                  The next <strong>Calculate PO</strong> adds them to <strong>effective pipeline</strong> (14-day lookback, with <strong>Raised yesterday</strong> / <strong>Raised today</strong> columns for visibility) so recommendations step down instead of repeating the same release daily.
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

            {raiseConfirmErr && (
              <div className="px-4 py-2 mx-4 mb-0 text-xs text-rose-800 bg-rose-50 border border-rose-200 rounded">
                {raiseConfirmErr}
              </div>
            )}

            <div className="p-4 border-t border-gray-200 flex gap-3 justify-end">
              <button
                onClick={() => setRaiseModal(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 hover:bg-gray-50"
              >
                Cancel
              </button>
              {selectedRows.length > 0 && (
                <button
                  type="button"
                  onClick={() => void confirmRaiseAndExport(selectedRows)}
                  disabled={raiseConfirmBusy}
                  className="px-5 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {raiseConfirmBusy ? 'Saving…' : '⬇ Export & Confirm PO'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Eff-Days inventory-history drill-down */}
      {effInvSku && (
        <div className="fixed inset-0 z-50 bg-black/40 flex items-center justify-center p-4" onClick={closeEffInvDrawer}>
          <div
            className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[85vh] flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-5 py-4 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-gray-900">📅 In-stock days · {effInvSku}</h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  Day-by-day inventory snapshots used by the engine to compute <code className="font-mono">Eff_Days</code>.
                  Days with on-hand qty ≥ <strong>{effInvData?.in_stock_min_qty ?? 1}</strong> count toward effective
                  days.
                </p>
              </div>
              <button onClick={closeEffInvDrawer} className="text-gray-400 hover:text-gray-700 text-xl leading-none">×</button>
            </div>

            <div className="px-5 py-3 border-b border-gray-100 bg-gray-50 text-xs text-gray-700 flex flex-wrap gap-x-6 gap-y-1">
              {effInvLoading ? (
                <span>Loading…</span>
              ) : !effInvData ? (
                <span className="text-amber-700">
                  No daily inventory history uploaded for this SKU. Upload the “Daily Inventory History” matrix on the{' '}
                  <Link to="/upload" className="underline font-medium text-amber-900">
                    Upload Data
                  </Link>{' '}
                  page (History &amp; setup → PO Engine baselines) to enable verification.
                </span>
              ) : (
                <>
                  <span><strong>Window:</strong> {effInvData.window_start} → {effInvData.window_end} ({effInvData.window_days}d)</span>
                  <span><strong>Sheet coverage:</strong> {effInvData.covered_days ?? effInvData.rows.length}d</span>
                  <span><strong>In-stock:</strong> <span className="text-emerald-700 font-semibold">{effInvData.in_stock_days}d</span></span>
                  <span><strong>Out-of-stock:</strong> <span className="text-rose-700 font-semibold">{effInvData.out_of_stock_days}d</span></span>
                  {(effInvData.derived_days ?? 0) > 0 && (
                    <span className="w-full text-sky-800 bg-sky-50 border border-sky-200 rounded px-2 py-1">
                      ℹ {effInvData.uploaded_days ?? 0} days from uploaded sheet, <strong>{effInvData.derived_days}</strong> days auto-derived from daily sales activity (shipments down, refunds back up). Upload the baseline matrix once on{' '}
                      <Link to="/upload" className="underline font-medium">
                        Upload → History &amp; setup
                      </Link>{' '}
                      — subsequent days roll forward automatically as sales come in.
                    </span>
                  )}
                  {(effInvData.covered_days ?? 0) < effInvData.window_days && (effInvData.covered_days ?? 0) > 0 && (
                    <span className="text-amber-700 w-full">
                      ⚠ Coverage only spans {effInvData.covered_days} of {effInvData.window_days} days. Engine extrapolates:
                      Eff_Days ≈ <strong>{Math.min(effInvData.window_days, Math.round(effInvData.in_stock_days * effInvData.window_days / Math.max(1, effInvData.covered_days ?? 1)))}</strong>
                      &nbsp;(in-stock rate × window).
                    </span>
                  )}
                  {effInvData.parent_used && (
                    <span className="text-amber-700">⚠ Showing parent-rollup (no exact-variant history found)</span>
                  )}
                </>
              )}
            </div>

            <div className="flex-1 overflow-auto">
              {effInvData && effInvData.rows.length > 0 ? (
                <table className="w-full text-xs">
                  <thead className="bg-white sticky top-0 border-b border-gray-200">
                    <tr>
                      <th className="text-left px-4 py-2 font-semibold text-gray-600">Date</th>
                      <th className="text-right px-4 py-2 font-semibold text-gray-600">On-hand qty</th>
                      <th className="text-left px-4 py-2 font-semibold text-gray-600">Source</th>
                      <th className="text-left px-4 py-2 font-semibold text-gray-600">Counted as in-stock?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {effInvData.rows.map(r => (
                      <tr key={r.date} className={r.in_stock ? '' : 'bg-rose-50/40'}>
                        <td className="px-4 py-1.5 font-mono text-gray-700">{r.date}</td>
                        <td className="px-4 py-1.5 text-right font-mono">{r.qty.toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
                        <td className="px-4 py-1.5">
                          {r.source === 'derived'
                            ? <span className="inline-flex items-center gap-1 text-sky-700 bg-sky-50 border border-sky-200 rounded px-1.5 py-0.5 text-[10px] font-semibold">auto · from sales</span>
                            : <span className="inline-flex items-center gap-1 text-gray-600 bg-gray-50 border border-gray-200 rounded px-1.5 py-0.5 text-[10px] font-semibold">uploaded</span>}
                        </td>
                        <td className="px-4 py-1.5">
                          {r.in_stock
                            ? <span className="text-emerald-700 font-semibold">✓ yes</span>
                            : <span className="text-rose-700 font-semibold">✗ OOS</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (!effInvLoading && (
                <div className="p-8 text-center text-sm text-gray-500">
                  No daily snapshots available for this SKU within the selected window.
                </div>
              ))}
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
  const color = safe < 14 ? 'text-red-600 font-bold' : safe < 45 ? 'text-yellow-600 font-semibold' : 'text-gray-700'
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

const KpiCard = memo(function KpiCard({
  label,
  value,
  accent,
  title,
}: {
  label: string
  value: number
  accent?: string
  title?: string
}) {
  return (
    <div
      className={`bg-white rounded-xl border border-gray-200 p-4 shadow-sm border-l-4 ${accent ?? 'border-l-[#002B5B]'}`}
      title={title}
    >
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
  raisedDate: string,
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
  const csv = header + '\n' + body
  const day = raisedDate || calendarDateIST()
  trigger(csv, `po_recommendation ${day}.csv`)
  void archivePoExportOnServer(csv, day)
}

function exportRaisePO(rows: Array<PORow & { Final_PO_Qty: number }>, poNumber: string) {
  const cols  = ['PO_No', 'OMS_SKU', 'Priority', 'Days_Left', 'ADS', 'Gross_PO_Qty', 'PO_Pipeline_Total', 'Final_PO_Qty']
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => {
    if (c === 'PO_No') return JSON.stringify(poNumber)
    return JSON.stringify(r[c as keyof typeof r] ?? '')
  }).join(',')).join('\n')
  const csv = header + '\n' + body
  const safeNo = poNumber.replace(/[^\w-]+/g, '_')
  const fname = `${safeNo}_raise.csv`
  trigger(csv, fname)
  void archivePoExportOnServer(csv, calendarDateIST())
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

function exportRiskReviewCsv(rows: PORiskRow[]) {
  const cols = [
    'OMS_SKU', 'risk_reasons', 'risk_score', 'Priority',
    'Total_Inventory', 'Days_Left',
    'ADS', 'Recent_ADS', 'LY_ADS', 'Seasonal_Month_ADS', 'Flat30_ADS',
    'Sold_Units', 'Ship_Units_150d',
    'PO_Pipeline_Total', 'Gross_PO_Qty', 'PO_Qty', 'Lead_Time_Days',
  ]
  const header = cols.join(',')
  const body = rows.map(r => cols.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  trigger(header + '\n' + body, 'po_review_shortlist.csv')
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
