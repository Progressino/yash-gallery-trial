/**
 * PO Engine — full columns + quarterly; results persist across tab switches.
 */
import axios from 'axios'
import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import api, {
  getCoverage,
  getCoverageResilient,
  getPoReadiness,
  getDataParity,
  getPoSharedCacheAvailability,
  loadPoCalculateResultFromSession,
  resumePoCalculateIfRunning,
  startPoCalculate,
  type CoverageResponse,
  type DataParityReport,
  type PoSharedCacheAvailability,
  type PoReadinessResponse,
} from '../api/client'
import { PageLoadingStripe } from '../components/LoadingProgressBar'
import { calendarDateIST } from '../lib/dates'
import { salesDataGapNeedsWarning } from '../lib/reportingDates'
import { InventoryStalenessBanner } from '../components/InventoryStalenessBanner'
import {
  PO_OPERATIONAL_TOTAL,
  poOperationalLoaded,
  poPageHydrationReady,
} from '../lib/localSessionHint'
import {
  countQuarterColumns,
  EXPECTED_QUARTER_COLS,
  exportPoCsv,
  finalPoQtyForSku,
  formatPoCell,
  poCellClass,
  poColHeaderLabel,
  poColHelpText,
  poSkuKey,
  PRIORITY_ORDER,
  quarterColumnsFromApi,
  visiblePoColumns,
} from '../lib/poDisplay'
import { confirmPoRaiseOnServer, exportRaisePoCsv, type PoRaiseRow } from '../lib/poRaiseActions'
import { buildPoClientReportHtml, downloadPoClientReport } from '../utils/poClientReport'
import { PoQtyInput } from '../components/PoQtyInput'
import { PODashboardPanel } from '../components/PODashboardPanel'
import { mayResetSharedData, useAuth } from '../store/auth'
import {
  usePOFreshStore,
  type POFreshParams,
} from '../store/poFresh'
import './POFresh.css'

type Row = Record<string, string | number | undefined>

type QuarterlyResult = {
  loaded?: boolean
  status?: string
  progress?: number
  message?: string
  columns?: string[]
  rows?: Row[]
}


type Params = POFreshParams

const PAGE_SIZE = 100

function countLoaded(c: CoverageResponse) {
  const loaded = poOperationalLoaded(c)
  return {
    loaded,
    total: PO_OPERATIONAL_TOTAL,
    ready: poPageHydrationReady(c) || loaded === PO_OPERATIONAL_TOTAL,
  }
}

function num(v: unknown): number {
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}

function buildBody(p: Params, opts?: { useSharedCache?: boolean }) {
  return {
    period_days: p.period_days,
    lead_time: p.lead_time,
    target_days: p.target_days,
    grace_days: p.grace_days,
    demand_basis: p.demand_basis,
    group_by_parent: p.group_by_parent,
    safety_pct: p.safety_pct,
    use_seasonality: p.use_seasonality,
    use_ly_fallback: p.use_ly_fallback,
    seasonal_weight: 0.5,
    enforce_two_size_minimum: p.enforce_two_size_minimum,
    enforce_lead_time_release_gate: true,
    urgent_all_sizes_days: 45,
    planning_date: calendarDateIST(),
    raise_ledger_lookback_days: p.raise_ledger_lookback_days,
    raise_view_date: p.raise_view_date.trim() || undefined,
    auto_import_yesterday_ledger: true,
    use_shared_cache: opts?.useSharedCache ?? true,
  }
}

function priorityMeta(priority: unknown): { dot: string; label: string; cls: string } {
  const p = String(priority ?? '')
  if (p.includes('URGENT')) return { dot: 'bg-[var(--po-error)]', label: 'Urgent', cls: 'po-fresh-priority-urgent' }
  if (p.includes('HIGH')) return { dot: 'bg-amber-500', label: 'High', cls: 'po-fresh-priority-high' }
  if (p.includes('MEDIUM')) return { dot: 'bg-[var(--po-primary)]', label: 'Medium', cls: 'po-fresh-priority-medium' }
  if (p.includes('Pipeline')) return { dot: 'bg-[var(--po-secondary)]', label: 'In Pipeline', cls: 'po-fresh-priority-pipeline' }
  return { dot: 'bg-[var(--po-outline)]', label: p || '—', cls: 'text-[var(--po-outline)]' }
}

class POFreshErrorBoundary extends Component<
  { children: ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null }

  static getDerivedStateFromError(error: unknown) {
    return {
      error: error instanceof Error ? error.message : String(error ?? 'unknown'),
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="po-fresh-shell p-6 max-w-xl">
          <h1 className="po-font-display text-lg font-semibold text-[var(--po-error)]">PO page error</h1>
          <p className="mt-2 text-sm font-mono break-all">{this.state.error}</p>
          <button
            type="button"
            className="mt-4 px-4 py-2 rounded-lg po-fresh-btn-secondary text-sm"
            onClick={() => this.setState({ error: null })}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

function POFreshInner() {
  const authUser = useAuth(s => s.user)
  const canDeleteRaiseSkus = mayResetSharedData(authUser)
  const params = usePOFreshStore(s => s.params)
  const setParams = usePOFreshStore(s => s.setParams)
  const tab = usePOFreshStore(s => s.tab)
  const setTab = usePOFreshStore(s => s.setTab)
  const result = usePOFreshStore(s => s.result)
  const setResult = usePOFreshStore(s => s.setResult)
  const quarterly = usePOFreshStore(s => s.quarterly)
  const setQuarterly = usePOFreshStore(s => s.setQuarterly)
  const search = usePOFreshStore(s => s.search)
  const setSearch = usePOFreshStore(s => s.setSearch)
  const priorityFilter = usePOFreshStore(s => s.priorityFilter)
  const setPriorityFilter = usePOFreshStore(s => s.setPriorityFilter)
  const page = usePOFreshStore(s => s.page)
  const setPage = usePOFreshStore(s => s.setPage)
  const sortByPriority = usePOFreshStore(s => s.sortByPriority)
  const setSortByPriority = usePOFreshStore(s => s.setSortByPriority)
  const fromSharedCache = usePOFreshStore(s => s.fromSharedCache)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [dataStatus, setDataStatus] = useState<{ loaded: number; total: number; ready: boolean } | null>(
    null,
  )
  const [coverageDetail, setCoverageDetail] = useState<CoverageResponse | null>(null)
  const [pipelineReadiness, setPipelineReadiness] = useState<PoReadinessResponse | null>(null)
  const [checkingData, setCheckingData] = useState(false)
  const [coverageError, setCoverageError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<{ msg: string; pct: number | null } | null>(null)
  const [quarterlyLoading, setQuarterlyLoading] = useState(false)
  const [quarterlyMsg, setQuarterlyMsg] = useState<string>()
  const [quarterlyPage, setQuarterlyPage] = useState(0)
  const progressThrottle = useRef(0)
  const calcRunSeq = useRef(0)
  const [actionMsg, setActionMsg] = useState<string | null>(null)
  const [serverEngineVersion, setServerEngineVersion] = useState<number | null>(null)
  const [sharedCacheHint, setSharedCacheHint] = useState<PoSharedCacheAvailability | null>(null)
  const [parityReport, setParityReport] = useState<DataParityReport | null>(null)
  const [editedQty, setEditedQty] = useState<Record<string, number>>({})
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [raiseModal, setRaiseModal] = useState(false)
  const [raiseConfirmBusy, setRaiseConfirmBusy] = useState(false)
  const [raiseConfirmErr, setRaiseConfirmErr] = useState<string | null>(null)

  const quarterCols = useMemo(
    () => quarterColumnsFromApi(quarterly?.columns),
    [quarterly?.columns],
  )

  const quarterMap = useMemo(() => {
    const map: Record<string, Record<string, number | string>> = {}
    for (const row of quarterly?.rows ?? []) {
      const key = poSkuKey(String(row.OMS_SKU))
      map[key] = {}
      for (const c of quarterCols) map[key][c] = Number(row[c] ?? 0)
    }
    return map
  }, [quarterly?.rows, quarterCols])

  const refreshDataStatus = useCallback(async () => {
    setCheckingData(true)
    setCoverageError(null)
    try {
      const c = await getCoverageResilient({ light: true, timeout: 90_000 })
      setCoverageDetail(c)
      setDataStatus(countLoaded(c))
      try {
        const pr = await getPoReadiness({ timeout: 30_000 })
        setPipelineReadiness(pr)
      } catch {
        setPipelineReadiness(null)
      }
      try {
        const parity = await getDataParity(calendarDateIST())
        setParityReport(parity)
      } catch {
        setParityReport(null)
      }
    } catch (e: unknown) {
      setCoverageError(e instanceof Error ? e.message : 'Failed to read data coverage')
    } finally {
      setCheckingData(false)
    }
  }, [])

  const onProgress = useCallback((msg: string, pct?: number) => {
    const now = Date.now()
    if (pct != null && pct >= 100) {
      setProgress({ msg, pct })
      return
    }
    if (now - progressThrottle.current < 300) return
    progressThrottle.current = now
    setProgress({ msg, pct: pct ?? null })
  }, [])

  useEffect(() => {
    const loadVer = () => {
      void api
        .get<{ po_merge_version?: number }>('/health')
        .then(r => {
          if (r.data.po_merge_version != null) setServerEngineVersion(r.data.po_merge_version)
        })
        .catch(() => {})
    }
    loadVer()
    const id = window.setInterval(loadVer, 20_000)
    const onVis = () => {
      if (document.visibilityState === 'visible') loadVer()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      window.clearInterval(id)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [])

  useEffect(() => {
    void refreshDataStatus()
    // Existing PO attaches async after hydrate-warm — retry briefly so the chip updates.
    const id = window.setInterval(() => void refreshDataStatus(), 3000)
    const stop = window.setTimeout(() => window.clearInterval(id), 30_000)
    return () => {
      window.clearInterval(id)
      window.clearTimeout(stop)
    }
  }, [refreshDataStatus])

  const existingPoUploadBusy = coverageDetail?.existing_po_upload_status === 'running'

  /** Hint when a matching shared PO run exists (same planning date + settings). */
  useEffect(() => {
    if (!dataStatus?.ready || loading) return
    let cancelled = false
    void (async () => {
      try {
        const hint = await getPoSharedCacheAvailability(buildBody(params))
        if (!cancelled) setSharedCacheHint(hint)
      } catch {
        if (!cancelled) setSharedCacheHint(null)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [dataStatus?.ready, loading, params])

  /** Restore finished PO from server session only — never block the Calculate button. */
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const resumed = await resumePoCalculateIfRunning(onProgress)
        if (cancelled) return
        if (resumed?.ok) {
          setResult(resumed)
          return
        }
        const session = await loadPoCalculateResultFromSession(onProgress)
        if (cancelled) return
        if (session?.ok && (session.rows?.length ?? 0) > 0) {
          setResult(session)
        }
      } catch {
        /* optional — user can still click Calculate */
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only session restore
  }, [])

  /** After a backend deploy, prompt recalculate — do not auto-run (OOM on large local catalogs). */
  useEffect(() => {
    if (!dataStatus?.ready || !serverEngineVersion || !result?.ok) return
    const resultVer = result.po_merge_version ?? 0
    if (resultVer >= serverEngineVersion) return
    const tag = `PO engine updated to v${serverEngineVersion}`
    if (result.message?.includes(tag)) return
    setResult({
      ...result,
      message: (result.message ? `${result.message} ` : '') + `${tag} — click Calculate PO to refresh.`,
    })
  }, [dataStatus?.ready, result, serverEngineVersion, setResult])

  const loadQuarterly = useCallback(async () => {
    setQuarterlyLoading(true)
    setQuarterlyMsg('Loading quarterly history…')
    setQuarterly(null)
    try {
      const maxPolls = 90
      for (let poll = 0; poll < maxPolls; poll++) {
        const { data } = await api.get<QuarterlyResult>('/po/quarterly', {
          params: { group_by_parent: params.group_by_parent, n_quarters: 8 },
          timeout: 90_000,
        })
        if (data.status === 'warming' || (!data.loaded && !data.rows?.length && poll < maxPolls - 1)) {
          setQuarterlyMsg(data.message || 'Building quarterly history…')
          await new Promise(r => setTimeout(r, 3000))
          continue
        }
        if (data.status === 'error') {
          setQuarterly({ loaded: false, rows: [], columns: [], message: data.message })
          return
        }
        const qCount = countQuarterColumns(data.columns)
        if (data.loaded && qCount < EXPECTED_QUARTER_COLS && poll < maxPolls - 1) {
          setQuarterlyMsg(
            data.message || `Quarterly incomplete (${qCount}/${EXPECTED_QUARTER_COLS}) — retrying…`,
          )
          await new Promise(r => setTimeout(r, 3000))
          continue
        }
        setQuarterly(data)
        setQuarterlyMsg(undefined)
        setQuarterlyPage(0)
        return
      }
      setQuarterly({ loaded: false, message: 'Quarterly load timed out' })
    } catch (e: unknown) {
      const retry =
        axios.isAxiosError(e) &&
        (e.response?.status === 502 || e.code === 'ECONNABORTED')
      setQuarterly({
        loaded: false,
        message: retry
          ? 'Server busy — try Load quarterly again in a minute.'
          : e instanceof Error
            ? e.message
            : 'Quarterly fetch failed',
      })
    } finally {
      setQuarterlyLoading(false)
    }
  }, [params.group_by_parent, setQuarterly])

  /** Load quarterly history after PO calculate or session restore (inline quarter columns). */
  useEffect(() => {
    if (!result?.ok || quarterlyLoading) return
    if (quarterly?.loaded && (quarterly.rows?.length ?? 0) > 0 && quarterCols.length >= EXPECTED_QUARTER_COLS) {
      return
    }
    void loadQuarterly()
  }, [result?.ok, result?.row_count, result?.rows?.length, quarterlyLoading, quarterly?.loaded, quarterly?.rows?.length, quarterCols.length, loadQuarterly])

  const runCalculate = useCallback(async (opts?: { force?: boolean }) => {
    const force = Boolean(opts?.force)
    const seq = ++calcRunSeq.current
    setLoading(true)
    setActionMsg(force ? 'Force recalculating PO…' : 'Starting PO calculation…')
    setProgress({ msg: 'Checking data coverage…', pct: 2 })
    setResult(null)
    setQuarterly(null)
    setPage(0)
    try {
      let status = dataStatus
      if (!status?.ready) {
        const c = await getCoverage({ light: true, timeout: 45_000 })
        if (seq !== calcRunSeq.current) return
        setCoverageDetail(c)
        status = countLoaded(c)
        setDataStatus(status)
      }
      if (!status.ready) {
        setResult({
          ok: false,
          message: `Data not ready (${status.loaded}/${status.total}). Sidebar → Reload from GitHub, then retry.`,
        })
        return
      }
      try {
        const pr = await getPoReadiness({ timeout: 30_000 })
        setPipelineReadiness(pr)
        if (pr.calculate_allowed === false && pr.pipeline_blockers?.length) {
          setResult({
            ok: false,
            message: pr.pipeline_blockers[0],
          })
          return
        }
      } catch {
        /* readiness optional — server gate still applies */
      }
      const useSharedCache = !force
      setProgress({
        msg: useSharedCache
          ? 'Checking shared PO cache, then calculating if needed…'
          : 'Starting fresh PO calculation on server…',
        pct: 5,
      })
      setEditedQty({})
      setSelected(new Set())
      const out = await startPoCalculate(
        buildBody(params, { useSharedCache }),
        (msg, pct) => {
          if (seq !== calcRunSeq.current) return
          onProgress(msg, pct)
        },
      )
      if (seq !== calcRunSeq.current) return
      setResult(out)
      if (!out.ok) {
        setActionMsg(out.message || 'PO calculation failed')
      } else {
        setSharedCacheHint(null)
        if (!force) {
          void getPoSharedCacheAvailability(buildBody(params))
            .then(setSharedCacheHint)
            .catch(() => setSharedCacheHint(null))
        }
      }
    } catch (e: unknown) {
      if (seq !== calcRunSeq.current) return
      const message = e instanceof Error ? e.message : 'PO calculation failed'
      setResult({ ok: false, message })
      setActionMsg(message)
    } finally {
      if (seq === calcRunSeq.current) {
        setLoading(false)
        setProgress(null)
        setActionMsg(null)
      }
    }
  }, [dataStatus, onProgress, params, setPage, setQuarterly, setResult])

  const allRows = useMemo(() => {
    const rows = (result?.rows ?? []) as Row[]
    let filtered = rows
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      filtered = filtered.filter(r => String(r.OMS_SKU ?? '').toLowerCase().includes(q))
    }
    if (priorityFilter === 'urgent') {
      filtered = filtered.filter(r => String(r.Priority ?? '').includes('URGENT'))
    } else if (priorityFilter === 'with_po') {
      filtered = filtered.filter(r => num(r.PO_Qty) > 0)
    } else if (priorityFilter === 'blocked') {
      filtered = filtered.filter(r => {
        const st = String(r.SKU_Sheet_Status ?? '').toLowerCase()
        return st.includes('closed') || st.includes('doubt') || st.includes('sales after closed')
      })
    }
    if (!sortByPriority) return filtered
    return [...filtered].sort(
      (a, b) =>
        (PRIORITY_ORDER[String(a.Priority)] ?? 9) - (PRIORITY_ORDER[String(b.Priority)] ?? 9),
    )
  }, [result?.rows, search, sortByPriority, priorityFilter])

  const pageRows = useMemo(() => {
    const start = page * PAGE_SIZE
    return allRows.slice(start, start + PAGE_SIZE)
  }, [allRows, page])

  const totalPO = useMemo(() => {
    const server = result?.summary?.new_po_qty_sum
    if (typeof server === 'number' && Number.isFinite(server)) return server
    return allRows.reduce((s, r) => s + num(r.PO_Qty), 0)
  }, [allRows, result?.summary?.new_po_qty_sum])

  const kpis = useMemo(
    () => ({
      urgent: allRows.filter(r => r.Priority === '🔴 URGENT').length,
      high: allRows.filter(r => r.Priority === '🟡 HIGH').length,
      medium: allRows.filter(r => r.Priority === '🟢 MEDIUM').length,
      pipeline: allRows.filter(r => r.Priority === '🔄 In Pipeline').length,
      withPo: allRows.filter(r => num(r.PO_Qty) > 0).length,
      pipelineUnits:
        typeof result?.summary?.pipeline_qty_sum === 'number' &&
        Number.isFinite(result.summary.pipeline_qty_sum)
          ? result.summary.pipeline_qty_sum
          : allRows.reduce((s, r) => s + num(r.PO_Pipeline_Total), 0),
      grossPo: allRows.reduce((s, r) => s + num(r.Gross_PO_Qty), 0),
    }),
    [allRows],
  )

  const dashboardParams = useMemo(() => buildBody(params), [params])

  const filteredQuarterlyRows = useMemo(() => {
    const rows = quarterly?.rows ?? []
    const q = search.trim().toLowerCase()
    if (!q) return rows
    return rows.filter(r => String(r.OMS_SKU ?? '').toLowerCase().includes(q))
  }, [quarterly?.rows, search])

  const quarterRowCount = filteredQuarterlyRows.length
  const quarterPageCount = Math.max(1, Math.ceil(quarterRowCount / PAGE_SIZE))
  const quarterPageRows = useMemo(() => {
    const start = quarterlyPage * PAGE_SIZE
    return filteredQuarterlyRows.slice(start, start + PAGE_SIZE)
  }, [filteredQuarterlyRows, quarterlyPage])

  const tableCols = useMemo(
    () => visiblePoColumns(result?.columns, quarterCols),
    [result?.columns, quarterCols],
  )

  const pageCount = Math.max(1, Math.ceil(allRows.length / PAGE_SIZE))

  const visibleSkus = useMemo(() => pageRows.map(r => String(r.OMS_SKU ?? '')), [pageRows])
  const allVisibleSelected =
    visibleSkus.length > 0 && visibleSkus.every(s => selected.has(s))
  const someSelected = selected.size > 0

  const toggleRowSelect = useCallback((sku: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(sku)) next.delete(sku)
      else next.add(sku)
      return next
    })
  }, [])

  const toggleAllVisible = useCallback(() => {
    setSelected(prev => {
      const next = new Set(prev)
      if (visibleSkus.length > 0 && visibleSkus.every(s => next.has(s))) {
        visibleSkus.forEach(s => next.delete(s))
      } else {
        visibleSkus.forEach(s => next.add(s))
      }
      return next
    })
  }, [visibleSkus])

  const selectedRows = useMemo((): PoRaiseRow[] => {
    return allRows
      .filter(r => selected.has(String(r.OMS_SKU ?? '')))
      .map(r => ({ ...r, Final_PO_Qty: finalPoQtyForSku(r, editedQty) }))
      .filter(r => r.Final_PO_Qty > 0)
  }, [allRows, selected, editedQty])

  const totalRaiseUnits = useMemo(
    () => selectedRows.reduce((s, r) => s + r.Final_PO_Qty, 0),
    [selectedRows],
  )

  const exportCsv = useCallback(() => {
    const poCols = visiblePoColumns(result?.columns, [])
    const day = calendarDateIST()
    const fname = `po_recommendation ${day}.csv`
    exportPoCsv(allRows, poCols, quarterCols, quarterMap, fname, editedQty)
  }, [allRows, editedQty, quarterCols, quarterMap, result?.columns])

  const confirmRaiseAndExport = useCallback(async () => {
    setRaiseConfirmErr(null)
    setRaiseConfirmBusy(true)
    try {
      const raisedDate = params.raise_view_date.trim() || calendarDateIST()
      const res = await confirmPoRaiseOnServer({
        rows: selectedRows,
        raisedDate,
        groupByParent: params.group_by_parent,
      })
      if (!res.ok) {
        setRaiseConfirmErr(res.message || 'Could not save raise ledger.')
        return
      }
      const poNumber = res.poNumber || `PO-${raisedDate}`
      exportRaisePoCsv(selectedRows, poNumber)
      const { buildPoRaiseReportHtml, downloadPoRaiseReport } = await import('../utils/poRaiseReport')
      const reportRows = selectedRows.map(r => ({
        OMS_SKU: String(r.OMS_SKU ?? ''),
        Priority: r.Priority as string | undefined,
        Days_Left: Number(r.Days_Left ?? 0),
        ADS: Number(r.ADS ?? 0),
        Gross_PO_Qty: Number(r.Gross_PO_Qty ?? 0),
        PO_Pipeline_Total: Number(r.PO_Pipeline_Total ?? 0),
        Final_PO_Qty: r.Final_PO_Qty,
      }))
      downloadPoRaiseReport(
        buildPoRaiseReportHtml({
          poNumber,
          raisedDate: res.raisedDate || raisedDate,
          rows: reportRows,
          totalQty: res.totalQty ?? totalRaiseUnits,
        }),
        poNumber,
      )
      setRaiseModal(false)
      setSelected(new Set())
      setActionMsg('Raise recorded — recalculating PO with updated ledger…')
      await runCalculate({ force: true })
      setActionMsg(null)
    } catch (e: unknown) {
      setRaiseConfirmErr(e instanceof Error ? e.message : 'Raise ledger save failed')
    } finally {
      setRaiseConfirmBusy(false)
    }
  }, [params.group_by_parent, params.raise_view_date, runCalculate, selectedRows, totalRaiseUnits])

  const exportClientReport = useCallback(() => {
    const day = calendarDateIST()
    const html = buildPoClientReportHtml({
      rows: allRows,
      params: {
        period_days: params.period_days,
        lead_time: params.lead_time,
        target_days: params.target_days,
        demand_basis: params.demand_basis,
        use_seasonality: params.use_seasonality,
        use_ly_fallback: params.use_ly_fallback,
      },
      meta: {
        reportDate: day,
        salesThrough: result?.sales_through,
        planningDate: result?.planning_date,
        poMergeVersion: result?.po_merge_version,
        totalRows: allRows.length,
      },
    })
    downloadPoClientReport(html, day)
  }, [allRows, params, result?.planning_date, result?.po_merge_version, result?.sales_through])

  const readinessLabel = dataStatus
    ? dataStatus.ready
      ? pipelineReadiness?.calculate_allowed === false
        ? pipelineReadiness.pipeline_blockers?.[0] ??
          'Inputs still processing — wait before calculating.'
        : `All ${dataStatus.total} data sources loaded — ready to calculate.`
      : `${dataStatus.loaded}/${dataStatus.total} sources loaded — reload data before calculating.`
    : 'Check data coverage before your first calculation.'

  const canCalculate =
    Boolean(dataStatus?.ready) &&
    pipelineReadiness?.calculate_allowed !== false &&
    !loading

  return (
    <div className="po-fresh-shell -m-4 md:-m-6 p-4 md:p-8" data-testid="po-fresh-root">
      <div className="max-w-7xl mx-auto space-y-8">
        <PageLoadingStripe
          active={loading}
          label={progress?.msg || actionMsg || 'Calculating PO recommendations…'}
          percent={progress?.pct}
          className="sticky top-0 z-40"
        />
        <InventoryStalenessBanner />
        {/* Page header */}
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="po-font-display text-3xl md:text-4xl font-bold tracking-tight">
              PO Engine
            </h1>
            <p className="mt-1 text-sm font-medium text-[var(--po-outline)]">
              Real-time demand calculation and procurement planning.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {someSelected && (
              <button
                type="button"
                onClick={() => {
                  setRaiseConfirmErr(null)
                  setRaiseModal(true)
                }}
                className="px-6 py-2.5 text-sm font-semibold rounded-xl text-white bg-green-600 hover:bg-green-700 shadow-sm"
              >
                Raise PO ({selected.size} SKU{selected.size === 1 ? '' : 's'}, {totalRaiseUnits.toLocaleString()} units)
              </button>
            )}
            <button
              type="button"
              disabled={!result?.ok || allRows.length === 0}
              onClick={exportClientReport}
              className="po-fresh-btn-primary px-6 py-2.5 text-sm disabled:opacity-40"
            >
              Client Report
            </button>
            <button
              type="button"
              disabled={!result?.ok || allRows.length === 0}
              onClick={exportCsv}
              className="po-fresh-btn-secondary px-6 py-2.5 text-sm disabled:opacity-40"
              title="Download CSV for review — does not record raises (use Raise PO → Export & Confirm when ready)."
            >
              Export CSV (test)
            </button>
          </div>
        </div>

        {/* Status chips */}
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ${
              dataStatus?.ready
                ? 'bg-[var(--po-primary-container)]/20 text-[var(--po-on-primary-container)]'
                : 'bg-[var(--po-surface-container)] text-[var(--po-outline)]'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${dataStatus?.ready ? 'bg-[var(--po-primary)] animate-pulse' : 'bg-amber-500'}`}
            />
            Data coverage:{' '}
            {dataStatus ? (
              <strong>
                {dataStatus.loaded}/{dataStatus.total}
              </strong>
            ) : (
              'not checked'
            )}
          </span>
          <button
            type="button"
            disabled={checkingData}
            onClick={() => void refreshDataStatus()}
            className="text-xs font-semibold text-[var(--po-secondary)] hover:underline disabled:opacity-50"
          >
            {checkingData ? 'Checking…' : 'Refresh status'}
          </button>
          {coverageError && (
            <span className="text-xs font-semibold text-amber-700">{coverageError}</span>
          )}
          {coverageDetail?.existing_po ? (
            <span
              className="inline-flex items-center gap-2 rounded-full bg-[var(--po-surface-container)] px-3 py-1.5 text-xs font-semibold text-[var(--po-on-primary-container)]"
              title="Manufacturing / pipeline sheet (not a list of today's PO raises). Pipeline SKUs reduce New PO via projected cover; New Order rows seed the raise ledger for the sheet date."
            >
              Existing PO: {(coverageDetail.existing_po_rows ?? 0).toLocaleString()} tracked
              {(coverageDetail.existing_po_pipeline_skus ?? 0) > 0 && (
                <>
                  {' '}
                  · {(coverageDetail.existing_po_pipeline_skus ?? 0).toLocaleString()} w/ pipeline
                </>
              )}
              {(coverageDetail.existing_po_new_order_skus ?? 0) > 0 && (
                <>
                  {' '}
                  · {(coverageDetail.existing_po_new_order_skus ?? 0).toLocaleString()} new order
                </>
              )}
              {coverageDetail.existing_po_filename ? ` · ${coverageDetail.existing_po_filename}` : ''}
            </span>
          ) : existingPoUploadBusy ? (
            <span className="inline-flex items-center gap-2 rounded-full bg-sky-50 px-3 py-1.5 text-xs font-semibold text-sky-800">
              Loading existing PO sheet…
              {coverageDetail?.existing_po_upload_progress
                ? ` (${coverageDetail.existing_po_upload_progress}%)`
                : ''}
            </span>
          ) : (
            <span className="inline-flex items-center gap-2 rounded-full bg-amber-50 px-3 py-1.5 text-xs font-semibold text-amber-800">
              No existing PO sheet — upload on Upload page
            </span>
          )}
          {serverEngineVersion != null && (
            <span
              className="inline-flex items-center gap-2 rounded-full bg-[var(--po-secondary-container)]/30 px-3 py-1.5 text-xs font-semibold text-[var(--po-on-primary-container)]"
              title="PO calculation logic version on the server"
            >
              PO engine v<strong>{serverEngineVersion}</strong>
              {result?.ok && result.po_merge_version != null && result.po_merge_version < serverEngineVersion && (
                <span className="text-[var(--po-error)]">· recalculate needed</span>
              )}
              {result?.ok && result.po_merge_version != null && result.po_merge_version >= serverEngineVersion && (
                <span className="opacity-70">· result v{result.po_merge_version}</span>
              )}
            </span>
          )}
        </div>

        {/* Bento grid: params + engine */}
        <div className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-8 po-fresh-card p-6">
            <h3 className="po-label mb-6 flex items-center gap-2">
              <span aria-hidden>⚙</span> Demand Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <NumField
                label="Period (days)"
                value={params.period_days}
                onChange={v => setParams(p => ({ ...p, period_days: v }))}
              />
              <NumField
                label="Lead time (days)"
                value={params.lead_time}
                onChange={v => setParams(p => ({ ...p, lead_time: v }))}
              />
              <NumField
                label="Safety %"
                value={params.safety_pct}
                onChange={v => setParams(p => ({ ...p, safety_pct: v }))}
              />
              <label className="space-y-2">
                <span className="po-label ml-1">Demand basis</span>
                <select
                  value={params.demand_basis}
                  onChange={e => setParams(p => ({ ...p, demand_basis: e.target.value as 'Sold' | 'Net' }))}
                  className="po-fresh-input"
                >
                  <option value="Sold">Sold</option>
                  <option value="Net">Net</option>
                </select>
              </label>
              <label className="space-y-2">
                <span className="po-label ml-1">Raise date</span>
                <input
                  type="date"
                  value={params.raise_view_date}
                  onChange={e => setParams(p => ({ ...p, raise_view_date: e.target.value }))}
                  className="po-fresh-input"
                />
              </label>
              <div className="flex items-end pb-1">
                <label className="flex items-center gap-3 cursor-pointer text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={sortByPriority}
                    onChange={e => setSortByPriority(e.target.checked)}
                    className="w-4 h-4 rounded accent-[var(--po-primary)]"
                  />
                  Sort by priority
                </label>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setShowAdvanced(v => !v)}
              className="mt-5 text-xs font-semibold text-[var(--po-secondary)] hover:underline"
            >
              {showAdvanced ? 'Hide' : 'Show'} advanced parameters
            </button>
            {showAdvanced && (
              <div className="mt-4 pt-4 border-t border-[var(--po-outline-ghost)] space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  <NumField
                    label="Post-PO days"
                    value={params.target_days}
                    onChange={v => setParams(p => ({ ...p, target_days: v }))}
                  />
                  <NumField
                    label="Grace days"
                    value={params.grace_days}
                    onChange={v => setParams(p => ({ ...p, grace_days: v }))}
                  />
                  <NumField
                    label="Raise lookback (days)"
                    value={params.raise_ledger_lookback_days}
                    onChange={v => setParams(p => ({ ...p, raise_ledger_lookback_days: v }))}
                  />
                </div>
                <div className="flex flex-wrap gap-5 text-sm">
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={params.enforce_two_size_minimum}
                      onChange={e => setParams(p => ({ ...p, enforce_two_size_minimum: e.target.checked }))}
                      className="accent-[var(--po-primary)]"
                    />
                    Two-size minimum
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={params.group_by_parent}
                      onChange={e => setParams(p => ({ ...p, group_by_parent: e.target.checked }))}
                      className="accent-[var(--po-primary)]"
                    />
                    Group by parent SKU
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={params.use_seasonality}
                      onChange={e => setParams(p => ({ ...p, use_seasonality: e.target.checked }))}
                      className="accent-[var(--po-primary)]"
                    />
                    YoY seasonality
                  </label>
                  <label className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={params.use_ly_fallback}
                      onChange={e => setParams(p => ({ ...p, use_ly_fallback: e.target.checked }))}
                      className="accent-[var(--po-primary)]"
                    />
                    LY ADS fallback
                    <span className="text-xs opacity-60 ml-1">(uses last-year demand for zero-sales SKUs)</span>
                  </label>
                </div>
              </div>
            )}
          </div>

          <div className="col-span-12 lg:col-span-4 po-fresh-engine-card p-8 flex flex-col justify-between gap-6 min-h-[220px]">
            <div>
              <h2 className="po-font-display text-2xl font-bold">Engine Readiness</h2>
              <p className="text-sm mt-2 opacity-90 leading-relaxed">{readinessLabel}</p>
              {serverEngineVersion != null && (
                <p className="text-xs mt-2 opacity-80">
                  PO logic v<strong>{serverEngineVersion}</strong>
                  {result?.ok && result.po_merge_version != null && result.po_merge_version < serverEngineVersion
                    ? ' — click Calculate PO to apply the latest rules'
                    : result?.ok && result.po_merge_version != null
                      ? ` · last run v${result.po_merge_version}`
                      : ' — visible before first calculate'}
                </p>
              )}
              {progress && (
                <div className="mt-4 space-y-1">
                  <div className="h-1.5 rounded-full bg-white/20 overflow-hidden">
                    <div
                      className="h-full bg-[var(--po-primary-container)] transition-[width]"
                      style={{ width: `${progress.pct ?? 12}%` }}
                    />
                  </div>
                  <p className="text-xs opacity-80">{progress.msg}</p>
                </div>
              )}
            </div>
            <div className="space-y-3">
              {sharedCacheHint?.available && !loading && (
                <p className="text-xs text-white/85 leading-relaxed">
                  Shared PO available
                  {sharedCacheHint.computed_at ? ` (${sharedCacheHint.computed_at})` : ''}
                  {sharedCacheHint.row_count
                    ? ` · ${sharedCacheHint.row_count.toLocaleString()} rows`
                    : ''}
                  — Calculate PO loads it in seconds when settings match.
                </p>
              )}
              <button
                type="button"
                onClick={() => void runCalculate()}
                disabled={!canCalculate}
                className="po-fresh-btn-primary w-full py-4 text-lg disabled:opacity-50 disabled:cursor-not-allowed"
                aria-busy={loading}
                title={
                  pipelineReadiness?.calculate_allowed === false
                    ? pipelineReadiness.pipeline_blockers?.[0]
                    : undefined
                }
              >
                {loading ? 'Calculating…' : 'Calculate PO'}
              </button>
              {pipelineReadiness?.pipeline_warnings?.length ? (
                <p className="text-xs text-amber-100/90 leading-relaxed">
                  {pipelineReadiness.pipeline_warnings[0]}
                </p>
              ) : null}
              {loading && (
                <p className="text-xs text-white/80 text-center">
                  Usually completes in under a minute — keep this tab open.
                </p>
              )}
              <button
                type="button"
                disabled={loading || !canCalculate}
                onClick={() => void runCalculate({ force: true })}
                className="w-full text-sm font-semibold text-white/90 hover:text-white disabled:opacity-40"
              >
                Force recalculate
              </button>
              <p className="text-[11px] text-white/70 text-center leading-snug">
                Skips shared cache — use after data uploads or when you need a full engine run.
              </p>
              <button
                type="button"
                disabled={quarterlyLoading || !result?.ok}
                onClick={() => void loadQuarterly()}
                className="w-full text-sm font-semibold text-white/90 hover:text-white disabled:opacity-40"
              >
                {quarterlyLoading ? 'Loading quarterly…' : 'Load quarterly columns'}
              </button>
            </div>
          </div>
        </div>

        {quarterlyMsg && <p className="text-xs text-[var(--po-outline)] -mt-4">{quarterlyMsg}</p>}

        {parityReport && !parityReport.ok && parityReport.warnings.length > 0 ? (
          <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 space-y-1">
            {parityReport.warnings.map((w, i) => (
              <p key={i}>{w}</p>
            ))}
          </div>
        ) : null}

        {/* Sales window banner — shows after every successful calculate */}
        {result?.ok && (
          <SalesWindowBanner
            result={result}
            periodDays={params.period_days}
            serverEngineVersion={serverEngineVersion}
          />
        )}
        {result?.ok && fromSharedCache && (
          <p className="text-sm text-[var(--po-secondary)] bg-sky-50 rounded-xl px-4 py-3 mb-4">
            Loaded from a shared PO run
            {result.shared_cache_at ? ` (${result.shared_cache_at})` : ''}
            — same planning date and settings as an earlier calculation today.
          </p>
        )}

        {result && !result.ok && (
          <p className="text-sm text-[var(--po-error)] bg-red-50 rounded-xl px-4 py-3">{result.message}</p>
        )}

        {/* KPI metrics */}
        {tab === 'po' && result?.ok && allRows.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            <MetricCard
              label="New PO Units"
              value={totalPO.toLocaleString()}
              hint={`Gross ${kpis.grossPo.toLocaleString()} pre-pipe`}
              help="Sum of New PO Qty across the filtered rows (after pipeline + confirmed raise deductions and pack rounding)."
            />
            <MetricCard
              label="SKUs with PO"
              value={kpis.withPo.toLocaleString()}
              hint={`${allRows.length.toLocaleString()} total rows`}
              help="Count of rows where New PO Qty > 0 (after all gates/blocks and deductions)."
            />
            <MetricCard
              label="Urgent Reorder"
              value={kpis.urgent.toLocaleString()}
              hint="Requires immediate action"
              accent="text-[var(--po-error)]"
              help="Count of rows in the 🔴 URGENT priority bucket."
            />
            <MetricCard
              label="Sheet Pipeline"
              value={kpis.pipelineUnits.toLocaleString()}
              hint={`${kpis.pipeline.toLocaleString()} SKUs in pipeline`}
              accent="text-[var(--po-secondary)]"
              help="Total open pipeline from the uploaded Existing PO sheet (deduped at source — not added to New PO)."
            />
          </div>
        )}

        {/* Main content card with tabs */}
        <div className="po-fresh-table-wrap">
          <div className="px-6 pt-5 pb-0 flex flex-wrap items-center justify-between gap-4 border-b border-[var(--po-outline-ghost)]">
            <div className="po-fresh-tabs flex items-center gap-6">
              {(
                [
                  ['po', 'PO Results'],
                  ['dashboard', 'PO Dashboard'],
                  ['quarterly', 'Quarterly History'],
                ] as const
              ).map(([t, label]) => (
                <button
                  key={t}
                  type="button"
                  data-active={tab === t}
                  onClick={() => setTab(t)}
                >
                  {label}
                </button>
              ))}
            </div>
            {tab === 'po' && result?.ok && allRows.length > 0 && (
              <div className="relative pb-3">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--po-outline)] text-sm">⌕</span>
                <input
                  value={search}
                  onChange={e => {
                    setSearch(e.target.value)
                    setPage(0)
                  }}
                  placeholder="Search SKU…"
                  className="po-fresh-search"
                />
              </div>
            )}
          </div>

          {tab === 'dashboard' && (
            <div className="p-6">
              <PODashboardPanel
                embedded
                poParams={dashboardParams}
                canDeleteRaiseSkus={canDeleteRaiseSkus}
                onRaiseLedgerChanged={async msg => {
                  if (msg) setResult({ ok: false, message: msg })
                }}
              />
            </div>
          )}

          {tab === 'po' && result?.ok && (result?.rows?.length ?? 0) > 0 && (
            <div className="px-6 pt-4 pb-2 flex flex-wrap items-center justify-between gap-3 border-b border-[var(--po-outline-ghost)]">
              <p className="text-xs text-[var(--po-outline)]">
                Edit <strong>PO Qty</strong> inline; <strong>Export CSV (test)</strong> downloads only.
                Select rows and use <strong>Raise PO</strong> → <strong>Export &amp; Confirm</strong> to record the ledger.
              </p>
              <div className="flex flex-wrap items-center gap-2">
                {(
                  [
                    ['all', 'All'],
                    ['urgent', 'Urgent'],
                    ['with_po', 'Has PO'],
                    ['blocked', 'Closed / blocked'],
                  ] as const
                ).map(([key, label]) => (
                  <button
                    key={key}
                    type="button"
                    data-active={priorityFilter === key}
                    className="po-fresh-filter-chip"
                    onClick={() => {
                      setPriorityFilter(key)
                      setPage(0)
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {tab === 'po' && result?.ok && allRows.length > 0 && (
            <>
              <PoTable
                cols={tableCols}
                rows={pageRows}
                quarterCols={quarterCols}
                quarterMap={quarterMap}
                periodDays={params.period_days}
                targetDays={params.target_days}
                enteredLeadDays={params.lead_time}
                editedQty={editedQty}
                onEditedQtyChange={(sku, qty) => setEditedQty(prev => ({ ...prev, [sku]: qty }))}
                onEditedQtyReset={sku =>
                  setEditedQty(prev => {
                    const next = { ...prev }
                    delete next[sku]
                    return next
                  })
                }
                selected={selected}
                onToggleRow={toggleRowSelect}
                allVisibleSelected={allVisibleSelected}
                onToggleAllVisible={toggleAllVisible}
              />
              <TableFooter
                page={page}
                pageCount={pageCount}
                pageRows={pageRows.length}
                totalRows={allRows.length}
                onPrev={() => setPage(Math.max(0, page - 1))}
                onNext={() => setPage(page + 1)}
                onExport={exportCsv}
                onClientReport={exportClientReport}
                quarterCount={quarterCols.length}
              />
            </>
          )}

          {tab === 'po' && result?.ok && allRows.length === 0 && (
            <p className="p-6 text-sm text-[var(--po-outline)]">
              {(result?.rows?.length ?? 0) > 0
                ? 'No rows match the current search or filter. Try clearing filters.'
                : 'Calculation finished with zero rows.'}
            </p>
          )}

          {tab === 'po' && !result?.ok && (
            <p className="p-6 text-sm text-[var(--po-outline)]">
              Run Calculate PO to see procurement recommendations.
            </p>
          )}

          {tab === 'quarterly' && (
            <div className="p-6 space-y-4">
              {!quarterly?.rows?.length ? (
                <p className="text-sm text-[var(--po-outline)]">
                  {quarterlyLoading
                    ? quarterlyMsg || 'Loading…'
                    : 'Run Calculate PO, then click Load quarterly columns.'}
                </p>
              ) : (
                <>
                  <p className="text-xs text-[var(--po-outline)]">
                    {quarterRowCount.toLocaleString()} SKUs · {quarterCols.length} quarter columns
                  </p>
                  <PoTable
                    cols={['OMS_SKU', 'Status', 'Avg_Monthly', ...quarterCols].filter(c =>
                      (quarterly.columns ?? []).includes(c) || c === 'OMS_SKU',
                    )}
                    rows={quarterPageRows}
                    quarterCols={quarterCols}
                    quarterMap={quarterMap}
                    periodDays={params.period_days}
                  />
                  <TableFooter
                    page={quarterlyPage}
                    pageCount={quarterPageCount}
                    pageRows={quarterPageRows.length}
                    totalRows={quarterRowCount}
                    onPrev={() => setQuarterlyPage(p => Math.max(0, p - 1))}
                    onNext={() => setQuarterlyPage(p => p + 1)}
                    quarterCount={quarterCols.length}
                  />
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {raiseModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col">
            <div className="p-5 border-b border-gray-200 flex items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-bold text-[var(--po-secondary)]">Raise Purchase Order</h3>
                <p className="text-sm text-[var(--po-outline)] mt-0.5">
                  {selectedRows.length} SKU{selectedRows.length === 1 ? '' : 's'} · {totalRaiseUnits.toLocaleString()} units
                </p>
                <p className="text-[11px] text-sky-900 bg-sky-50 border border-sky-200 rounded px-2 py-1.5 mt-2 leading-snug">
                  <strong>Export &amp; Confirm</strong> downloads the raise CSV and records quantities in the raise ledger for{' '}
                  <strong>{params.raise_view_date || calendarDateIST()}</strong>. The next Calculate PO treats them as confirmed pipeline.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setRaiseModal(false)}
                className="text-gray-400 hover:text-gray-600 text-2xl leading-none"
              >
                ×
              </button>
            </div>
            <div className="overflow-auto flex-1 p-4">
              {selectedRows.length === 0 ? (
                <p className="text-center text-[var(--po-outline)] py-8 text-sm">
                  No SKUs with PO Qty &gt; 0 in selection. Edit quantities or select other rows.
                </p>
              ) : (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b border-gray-200">
                      <th className="text-left px-3 py-2 font-semibold text-gray-600">SKU</th>
                      <th className="text-left px-3 py-2 font-semibold text-gray-600">Priority</th>
                      <th className="text-right px-3 py-2 font-semibold text-orange-600">PO Qty</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedRows.map(r => (
                      <tr key={String(r.OMS_SKU)} className="border-b border-gray-100">
                        <td className="px-3 py-2 font-medium">{String(r.OMS_SKU)}</td>
                        <td className="px-3 py-2">
                          <PriorityPill priority={r.Priority} />
                        </td>
                        <td className="px-3 py-2 text-right font-bold text-orange-600">
                          {r.Final_PO_Qty.toLocaleString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
            {raiseConfirmErr && (
              <div className="px-4 py-2 mx-4 text-xs text-rose-800 bg-rose-50 border border-rose-200 rounded">
                {raiseConfirmErr}
              </div>
            )}
            <div className="p-4 border-t border-gray-200 flex gap-3 justify-end">
              <button
                type="button"
                onClick={() => setRaiseModal(false)}
                className="px-4 py-2 rounded-lg text-sm font-medium border border-gray-300 hover:bg-gray-50"
              >
                Cancel
              </button>
              {selectedRows.length > 0 && (
                <button
                  type="button"
                  onClick={() => void confirmRaiseAndExport()}
                  disabled={raiseConfirmBusy}
                  className="px-5 py-2 rounded-lg text-sm font-semibold text-white bg-green-600 hover:bg-green-700 disabled:opacity-50"
                >
                  {raiseConfirmBusy ? 'Saving…' : 'Export & Confirm PO'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function POFresh() {
  return (
    <POFreshErrorBoundary>
      <POFreshInner />
    </POFreshErrorBoundary>
  )
}

/** Mirror of backend ``(Total_Inventory + PO_Pipeline_Effective + PO_Qty) / ADS``. */
function computePostPoCover(row: Row, finalQty: number): number {
  const ads = Number(row['ADS'] ?? 0)
  if (!Number.isFinite(ads) || ads <= 0) return 999
  const inv = Number(row['Total_Inventory'] ?? 0)
  const pipeEff = Number(row['PO_Pipeline_Effective'] ?? row['PO_Pipeline_Total'] ?? 0)
  const q = Math.max(0, Math.floor(finalQty))
  return Math.round(((inv + pipeEff + q) / ads) * 10) / 10
}

function PoTable({
  cols,
  rows,
  quarterCols,
  quarterMap,
  periodDays,
  targetDays = 180,
  enteredLeadDays = 60,
  editedQty,
  onEditedQtyChange,
  onEditedQtyReset,
  selected,
  onToggleRow,
  allVisibleSelected,
  onToggleAllVisible,
}: {
  cols: string[]
  rows: Row[]
  quarterCols: string[]
  quarterMap: Record<string, Record<string, number | string>>
  periodDays: number
  targetDays?: number
  enteredLeadDays?: number
  editedQty?: Record<string, number>
  onEditedQtyChange?: (sku: string, qty: number) => void
  onEditedQtyReset?: (sku: string) => void
  selected?: Set<string>
  onToggleRow?: (sku: string) => void
  allVisibleSelected?: boolean
  onToggleAllVisible?: () => void
}) {
  const [selectedRow, setSelectedRow] = useState<Row | null>(null)
  const selectable = Boolean(selected && onToggleRow && onToggleAllVisible)

  return (
    <>
      <div className="overflow-auto po-fresh-scroll max-h-[62vh]">
        <table className="po-fresh-table min-w-max w-full">
          <thead className="sticky top-0 z-10">
            <tr>
              {selectable && (
                <th className="w-10 text-center">
                  <input
                    type="checkbox"
                    checked={Boolean(allVisibleSelected)}
                    onChange={onToggleAllVisible}
                    title="Select all on this page"
                  />
                </th>
              )}
              {cols.map(c => (
                <th key={c} className="text-left">
                  <div className="flex items-center gap-2">
                    <span>{poColHeaderLabel(c)}</span>
                    <HelpTip text={poColHelpText(c)} />
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(r => {
              const sku = String(r.OMS_SKU ?? '')
              const isSelected = selectedRow?.OMS_SKU === r.OMS_SKU
              const isUrgent = String(r.Priority ?? '').includes('URGENT')
              const computedQty = num(r.PO_Qty)
              const finalQty = finalPoQtyForSku(r, editedQty)
              return (
              <tr
                key={sku}
                onClick={() => setSelectedRow(isSelected ? null : r)}
                className={[
                  isSelected ? 'po-row-selected' : '',
                  isUrgent ? 'po-row-urgent' : '',
                ].filter(Boolean).join(' ')}
              >
                {selectable && (
                  <td className="text-center" onClick={e => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={selected?.has(sku) ?? false}
                      onChange={() => onToggleRow?.(sku)}
                    />
                  </td>
                )}
                {cols.map(c => {
                  const qsku = poSkuKey(sku)
                  const v = quarterCols.includes(c)
                    ? (quarterMap[qsku]?.[c] ?? r[c])
                    : r[c]
                  if (c === 'Priority') {
                    return (
                      <td key={c}>
                        <PriorityPill priority={v} />
                      </td>
                    )
                  }
                  if (c === 'SKU_Sheet_Status') {
                    return (
                      <td key={c}>
                        <StatusBadge status={String(v ?? '')} />
                      </td>
                    )
                  }
                  if (c === 'OMS_SKU') {
                    return (
                      <td key={c}>
                        <span className="po-fresh-sku-link text-sm">
                          {formatPoCell(c, v)}
                          <span className="po-fresh-sku-chevron">View breakdown →</span>
                        </span>
                      </td>
                    )
                  }
                  if (c === 'PO_Qty' && onEditedQtyChange && onEditedQtyReset) {
                    return (
                      <td key={c} onClick={e => e.stopPropagation()}>
                        <PoQtyInput
                          value={finalQty}
                          computed={computedQty}
                          onChange={v => onEditedQtyChange(sku, v)}
                          onReset={() => onEditedQtyReset(sku)}
                        />
                      </td>
                    )
                  }
                  if (c === 'PO_Qty' && finalQty > 0) {
                    return (
                      <td key={c} className="font-bold text-[var(--po-primary)]">
                        {formatPoCell(c, v)}
                      </td>
                    )
                  }
                  if (c === 'Post_PO_Cover_Days_Capped' && editedQty?.[poSkuKey(sku)] !== undefined) {
                    const live = computePostPoCover(r, finalQty)
                    const display = live >= 999 ? '—' : live.toFixed(1)
                    return (
                      <td key={c} className={`${poCellClass(c, live)} font-semibold`} title="Recalculated from your edited qty">
                        {display} <span className="text-[10px] opacity-60">↺</span>
                      </td>
                    )
                  }
                  if (c === 'PO_Block_Reason' && v) {
                    return (
                      <td key={c} className="max-w-[12rem] truncate text-xs text-[var(--po-outline)]" title={String(v)}>
                        {formatPoCell(c, v)}
                      </td>
                    )
                  }
                  return (
                    <td key={c} className={poCellClass(c, v)}>
                      {formatPoCell(c, v)}
                    </td>
                  )
                })}
              </tr>
            )})}
          </tbody>
        </table>
      </div>
      {selectedRow && (
        <SkuDetailDrawer
          row={selectedRow}
          periodDays={periodDays}
          targetDays={targetDays}
          enteredLeadDays={enteredLeadDays}
          editedQty={editedQty?.[poSkuKey(String(selectedRow.OMS_SKU ?? ''))]}
          onClose={() => setSelectedRow(null)}
        />
      )}
    </>
  )
}

/** Step-by-step calculation breakdown drawer for a selected PO row */
function SkuDetailDrawer({
  row,
  periodDays,
  targetDays,
  enteredLeadDays,
  editedQty,
  onClose,
}: {
  row: Row
  periodDays: number
  targetDays: number
  enteredLeadDays: number
  editedQty?: number
  onClose: () => void
}) {
  const n = (k: string) => Number(row[k] ?? 0)
  const s = (k: string) => String(row[k] ?? '—')
  const fmt = (v: number, dec = 1) => isNaN(v) ? '—' : v.toFixed(dec)
  const fmtInt = (v: number) => isNaN(v) ? '—' : Math.round(v).toLocaleString()

  const sold = n('Sold_Units')
  const returns = n('Return_Units')
  const overlay = n('Return_Overlay_Units')
  const netUnits = n('Net_Units')
  const effDays = n('Eff_Days')
  const recentAds = n('Recent_ADS')
  const lyAds = n('LY_ADS')
  const seasonAds = n('Seasonal_Month_ADS')
  const flat30Ads = n('Flat30_ADS')
  const ads = n('ADS')
  const totalInv = n('Total_Inventory')
  const pipeline = n('PO_Pipeline_Effective') || n('PO_Pipeline_Total')
  const projDays = n('Projected_Running_Days')
  const daysLeft = n('Days_Left')
  const grossPo = n('Gross_PO_Qty')
  const poQtyBase = n('PO_Qty')
  const poQty = editedQty !== undefined ? editedQty : poQtyBase
  const postCoverBase = n('Post_PO_Cover_Days_Capped')
  const postCover = editedQty !== undefined ? computePostPoCover(row, editedQty) : postCoverBase
  const drawerQtyEdited = editedQty !== undefined && editedQty !== poQtyBase
  const sheetLeadDays = n('Lead_Time_Days') || 0
  const gateLeadDays = enteredLeadDays > 0 ? enteredLeadDays : sheetLeadDays
  const gateLeadLabel = enteredLeadDays > 0 ? 'entered lead gate' : 'sheet lead gate'
  const blockReason = s('PO_Block_Reason')
  const priority = s('Priority')

  type Step = { label: string; formula: string; result: string; note?: string; highlight?: boolean }
  const steps: Step[] = [
    {
      label: `① Sales (${periodDays}-day window)`,
      formula: `Sold Units − Returns − Overlay`,
      result: `${fmtInt(sold)} − ${fmtInt(returns)} − ${fmtInt(overlay)} = ${fmtInt(netUnits)} net units`,
      note: 'Only shipment rows count as sold. Returns & overlay reduce demand.',
    },
    {
      label: '② Effective Days',
      formula: `Active sell-days in ${periodDays}-day ADS window`,
      result: `${fmt(effDays, 0)} days`,
      note:
        netUnits <= 0
          ? `No shipments in the last ${periodDays} days — Recent ADS = 0; ADS uses LY/seasonal/Flat30 floors.`
          : effDays < 25
            ? `⚠ Only ${fmt(effDays, 0)} active days — ADS capped at sold÷${periodDays} (${fmt(sold / periodDays, 2)}/day) when higher.`
            : `${fmt(effDays, 0)} of ${periodDays} days active`,
    },
    {
      label: '③ ADS Signals',
      formula: `max(Recent, Last Year, Seasonal, Flat30)`,
      result: `Recent=${fmt(recentAds)} · LY=${fmt(lyAds)} · Season=${fmt(seasonAds)} · Flat30=${fmt(flat30Ads)}`,
      note: `→ ADS used = ${fmt(ads, 2)}/day (capped primary + floors)`,
      highlight: true,
    },
    {
      label: '④ Inventory Cover',
      formula: `Total Inventory ÷ ADS`,
      result: `${fmtInt(totalInv)} ÷ ${fmt(ads, 2)} = ${fmt(daysLeft, 0)} days left`,
      note: daysLeft > 90 ? '✅ Covered without new PO' : daysLeft < 30 ? '🔴 Critically low stock' : '🟡 Approaching reorder point',
    },
    {
      label: '⑤ Pipeline',
      formula: `Existing PO + Confirmed Raises`,
      result: `${fmtInt(pipeline)} units in pipeline`,
      note: `Projected cover incl. pipeline: ${fmt(projDays, 0)} days · ${gateLeadLabel}: ${fmt(gateLeadDays, 0)} days${enteredLeadDays > 0 && sheetLeadDays > 0 && sheetLeadDays !== enteredLeadDays ? ` (sheet ${fmt(sheetLeadDays, 0)}d — display only)` : enteredLeadDays <= 0 && sheetLeadDays > 0 ? ' (from SKU status sheet)' : ''}`,
    },
    {
      label: '⑥ Gross PO Need',
      formula: `ADS × (Target Cover − Projected Days)`,
      result: grossPo > 0
        ? `${fmt(ads, 2)} × (${fmt(targetDays, 0)} − ${fmt(projDays, 0)}) → ${fmtInt(grossPo)} units (pack-rounded)`
        : projDays >= gateLeadDays && projDays < targetDays
          ? `${fmt(ads, 2)} × (${fmt(targetDays, 0)} − ${fmt(projDays, 0)}) → blocked (projected ${fmt(projDays, 0)}d ≥ ${gateLeadLabel} ${fmt(gateLeadDays, 0)}d)`
          : projDays >= targetDays
            ? `${fmt(ads, 2)} × (${fmt(targetDays, 0)} − ${fmt(projDays, 0)}) → 0 (already at target)`
            : `${fmt(ads, 2)} × (${fmt(targetDays, 0)} − ${fmt(projDays, 0)}) → 0`,
      note: projDays < gateLeadDays
        ? `Release gate: projected ${fmt(projDays, 0)}d < ${gateLeadLabel} ${fmt(gateLeadDays, 0)}d → PO allowed.`
        : `Release gate: projected ${fmt(projDays, 0)}d ≥ ${gateLeadLabel} ${fmt(gateLeadDays, 0)}d → no PO until cover drops.`,
      highlight: grossPo > 0,
    },
    {
      label: drawerQtyEdited ? '⑦ Final PO Qty ✏️ (edited)' : '⑦ Final PO Qty',
      formula: overlay > 0 ? `Gross PO − Return Overlay` : `= Gross PO`,
      result: overlay > 0
        ? `${fmtInt(grossPo)} − ${fmtInt(overlay)} = ${fmtInt(poQty)} units`
        : `${fmtInt(poQty)} units`,
      note: blockReason && blockReason !== '—'
        ? `🚫 Blocked: ${blockReason}`
        : postCover < 999
          ? `Post-PO cover: ${fmt(postCover, 0)} days${drawerQtyEdited ? ' (updated from your edit)' : ''}`
          : undefined,
      highlight: true,
    },
  ]

  const priorityMeta_ = priorityMeta(priority)

  return (
    <div className="fixed inset-0 z-50 flex justify-end po-fresh-drawer-backdrop" onClick={onClose}>
      <div
        className="relative w-full max-w-md bg-[var(--po-surface-highest)] shadow-2xl flex flex-col h-full overflow-y-auto"
        onClick={e => e.stopPropagation()}
        style={{ borderLeft: '2px solid var(--po-outline-ghost)' }}
      >
        {/* Header */}
        <div className="sticky top-0 bg-[var(--po-surface-highest)] z-10 px-5 py-4 border-b border-[var(--po-outline-ghost)]">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs po-label mb-1">PO Calculation Breakdown</p>
              <p className="text-lg font-bold po-font-display text-[var(--po-on-surface)]">{s('OMS_SKU')}</p>
              <div className="flex items-center gap-2 mt-1">
                <span className={`w-2 h-2 rounded-full shrink-0 ${priorityMeta_.dot}`} />
                <span className={`text-xs font-semibold ${priorityMeta_.cls}`}>{priorityMeta_.label}</span>
                {poQty > 0 && (
                  <span className="ml-2 px-2 py-0.5 rounded-full text-xs font-bold bg-orange-100 text-orange-700">
                    PO = {fmtInt(poQty)} units
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={onClose}
              className="text-[var(--po-outline)] hover:text-[var(--po-on-surface)] text-xl leading-none mt-0.5"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Steps */}
        <div className="px-5 py-4 flex flex-col gap-3">
          {steps.map(step => (
            <div
              key={step.label}
              className={`rounded-xl p-4 border ${step.highlight ? 'border-[var(--po-primary)] bg-[#f0faf4]' : 'border-[var(--po-outline-ghost)] bg-[var(--po-surface-low)]'}`}
            >
              <p className="text-xs font-bold text-[var(--po-outline)] uppercase tracking-wide mb-1">{step.label}</p>
              <p className="text-xs text-[var(--po-outline)] mb-1 font-mono">{step.formula}</p>
              <p className="text-sm font-semibold text-[var(--po-on-surface)]">{step.result}</p>
              {step.note && (
                <p className="text-xs text-[var(--po-outline)] mt-1">{step.note}</p>
              )}
            </div>
          ))}

          {/* All values reference */}
          <details className="mt-2">
            <summary className="text-xs text-[var(--po-outline)] cursor-pointer select-none hover:text-[var(--po-on-surface)] font-semibold">
              All column values for this SKU
            </summary>
            <div className="mt-2 rounded-xl border border-[var(--po-outline-ghost)] overflow-hidden">
              <table className="w-full text-xs">
                <tbody>
                  {Object.entries(row).filter(([, v]) => v != null && v !== '').map(([k, v]) => (
                    <tr key={k} className="border-b border-[var(--po-outline-ghost)] last:border-0 even:bg-[var(--po-surface-low)]">
                      <td className="px-3 py-1.5 text-[var(--po-outline)] font-medium w-1/2">{k.replace(/_/g, ' ')}</td>
                      <td className="px-3 py-1.5 font-semibold text-right">{String(v)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </div>
      </div>
    </div>
  )
}

function SalesWindowBanner({
  result,
  periodDays,
  serverEngineVersion,
}: {
  result: import('../api/client').POCalculateResult
  periodDays: number
  serverEngineVersion: number | null
}) {
  const through = result.sales_through
  const planDate = result.planning_date

  // compute start = through - periodDays (best guess if no exact date)
  const endLabel = through ?? planDate ?? '—'
  let startLabel = '—'
  if (through) {
    try {
      const end = new Date(through)
      end.setDate(end.getDate() - periodDays + 1)
      startLabel = end.toISOString().slice(0, 10)
    } catch {
      startLabel = '—'
    }
  }

  const stale =
    through && planDate && salesDataGapNeedsWarning(planDate, through)
  return (
    <div
      className={`flex flex-wrap items-center gap-4 rounded-xl px-4 py-3 text-xs font-semibold ${
        stale
          ? 'bg-amber-50 text-amber-800'
          : 'bg-[var(--po-surface-container)] text-[var(--po-on-primary-container)]'
      }`}
    >
      <span>
        📅 Sales window:{' '}
        <strong>
          {startLabel} → {endLabel}
        </strong>{' '}
        ({periodDays}d)
      </span>
      {stale && (
        <span className="font-bold">
          ⚠ Platform data ends {through} but planning date is {planDate} — reload data for a
          complete sales window.
        </span>
      )}
      {!through && (
        <span className="opacity-70">Platform data date unknown — verify uploads are current.</span>
      )}
      {result.raise_ledger_rows != null && result.raise_ledger_rows > 0 && (
        <span className="text-[var(--po-primary)]">
          ✓ {result.raise_ledger_rows} raise ledger rows loaded
        </span>
      )}
      {result.po_merge_version != null && (
        <span>
          Engine v<strong>{result.po_merge_version}</strong>
          {serverEngineVersion != null && result.po_merge_version < serverEngineVersion && (
            <span className="text-[var(--po-error)]">
              {' '}
              (server v{serverEngineVersion} — click Calculate PO)
            </span>
          )}
          {result.summary?.new_po_qty_sum != null && (
            <>
              {' '}
              · new PO to raise{' '}
              <strong>{result.summary.new_po_qty_sum.toLocaleString()}</strong> units
              {result.summary.pipeline_qty_sum != null && result.summary.pipeline_qty_sum > 0 && (
                <>
                  {' '}
                  <span className="opacity-80">
                    (Existing PO pipeline already loaded:{' '}
                    <strong>{result.summary.pipeline_qty_sum.toLocaleString()}</strong> units on{' '}
                    {result.summary.pipeline_sku_count?.toLocaleString() ?? '—'} SKUs — not added
                    again)
                  </span>
                </>
              )}
            </>
          )}
        </span>
      )}
    </div>
  )
}

function HelpTip({ text }: { text?: string }) {
  const anchorRef = useRef<HTMLSpanElement>(null)
  const [visible, setVisible] = useState(false)
  const [pos, setPos] = useState({ top: 0, left: 0, arrowLeft: 144 })

  if (!text) return null

  const show = () => {
    const el = anchorRef.current
    if (!el) return
    const rect = el.getBoundingClientRect()
    const width = 288
    const left = Math.min(Math.max(8, rect.left + rect.width / 2 - width / 2), window.innerWidth - width - 8)
    const arrowLeft = Math.min(Math.max(16, rect.left + rect.width / 2 - left), width - 16)
    const top = rect.bottom + 10
    setPos({ top, left, arrowLeft })
    setVisible(true)
  }

  const hide = () => setVisible(false)

  return (
    <>
      <span
        ref={anchorRef}
        className="po-fresh-helptip-trigger"
        role="button"
        tabIndex={0}
        aria-label={text}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
      >
        i
      </span>
      {visible &&
        createPortal(
          <div
            className="po-fresh-helptip"
            style={{
              top: pos.top,
              left: pos.left,
              width: 288,
              ['--tip-arrow-left' as string]: `${pos.arrowLeft}px`,
            }}
          >
            {text}
          </div>,
          document.body,
        )}
    </>
  )
}

function PriorityPill({ priority }: { priority: unknown }) {
  const meta = priorityMeta(priority)
  const pillClass =
    meta.label === 'Urgent'
      ? 'po-priority-pill-urgent'
      : meta.label === 'High'
        ? 'po-priority-pill-high'
        : meta.label === 'Medium'
          ? 'po-priority-pill-medium'
          : meta.label === 'In Pipeline'
            ? 'po-priority-pill-pipeline'
            : 'po-priority-pill-ok'
  return (
    <span className={`po-priority-pill ${pillClass}`}>
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${meta.dot}`} />
      {meta.label}
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  const s = status.trim()
  if (!s) return <span className="po-status-empty">—</span>
  const lower = s.toLowerCase()
  let cls = 'po-status-neutral'
  if (lower.includes('closed')) cls = 'po-status-closed'
  else if (lower.includes('doubt')) cls = 'po-status-doubt'
  else if (lower.includes('high') || lower.includes('fast') || lower.includes('new sku')) cls = 'po-status-active'
  return (
    <span className={`po-status-badge ${cls}`} title={s}>
      {s}
    </span>
  )
}

function TableFooter({
  page,
  pageCount,
  pageRows,
  totalRows,
  onPrev,
  onNext,
  onExport,
  onClientReport,
  quarterCount,
}: {
  page: number
  pageCount: number
  pageRows: number
  totalRows: number
  onPrev: () => void
  onNext: () => void
  onExport?: () => void
  onClientReport?: () => void
  quarterCount?: number
}) {
  return (
    <div className="px-6 py-4 flex flex-wrap items-center justify-between gap-3 bg-[var(--po-surface-low)]/50">
      <span className="text-xs text-[var(--po-outline)]">
        Showing {pageRows} of {totalRows.toLocaleString()} · Page {page + 1}/{pageCount}
      </span>
      <div className="flex items-center gap-2">
        {onClientReport ? (
        <button
          type="button"
          onClick={onClientReport}
          className="text-xs font-semibold text-[var(--po-secondary)] hover:underline"
        >
          Client Report
        </button>
        ) : null}
        {onExport ? (
        <button
          type="button"
          onClick={onExport}
          className="text-xs font-semibold text-[var(--po-secondary)] hover:underline"
        >
          Export CSV{quarterCount && quarterCount > 0 ? ` (+${quarterCount} Q)` : ''}
        </button>
        ) : null}
        <button
          type="button"
          disabled={page <= 0}
          onClick={onPrev}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[var(--po-surface-container)] disabled:opacity-40"
        >
          Prev
        </button>
        <button
          type="button"
          disabled={page + 1 >= pageCount}
          onClick={onNext}
          className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-[var(--po-surface-container)] disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  )
}

function NumField({
  label,
  value,
  onChange,
}: {
  label: string
  value: number
  onChange: (v: number) => void
}) {
  return (
    <label className="space-y-2">
      <span className="po-label ml-1">{label}</span>
      <input
        type="number"
        value={value}
        onChange={e => onChange(+e.target.value)}
        className="po-fresh-input"
      />
    </label>
  )
}

function MetricCard({
  label,
  value,
  hint,
  accent,
  help,
}: {
  label: string
  value: string
  hint?: string
  accent?: string
  help?: string
}) {
  return (
    <div className="po-fresh-metric-card">
      <p className={`po-label ${accent ?? ''}`}>
        <span className="inline-flex items-center gap-2">
          <span>{label}</span>
          <HelpTip text={help} />
        </span>
      </p>
      <p className={`po-fresh-metric-value mt-2 ${accent ?? 'text-[var(--po-on-surface)]'}`}>
        {value}
      </p>
      {hint && <p className="mt-2 text-xs font-semibold text-[var(--po-outline)]">{hint}</p>}
    </div>
  )
}
