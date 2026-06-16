/**
 * PO Engine (Fresh) — isolated purchase-order UI for local testing.
 * No zustand, no session subscriptions, no mount effects, lazy-loaded only.
 */
import { Component, useCallback, useMemo, useRef, useState, type ReactNode } from 'react'
import { getCoverage, startPoCalculate, type CoverageResponse, type POCalculateResult } from '../api/client'
import { calendarDateIST } from '../lib/dates'

const DATA_KEYS = [
  'sku_mapping',
  'mtr',
  'sales',
  'inventory',
  'myntra',
  'meesho',
  'flipkart',
  'snapdeal',
] as const

const DISPLAY_COLS = [
  'Priority',
  'OMS_SKU',
  'Total_Inventory',
  'Projected_Running_Days',
  'Post_PO_Cover_Days_Capped',
  'ADS',
  'PO_Qty',
] as const

const PAGE_SIZE = 100

type Row = Record<string, string | number | undefined>

type Params = {
  period_days: number
  lead_time: number
  target_days: number
  demand_basis: 'Sold' | 'Net'
}

const DEFAULT_PARAMS: Params = {
  period_days: 30,
  lead_time: 60,
  target_days: 180,
  demand_basis: 'Sold',
}

function countLoaded(c: CoverageResponse) {
  const loaded = DATA_KEYS.filter(k => Boolean(c[k])).length
  return { loaded, total: DATA_KEYS.length, ready: loaded === DATA_KEYS.length }
}

function num(v: unknown): number {
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}

function buildBody(p: Params) {
  return {
    period_days: p.period_days,
    lead_time: p.lead_time,
    target_days: p.target_days,
    grace_days: 0,
    demand_basis: p.demand_basis,
    group_by_parent: false,
    safety_pct: 0,
    use_seasonality: false,
    seasonal_weight: 0.5,
    enforce_two_size_minimum: true,
    enforce_lead_time_release_gate: false,
    urgent_all_sizes_days: 45,
    planning_date: calendarDateIST(),
    raise_ledger_lookback_days: 14,
    auto_import_yesterday_ledger: true,
    use_shared_cache: false,
  }
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
        <div className="p-6 max-w-xl">
          <h1 className="text-lg font-semibold text-red-700">PO page error</h1>
          <p className="mt-2 text-sm text-slate-700 font-mono break-all">{this.state.error}</p>
          <button
            type="button"
            className="mt-4 px-4 py-2 rounded-lg bg-[#002B5B] text-white text-sm"
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
  const [params, setParams] = useState(DEFAULT_PARAMS)
  const [dataStatus, setDataStatus] = useState<{ loaded: number; total: number; ready: boolean } | null>(
    null,
  )
  const [checkingData, setCheckingData] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState<{ msg: string; pct: number | null } | null>(null)
  const [result, setResult] = useState<POCalculateResult | null>(null)
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const progressThrottle = useRef(0)

  const refreshDataStatus = useCallback(async () => {
    setCheckingData(true)
    try {
      const c = await getCoverage({ light: true, timeout: 45_000 })
      setDataStatus(countLoaded(c))
    } catch (e: unknown) {
      setResult({
        ok: false,
        message: e instanceof Error ? e.message : 'Failed to read data coverage',
      })
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

  const runCalculate = useCallback(async () => {
    setLoading(true)
    setProgress({ msg: 'Checking data coverage…', pct: 1 })
    setResult(null)
    setPage(0)
    try {
      const c = await getCoverage({ light: true, timeout: 45_000 })
      const status = countLoaded(c)
      setDataStatus(status)
      if (!status.ready) {
        setResult({
          ok: false,
          message: `Data not ready (${status.loaded}/${status.total}). Load cache from sidebar, then retry.`,
        })
        return
      }
      const out = await startPoCalculate(buildBody(params), onProgress)
      setResult(out)
    } catch (e: unknown) {
      setResult({
        ok: false,
        message: e instanceof Error ? e.message : 'PO calculation failed',
      })
    } finally {
      setLoading(false)
      setProgress(null)
    }
  }, [onProgress, params])

  const allRows = useMemo(() => {
    const rows = (result?.rows ?? []) as Row[]
    if (!search.trim()) return rows
    const q = search.trim().toLowerCase()
    return rows.filter(r => String(r.OMS_SKU ?? '').toLowerCase().includes(q))
  }, [result?.rows, search])

  const pageRows = useMemo(() => {
    const start = page * PAGE_SIZE
    return allRows.slice(start, start + PAGE_SIZE)
  }, [allRows, page])

  const totalPO = useMemo(
    () => allRows.reduce((s, r) => s + num(r.PO_Qty), 0),
    [allRows],
  )

  const visibleCols = useMemo(() => {
    const fromApi = result?.columns ?? []
    return DISPLAY_COLS.filter(c => fromApi.length === 0 || fromApi.includes(c))
  }, [result?.columns])

  const pageCount = Math.max(1, Math.ceil(allRows.length / PAGE_SIZE))

  return (
    <div className="p-4 md:p-6 max-w-[100vw] space-y-5" data-testid="po-fresh-root">
      <div>
        <h1 className="text-2xl font-bold text-[#002B5B]">PO Engine (Fresh)</h1>
        <p className="text-sm text-slate-600 mt-1">
          Minimal PO UI — app data only, explicit calculate, no auto-load on mount.
        </p>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 flex flex-wrap items-center gap-3">
        <span className="text-sm text-slate-700">
          Data coverage:{' '}
          {dataStatus ? (
            <strong className={dataStatus.ready ? 'text-emerald-700' : 'text-amber-700'}>
              {dataStatus.loaded}/{dataStatus.total}
            </strong>
          ) : (
            <span className="text-slate-400">not checked yet</span>
          )}
        </span>
        <button
          type="button"
          disabled={checkingData}
          onClick={() => void refreshDataStatus()}
          className="text-sm px-3 py-1.5 rounded-lg border border-slate-300 hover:bg-slate-50 disabled:opacity-50"
        >
          {checkingData ? 'Checking…' : 'Check data status'}
        </button>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
            label="Post-PO running days"
            value={params.target_days}
            onChange={v => setParams(p => ({ ...p, target_days: v }))}
          />
          <label className="text-xs font-semibold text-slate-500 uppercase">
            Demand basis
            <select
              value={params.demand_basis}
              onChange={e =>
                setParams(p => ({ ...p, demand_basis: e.target.value as 'Sold' | 'Net' }))
              }
              className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            >
              <option value="Sold">Sold</option>
              <option value="Net">Net</option>
            </select>
          </label>
        </div>

        <button
          type="button"
          disabled={loading}
          onClick={() => void runCalculate()}
          className="px-5 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
        >
          {loading ? 'Calculating…' : 'Calculate PO'}
        </button>

        {progress && (
          <div className="space-y-1 max-w-md">
            <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
              <div
                className="h-full bg-[#002B5B] transition-[width]"
                style={{ width: `${progress.pct ?? 8}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">{progress.msg}</p>
          </div>
        )}
      </div>

      {result && !result.ok && (
        <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {result.message}
        </p>
      )}

      {result?.ok && allRows.length > 0 && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            <Stat label="New PO units" value={totalPO.toLocaleString()} />
            <Stat
              label="SKUs with PO"
              value={allRows.filter(r => num(r.PO_Qty) > 0).length.toLocaleString()}
            />
            <Stat label="Total rows" value={allRows.length.toLocaleString()} />
          </div>

          <div className="flex flex-wrap gap-3 items-center">
            <input
              value={search}
              onChange={e => {
                setSearch(e.target.value)
                setPage(0)
              }}
              placeholder="Search SKU…"
              className="border border-slate-300 rounded-lg px-3 py-2 text-sm w-56"
            />
            <span className="text-xs text-slate-400">
              Page {page + 1} / {pageCount} · {pageRows.length} shown
            </span>
            <button
              type="button"
              disabled={page <= 0}
              onClick={() => setPage(p => Math.max(0, p - 1))}
              className="text-sm px-2 py-1 border rounded disabled:opacity-40"
            >
              Prev
            </button>
            <button
              type="button"
              disabled={page + 1 >= pageCount}
              onClick={() => setPage(p => p + 1)}
              className="text-sm px-2 py-1 border rounded disabled:opacity-40"
            >
              Next
            </button>
          </div>

          <div className="overflow-auto rounded-xl border border-slate-200 bg-white max-h-[65vh]">
            <table className="min-w-full text-xs">
              <thead className="sticky top-0 bg-slate-100">
                <tr>
                  {visibleCols.map(c => (
                    <th key={c} className="px-3 py-2 text-left font-semibold whitespace-nowrap">
                      {c.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map(r => (
                  <tr key={String(r.OMS_SKU)} className="border-t border-slate-100">
                    {visibleCols.map(c => (
                      <td key={c} className="px-3 py-1.5 whitespace-nowrap">
                        {formatCell(c, r[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {result?.ok && allRows.length === 0 && (
        <p className="text-sm text-slate-600">Calculation finished with zero rows.</p>
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
    <label className="text-xs font-semibold text-slate-500 uppercase">
      {label}
      <input
        type="number"
        value={value}
        onChange={e => onChange(+e.target.value)}
        className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
      />
    </label>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-4 py-3">
      <div className="text-[10px] font-semibold uppercase text-slate-500">{label}</div>
      <div className="text-xl font-bold text-[#002B5B]">{value}</div>
    </div>
  )
}

function formatCell(col: string, v: unknown): string {
  if (v == null || v === '') return '—'
  if (typeof v === 'number') {
    if (col.includes('Days') || col.includes('ADS')) return v.toFixed(1)
    return Number.isInteger(v) ? String(v) : v.toFixed(2)
  }
  return String(v)
}
