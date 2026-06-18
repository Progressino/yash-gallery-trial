/**
 * PO Engine 2 — fresh, minimal purchase-order recommendations.
 * App data only; no mount-time auto-calculate; no Excel PO upload dependency.
 */
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import api, { startPoCalculate, type POCalculateResult } from '../api/client'
import { calendarDateIST } from '../lib/dates'
import { poOperationalReady, poOperationalLoaded, PO_OPERATIONAL_TOTAL } from '../lib/localSessionHint'
import { useSession } from '../store/session'
import { PO2_DEFAULT_PARAMS, usePO2Store, type PO2Row } from '../store/po2'

const DISPLAY_COLS = [
  'Priority',
  'OMS_SKU',
  'Total_Inventory',
  'Projected_Running_Days',
  'Post_PO_Cover_Days_Capped',
  'ADS',
  'PO_Qty',
] as const

const PRIORITY_ORDER: Record<string, number> = {
  '🔴 URGENT': 0,
  '🟡 HIGH': 1,
  '🟢 MEDIUM': 2,
  '🔄 In Pipeline': 3,
  '⚪ OK': 4,
}

function num(v: unknown): number {
  const n = Number(v)
  return Number.isFinite(n) ? n : 0
}

function buildCalculateBody(params: typeof PO2_DEFAULT_PARAMS) {
  return {
    period_days: params.period_days,
    lead_time: params.lead_time,
    target_days: params.target_days,
    grace_days: params.grace_days,
    demand_basis: params.demand_basis,
    group_by_parent: false,
    safety_pct: 0,
    use_seasonality: false,
    seasonal_weight: 0.5,
    enforce_two_size_minimum: params.enforce_two_size_minimum,
    enforce_lead_time_release_gate: false,
    urgent_all_sizes_days: params.urgent_all_sizes_days,
    planning_date: calendarDateIST(),
    raise_ledger_lookback_days: 14,
    auto_import_yesterday_ledger: true,
    use_shared_cache: false,
  }
}

function downloadCsv(rows: PO2Row[], columns: string[]) {
  const header = columns.join(',')
  const body = rows
    .map(r => columns.map(c => JSON.stringify(r[c] ?? '')).join(','))
    .join('\n')
  const blob = new Blob(['\ufeff' + header + '\n' + body], { type: 'text/csv;charset=utf-8;' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `po_recommendation_${calendarDateIST()}.csv`
  a.click()
}

export default function PO2() {
  const { params, setParams } = usePO2Store()
  const dataReady = useSession(s => poOperationalReady(s))
  const dataLoadLoaded = useSession(s => poOperationalLoaded(s))
  const dataLoadTotal = PO_OPERATIONAL_TOTAL

  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState('')
  const [progressPct, setProgressPct] = useState<number | null>(null)
  const [result, setResult] = useState<POCalculateResult | null>(null)
  const [search, setSearch] = useState('')
  const [buildLabel, setBuildLabel] = useState<string | null>(null)

  useEffect(() => {
    api
      .get<{ git_sha?: string; built_at?: string }>('/health')
      .then(r => {
        const sha = r.data.git_sha
        if (!sha) return
        const built = r.data.built_at ? ` · ${r.data.built_at.slice(0, 10)}` : ''
        setBuildLabel(`${sha}${built}`)
      })
      .catch(() => {})
  }, [])

  const rows = useMemo(() => {
    const all = (result?.rows ?? []) as PO2Row[]
    const filtered = search
      ? all.filter(r => String(r.OMS_SKU ?? '').toLowerCase().includes(search.toLowerCase()))
      : all
    return [...filtered].sort(
      (a, b) =>
        (PRIORITY_ORDER[String(a.Priority)] ?? 9) - (PRIORITY_ORDER[String(b.Priority)] ?? 9),
    )
  }, [result?.rows, search])

  const totalPO = useMemo(
    () => rows.reduce((s, r) => s + num(r.PO_Qty), 0),
    [rows],
  )

  const urgentCount = useMemo(
    () => rows.filter(r => r.Priority === '🔴 URGENT').length,
    [rows],
  )

  const visibleCols = useMemo(() => {
    const fromApi = result?.columns ?? []
    return DISPLAY_COLS.filter(c => fromApi.length === 0 || fromApi.includes(c))
  }, [result?.columns])

  const runCalculate = useCallback(async () => {
    if (!dataReady) {
      setResult({
        ok: false,
        message: `Data still loading (${dataLoadLoaded}/${dataLoadTotal}). Wait for sidebar ${dataLoadTotal}/${dataLoadTotal}, then try again.`,
      })
      return
    }
    setLoading(true)
    setProgress('Starting PO calculation…')
    setProgressPct(2)
    setResult(null)
    try {
      const out = await startPoCalculate(buildCalculateBody(params), (msg, pct) => {
        setProgress(msg)
        if (pct != null && Number.isFinite(pct)) setProgressPct(pct)
      })
      setResult(out)
    } catch (e: unknown) {
      setResult({
        ok: false,
        message: e instanceof Error ? e.message : 'PO calculation failed',
      })
    } finally {
      setLoading(false)
      setProgress('')
      setProgressPct(null)
    }
  }, [dataReady, dataLoadLoaded, dataLoadTotal, params])

  return (
    <div className="p-4 md:p-6 max-w-[100vw] space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-[#002B5B]">PO Engine 2</h1>
          <p className="text-sm text-slate-600 mt-1">
            App-data recommendations — fills toward post-PO target days from sales, inventory, and raise ledger.
          </p>
        </div>
        {buildLabel ? (
          <span className="text-[11px] font-mono text-slate-600 bg-white border border-slate-200 rounded px-2 py-1">
            Build {buildLabel}
          </span>
        ) : null}
      </div>

      {!dataReady ? (
        <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          Loading server data ({dataLoadLoaded}/{dataLoadTotal}). PO calculation unlocks at{' '}
          <strong>{dataLoadTotal}/{dataLoadTotal}</strong>. Use sidebar <strong>Load cache</strong> if this stays below {dataLoadTotal}/{dataLoadTotal}.
        </div>
      ) : (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
          Data ready ({dataLoadLoaded}/{dataLoadTotal}). You can calculate PO.
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-800 mb-3">Parameters</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <label className="text-xs font-semibold text-slate-500 uppercase">
            Period (days)
            <input
              type="number"
              min={7}
              max={365}
              value={params.period_days}
              onChange={e => setParams({ ...params, period_days: +e.target.value })}
              className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs font-semibold text-slate-500 uppercase">
            Lead time (days)
            <input
              type="number"
              min={1}
              max={180}
              value={params.lead_time}
              onChange={e => setParams({ ...params, lead_time: +e.target.value })}
              className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs font-semibold text-slate-500 uppercase">
            Post-PO running days
            <input
              type="number"
              min={30}
              max={365}
              value={params.target_days}
              onChange={e => setParams({ ...params, target_days: +e.target.value })}
              className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            />
          </label>
          <label className="text-xs font-semibold text-slate-500 uppercase">
            Demand basis
            <select
              value={params.demand_basis}
              onChange={e =>
                setParams({ ...params, demand_basis: e.target.value as 'Sold' | 'Net' })
              }
              className="mt-1 w-full border border-slate-300 rounded-lg px-3 py-2 text-sm"
            >
              <option value="Sold">Sold</option>
              <option value="Net">Net</option>
            </select>
          </label>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={loading || !dataReady}
            onClick={() => void runCalculate()}
            className="px-5 py-2.5 rounded-lg text-sm font-semibold text-white bg-[#002B5B] hover:bg-blue-800 disabled:opacity-50"
          >
            {loading ? 'Calculating…' : 'Calculate PO'}
          </button>
          <span className="text-xs text-slate-500">
            Formula: PO fills toward {params.target_days + params.grace_days}d post-PO cover (no Excel PO sheet).
          </span>
        </div>
        {loading && (
          <div className="mt-3 space-y-1">
            <div className="h-2 w-full max-w-md overflow-hidden rounded-full bg-slate-200">
              <div
                className="h-full rounded-full bg-[#002B5B] transition-[width]"
                style={{ width: `${progressPct ?? 5}%` }}
              />
            </div>
            <p className="text-xs text-slate-500">{progress || 'Running…'}</p>
          </div>
        )}
      </div>

      {result && !result.ok && (
        <p className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2">
          {result.message}
        </p>
      )}

      {result?.ok && rows.length > 0 && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Kpi label="New PO units" value={totalPO.toLocaleString()} />
            <Kpi label="SKUs with PO" value={rows.filter(r => num(r.PO_Qty) > 0).length.toLocaleString()} />
            <Kpi label="Urgent SKUs" value={urgentCount.toLocaleString()} accent="text-red-600" />
            <Kpi label="Rows" value={rows.length.toLocaleString()} />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <input
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search SKU…"
              className="border border-slate-300 rounded-lg px-3 py-2 text-sm w-56"
            />
            <button
              type="button"
              onClick={() => downloadCsv(rows, [...visibleCols])}
              className="text-sm px-3 py-2 rounded-lg border border-slate-300 hover:bg-slate-50"
            >
              Export CSV
            </button>
            <span className="text-xs text-slate-400">{rows.length} SKUs shown</span>
          </div>

          <div className="overflow-auto rounded-xl border border-slate-200 bg-white shadow-sm max-h-[70vh]">
            <table className="min-w-full text-xs">
              <thead className="sticky top-0 bg-slate-100 z-10">
                <tr>
                  {visibleCols.map(c => (
                    <th key={c} className="px-3 py-2 text-left font-semibold text-slate-700 whitespace-nowrap">
                      {c.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map(r => (
                  <tr key={String(r.OMS_SKU)} className="border-t border-slate-100 hover:bg-slate-50">
                    {visibleCols.map(c => (
                      <td
                        key={c}
                        className={`px-3 py-1.5 whitespace-nowrap ${
                          c === 'PO_Qty' && num(r.PO_Qty) > 0 ? 'font-semibold text-orange-700' : ''
                        }`}
                      >
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

      <p className="text-xs text-slate-500">
        Legacy PO Engine still available at{' '}
        <Link to="/po-legacy" className="underline text-[#002B5B]">/po-legacy</Link>.
      </p>
    </div>
  )
}

function Kpi({
  label,
  value,
  accent,
}: {
  label: string
  value: string
  accent?: string
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm">
      <div className="text-[10px] font-semibold uppercase text-slate-500">{label}</div>
      <div className={`text-xl font-bold ${accent ?? 'text-[#002B5B]'}`}>{value}</div>
    </div>
  )
}

function formatCell(col: string, v: unknown): string {
  if (v == null || v === '') return '—'
  if (typeof v === 'number') {
    if (col.includes('Days') || col.includes('ADS')) return Number(v).toFixed(1)
    return Number.isInteger(v) ? String(v) : Number(v).toFixed(2)
  }
  return String(v)
}
