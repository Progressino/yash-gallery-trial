import { useCallback, useMemo, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../api/client'
import { usePOStore } from '../store/po'
import { PageLoadingStripe } from './LoadingProgressBar'
import { calendarDateIST } from '../lib/dates'

/** Full PO-engine run; nginx `/api` allows up to 900s — keep client in sync so axios does not abort first. */
const PO_DASHBOARD_TIMEOUT_MS = 300_000
const PO_DASHBOARD_TIMEOUT_MIN = PO_DASHBOARD_TIMEOUT_MS / 60_000

type DashRow = Record<string, string | number>

interface DashboardPayload {
  ok: boolean
  message?: string
  summary?: Record<string, number>
  windows?: Record<string, string | number | undefined>
  in_production?: DashRow[]
  open_po?: DashRow[]
  spike_attention?: DashRow[]
  running_tight?: DashRow[]
  raised_ledger_active?: { oms_sku: string; qty: number; last_raised_date?: string }[]
  raised_ledger_skus?: number
  raised_ledger_units?: number
}

function SectionTable({
  title,
  subtitle,
  rows,
  accent,
}: {
  title: string
  subtitle: string
  rows: DashRow[]
  accent: 'slate' | 'amber' | 'rose' | 'emerald'
}) {
  const ring =
    accent === 'amber'
      ? 'border-amber-200'
      : accent === 'rose'
        ? 'border-rose-200'
        : accent === 'emerald'
          ? 'border-emerald-200'
          : 'border-slate-200'
  const cols = useMemo(() => (rows.length > 0 ? Object.keys(rows[0]) : []), [rows])
  return (
    <section className={`rounded-xl border bg-white shadow-sm ${ring}`}>
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/80">
        <h2 className="text-sm font-bold text-[#002B5B]">{title}</h2>
        <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>
      </div>
      {rows.length === 0 ? (
        <p className="text-sm text-gray-400 px-4 py-6 text-center">No rows match right now.</p>
      ) : (
        <div className="overflow-x-auto max-h-[min(420px,50vh)]">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-white border-b border-gray-200 z-10">
              <tr>
                {cols.map(c => (
                  <th key={c} className="text-left px-3 py-2 font-semibold text-gray-600 whitespace-nowrap">
                    {c.replace(/_/g, ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/60">
                  {cols.map(c => (
                    <td key={c} className="px-3 py-1.5 text-gray-800 whitespace-nowrap max-w-[220px] truncate">
                      {String(r[c] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export interface PODashboardPanelProps {
  /** When true, omit standalone page chrome (used inside PO Engine). */
  embedded?: boolean
}

export function PODashboardPanel({ embedded = false }: PODashboardPanelProps) {
  const params = usePOStore(s => s.params)
  const [tuning, setTuning] = useState({
    recent_days: 7,
    prev_days: 7,
    spike_ratio: 1.35,
    min_recent_units: 5,
    low_run_days: 40,
    max_rows_per_section: 80,
  })
  const [data, setData] = useState<DashboardPayload | null>(null)

  const mut = useMutation({
    mutationFn: async () => {
      const { data } = await api.post<DashboardPayload>(
        '/po/dashboard',
        {
          ...params,
          min_denominator: 7,
          ...tuning,
          planning_date: calendarDateIST(),
          raise_ledger_lookback_days: 14,
        },
        { timeout: PO_DASHBOARD_TIMEOUT_MS },
      )
      return data
    },
    onSuccess: d => setData(d),
  })

  // useMutation's `mutate` is referentially stable in v5; depending on the whole
  // `mut` object causes an effect-rerun-per-render loop (the panel previously
  // fired requests in a tight loop on first activation).
  const mutate = mut.mutate
  const load = useCallback(() => void mutate(), [mutate])

  const errorMsg = (() => {
    if (!mut.isError) return null
    const e = mut.error as { code?: string; message?: string; response?: { status?: number; data?: { message?: string } } } | null
    if (!e) return 'Request failed.'
    if (e.code === 'ECONNABORTED' || /timeout/i.test(e.message || '')) {
      return `Dashboard request timed out after ${PO_DASHBOARD_TIMEOUT_MIN} minutes. The catalog run can be very heavy — try again in a moment, or ask an admin if this keeps happening.`
    }
    if (e.response?.data?.message) return e.response.data.message
    if (!e.response) return 'Cannot reach the server. Check your connection and try again.'
    return `Request failed (status ${e.response.status}).`
  })()

  const summary = data?.summary
  const win = data?.windows

  const rootCls = embedded
    ? 'space-y-6 max-w-[1600px]'
    : 'p-4 md:p-6 max-w-[1600px] mx-auto space-y-6'

  return (
    <div className={rootCls}>
      <PageLoadingStripe
        active={mut.isPending}
        label="Running PO dashboard…"
        className="sticky top-0 z-20 mb-3"
      />
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          {embedded ? (
            <h3 className="text-lg font-bold text-[#002B5B]">📋 PO Dashboard</h3>
          ) : (
            <h1 className="text-2xl font-bold text-[#002B5B]">PO Dashboard</h1>
          )}
          <p className="text-sm text-gray-600 mt-1 max-w-3xl">
            Pipeline vs open recommendations, short-horizon sell-through vs the prior week, and SKUs that are spiking or
            running tight on cover. Uses the same PO engine parameters as the{' '}
            {embedded ? (
              <strong className="text-gray-800">PO Recommendation</strong>
            ) : (
              <span className="text-gray-800 font-medium">PO Engine</span>
            )}{' '}
            tab.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          <button
            type="button"
            onClick={load}
            disabled={mut.isPending}
            className="text-sm px-5 py-2 rounded-lg bg-[#002B5B] text-white font-semibold hover:bg-blue-900 disabled:opacity-50"
          >
            {mut.isPending ? 'Loading…' : data ? 'Refresh dashboard' : 'Load dashboard'}
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50/80 p-4 grid sm:grid-cols-2 lg:grid-cols-3 gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Recent window (days)</span>
          <input
            type="number"
            min={1}
            max={30}
            className="border rounded px-2 py-1"
            value={tuning.recent_days}
            onChange={e => setTuning(t => ({ ...t, recent_days: +e.target.value || 7 }))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Compare vs previous (days)</span>
          <input
            type="number"
            min={1}
            max={30}
            className="border rounded px-2 py-1"
            value={tuning.prev_days}
            onChange={e => setTuning(t => ({ ...t, prev_days: +e.target.value || 7 }))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Spike ratio (recent ÷ prev)</span>
          <input
            type="number"
            step={0.05}
            min={1}
            className="border rounded px-2 py-1"
            value={tuning.spike_ratio}
            onChange={e => setTuning(t => ({ ...t, spike_ratio: +e.target.value || 1.35 }))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Min recent units (noise floor)</span>
          <input
            type="number"
            min={0}
            className="border rounded px-2 py-1"
            value={tuning.min_recent_units}
            onChange={e => setTuning(t => ({ ...t, min_recent_units: +e.target.value || 0 }))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Tight cover threshold (days)</span>
          <input
            type="number"
            min={1}
            className="border rounded px-2 py-1"
            value={tuning.low_run_days}
            onChange={e => setTuning(t => ({ ...t, low_run_days: +e.target.value || 40 }))}
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="font-semibold text-gray-700">Max rows / section</span>
          <input
            type="number"
            min={10}
            max={300}
            className="border rounded px-2 py-1"
            value={tuning.max_rows_per_section}
            onChange={e => setTuning(t => ({ ...t, max_rows_per_section: +e.target.value || 80 }))}
          />
        </label>
        <p className="sm:col-span-2 lg:col-span-3 text-gray-500">
          PO parameters (period, lead, target, two-size rule, etc.) follow your saved settings from the{' '}
          <strong>PO Recommendation</strong> tab{embedded ? '' : ' in PO Engine'}.
        </p>
      </div>

      {!data && !mut.isPending && !mut.isError && (
        <p className="text-sm text-gray-600 rounded-lg border border-slate-200 bg-slate-50/80 px-4 py-3">
          Nothing is fetched until you choose <strong className="text-[#002B5B]">Load dashboard</strong> — tune the windows above if needed, then run when you are ready (large catalogs can take several minutes; the browser waits up to {PO_DASHBOARD_TIMEOUT_MIN} minutes before timing out).
        </p>
      )}

      {errorMsg && (
        <div className="rounded-lg bg-rose-50 text-rose-800 text-sm px-4 py-3 border border-rose-200">
          {errorMsg}
        </div>
      )}

      {data && !data.ok && (
        <div className="rounded-lg bg-amber-50 text-amber-900 text-sm px-4 py-3 border border-amber-200">{data.message}</div>
      )}

      {data?.ok && summary && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {[
              ['SKU rows', summary.sku_rows],
              ['In production', summary.in_production_skus],
              ['Open PO rec.', summary.open_po_skus],
              ['Spike attention', summary.spike_attention_skus],
              ['Running tight', summary.running_tight_skus],
              ['OOS + demand', summary.oos_skus],
            ].map(([k, v]) => (
              <div key={String(k)} className="rounded-lg border border-gray-200 bg-white p-3 shadow-sm">
                <div className="text-[10px] uppercase tracking-wide text-gray-500 font-semibold">{k}</div>
                <div className="text-2xl font-bold text-[#002B5B] mt-1">{v}</div>
              </div>
            ))}
          </div>
          <div className="grid md:grid-cols-3 gap-3 text-xs text-gray-600">
            <div className="rounded-lg border bg-white p-3">
              <strong className="text-gray-800">Pipeline units</strong>
              <div>Total pipeline (sheet + raised): {summary.total_pipeline_units?.toLocaleString()}</div>
              <div>From sheet only: {summary.total_sheet_pipeline_units?.toLocaleString()}</div>
              <div>Raised in app (active): {summary.total_raised_recent_units?.toLocaleString()}</div>
            </div>
            <div className="rounded-lg border bg-white p-3">
              <strong className="text-gray-800">Open PO</strong>
              <div>Net recommended units: {summary.total_open_po_units?.toLocaleString()}</div>
            </div>
            <div className="rounded-lg border bg-white p-3">
              <strong className="text-gray-800">Sales window</strong>
              <div>Recent: {win?.recent_days}d vs prev: {win?.prev_days}d</div>
              <div>Latest sales date: {win?.sales_max_date ?? '—'}</div>
            </div>
          </div>
        </>
      )}

      {data?.ok && (
        <div className="grid xl:grid-cols-2 gap-6">
          <SectionTable
            title="Under production / in pipeline"
            subtitle="SKUs with pipeline units (existing PO sheet + quantities raised and confirmed in the app). Pipeline_From_Sheet excludes app-raised-only layering."
            rows={data.in_production ?? []}
            accent="slate"
          />
          <SectionTable
            title="Open PO recommendations"
            subtitle="Engine net PO_Qty > 0 — still to raise after pipeline and rules."
            rows={data.open_po ?? []}
            accent="emerald"
          />
          <SectionTable
            title="Spike + needs attention"
            subtitle={`Recent ${win?.recent_days ?? 7}d shipments up sharply vs prior ${win?.prev_days ?? 7}d, and cover below threshold (running days or days left).`}
            rows={data.spike_attention ?? []}
            accent="amber"
          />
          <SectionTable
            title="Running tight on cover"
            subtitle={`Projected running days below ${win?.low_run_days_threshold ?? tuning.low_run_days} (with ADS > 0) — review even if not a spike.`}
            rows={data.running_tight ?? []}
            accent="rose"
          />
        </div>
      )}

      {data?.ok && (data.raised_ledger_active?.length ?? 0) > 0 && (
        <section className="rounded-xl border border-sky-200 bg-sky-50/50 p-4">
          <h3 className="text-sm font-bold text-sky-900">Active raised PO ledger (app)</h3>
          <p className="text-xs text-sky-800 mt-1 mb-3">
            {data.raised_ledger_skus} SKUs · {data.raised_ledger_units?.toLocaleString()} units — cleared automatically after
            lead time or from PO Engine / API.
          </p>
          <ul className="text-xs grid sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-40 overflow-y-auto">
            {(data.raised_ledger_active ?? []).map((r, i) => (
              <li key={i} className="font-mono bg-white/80 rounded px-2 py-1 border border-sky-100">
                {r.oms_sku} · {r.qty} · {r.last_raised_date ?? ''}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
