import { Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import { mayAccessErpAdmin, useAuth } from '../store/auth'

interface DaySummary {
  count: number
  avg_sec: number | null
  p95_sec: number | null
  max_sec: number | null
}

interface SlowRow {
  name: string
  count: number
  avg_sec: number
  max_sec: number
  p95_sec: number
}

interface PoRecent {
  at: string
  duration_sec: number
  ok?: boolean
  total_rows?: number
  ads_source?: string
}

interface PerfDashboard {
  generated_at: string
  window_hours: number
  samples_in_window: number
  po_calculate: {
    today: DaySummary
    yesterday: DaySummary
    recent: PoRecent[]
  }
  slowest_endpoints: SlowRow[]
  slowest_queries: SlowRow[]
  session_restore: {
    summary: DaySummary
    slowest: SlowRow[]
  }
  cache: {
    hits: number
    misses: number
    hit_rate: number | null
    by_source: Record<string, number>
  }
  postgres: {
    available?: boolean
    settings?: Record<string, string>
    connections?: { active: number; idle: number; total: number }
    buffer_cache_hit_ratio?: number | null
    database_size?: string
    backend_memory_pretty?: string
    container_memory_pretty?: string
    error?: string
  }
}

function fmtSec(v: number | null | undefined, fallback = '—') {
  if (v == null || Number.isNaN(v)) return fallback
  return `${v}s`
}

function PoCompareCard({ label, data }: { label: string; data: DaySummary }) {
  return (
    <div className="bg-white rounded-xl p-4 border border-gray-100 shadow-sm">
      <p className="text-xs font-semibold tracking-wide text-gray-500 uppercase">{label}</p>
      <p className="text-3xl font-bold text-[#002B5B] mt-1">{fmtSec(data.avg_sec)}</p>
      <p className="text-xs text-gray-400 mt-2">
        {data.count} run{data.count === 1 ? '' : 's'}
        {data.p95_sec != null ? ` · p95 ${fmtSec(data.p95_sec)}` : ''}
        {data.max_sec != null ? ` · max ${fmtSec(data.max_sec)}` : ''}
      </p>
    </div>
  )
}

function SlowTable({ title, rows, nameLabel }: { title: string; rows: SlowRow[]; nameLabel: string }) {
  return (
    <div className="bg-white rounded-xl border p-4 overflow-hidden">
      <h3 className="font-semibold text-gray-700 mb-3 text-sm">{title}</h3>
      {rows.length === 0 ? (
        <p className="text-sm text-gray-400">No samples in window.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-400 border-b">
                <th className="py-2 pr-3">{nameLabel}</th>
                <th className="py-2 pr-3 text-right">Count</th>
                <th className="py-2 pr-3 text-right">Avg</th>
                <th className="py-2 pr-3 text-right">P95</th>
                <th className="py-2 text-right">Max</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.name} className="border-b border-gray-50 last:border-0">
                  <td className="py-2 pr-3 font-mono text-xs text-gray-700 max-w-md truncate" title={r.name}>
                    {r.name}
                  </td>
                  <td className="py-2 pr-3 text-right text-gray-600">{r.count}</td>
                  <td className="py-2 pr-3 text-right">{fmtSec(r.avg_sec)}</td>
                  <td className="py-2 pr-3 text-right">{fmtSec(r.p95_sec)}</td>
                  <td className="py-2 text-right font-medium text-amber-700">{fmtSec(r.max_sec)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function AdminPerformance() {
  const authUser = useAuth(s => s.user)

  const { data, isLoading, isError, refetch, isFetching } = useQuery<PerfDashboard>({
    queryKey: ['admin-performance'],
    queryFn: () => api.get('/admin/performance', { params: { hours: 48 } }).then(r => r.data),
    enabled: !!mayAccessErpAdmin(authUser),
    refetchInterval: 30_000,
  })

  if (!mayAccessErpAdmin(authUser)) {
    return (
      <div className="p-8 text-center text-gray-600">
        <p className="font-medium">Admin access required</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
            <Link to="/admin" className="hover:text-[#002B5B]">Admin</Link>
            <span>/</span>
            <span>Performance</span>
          </div>
          <h1 className="text-xl font-bold text-gray-800">Performance Dashboard</h1>
          <p className="text-sm text-gray-500">
            PO timings, slow endpoints, queries, cache, snapshot restore, and PostgreSQL — no SSH.
          </p>
        </div>
        <button
          type="button"
          onClick={() => refetch()}
          disabled={isFetching}
          className="px-3 py-1.5 border border-gray-200 rounded-lg text-sm text-gray-600 hover:bg-gray-50 disabled:opacity-50"
        >
          {isFetching ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>

      {isLoading && <p className="text-sm text-gray-500">Loading metrics…</p>}
      {isError && <p className="text-sm text-red-600">Failed to load performance data.</p>}

      {data && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <PoCompareCard label="PO avg — today (IST)" data={data.po_calculate.today} />
            <PoCompareCard label="PO avg — yesterday (IST)" data={data.po_calculate.yesterday} />
          </div>

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-2 text-sm">Cache hit rate</h3>
              <p className="text-2xl font-bold text-green-700">
                {data.cache.hit_rate != null ? `${(data.cache.hit_rate * 100).toFixed(1)}%` : '—'}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {data.cache.hits} hits · {data.cache.misses} misses
              </p>
            </div>
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-2 text-sm">Snapshot restore (today)</h3>
              <p className="text-2xl font-bold text-[#002B5B]">{fmtSec(data.session_restore.summary.avg_sec)}</p>
              <p className="text-xs text-gray-400 mt-1">{data.session_restore.summary.count} restores</p>
            </div>
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-2 text-sm">PostgreSQL</h3>
              {data.postgres.available ? (
                <>
                  <p className="text-sm text-gray-700">
                    DB {data.postgres.database_size ?? '—'}
                    {data.postgres.container_memory_pretty
                      ? ` · container ${data.postgres.container_memory_pretty}`
                      : data.postgres.backend_memory_pretty
                        ? ` · backends ${data.postgres.backend_memory_pretty}`
                        : ''}
                  </p>
                  {data.postgres.connections && (
                    <p className="text-xs text-gray-400 mt-1">
                      {data.postgres.connections.active} active · {data.postgres.connections.idle} idle
                      {data.postgres.buffer_cache_hit_ratio != null
                        ? ` · buffer hit ${(data.postgres.buffer_cache_hit_ratio * 100).toFixed(1)}%`
                        : ''}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-sm text-gray-400">{data.postgres.error || 'PG ops not available'}</p>
              )}
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-4">
            <SlowTable title="Slowest endpoints" rows={data.slowest_endpoints} nameLabel="Route" />
            <SlowTable title="Slowest queries" rows={data.slowest_queries} nameLabel="Query" />
          </div>

          <div className="grid lg:grid-cols-2 gap-4">
            <div className="bg-white rounded-xl border p-4">
              <h3 className="font-semibold text-gray-700 mb-3 text-sm">Recent PO calculations</h3>
              {data.po_calculate.recent.length === 0 ? (
                <p className="text-sm text-gray-400">No PO runs in window.</p>
              ) : (
                <div className="space-y-2 max-h-72 overflow-y-auto">
                  {data.po_calculate.recent.map((r, i) => (
                    <div key={`${r.at}-${i}`} className="flex justify-between text-sm border-b border-gray-50 pb-2">
                      <div>
                        <p className="text-gray-700">{fmtSec(r.duration_sec)}</p>
                        <p className="text-xs text-gray-400">{r.at}</p>
                      </div>
                      <div className="text-right text-xs text-gray-500">
                        <p className={r.ok ? 'text-green-600' : 'text-red-600'}>{r.ok ? 'ok' : 'failed'}</p>
                        {r.total_rows != null && <p>{r.total_rows.toLocaleString()} rows</p>}
                        {r.ads_source && <p>{r.ads_source}</p>}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <SlowTable title="Slowest snapshot restores" rows={data.session_restore.slowest} nameLabel="Source" />
          </div>

          <p className="text-xs text-gray-400">
            Window: {data.window_hours}h · {data.samples_in_window} samples · updated {data.generated_at}
          </p>
        </>
      )}
    </div>
  )
}
