"""Master Production Status Report UI — quantity by stage for the same SKU."""

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api'

type MasterLine = {
  so_number: string
  sku: string
  main_sku: string
  component_code: string
  process: string
  available_qty: number
  total_in: number
  total_out: number
  jo_numbers: string
  jo_planned: number
  jo_issued: number
  jo_received: number
  jo_balance: number
  updated_at?: string
}

type OverviewRow = {
  so_number: string
  sku: string
  main_sku: string
  component_code: string
  stage_qty: Record<string, { available: number; jo_balance: number }>
  total_available: number
  total_jo_balance: number
  jo_numbers: string
}

function fmt(n: number | undefined) {
  return Number(n || 0).toLocaleString()
}

export default function MasterProductionStatusPanel() {
  const [so, setSo] = useState('')
  const [sku, setSku] = useState('')
  const [mainSku, setMainSku] = useState('')
  const [component, setComponent] = useState('')
  const [jo, setJo] = useState('')
  const [process, setProcess] = useState('')
  const [q, setQ] = useState('')
  const [view, setView] = useState<'overview' | 'detail'>('overview')
  const [page, setPage] = useState(0)
  const pageSize = 100

  const params = useMemo(
    () => ({
      so_number: so.trim() || undefined,
      sku: sku.trim() || undefined,
      main_sku: mainSku.trim() || undefined,
      component: component.trim() || undefined,
      jo_number: jo.trim() || undefined,
      process: process.trim() || undefined,
      q: q.trim() || undefined,
      limit: pageSize,
      offset: page * pageSize,
    }),
    [so, sku, mainSku, component, jo, process, q, page],
  )

  const reportQ = useQuery({
    queryKey: ['master-status-report', params],
    queryFn: () => api.get('/production/master-status-report', { params }).then(r => r.data),
    staleTime: 30_000,
  })

  const stages: string[] = reportQ.data?.stages || []
  const lines: MasterLine[] = reportQ.data?.lines || []
  const overview: OverviewRow[] = reportQ.data?.overview || []
  const total = Number(reportQ.data?.total || 0)
  const pages = Math.max(1, Math.ceil(total / pageSize))

  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-semibold text-gray-800">Master Production Status</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Same SKU can appear at multiple stages (sizes / components / partial qty). Data from{' '}
          <code className="text-[10px]">process_stock</code> + open Job Orders — not a duplicate ledger.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
        <input className="border rounded-lg px-2 py-1.5 text-xs" placeholder="SO No." value={so} onChange={e => { setSo(e.target.value); setPage(0) }} />
        <input className="border rounded-lg px-2 py-1.5 text-xs" placeholder="Main / Parent SKU" value={mainSku} onChange={e => { setMainSku(e.target.value); setPage(0) }} />
        <input className="border rounded-lg px-2 py-1.5 text-xs font-mono" placeholder="Size / SKU" value={sku} onChange={e => { setSku(e.target.value); setPage(0) }} />
        <input className="border rounded-lg px-2 py-1.5 text-xs" placeholder="Component" value={component} onChange={e => { setComponent(e.target.value); setPage(0) }} />
        <input className="border rounded-lg px-2 py-1.5 text-xs" placeholder="JO No." value={jo} onChange={e => { setJo(e.target.value); setPage(0) }} />
        <select className="border rounded-lg px-2 py-1.5 text-xs" value={process} onChange={e => { setProcess(e.target.value); setPage(0) }}>
          <option value="">All stages</option>
          {stages.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <input className="border rounded-lg px-2 py-1.5 text-xs" placeholder="Search…" value={q} onChange={e => { setQ(e.target.value); setPage(0) }} />
      </div>

      <div className="flex items-center gap-2 flex-wrap">
        {([['overview', 'SKU overview (qty by stage)'], ['detail', 'Quantity detail']] as const).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setView(key)}
            className={`px-3 py-1.5 text-xs rounded-lg font-medium ${view === key ? 'bg-[#002B5B] text-white' : 'bg-white border text-gray-600'}`}
          >
            {label}
          </button>
        ))}
        <span className="text-xs text-gray-400 ml-auto">
          {reportQ.isFetching ? 'Loading…' : `${fmt(total)} stage lines · ${fmt(reportQ.data?.overview_total)} SKUs`}
        </span>
      </div>

      {reportQ.isError && (
        <p className="text-sm text-red-600">Failed to load master status report.</p>
      )}

      {view === 'overview' ? (
        <div className="bg-white rounded-xl border overflow-x-auto">
          <table className="w-full text-sm min-w-[720px]">
            <thead className="text-gray-400 text-xs uppercase bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2 sticky left-0 bg-gray-50">SO</th>
                <th className="text-left px-3 py-2">Main</th>
                <th className="text-left px-3 py-2">SKU</th>
                <th className="text-left px-3 py-2">Comp</th>
                {stages.map(s => (
                  <th key={s} className="text-right px-2 py-2 whitespace-nowrap">{s}</th>
                ))}
                <th className="text-right px-3 py-2">Total at stages</th>
              </tr>
            </thead>
            <tbody>
              {overview.map((r, i) => (
                <tr key={`${r.so_number}-${r.sku}-${i}`} className="border-t border-gray-50 hover:bg-slate-50">
                  <td className="px-3 py-1.5 font-semibold text-[#002B5B] sticky left-0 bg-white">{r.so_number}</td>
                  <td className="px-3 py-1.5 font-mono text-[11px]">{r.main_sku || '—'}</td>
                  <td className="px-3 py-1.5 font-mono text-[11px]">{r.sku}</td>
                  <td className="px-3 py-1.5 text-xs text-gray-500">{r.component_code || '—'}</td>
                  {stages.map(s => {
                    const cell = r.stage_qty?.[s]
                    const avail = cell?.available || 0
                    const bal = cell?.jo_balance || 0
                    return (
                      <td key={s} className={`px-2 py-1.5 text-right tabular-nums ${avail > 0 ? 'text-emerald-700 font-semibold' : bal > 0 ? 'text-amber-600' : 'text-gray-300'}`}>
                        {avail > 0 ? fmt(avail) : bal > 0 ? `(${fmt(bal)})` : '—'}
                      </td>
                    )
                  })}
                  <td className="px-3 py-1.5 text-right font-bold tabular-nums">{fmt(r.total_available)}</td>
                </tr>
              ))}
              {overview.length === 0 && !reportQ.isFetching && (
                <tr><td colSpan={stages.length + 5} className="text-center text-gray-400 py-8 text-sm">No WIP / JO rows match filters.</td></tr>
              )}
            </tbody>
          </table>
          <p className="text-[10px] text-gray-400 px-3 py-2">
            Green = available at stage (process_stock). Amber (n) = open JO balance with no stock row yet.
          </p>
        </div>
      ) : (
        <div className="bg-white rounded-xl border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="text-gray-400 text-xs uppercase bg-gray-50">
              <tr>
                <th className="text-left px-3 py-2">SO</th>
                <th className="text-left px-3 py-2">SKU</th>
                <th className="text-left px-3 py-2">Stage</th>
                <th className="text-right px-3 py-2">At stage</th>
                <th className="text-right px-3 py-2">In</th>
                <th className="text-right px-3 py-2">Out</th>
                <th className="text-right px-3 py-2">JO bal</th>
                <th className="text-left px-3 py-2">JO</th>
              </tr>
            </thead>
            <tbody>
              {lines.map((r, i) => (
                <tr key={`${r.so_number}-${r.sku}-${r.process}-${i}`} className="border-t border-gray-50">
                  <td className="px-3 py-1.5 font-semibold text-[#002B5B]">{r.so_number}</td>
                  <td className="px-3 py-1.5 font-mono text-[11px]">
                    {r.sku}
                    {r.component_code ? <span className="text-gray-400 ml-1">({r.component_code})</span> : null}
                  </td>
                  <td className="px-3 py-1.5">{r.process}</td>
                  <td className="px-3 py-1.5 text-right font-semibold text-emerald-700">{fmt(r.available_qty)}</td>
                  <td className="px-3 py-1.5 text-right text-blue-600">{fmt(r.total_in)}</td>
                  <td className="px-3 py-1.5 text-right text-gray-600">{fmt(r.total_out)}</td>
                  <td className="px-3 py-1.5 text-right text-amber-600">{fmt(r.jo_balance)}</td>
                  <td className="px-3 py-1.5 font-mono text-[10px] text-gray-500">{r.jo_numbers || '—'}</td>
                </tr>
              ))}
              {lines.length === 0 && !reportQ.isFetching && (
                <tr><td colSpan={8} className="text-center text-gray-400 py-8 text-sm">No rows.</td></tr>
              )}
            </tbody>
          </table>
          <div className="flex items-center justify-between px-3 py-2 border-t text-xs text-gray-500">
            <button type="button" disabled={page <= 0} className="px-2 py-1 border rounded disabled:opacity-40" onClick={() => setPage(p => Math.max(0, p - 1))}>← Prev</button>
            <span>Page {page + 1} / {pages}</span>
            <button type="button" disabled={page + 1 >= pages} className="px-2 py-1 border rounded disabled:opacity-40" onClick={() => setPage(p => p + 1)}>Next →</button>
          </div>
        </div>
      )}
    </div>
  )
}
