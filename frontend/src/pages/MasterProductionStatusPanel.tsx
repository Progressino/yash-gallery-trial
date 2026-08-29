/** Master Production Status Report — quantity by stage for the same SKU. */

import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

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

type MasterReport = {
  stages?: string[]
  lines?: MasterLine[]
  overview?: OverviewRow[]
  total?: number
  overview_total?: number
}

type CellDetail = {
  ok?: boolean
  so_number?: string
  sku?: string
  process?: string
  jos?: Array<{
    jo_number: string
    process: string
    status: string
    sku: string
    component_code?: string
    planned_qty: number
    issued_qty: number
    received_qty: number
    balance_qty: number
    vendor_name?: string
  }>
  stock?: Array<{
    sku: string
    process: string
    available_qty: number
    total_in: number
    total_out: number
    jo_number?: string
  }>
  components?: Array<{
    jo_number: string
    process: string
    sku: string
    component_code?: string
    planned_qty: number
    balance_qty: number
    status: string
  }>
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
  const [drill, setDrill] = useState<{ so_number: string; sku: string; process: string } | null>(null)
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
    queryFn: async () => {
      const r = await api.get('/production/master-status-report', { params })
      return r.data as MasterReport
    },
    staleTime: 30_000,
  })

  const detailQ = useQuery({
    queryKey: ['master-status-detail', drill],
    enabled: !!drill,
    queryFn: async () => {
      const r = await api.get('/production/master-status-report/detail', {
        params: {
          so_number: drill!.so_number,
          sku: drill!.sku,
          process: drill!.process || undefined,
        },
      })
      return r.data as CellDetail
    },
  })

  const data = reportQ.data
  const stages: string[] = data?.stages || []
  const lines: MasterLine[] = data?.lines || []
  const overview: OverviewRow[] = data?.overview || []
  const total = Number(data?.total || 0)
  const pages = Math.max(1, Math.ceil(total / pageSize))
  const overviewTotal = Number(data?.overview_total || 0)
  const detail = detailQ.data

  return (
    <div className="space-y-4">
      <div>
        <h3 className="font-semibold text-gray-800">Master Production Status</h3>
        <p className="text-xs text-gray-500 mt-0.5">
          Overview keeps one row per Size SKU with stage-wise quantities. Click a qty to open JO / component breakup.
          Data from <code className="text-[10px]">process_stock</code> + open Job Orders.
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
          {reportQ.isFetching ? 'Loading…' : `${fmt(total)} stage lines · ${fmt(overviewTotal)} SKUs`}
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
                    const has = avail > 0 || bal > 0
                    return (
                      <td key={s} className={`px-2 py-1.5 text-right tabular-nums ${avail > 0 ? 'text-emerald-700 font-semibold' : bal > 0 ? 'text-amber-600' : 'text-gray-300'}`}>
                        {has ? (
                          <button
                            type="button"
                            className="underline-offset-2 hover:underline"
                            title={`Open JO / component breakup — ${s}`}
                            onClick={() => setDrill({ so_number: r.so_number, sku: r.sku, process: s })}
                          >
                            {avail > 0 ? fmt(avail) : `(${fmt(bal)})`}
                          </button>
                        ) : (
                          '—'
                        )}
                      </td>
                    )
                  })}
                  <td className="px-3 py-1.5 text-right font-bold tabular-nums">
                    {r.total_available > 0 || r.total_jo_balance > 0 ? (
                      <button
                        type="button"
                        className="underline-offset-2 hover:underline"
                        onClick={() => setDrill({ so_number: r.so_number, sku: r.sku, process: '' })}
                      >
                        {fmt(r.total_available)}
                      </button>
                    ) : (
                      fmt(r.total_available)
                    )}
                  </td>
                </tr>
              ))}
              {overview.length === 0 && !reportQ.isFetching && (
                <tr><td colSpan={stages.length + 5} className="text-center text-gray-400 py-8 text-sm">No WIP / JO rows match filters.</td></tr>
              )}
            </tbody>
          </table>
          <p className="text-[10px] text-gray-400 px-3 py-2">
            Green = available at stage (process_stock). Amber (n) = open JO balance with no stock row yet. Click a quantity for JO-wise breakup.
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

      {drill && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setDrill(null)}>
          <div className="bg-white rounded-xl shadow-xl max-w-3xl w-full max-h-[85vh] overflow-auto" onClick={e => e.stopPropagation()}>
            <div className="flex items-start justify-between gap-3 px-4 py-3 border-b">
              <div>
                <h4 className="font-semibold text-[#002B5B] text-sm">Stage breakup</h4>
                <p className="text-xs text-gray-500 font-mono mt-0.5">
                  {drill.so_number} · {drill.sku}{drill.process ? ` · ${drill.process}` : ' · all stages'}
                </p>
              </div>
              <button type="button" className="text-gray-400 hover:text-gray-700 text-sm px-2" onClick={() => setDrill(null)}>✕</button>
            </div>
            <div className="p-4 space-y-4 text-sm">
              {detailQ.isFetching && <p className="text-gray-400 text-xs">Loading…</p>}
              {detailQ.isError && <p className="text-red-600 text-xs">Failed to load breakup.</p>}
              {detail && (
                <>
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Job orders</p>
                    {(detail.jos || []).length === 0 ? (
                      <p className="text-xs text-gray-400">No open JOs for this cell.</p>
                    ) : (
                      <table className="w-full text-xs">
                        <thead className="text-gray-400 uppercase bg-gray-50">
                          <tr>
                            <th className="text-left px-2 py-1">JO</th>
                            <th className="text-left px-2 py-1">Stage</th>
                            <th className="text-left px-2 py-1">Status</th>
                            <th className="text-right px-2 py-1">Planned</th>
                            <th className="text-right px-2 py-1">Issued</th>
                            <th className="text-right px-2 py-1">Recv</th>
                            <th className="text-right px-2 py-1">Balance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(detail.jos || []).map((j, i) => (
                            <tr key={`${j.jo_number}-${i}`} className="border-t">
                              <td className="px-2 py-1 font-mono font-semibold text-[#002B5B]">{j.jo_number}</td>
                              <td className="px-2 py-1">{j.process}</td>
                              <td className="px-2 py-1">{j.status}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(j.planned_qty)}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(j.issued_qty)}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(j.received_qty)}</td>
                              <td className="px-2 py-1 text-right tabular-nums font-semibold">{fmt(j.balance_qty)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Process stock</p>
                    {(detail.stock || []).length === 0 ? (
                      <p className="text-xs text-gray-400">No stock rows.</p>
                    ) : (
                      <table className="w-full text-xs">
                        <thead className="text-gray-400 uppercase bg-gray-50">
                          <tr>
                            <th className="text-left px-2 py-1">SKU</th>
                            <th className="text-left px-2 py-1">Stage</th>
                            <th className="text-right px-2 py-1">Available</th>
                            <th className="text-right px-2 py-1">In</th>
                            <th className="text-right px-2 py-1">Out</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(detail.stock || []).map((s, i) => (
                            <tr key={`${s.sku}-${s.process}-${i}`} className="border-t">
                              <td className="px-2 py-1 font-mono">{s.sku}</td>
                              <td className="px-2 py-1">{s.process}</td>
                              <td className="px-2 py-1 text-right tabular-nums text-emerald-700 font-semibold">{fmt(s.available_qty)}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(s.total_in)}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(s.total_out)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                  {(detail.components || []).length > 0 && (
                    <div>
                      <p className="text-xs font-semibold text-gray-500 uppercase mb-1">Component / panel JOs</p>
                      <table className="w-full text-xs">
                        <thead className="text-gray-400 uppercase bg-gray-50">
                          <tr>
                            <th className="text-left px-2 py-1">JO</th>
                            <th className="text-left px-2 py-1">SKU</th>
                            <th className="text-left px-2 py-1">Comp</th>
                            <th className="text-left px-2 py-1">Stage</th>
                            <th className="text-right px-2 py-1">Planned</th>
                            <th className="text-right px-2 py-1">Balance</th>
                          </tr>
                        </thead>
                        <tbody>
                          {(detail.components || []).map((c, i) => (
                            <tr key={`${c.jo_number}-${i}`} className="border-t">
                              <td className="px-2 py-1 font-mono">{c.jo_number}</td>
                              <td className="px-2 py-1 font-mono">{c.sku}</td>
                              <td className="px-2 py-1">{c.component_code || '—'}</td>
                              <td className="px-2 py-1">{c.process}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(c.planned_qty)}</td>
                              <td className="px-2 py-1 text-right tabular-nums">{fmt(c.balance_qty)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
