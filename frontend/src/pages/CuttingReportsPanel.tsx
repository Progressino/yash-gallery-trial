import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import { downloadCsv } from '../lib/exportCsv'

type CuttingRow = {
  so_number: string
  so_date: string
  delivery_date: string
  jo_number: string
  jo_date: string
  parent_style: string
  sku: string
  size: string
  component: string
  fabric_code: string
  planned_qty: number
  issued_qty: number
  received_qty: number
  balance_qty: number
  opening_balance?: number
  closing_balance?: number
  received_on_date?: number | null
  issued_on_date?: number | null
  production_mode?: string
  qty_variance: number
  status: string
  last_activity_date: string
  aging_days: number | null
  aging_bucket: string
  planned_fabric: number
  actual_fabric: number
  bom_avg: number | null
  actual_avg: number | null
  avg_diff: number | null
  fabric_saving: number | null
  fabric_saving_pct: number | null
}

const fmt = (n: number | null | undefined) => {
  const v = Number(n || 0)
  return v.toLocaleString()
}
const fmtN = (n: number | null | undefined, d = 2) => {
  if (n == null || Number.isNaN(Number(n))) return '—'
  return Number(n).toFixed(d)
}

const statusClass: Record<string, string> = {
  pending: 'bg-amber-100 text-amber-800',
  over: 'bg-rose-100 text-rose-800',
  under: 'bg-orange-100 text-orange-800',
  exact: 'bg-emerald-100 text-emerald-800',
}

const EXPORT_HEADERS = [
  'SO No', 'SO Date', 'Delivery Date', 'Cutting JO', 'JO Date', 'Parent Style', 'SKU', 'Size',
  'Component', 'Fabric / P-Code', 'Planned Qty', 'Issued Qty', 'Received Qty', 'Balance Qty',
  'Qty Variance', 'Status', 'Last Activity', 'Aging Days', 'Aging Bucket',
  'Planned Fabric', 'Actual Fabric', 'BOM Avg', 'Actual Avg', 'Avg Diff', 'Saving/Excess', 'Saving %',
]

export default function CuttingReportsPanel() {
  const [filters, setFilters] = useState({
    date_from: '', date_to: '', so_number: '', parent_style: '', sku: '', size: '',
    jo_number: '', component: '', fabric_code: '', status: '', aging_bucket: '',
    aging_basis: 'jo_date', variance: '', brand: '', search: '', group_by: 'so',
    as_of_date: '', activity_date: '',
  })
  const [page, setPage] = useState(1)
  const params = useMemo(() => ({ ...filters, page, page_size: 150 }), [filters, page])

  const { data, isFetching } = useQuery({
    queryKey: ['cutting-report', params],
    queryFn: () => api.get('/production/cutting-report', { params }).then(r => r.data),
  })

  const kpis = data?.kpis || {}
  const rows: CuttingRow[] = data?.rows || []
  const groups = data?.groups || []
  const total = Number(data?.total || 0)

  const set = (k: string, v: string) => {
    setPage(1)
    setFilters(f => ({ ...f, [k]: v }))
  }

  const exportAll = async () => {
    const res = await api.get('/production/cutting-report', { params: { ...filters, export: true, page_size: 0 } })
    const all: CuttingRow[] = res.data?.rows || []
    downloadCsv(
      `cutting_report_${new Date().toISOString().slice(0, 10)}.csv`,
      EXPORT_HEADERS,
      all.map(r => [
        r.so_number, r.so_date, r.delivery_date, r.jo_number, r.jo_date, r.parent_style, r.sku, r.size,
        r.component, r.fabric_code, r.planned_qty, r.issued_qty, r.received_qty, r.balance_qty,
        r.qty_variance, r.status, r.last_activity_date, r.aging_days ?? '', r.aging_bucket,
        r.planned_fabric, r.actual_fabric, r.bom_avg ?? '', r.actual_avg ?? '', r.avg_diff ?? '',
        r.fabric_saving ?? '', r.fabric_saving_pct ?? '',
      ]),
    )
  }

  const kpiCards: [string, string, string][] = [
    ['Total Planned', fmt(kpis.planned_qty), 'text-[#002B5B]'],
    ['Total Received', fmt(kpis.received_qty), 'text-emerald-700'],
    ['Total Balance', fmt(kpis.balance_qty), 'text-amber-700'],
    ...(filters.activity_date ? [
      ['Opening Bal.', fmt(kpis.opening_balance), 'text-slate-700'],
      ['Cut Today', fmt(kpis.received_on_date), 'text-emerald-800'],
      ['Closing Bal.', fmt(kpis.closing_balance), 'text-amber-800'],
    ] as [string, string, string][] : []),
    ['Pending JOs', fmt(kpis.pending_jos), 'text-amber-800'],
    ['Over Cutting', fmt(kpis.over_qty), 'text-rose-700'],
    ['Under Cutting', fmt(kpis.under_qty), 'text-orange-700'],
    ['Planned Fabric', fmtN(kpis.planned_fabric), 'text-sky-800'],
    ['Actual Fabric', fmtN(kpis.actual_fabric), 'text-indigo-800'],
    ['Saving / Excess', fmtN(kpis.fabric_saving), Number(kpis.fabric_saving) >= 0 ? 'text-emerald-700' : 'text-rose-700'],
  ]

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-gray-800">Cutting summary &amp; balance</h3>
          <p className="text-[11px] text-gray-500">
            Balance = Planned − Received (negative = over-receipt). Set <b>Activity date</b> for daily opening / cut / closing balance.
            {filters.as_of_date ? ` As-of ${filters.as_of_date}.` : ''}
            {filters.activity_date ? ` Daily view: ${filters.activity_date}.` : ''}
          </p>
        </div>
        <button type="button" onClick={() => void exportAll()} className="px-3 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg">
          ↓ Export Excel (CSV)
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-9 gap-2">
        {kpiCards.map(([l, v, c]) => (
          <div key={l} className="bg-white border rounded-lg p-2 text-center">
            <p className={`text-sm font-bold ${c}`}>{v}</p>
            <p className="text-[10px] text-gray-500">{l}</p>
          </div>
        ))}
      </div>

      <div className="bg-white border rounded-xl p-3 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
        {[
          ['date_from', 'JO from', 'date'], ['date_to', 'JO to', 'date'],
          ['as_of_date', 'As-of date', 'date'], ['activity_date', 'Activity date', 'date'],
          ['so_number', 'SO No', 'text'], ['parent_style', 'Parent style', 'text'],
          ['sku', 'SKU', 'text'], ['size', 'Size', 'text'],
          ['jo_number', 'Cutting JO', 'text'], ['component', 'Component', 'text'],
          ['fabric_code', 'Fabric / P-Code', 'text'], ['brand', 'Brand', 'text'],
          ['search', 'Search', 'text'],
        ].map(([k, label, type]) => (
          <label key={k} className="block">
            <span className="text-gray-500">{label}</span>
            <input type={type} value={(filters as any)[k]} onChange={e => set(k, e.target.value)}
              className="mt-0.5 w-full border rounded px-2 py-1" />
          </label>
        ))}
        <label className="block">
          <span className="text-gray-500">Status</span>
          <select value={filters.status} onChange={e => set('status', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="exact">Exact / completed</option>
            <option value="over">Over cutting</option>
            <option value="under">Under cutting</option>
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Over / Under</span>
          <select value={filters.variance} onChange={e => set('variance', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            <option value="pending">Pending</option>
            <option value="exact">Exact</option>
            <option value="over">Over</option>
            <option value="under">Under</option>
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Aging basis</span>
          <select value={filters.aging_basis} onChange={e => set('aging_basis', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="jo_date">Cutting JO date</option>
            <option value="so_date">SO date</option>
            <option value="delivery_date">Delivery / due date</option>
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Aging bucket</span>
          <select value={filters.aging_bucket} onChange={e => set('aging_bucket', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            {['0-2', '3-5', '6-10', '11-15', '15+'].map(b => <option key={b} value={b}>{b} days</option>)}
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Group by</span>
          <select value={filters.group_by} onChange={e => set('group_by', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="so">SO</option>
            <option value="parent_style">Parent style</option>
            <option value="sku">SKU / size</option>
            <option value="component">Component</option>
            <option value="jo">Cutting JO</option>
          </select>
        </label>
      </div>

      {groups.length > 0 && (
        <div className="bg-white border rounded-xl overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-gray-50 text-gray-500 uppercase">
              <tr>
                <th className="text-left px-3 py-2">Group</th>
                <th className="text-right px-3 py-2">Planned</th>
                <th className="text-right px-3 py-2">Issued</th>
                <th className="text-right px-3 py-2">Received</th>
                <th className="text-right px-3 py-2">Balance</th>
                <th className="text-right px-3 py-2">Variance</th>
                <th className="text-right px-3 py-2">Pending JOs</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g: any) => (
                <tr key={g.group} className="border-t">
                  <td className="px-3 py-1.5 font-semibold text-[#002B5B]">{g.group}</td>
                  <td className="px-3 py-1.5 text-right">{fmt(g.planned_qty)}</td>
                  <td className="px-3 py-1.5 text-right">{fmt(g.issued_qty)}</td>
                  <td className="px-3 py-1.5 text-right text-emerald-700">{fmt(g.received_qty)}</td>
                  <td className={`px-3 py-1.5 text-right font-semibold ${g.balance_qty > 0 ? 'text-amber-700' : 'text-emerald-700'}`}>{fmt(g.balance_qty)}</td>
                  <td className={`px-3 py-1.5 text-right ${g.qty_variance > 0 ? 'text-rose-700' : g.qty_variance < 0 ? 'text-orange-700' : ''}`}>{fmt(g.qty_variance)}</td>
                  <td className="px-3 py-1.5 text-right">{fmt(g.pending_jos)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="bg-white border rounded-xl overflow-auto">
        <div className="px-3 py-2 text-xs text-gray-500 flex justify-between">
          <span>{isFetching ? 'Loading…' : `${total.toLocaleString()} rows`}</span>
          <span>Page {page}</span>
        </div>
        <table className="w-full text-[11px]">
          <thead className="bg-gray-50 text-gray-500 uppercase sticky top-0">
            <tr>
              {['SO', 'JO', 'Path', 'Style', 'SKU', 'Size', 'Comp', 'Fabric', 'Plan', 'Iss', 'Rec', 'Bal', 'Open', 'Close', 'Today', 'Var', 'Status', 'Age', 'BOM Avg', 'Act Avg', 'Save %'].map(h => (
                <th key={h} className="text-left px-2 py-2 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={`${r.jo_number}-${r.sku}-${i}`} className="border-t border-gray-50 hover:bg-slate-50">
                <td className="px-2 py-1.5 font-semibold text-[#002B5B]">{r.so_number}</td>
                <td className="px-2 py-1.5 font-mono">{r.jo_number}</td>
                <td className="px-2 py-1.5 text-[10px]">{r.production_mode || '—'}</td>
                <td className="px-2 py-1.5">{r.parent_style}</td>
                <td className="px-2 py-1.5 font-mono">{r.sku}</td>
                <td className="px-2 py-1.5">{r.size || '—'}</td>
                <td className="px-2 py-1.5">{r.component || '—'}</td>
                <td className="px-2 py-1.5 font-mono">{r.fabric_code || '—'}</td>
                <td className="px-2 py-1.5 text-right">{fmt(r.planned_qty)}</td>
                <td className="px-2 py-1.5 text-right">{fmt(r.issued_qty)}</td>
                <td className="px-2 py-1.5 text-right text-emerald-700">{fmt(r.received_qty)}</td>
                <td className={`px-2 py-1.5 text-right font-semibold ${r.balance_qty > 0 ? 'text-amber-700' : r.balance_qty < 0 ? 'text-rose-700' : 'text-emerald-700'}`}>{fmt(r.balance_qty)}</td>
                <td className="px-2 py-1.5 text-right text-slate-600">{fmt(r.opening_balance)}</td>
                <td className="px-2 py-1.5 text-right text-slate-800">{fmt(r.closing_balance)}</td>
                <td className="px-2 py-1.5 text-right text-emerald-800 font-medium">{fmt(r.received_on_date)}</td>
                <td className={`px-2 py-1.5 text-right ${r.qty_variance > 0 ? 'text-rose-700' : r.qty_variance < 0 ? 'text-orange-700' : ''}`}>{fmt(r.qty_variance)}</td>
                <td className="px-2 py-1.5"><span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${statusClass[r.status] || 'bg-gray-100'}`}>{r.status}</span></td>
                <td className="px-2 py-1.5">{r.aging_bucket ? `${r.aging_bucket} (${r.aging_days}d)` : '—'}</td>
                <td className="px-2 py-1.5 text-right">{fmtN(r.bom_avg, 3)}</td>
                <td className="px-2 py-1.5 text-right">{fmtN(r.actual_avg, 3)}</td>
                <td className={`px-2 py-1.5 text-right ${(r.fabric_saving_pct || 0) >= 0 ? 'text-emerald-700' : 'text-rose-700'}`}>
                  {r.fabric_saving_pct == null ? '—' : `${fmtN(r.fabric_saving_pct)}%`}
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={21} className="text-center text-gray-400 py-8">No Cutting JOs match these filters.</td></tr>
            )}
          </tbody>
        </table>
        <div className="p-2 flex gap-2 justify-end">
          <button type="button" disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="px-2 py-1 border rounded text-xs disabled:opacity-40">Prev</button>
          <button type="button" disabled={page * 150 >= total} onClick={() => setPage(p => p + 1)} className="px-2 py-1 border rounded text-xs disabled:opacity-40">Next</button>
        </div>
      </div>
    </div>
  )
}
