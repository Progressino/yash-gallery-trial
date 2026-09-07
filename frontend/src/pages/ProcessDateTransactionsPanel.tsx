import { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../api/client'
import { downloadCsv } from '../lib/exportCsv'

const fmt = (n: number | null | undefined) => Number(n || 0).toLocaleString()

const PROCESSES = [
  '', 'Cutting', 'Printing', 'Embroidery', 'Stitching', 'Kajh Button', 'Handwork', 'Finishing', 'Packing',
]

const EXPORT_HEADERS = [
  'Txn Date', 'Type', 'Process', 'Qty', 'SO', 'JO', 'SKU', 'Component',
  'Vendor', 'From', 'To', 'Remarks',
]

/**
 * Cross-stage date-wise transaction ledger.
 * One row per create / receive / issue event with that transaction's qty.
 */
export default function ProcessDateTransactionsPanel() {
  const [filters, setFilters] = useState({
    txn_date: '',
    date_from: '',
    date_to: '',
    process: '',
    so_number: '',
    sku: '',
    jo_number: '',
    component: '',
    vendor_name: '',
    txn_type: '',
  })
  const [page, setPage] = useState(1)
  const params = useMemo(() => ({ ...filters, page, page_size: 200 }), [filters, page])

  const { data, isFetching } = useQuery({
    queryKey: ['process-date-txns', params],
    queryFn: () => api.get('/production/process-date-transactions', { params }).then(r => r.data),
  })

  const rows = data?.rows || []
  const totals = data?.totals || {}
  const total = Number(data?.total || 0)

  const set = (k: string, v: string) => {
    setPage(1)
    setFilters(f => ({ ...f, [k]: v }))
  }

  const exportAll = async () => {
    const res = await api.get('/production/process-date-transactions', {
      params: { ...filters, export: true, page_size: 0 },
    })
    const all = res.data?.rows || []
    downloadCsv(
      `process_date_txns_${new Date().toISOString().slice(0, 10)}.csv`,
      EXPORT_HEADERS,
      all.map((r: any) => [
        r.txn_date, r.txn_type, r.process, r.qty, r.so_number, r.jo_number, r.sku,
        r.component, r.vendor_name, r.from_process, r.to_process, r.remarks,
      ]),
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="font-semibold text-gray-800">Date-wise production transactions</h3>
          <p className="text-[11px] text-gray-500">
            Audit individual creates, receipts and issues across all stages — not cumulative JO balances.
            Example: Stitching JO of 100 with receipts 50 + 20 + 30 on three dates appears as three rows.
          </p>
        </div>
        <button type="button" onClick={() => void exportAll()} className="px-3 py-1.5 text-xs bg-[#002B5B] text-white rounded-lg">
          ↓ Export Excel (CSV)
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {[
          ['Created qty', fmt(totals.created_qty), 'text-[#002B5B]'],
          ['Received qty', fmt(totals.received_qty), 'text-emerald-700'],
          ['Issued qty', fmt(totals.issued_qty), 'text-blue-700'],
          ['Txn rows', fmt(totals.row_count), 'text-slate-700'],
        ].map(([l, v, c]) => (
          <div key={l} className="bg-white border rounded-lg p-2 text-center">
            <p className={`text-sm font-bold ${c}`}>{v}</p>
            <p className="text-[10px] text-gray-500">{l}</p>
          </div>
        ))}
      </div>

      <div className="bg-white border rounded-xl p-3 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2 text-xs">
        <label className="block">
          <span className="text-gray-500">Txn date (single day)</span>
          <input type="date" value={filters.txn_date} onChange={e => set('txn_date', e.target.value)}
            className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
        <label className="block">
          <span className="text-gray-500">From</span>
          <input type="date" value={filters.date_from} onChange={e => set('date_from', e.target.value)}
            className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
        <label className="block">
          <span className="text-gray-500">To</span>
          <input type="date" value={filters.date_to} onChange={e => set('date_to', e.target.value)}
            className="mt-0.5 w-full border rounded px-2 py-1" />
        </label>
        <label className="block">
          <span className="text-gray-500">Process</span>
          <select value={filters.process} onChange={e => set('process', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All processes</option>
            {PROCESSES.filter(Boolean).map(p => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <label className="block">
          <span className="text-gray-500">Txn type</span>
          <select value={filters.txn_type} onChange={e => set('txn_type', e.target.value)} className="mt-0.5 w-full border rounded px-2 py-1">
            <option value="">All</option>
            <option value="created">JO created</option>
            <option value="received">Received</option>
            <option value="issued">Issued / moved</option>
          </select>
        </label>
        {[
          ['so_number', 'SO No'], ['sku', 'SKU'], ['jo_number', 'JO No'],
          ['component', 'Component'], ['vendor_name', 'Vendor'],
        ].map(([k, label]) => (
          <label key={k} className="block">
            <span className="text-gray-500">{label}</span>
            <input type="text" value={(filters as any)[k]} onChange={e => set(k, e.target.value)}
              className="mt-0.5 w-full border rounded px-2 py-1" />
          </label>
        ))}
      </div>

      <div className="bg-white border rounded-xl overflow-auto">
        <div className="px-3 py-2 text-xs text-gray-500 flex justify-between">
          <span>{isFetching ? 'Loading…' : `${total.toLocaleString()} transactions`}</span>
          <span>Page {page}</span>
        </div>
        <table className="w-full text-[11px]">
          <thead className="bg-gray-50 text-gray-500 uppercase sticky top-0">
            <tr>
              {['Date', 'Type', 'Process', 'Qty', 'SO', 'JO', 'SKU', 'Comp', 'Vendor', 'From → To', 'Remarks'].map(h => (
                <th key={h} className="text-left px-2 py-2 whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r: any, i: number) => (
              <tr key={`${r.txn_type}-${r.ref_id}-${i}`} className="border-t border-gray-50 hover:bg-slate-50">
                <td className="px-2 py-1.5 font-mono">{r.txn_date || '—'}</td>
                <td className="px-2 py-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold ${
                    r.txn_type === 'received' ? 'bg-emerald-100 text-emerald-800'
                      : r.txn_type === 'issued' ? 'bg-blue-100 text-blue-800'
                        : 'bg-slate-100 text-slate-800'
                  }`}>{r.txn_type}</span>
                </td>
                <td className="px-2 py-1.5">{r.process || '—'}</td>
                <td className="px-2 py-1.5 text-right font-semibold">{fmt(r.qty)}</td>
                <td className="px-2 py-1.5 font-semibold text-[#002B5B]">{r.so_number || '—'}</td>
                <td className="px-2 py-1.5 font-mono">{r.jo_number || '—'}</td>
                <td className="px-2 py-1.5 font-mono">{r.sku || '—'}</td>
                <td className="px-2 py-1.5">{r.component || '—'}</td>
                <td className="px-2 py-1.5">{r.vendor_name || '—'}</td>
                <td className="px-2 py-1.5 text-[10px]">
                  {r.from_process || r.to_process ? `${r.from_process || '—'} → ${r.to_process || '—'}` : '—'}
                </td>
                <td className="px-2 py-1.5 text-gray-500 max-w-[12rem] truncate">{r.remarks || '—'}</td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr><td colSpan={11} className="text-center text-gray-400 py-8">No transactions for these filters.</td></tr>
            )}
          </tbody>
          {rows.length > 0 && (
            <tfoot className="bg-slate-100 border-t-2 border-slate-300 font-semibold">
              <tr>
                <td className="px-2 py-2" colSpan={3}>Totals (filtered)</td>
                <td className="px-2 py-2 text-right">
                  C {fmt(totals.created_qty)} · R {fmt(totals.received_qty)} · I {fmt(totals.issued_qty)}
                </td>
                <td className="px-2 py-2" colSpan={7} />
              </tr>
            </tfoot>
          )}
        </table>
        <div className="p-2 flex gap-2 justify-end">
          <button type="button" disabled={page <= 1} onClick={() => setPage(p => p - 1)} className="px-2 py-1 border rounded text-xs disabled:opacity-40">Prev</button>
          <button type="button" disabled={page * 200 >= total} onClick={() => setPage(p => p + 1)} className="px-2 py-1 border rounded text-xs disabled:opacity-40">Next</button>
        </div>
      </div>
    </div>
  )
}
