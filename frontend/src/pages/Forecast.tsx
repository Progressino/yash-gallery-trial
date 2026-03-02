import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'

interface QuarterlyRow {
  OMS_SKU: string
  ADS: number
  Units_90d: number
  Units_30d: number
  Status: string
  Avg_Monthly: number
  [key: string]: string | number
}

interface QuarterlyData {
  loaded: boolean
  columns: string[]
  rows: QuarterlyRow[]
}

const STATUS_COLOR: Record<string, string> = {
  'Fast Moving':  'bg-green-100 text-green-800',
  'Moderate':     'bg-blue-100 text-blue-800',
  'Slow Selling': 'bg-amber-100 text-amber-800',
  'Not Moving':   'bg-red-100 text-red-800',
}

export default function Forecast() {
  const [groupByParent, setGroupByParent] = useState(false)
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<string | null>(null)

  const { data, isLoading, refetch } = useQuery<QuarterlyData>({
    queryKey: ['quarterly-history', groupByParent],
    queryFn: () => api.get(`/data/quarterly-history?group_by_parent=${groupByParent}&n_quarters=8`).then(r => r.data),
    refetchInterval: false,
  })

  if (isLoading) return <div className="flex items-center justify-center h-full text-gray-400">Loading…</div>

  if (!data?.loaded) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
      <p className="text-5xl">📈</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">AI Forecast</h2>
      <p className="text-gray-500 max-w-sm">
        Build the combined sales dataset first to enable quarterly trend analysis.
      </p>
      <a href="/" className="text-sm text-blue-600 underline">Go to Dashboard →</a>
    </div>
  )

  const qCols = (data.columns ?? []).filter(c =>
    !['OMS_SKU','ADS','Units_90d','Units_30d','Status','Avg_Monthly','Freq_30d'].includes(c)
  )

  const rows = (data.rows ?? []).filter(r =>
    !search || String(r['OMS_SKU'] ?? '').toLowerCase().includes(search.toLowerCase())
  )

  const selectedRow = rows.find(r => r['OMS_SKU'] === selected)

  // Build trend chart data for selected SKU
  const trendData = selectedRow
    ? qCols.map(q => ({ quarter: q, units: Number(selectedRow[q] ?? 0) }))
    : []

  // Status breakdown
  const statusCounts = (data.rows ?? []).reduce<Record<string, number>>((acc, r) => {
    const s = String(r['Status'] ?? '')
    acc[s] = (acc[s] ?? 0) + 1
    return acc
  }, {})

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-bold text-[#002B5B]">📈 Quarterly Trend & Forecast</h2>
          <p className="text-gray-400 text-sm mt-1">
            {(data.rows ?? []).length.toLocaleString()} SKUs across {qCols.length} quarters
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer mt-1">
          <input
            type="checkbox" checked={groupByParent}
            onChange={e => { setGroupByParent(e.target.checked); refetch() }}
            className="rounded"
          />
          Group by Parent SKU
        </label>
      </div>

      {/* Status summary */}
      <div className="grid grid-cols-4 gap-3">
        {Object.entries(statusCounts).map(([status, count]) => (
          <div key={status} className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
            <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide mb-1">{status}</p>
            <p className="text-2xl font-bold text-[#002B5B]">{count}</p>
            <span className={`text-xs px-2 py-0.5 rounded-full mt-1 inline-block ${STATUS_COLOR[status] ?? ''}`}>
              {((count / (data.rows?.length ?? 1)) * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      {/* SKU trend detail */}
      {selected && selectedRow && trendData.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="font-semibold text-[#002B5B]">{selected}</h3>
              <p className="text-xs text-gray-400 mt-0.5">
                ADS: {Number(selectedRow['ADS']).toFixed(3)} &nbsp;·&nbsp;
                90d: {selectedRow['Units_90d']} units &nbsp;·&nbsp;
                Avg Monthly: {selectedRow['Avg_Monthly']}
              </p>
            </div>
            <button onClick={() => setSelected(null)} className="text-gray-400 hover:text-gray-600 text-lg">×</button>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="quarter" tick={{ fontSize: 10 }} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="units" stroke="#002B5B" strokeWidth={2} dot name="Units Sold" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Table */}
      <div className="flex items-center gap-3">
        <input
          value={search} onChange={e => setSearch(e.target.value)}
          placeholder="Search SKU…"
          className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-56 focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
        <span className="text-xs text-gray-400">{rows.length} SKUs</span>
        <button
          onClick={() => downloadCsv(data.rows ?? [], data.columns ?? [])}
          className="ml-auto text-xs px-3 py-1.5 rounded border border-gray-300 hover:bg-gray-50"
        >
          ⬇ Export CSV
        </button>
      </div>

      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-4 py-3 font-semibold text-gray-600 sticky left-0 bg-gray-50 whitespace-nowrap">OMS SKU</th>
              <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">ADS</th>
              <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">90d Units</th>
              <th className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Avg/Mo</th>
              <th className="text-left px-3 py-3 font-semibold text-gray-600 whitespace-nowrap">Status</th>
              {qCols.map(q => (
                <th key={q} className="text-right px-3 py-3 font-semibold text-gray-600 whitespace-nowrap text-xs">{q}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 300).map((row, i) => (
              <tr
                key={i}
                onClick={() => setSelected(prev => prev === row['OMS_SKU'] ? null : String(row['OMS_SKU']))}
                className={`border-b border-gray-100 cursor-pointer hover:bg-blue-50
                  ${selected === row['OMS_SKU'] ? 'bg-blue-100' : ''}`}
              >
                <td className="px-4 py-2 font-medium text-gray-900 sticky left-0 bg-inherit whitespace-nowrap">
                  {row['OMS_SKU']}
                </td>
                <td className="px-3 py-2 text-right text-gray-700">{Number(row['ADS']).toFixed(3)}</td>
                <td className="px-3 py-2 text-right text-gray-700">{row['Units_90d']}</td>
                <td className="px-3 py-2 text-right text-gray-700">{row['Avg_Monthly']}</td>
                <td className="px-3 py-2">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${STATUS_COLOR[String(row['Status'])] ?? ''}`}>
                    {row['Status']}
                  </span>
                </td>
                {qCols.map(q => (
                  <td key={q} className="px-3 py-2 text-right text-gray-600 text-xs">
                    {Number(row[q] ?? 0).toLocaleString()}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length > 300 && (
          <p className="text-xs text-gray-400 text-center py-2">Showing 300 of {rows.length}</p>
        )}
      </div>
      <p className="text-xs text-gray-400">Click a row to view quarterly trend chart.</p>
    </div>
  )
}

function downloadCsv(rows: QuarterlyRow[], columns: string[]) {
  const header = columns.join(',')
  const body = rows.map(r => columns.map(c => JSON.stringify(r[c] ?? '')).join(',')).join('\n')
  const blob = new Blob([header + '\n' + body], { type: 'text/csv' })
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob)
  a.download = 'quarterly_history.csv'; a.click()
}
