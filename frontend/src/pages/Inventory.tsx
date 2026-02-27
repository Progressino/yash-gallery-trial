import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

interface InventoryData {
  loaded: boolean
  rows: Array<Record<string, number | string>>
  columns: string[]
}

export default function Inventory() {
  const [search, setSearch] = useState('')

  const { data, isLoading } = useQuery<InventoryData>({
    queryKey: ['inventory'],
    queryFn: () => api.get('/data/inventory').then(r => r.data),
    refetchInterval: 30000,
  })

  if (isLoading) return <div className="flex items-center justify-center h-full text-gray-400">Loading…</div>

  if (!data?.loaded || data.rows.length === 0) return (
    <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8">
      <p className="text-5xl">📦</p>
      <h2 className="text-2xl font-bold text-[#002B5B]">Inventory</h2>
      <p className="text-gray-500 max-w-sm">
        Upload OMS, Flipkart, Myntra, or Amazon inventory CSVs from the Dashboard.
      </p>
      <a href="/" className="text-sm text-blue-600 underline">Go to Dashboard →</a>
    </div>
  )

  const filtered = data.rows.filter(r =>
    !search || String(r['OMS_SKU'] ?? '').toLowerCase().includes(search.toLowerCase())
  )

  const invCols = (data.columns ?? []).filter(c => c !== 'OMS_SKU')
  const totalInventory = data.rows.reduce((s, r) => s + Number(r['Total_Inventory'] ?? 0), 0)
  const totalSkus = data.rows.length
  const zeroStock = data.rows.filter(r => Number(r['Total_Inventory'] ?? 0) <= 0).length

  return (
    <div className="p-6 space-y-5">
      <div>
        <h2 className="text-2xl font-bold text-[#002B5B]">📦 Inventory</h2>
        <p className="text-gray-400 text-sm mt-1">{totalSkus.toLocaleString()} active SKUs</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Total Units</p>
          <p className="text-2xl font-bold text-[#002B5B] mt-1">{totalInventory.toLocaleString()}</p>
        </div>
        <div className="bg-white rounded-xl border border-gray-200 p-4 shadow-sm">
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Active SKUs</p>
          <p className="text-2xl font-bold text-[#002B5B] mt-1">{totalSkus.toLocaleString()}</p>
        </div>
        <div className={`bg-white rounded-xl border p-4 shadow-sm ${zeroStock > 0 ? 'border-red-300' : 'border-gray-200'}`}>
          <p className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Out of Stock</p>
          <p className={`text-2xl font-bold mt-1 ${zeroStock > 0 ? 'text-red-600' : 'text-[#002B5B]'}`}>{zeroStock}</p>
        </div>
      </div>

      {/* Search */}
      <div>
        <input
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Search SKU…"
          className="w-full max-w-xs border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-300"
        />
        {search && <span className="text-xs text-gray-400 ml-2">{filtered.length} results</span>}
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 border-b border-gray-200">
              <th className="text-left px-4 py-3 font-semibold text-gray-600 sticky left-0 bg-gray-50 whitespace-nowrap">OMS SKU</th>
              {invCols.map(col => (
                <th key={col} className="text-right px-4 py-3 font-semibold text-gray-600 whitespace-nowrap">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 200).map((row, i) => {
              const total = Number(row['Total_Inventory'] ?? 0)
              return (
                <tr key={i} className={`border-b border-gray-100 hover:bg-blue-50 ${total <= 0 ? 'bg-red-50' : ''}`}>
                  <td className="px-4 py-2 font-medium text-gray-800 sticky left-0 bg-inherit whitespace-nowrap">
                    {row['OMS_SKU']}
                  </td>
                  {invCols.map(col => (
                    <td key={col} className="px-4 py-2 text-right text-gray-700">
                      {Number(row[col] ?? 0).toLocaleString()}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
        {filtered.length > 200 && (
          <p className="text-xs text-gray-400 text-center py-2">Showing 200 of {filtered.length} rows</p>
        )}
      </div>
    </div>
  )
}
